"""
Phase 5, step 4 -- the deliverable trio: `ami_raw` (site-level ground
truth), `ami_meter` (the synthetic smart meter), and `ami_raw_phaseseparate`
(per-phase ground truth, PV-allocated).
==============================================================================

Column/unit convention: this module follows `structured_data`'s naming style
(site_id, t_stamp, year, month, V, P_kw/P_kw_norm, Q_kvar, kWh registers)
rather than the raw `ts` units -- every W/var/Wh source column is divided by
1000 here. PV generation is normalized by site capacity (`S_99` or
`ac_capacity_kw`, matching `structured_data`'s own `normalization_basis`
choice) because output scales with installed capacity and there is a real
capacity metric behind it; load is NOT normalized, because there is no
equivalent capacity metric for a load circuit in `meta_up23c` -- `P_kw` for
load stays an absolute reading.

`ami_raw` gives a disaggregation algorithm's VALIDATOR everything it needs
to grade against: `P_kw_norm` (PV generation, capacity-normalized -- Phase 3
Section 9 already confirmed the source circuit reads as pure generation, no
correction needed), `P_kw` (`gross_load` -- the reconstruction formula the
user's real plots confirmed: `ac_load_net` signed and summed across phases,
plus `pv_site_net` signed), `V` (site voltage), and their reactive-power
counterparts `Q_kvar_norm`/`Q_kvar` -- built the same way but WITHOUT the
Phase 3 validation `P_kw`/`P_kw_norm` have (there is no night-time reactive-
power diagnostic), so `Q_kvar_norm` is signed via `circuit_polarity` rather
than assumed unsigned. All are site-level -- PV has no per-phase signal to
preserve, and `gross_load`/`gross_reactive_load` only mean anything once
every load phase and the PV side are combined.

`ami_meter` is what a disaggregation ALGORITHM actually gets to see: each
surviving `ac_load_net` circuit's own reading, kept PER PHASE (a real
3-phase meter reports phases separately, this doesn't pre-sum them), using
the RAW (uncorrected) `power` reading -- `ac_load_net` is already net-of-PV
in the sign convention as landed, so `ami_meter.P_kw` is what a real meter
would show, polarity quirks and all. `energy_import_kwh`/`energy_export_kwh`
are the real measured registers (not a derived W-clip split), matching a
genuine smart meter's import/export accumulators.

`ami_raw_phaseseparate` answers "what would the raw circuit readings look
like broken out by phase" -- one row per (site_id, device_id, circuit_id,
t_stamp) for EVERY kept `ac_load_net`/`pv_site_net` circuit, load AND PV
alike, tagged by `circuit_type`. `P_kw_signed`/`Q_kvar_signed` are
polarity-corrected (via `circuit_polarity`, the same correction `ami_raw`'s
own `gross_load`/`gross_reactive_load` reconstruction applies) but are
DELIBERATELY not the same values as `ami_meter.P_kw`/`Q_kvar`, which are
the raw, uncorrected sensor readings a real meter would show -- these two
tables answer different questions and are not meant to share a sign
convention. `V` is the raw per-circuit voltage (no polarity correction --
voltage has no sign ambiguity). This table does NOT allocate or split PV
across load phases, and does NOT compute a per-phase gross-load
reconciliation -- an earlier version did both, but that baked a debatable
per-site heuristic (splitting PV evenly across load phases whenever
circuit counts don't match) into a numeric column that looked like
measured ground truth. A load row's `P_kw_signed` is `ac_load_net`'s own
net-of-PV reading (see `build_ami_raw`'s docstring); a PV row's
`P_kw_signed` is that PV circuit's own undivided generation reading. Want
a PV-independent per-phase gross load? Compute it explicitly from these
raw rows, making your own matching choice for mismatched-count sites; for
the SITE-level equivalent (a different, validated method, not this
per-phase heuristic) use `ami_raw.P_kw`/`Q_kvar` directly.

`n_phases_at_site` (from the LOAD side's circuit count) and
`pv_allocation_method` are kept as lightweight, per-site descriptive tags,
decided once per site from that MONTH's kept circuit sets and copied onto
EVERY row for that site -- load or PV alike -- since they describe the
site's circuit topology, not a load-specific derived quantity: if the
number of kept `pv_site_net` circuits equals the number of kept
`ac_load_net` circuits, the site gets `"direct_matched_circuit"` (real,
unambiguous 1:1 circuit counts, though NOT a verified circuit-to-phase
(A/B/C) label -- see the project README on why even a count match is a
heuristic, not a proven pairing); otherwise (most commonly: one
`pv_site_net` circuit serving a multi-phase load) `"equal_split_across_load_phases"`;
a site with no surviving PV circuit gets `"no_pv_present"`. No Watts are
allocated based on this tag any more -- it's audit metadata only.

All three tables are built ONE LANDED MONTH AT A TIME (`run_build`,
`run_phase_split_build`), mirroring `ami_extract`/`ami_revalidate`'s
discipline: never hold more than one month's interval table in memory,
write each month's output immediately, and never guess at a store path --
the caller supplies where to write.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ami_resolution as Resolution
from . import ami_signal as Signal

__all__ = [
    "build_ami_raw",
    "build_ami_meter",
    "build_ami_raw_phaseseparate",
    "write_month_table",
    "run_build",
    "run_phase_split_build",
]


def _year_month(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """UTC year/month INT columns from a tz-aware timestamp series."""
    idx = pd.DatetimeIndex(series)
    return idx.year, idx.month


def build_ami_raw(
    interval_table: pd.DataFrame, circuit_polarity: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    time_column: str = "t_stamp",
    power_column: str = "power",
    reactive_power_column: str = "reactive_power",
    voltage_column: str = "voltage",
    pv_type: str = "pv_site_net",
    load_type: str = "ac_load_net",
    site_capacity: pd.DataFrame | None = None,
    normalization_basis: str = "s_99",
) -> pd.DataFrame:
    """
    One row per (site_id, t_stamp) -- house-load and PV-system columns kept
    SEPARATE, at site level, in two symmetric trios:

    - Load side: `P_load_kw`, `Q_load_kvar`, `V_load`.
    - PV side:   `P_pv_kw`,   `Q_pv_kvar`,   `V_pv`.

    All four power/reactive-power columns are `circuit_polarity`-corrected
    and SIGNED -- `P_load_kw`/`Q_load_kvar` are `ac_load_net`, summed across
    the site's load phases; `P_pv_kw`/`Q_pv_kvar` are `pv_site_net`, summed
    across the site's PV circuits. Both reuse
    `ami_signal.reconstruct_gross_load`'s own `load_signed`/`pv_signed`
    output -- the same validated per-side sums the `gross_load`
    reconstruction is built from, just exposed directly instead of only
    being visible pre-summed.

    IMPORTANT sign-convention note (fixed here): earlier versions of this
    function summed PV power UNSIGNED (raw `power`, no `circuit_polarity`
    applied), on the theory that Phase 3 Section 9 already confirmed PV
    needs no correction. That reading of Section 9 was wrong: Section 9
    confirmed `pv_site_net` is a generation-only signal (no LOAD netted
    into it) -- it says nothing about SIGN. Raw `power` for a PV circuit
    reads NEGATIVE while generating (confirmed against real fleet data),
    so summing it unsigned made `P_pv_kw`/`P_kw_norm` come out negative
    during generation -- inconsistent with `Q_kvar_norm` and `gross_load`,
    which already applied `circuit_polarity` correctly. `P_pv_kw` (and
    therefore `P_kw_norm`, which divides it by capacity) now applies the
    same `circuit_polarity` correction as everything else in this table,
    so it reads POSITIVE while the site is generating, matching
    `Q_pv_kvar`. If you compared `P_kw_norm` against an earlier build,
    expect it to have flipped sign.

    `P_kw`/`Q_kvar` (`gross_load`/`gross_reactive_load`) are kept as
    convenience columns -- `P_kw = P_load_kw + P_pv_kw` by construction
    (the "what would this site have consumed with no solar" counterfactual
    demand), not a third independent quantity.

    `P_kw_norm`/`Q_kvar_norm` are `P_pv_kw`/`Q_pv_kvar` divided by site
    capacity -- need `site_capacity` (a frame with `site_id`, `S_99`,
    `ac_capacity_kw`, one row per site, already deduplicated by the
    caller); without it, those two columns (and `S_99`/`ac_capacity_kw`
    themselves) are present but entirely null, so the column shape stays
    stable whether or not capacity metadata was supplied.
    `normalization_basis` selects which of `S_99`/`ac_capacity_kw` is the
    denominator; a site with a zero or missing value there gets a null
    normalized value, not a divide-by-zero inf.

    Reactive-power columns (`Q_load_kvar`, `Q_pv_kvar`, `Q_kvar_norm`,
    `Q_kvar`) carry a validation caveat active power doesn't: only the
    active-power reconstruction has been checked against real plots
    (Phase 4). Treat reactive power as reasonable-by-construction, not
    independently confirmed.

    `V_load` is the site's load-side voltage, averaged across kept
    `ac_load_net` circuits at that timestamp (matching `structured_data`'s
    own aggregation convention, just applied to the load side instead of
    the PV side it uses). `V_pv` is the analogous average across the
    site's kept `pv_site_net` circuits -- the inverter/PV-side voltage,
    which can genuinely differ from `V_load` (local voltage rise from the
    site's own export).

    Inner-joined on (site_id, t_stamp): a timestamp where only one side
    reported contributes no row.
    """
    columns = [
        site_column, time_column, "year", "month",
        "V_load", "V_pv",
        "P_load_kw", "P_pv_kw", "P_kw_norm", "P_kw",
        "Q_load_kvar", "Q_pv_kvar", "Q_kvar_norm", "Q_kvar",
        "S_99", "ac_capacity_kw", "normalization_basis",
    ]
    if interval_table is None or not len(interval_table):
        return pd.DataFrame(columns=columns)

    reconstructed_p = Signal.reconstruct_gross_load(
        interval_table, circuit_polarity,
        site_column=site_column, circuit_column=circuit_column,
        type_column=type_column, time_column=time_column, power_column=power_column,
    )
    p_frame = reconstructed_p[
        [site_column, time_column, "load_signed", "pv_signed", "reconstructed_load"]
    ].rename(columns={
        "load_signed": "load_w", "pv_signed": "pv_generation_w",
        "reconstructed_load": "gross_load_w",
    })
    merged = p_frame.dropna(subset=["load_w", "pv_generation_w"])
    if not len(merged):
        return pd.DataFrame(columns=columns)

    if reactive_power_column in interval_table.columns:
        reconstructed_q = Signal.reconstruct_gross_load(
            interval_table, circuit_polarity,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, time_column=time_column, power_column=reactive_power_column,
        )
        q_frame = reconstructed_q[
            [site_column, time_column, "load_signed", "pv_signed", "reconstructed_load"]
        ].rename(columns={
            "load_signed": "load_var", "pv_signed": "pv_reactive_generation_var",
            "reconstructed_load": "gross_reactive_load_var",
        })
    else:
        q_frame = pd.DataFrame(columns=[
            site_column, time_column, "load_var", "pv_reactive_generation_var",
            "gross_reactive_load_var",
        ])
    merged = merged.merge(q_frame, on=[site_column, time_column], how="left")

    load_side = interval_table[interval_table[type_column] == load_type]
    if len(load_side) and voltage_column in load_side.columns:
        v_load_agg = (
            load_side.groupby([site_column, time_column])[voltage_column]
            .mean()
            .reset_index()
            .rename(columns={voltage_column: "V_load"})
        )
        merged = merged.merge(v_load_agg, on=[site_column, time_column], how="left")
    else:
        merged["V_load"] = np.nan

    pv_side = interval_table[interval_table[type_column] == pv_type]
    if len(pv_side) and voltage_column in pv_side.columns:
        v_pv_agg = (
            pv_side.groupby([site_column, time_column])[voltage_column]
            .mean()
            .reset_index()
            .rename(columns={voltage_column: "V_pv"})
        )
        merged = merged.merge(v_pv_agg, on=[site_column, time_column], how="left")
    else:
        merged["V_pv"] = np.nan

    merged["P_load_kw"] = merged["load_w"] / 1000.0
    merged["P_pv_kw"] = merged["pv_generation_w"] / 1000.0
    merged["P_kw"] = merged["gross_load_w"] / 1000.0
    merged["Q_load_kvar"] = merged["load_var"] / 1000.0
    merged["Q_pv_kvar"] = merged["pv_reactive_generation_var"] / 1000.0
    merged["Q_kvar"] = merged["gross_reactive_load_var"] / 1000.0
    merged["year"], merged["month"] = _year_month(merged[time_column])

    if site_capacity is not None and len(site_capacity):
        merged = merged.merge(
            site_capacity[[site_column, "S_99", "ac_capacity_kw"]],
            on=site_column, how="left",
        )
    else:
        merged["S_99"] = np.nan
        merged["ac_capacity_kw"] = np.nan

    normalization_col = "S_99" if normalization_basis == "s_99" else "ac_capacity_kw"
    denominator = merged[normalization_col].where(merged[normalization_col] > 0)
    merged["P_kw_norm"] = merged["P_pv_kw"] / denominator
    merged["Q_kvar_norm"] = merged["Q_pv_kvar"] / denominator
    merged["normalization_basis"] = normalization_basis

    return merged[columns].sort_values([site_column, time_column]).reset_index(drop=True)


def build_ami_meter(
    interval_table: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    device_column: str = "device_id",
    type_column: str = "circuit_type",
    time_column: str = "t_stamp",
    power_column: str = "power",
    reactive_power_column: str = "reactive_power",
    voltage_column: str = "voltage",
    current_column: str = "current",
    power_factor_column: str = "power_factor",
    energy_import_column: str = "energy_import",
    energy_export_column: str = "energy_export",
    load_type: str = "ac_load_net",
) -> pd.DataFrame:
    """
    One row per (site_id, device_id, circuit_id, t_stamp): each surviving
    `load_type` circuit's own RAW reading -- no polarity correction here,
    `ac_load_net` is already net-of-PV as landed, and this table is meant
    to be what a real meter would show, quirks included (see module
    docstring on how this differs from `ami_raw_phaseseparate`). Every
    source column beyond `power`/`reactive_power` is passed straight
    through when present, and simply omitted (not erroring) when the
    interval table doesn't have it.
    """
    columns = [
        site_column, device_column, circuit_column, time_column, "year", "month",
        "V", "P_kw", "Q_kvar", "S_kva", "power_factor", "current_a",
        "energy_import_kwh", "energy_export_kwh",
    ]
    if interval_table is None or not len(interval_table):
        return pd.DataFrame(columns=columns)

    load_side = interval_table[interval_table[type_column] == load_type].copy()
    if not len(load_side):
        return pd.DataFrame(columns=columns)

    load_side["P_kw"] = load_side[power_column] / 1000.0
    if reactive_power_column in load_side.columns:
        load_side["Q_kvar"] = load_side[reactive_power_column] / 1000.0
        load_side["S_kva"] = np.sqrt(load_side["P_kw"] ** 2 + load_side["Q_kvar"] ** 2)
    if voltage_column in load_side.columns:
        load_side["V"] = load_side[voltage_column]
    if current_column in load_side.columns:
        load_side["current_a"] = load_side[current_column]
    if power_factor_column in load_side.columns:
        load_side["power_factor"] = load_side[power_factor_column]
    if energy_import_column in load_side.columns:
        load_side["energy_import_kwh"] = load_side[energy_import_column] / 1000.0
    if energy_export_column in load_side.columns:
        load_side["energy_export_kwh"] = load_side[energy_export_column] / 1000.0

    load_side["year"], load_side["month"] = _year_month(load_side[time_column])

    present_columns = [c for c in columns if c in load_side.columns]
    return (
        load_side[present_columns]
        .sort_values([site_column, circuit_column, time_column])
        .reset_index(drop=True)
    )


def build_ami_raw_phaseseparate(
    interval_table: pd.DataFrame, circuit_polarity: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    device_column: str = "device_id",
    type_column: str = "circuit_type",
    time_column: str = "t_stamp",
    power_column: str = "power",
    reactive_power_column: str = "reactive_power",
    voltage_column: str = "voltage",
    polarity_column: str = "circuit_polarity",
    load_type: str = "ac_load_net",
    pv_type: str = "pv_site_net",
) -> pd.DataFrame:
    """
    One row per (site_id, device_id, circuit_id, t_stamp), for EVERY kept
    `load_type`/`pv_type` circuit -- load AND PV circuits both get their
    own row, tagged by `circuit_type`. This is deliberately a thin,
    circuit-preserving view: `P_kw_signed`/`Q_kvar_signed` are that
    circuit's OWN reading, polarity-corrected (see module docstring for
    why this deliberately differs from `ami_meter.P_kw`/`Q_kvar`, which
    are raw/uncorrected). `V` is the raw per-circuit voltage (not
    polarity-corrected -- voltage has no sign-convention ambiguity).

    On a `load_type` row, `P_kw_signed` is `ac_load_net`'s own reading --
    already net-of-PV as landed (see `build_ami_raw`'s docstring), NOT a
    PV-independent gross load figure. On a `pv_type` row, `P_kw_signed` is
    that PV circuit's own generation reading, undivided. This table does
    NOT allocate or split PV across load phases, and does NOT compute a
    per-phase gross-load reconciliation -- earlier versions did both, but
    that baked a debatable heuristic (`equal_split_across_load_phases`,
    for the common case where PV/load circuit counts don't match) into a
    numeric column that looked like measured ground truth. If you need a
    PV-independent per-phase gross load, compute it explicitly yourself
    from these raw load/PV rows, making your own matching choice for
    mismatched-count sites; for the SITE-level equivalent (which uses a
    different, validated method, not this per-phase heuristic), use
    `ami_raw.P_kw`/`Q_kvar` directly.

    `n_phases_at_site` (from the LOAD side's circuit count) and
    `pv_allocation_method` (`direct_matched_circuit` when kept PV circuit
    count equals kept load circuit count, `equal_split_across_load_phases`
    when it doesn't, `no_pv_present` when there's no surviving PV circuit
    at all) are kept as lightweight, per-site descriptive tags -- decided
    once per site (same rule as before) and copied onto EVERY row for that
    site, load or PV alike, since they describe the site's circuit
    topology, not a load-specific derived quantity. They carry no
    allocated Watts with them any more.
    """
    columns = [
        site_column, device_column, circuit_column, type_column, time_column,
        "year", "month", "V", "P_kw_signed", "Q_kvar_signed",
        "n_phases_at_site", "pv_allocation_method",
    ]
    if interval_table is None or not len(interval_table):
        return pd.DataFrame(columns=columns)

    frame = interval_table.merge(circuit_polarity, on=circuit_column, how="left")
    frame["power_signed"] = frame[power_column] * frame[polarity_column]
    if reactive_power_column in frame.columns:
        frame["reactive_power_signed"] = frame[reactive_power_column] * frame[polarity_column]
    else:
        frame["reactive_power_signed"] = np.nan

    load_side = frame[frame[type_column] == load_type].copy()
    if not len(load_side):
        return pd.DataFrame(columns=columns)
    pv_side = frame[frame[type_column] == pv_type].copy()

    load_circuits_by_site = load_side.groupby(site_column)[circuit_column].unique()
    pv_circuits_by_site = (
        pv_side.groupby(site_column)[circuit_column].unique() if len(pv_side) else pd.Series(dtype=object)
    )

    site_plan_rows = []
    for site_id, load_ids in load_circuits_by_site.items():
        n_phases = len(sorted(load_ids))
        pv_ids_sorted = (
            sorted(pv_circuits_by_site.loc[site_id])
            if site_id in getattr(pv_circuits_by_site, "index", []) else []
        )
        if not pv_ids_sorted:
            method = "no_pv_present"
        elif len(pv_ids_sorted) == n_phases:
            method = "direct_matched_circuit"
        else:
            method = "equal_split_across_load_phases"
        site_plan_rows.append({
            site_column: site_id, "n_phases_at_site": n_phases, "pv_allocation_method": method,
        })
    site_plan = pd.DataFrame(site_plan_rows)

    rows = pd.concat([load_side, pv_side], ignore_index=True) if len(pv_side) else load_side.copy()
    rows["P_kw_signed"] = rows["power_signed"] / 1000.0
    rows["Q_kvar_signed"] = rows["reactive_power_signed"] / 1000.0
    if voltage_column in rows.columns:
        rows["V"] = rows[voltage_column]
    else:
        rows["V"] = np.nan
    rows = rows.merge(site_plan, on=site_column, how="left")

    rows["year"], rows["month"] = _year_month(rows[time_column])

    return (
        rows[columns]
        .sort_values([site_column, circuit_column, time_column])
        .reset_index(drop=True)
    )

def write_month_table(
    frame: pd.DataFrame, store_dir, year: int, month: int, *,
    partition_key: str = "dt_month",
    compression: str = "zstd",
) -> Path | None:
    """
    Write one month's already-built output table to
    `<store_dir>/<partition_key>=YYYY-MM/part-0000.parquet`.

    `store_dir` is treated as ALREADY the table's own directory (e.g.
    `ami_config.store_path("ami_raw")`) -- this function does not append a
    table-name segment of its own.

    Returns the path written, or None for an empty frame.
    """
    if frame is None or not len(frame):
        return None
    partition_dir = Path(store_dir) / f"{partition_key}={year}-{month:02d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    path = partition_dir / "part-0000.parquet"
    frame.to_parquet(path, compression=compression, index=False)
    return path


def run_build(
    month_frames, resolution: pd.DataFrame, circuit_polarity: pd.DataFrame,
    ami_raw_dir, ami_meter_dir, *,
    site_capacity: pd.DataFrame | None = None,
    normalization_basis: str = "s_99",
    apply_power_correction: bool = True,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    device_column: str = "device_id",
    time_column: str = "t_stamp",
    partition_key: str = "dt_month",
) -> pd.DataFrame:
    """
    Orchestrate `ami_raw` + `ami_meter` for every (year, month, frame) in
    `month_frames`, one month at a time. `site_capacity` is optional and
    keyword-only: omitting it still produces both tables, just with
    `ami_raw.P_kw_norm`/`S_99`/`ac_capacity_kw` left null -- this keeps a
    call written before capacity metadata was wired in still working
    unchanged.

    `apply_power_correction` is passed straight through to
    `Resolution.build_interval_table` (see its docstring) -- default True
    keeps the validated Phase 4 device/meter-model correction; pass False
    to use raw `power` for every circuit unconditionally, matching
    `structured_data`'s own treatment. To exclude the sites this correction
    affects entirely rather than just changing how they're treated, filter
    `resolution` down first via `Resolution.sites_with_power_correction`
    before calling this function -- see that helper's docstring for the
    exact pattern.

    Returns a provenance DataFrame, one row per month.
    """
    provenance_rows = []
    for year, month, frame in month_frames:
        interval_table = Resolution.build_interval_table(
            frame, resolution,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, device_column=device_column,
            time_column=time_column,
            apply_power_correction=apply_power_correction,
        )
        ami_raw = build_ami_raw(
            interval_table, circuit_polarity,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, time_column=time_column,
            site_capacity=site_capacity, normalization_basis=normalization_basis,
        )
        ami_meter = build_ami_meter(
            interval_table,
            site_column=site_column, circuit_column=circuit_column,
            device_column=device_column, type_column=type_column,
            time_column=time_column,
        )
        raw_path = write_month_table(ami_raw, ami_raw_dir, year, month, partition_key=partition_key)
        meter_path = write_month_table(ami_meter, ami_meter_dir, year, month, partition_key=partition_key)
        provenance_rows.append({
            "year": year, "month": month,
            "n_raw_rows": len(ami_raw), "n_meter_rows": len(ami_meter),
            "raw_path": raw_path, "meter_path": meter_path,
        })
        del interval_table, ami_raw, ami_meter

    return pd.DataFrame(provenance_rows)


def run_phase_split_build(
    month_frames, resolution: pd.DataFrame, circuit_polarity: pd.DataFrame,
    ami_raw_phaseseparate_dir, *,
    apply_power_correction: bool = True,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    device_column: str = "device_id",
    time_column: str = "t_stamp",
    partition_key: str = "dt_month",
) -> pd.DataFrame:
    """
    Orchestrate `ami_raw_phaseseparate` for every (year, month, frame) in
    `month_frames`, one month at a time -- same discipline as `run_build`,
    kept as a separate entry point so it can be run independently (e.g.
    against already-landed data, without re-running the `ami_raw`/
    `ami_meter` build).

    `apply_power_correction` is passed straight through to
    `Resolution.build_interval_table` -- see `run_build`'s docstring for
    what it does and how to instead exclude affected sites entirely via
    `Resolution.sites_with_power_correction`. Pass the SAME value here as
    you did to `run_build` if you want `ami_raw_phaseseparate` to reconcile
    with `ami_raw`/`ami_meter` for the same site/timestamp -- the two
    orchestrators don't share state, so a mismatched setting between them
    would silently break that reconciliation property.

    Returns a provenance DataFrame, one row per month: `year, month,
    n_rows, path`.
    """
    provenance_rows = []
    for year, month, frame in month_frames:
        interval_table = Resolution.build_interval_table(
            frame, resolution,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, device_column=device_column,
            time_column=time_column,
            apply_power_correction=apply_power_correction,
        )
        phase_split = build_ami_raw_phaseseparate(
            interval_table, circuit_polarity,
            site_column=site_column, circuit_column=circuit_column,
            device_column=device_column, type_column=type_column,
            time_column=time_column,
        )
        path = write_month_table(phase_split, ami_raw_phaseseparate_dir, year, month, partition_key=partition_key)
        provenance_rows.append({"year": year, "month": month, "n_rows": len(phase_split), "path": path})
        del interval_table, phase_split

    return pd.DataFrame(provenance_rows)
