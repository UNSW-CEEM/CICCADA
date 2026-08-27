"""
Phase 5, step 4 -- the deliverable pair: `ami_raw` (ground truth, separate
signals) and `ami_meter` (the synthetic smart meter).
=====================================================================

`ami_raw` gives a disaggregation algorithm's VALIDATOR the two things it
needs to grade against: `pv_generation` (already confirmed generation-only,
non-negative -- Phase 3 Section 9, no correction needed) and `gross_load`
(the reconstruction formula the user's real plots confirmed:
`ac_load_net (signed, summed across phases) + pv_site_net (signed)`). Both
are site-level, not per-circuit -- `gross_load` in particular only means
anything once every phase and the PV side are combined, so unlike the
interval-level table this is deliberately an aggregate, computed once,
here, as the answer key.

`ami_meter` is what a disaggregation ALGORITHM actually gets to see: each
surviving `ac_load_net` circuit's own reading, kept PER PHASE (per the
user's explicit call -- a real 3-phase meter reports phases separately,
this doesn't pre-sum them), in load convention (`ac_load_net` is already a
net reading, confirmed net-of-PV -- no sign flip needed). `net_import_w`/
`net_export_w` split the signed reading into the two non-negative registers
a real meter carries, but stay in WATTS, not kWh -- `ami_config`'s target
AMI interval (`TARGET_INTERVAL_MINUTES`) is explicitly still
`TARGET_INTERVAL_RESOLVED = False`, so resampling/kWh conversion is a
separate, not-yet-made decision this module does not force.

Both tables are built ONE LANDED MONTH AT A TIME (`run_build`), mirroring
`ami_extract`/`ami_revalidate`'s discipline: never hold more than one
month's interval table in memory, write each month's output immediately,
and never guess at a store path -- the caller supplies where to write.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import ami_resolution as Resolution
from . import ami_signal as Signal

__all__ = [
    "build_ami_raw",
    "build_ami_meter",
    "write_month_table",
    "run_build",
]


def build_ami_raw(
    interval_table: pd.DataFrame, circuit_polarity: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    time_column: str = "t_stamp",
    power_column: str = "power",
    pv_type: str = "pv_site_net",
) -> pd.DataFrame:
    """
    One row per (site_id, t_stamp): `pv_generation` (raw, unsigned -- the
    sum of any surviving `pv_type` circuits at that site; Section 9 already
    confirmed this needs no polarity correction to read as generation) and
    `gross_load` (`ami_signal.reconstruct_gross_load`'s `reconstructed_load`
    -- the ONLY place polarity is applied, since that formula needs both
    sides signed correctly to combine).

    Inner-joined on (site_id, t_stamp): a timestamp where only one side
    reported (a real, if partial, day-coverage gap -- Phase 3 found these)
    contributes no row here rather than a half-populated one, since a
    disaggregation ground-truth row needs BOTH signals to mean anything.
    That drop is silent at this function's level BY DESIGN -- callers that
    care about coverage loss should compare `len(result)` against
    `interval_table`'s own per-site timestamp counts themselves, not rely
    on this function to report it.
    """
    columns = [site_column, time_column, "pv_generation", "gross_load"]
    if interval_table is None or not len(interval_table):
        return pd.DataFrame(columns=columns)

    pv_side = interval_table[interval_table[type_column] == pv_type]
    if len(pv_side):
        pv_generation = (
            pv_side.groupby([site_column, time_column])[power_column]
            .sum()
            .reset_index()
            .rename(columns={power_column: "pv_generation"})
        )
    else:
        pv_generation = pd.DataFrame(columns=[site_column, time_column, "pv_generation"])

    reconstructed = Signal.reconstruct_gross_load(
        interval_table, circuit_polarity,
        site_column=site_column, circuit_column=circuit_column,
        type_column=type_column, time_column=time_column, power_column=power_column,
    )
    gross_load = reconstructed[[site_column, time_column, "reconstructed_load"]].rename(
        columns={"reconstructed_load": "gross_load"}
    )

    merged = pv_generation.merge(gross_load, on=[site_column, time_column], how="inner")
    merged = merged.dropna(subset=["pv_generation", "gross_load"])
    return merged[columns].sort_values([site_column, time_column]).reset_index(drop=True)


def build_ami_meter(
    interval_table: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    device_column: str = "device_id",
    type_column: str = "circuit_type",
    time_column: str = "t_stamp",
    power_column: str = "power",
    load_type: str = "ac_load_net",
) -> pd.DataFrame:
    """
    One row per (site_id, device_id, circuit_id, t_stamp): each surviving
    `load_type` circuit's own reading, kept PER PHASE (not summed across a
    site's circuits -- see module docstring). `net_power_w` is the raw,
    already-net (load-convention) reading; `net_import_w`/`net_export_w`
    split it into the two non-negative registers a real meter carries
    (`max(net, 0)` / `max(-net, 0)`), still in Watts -- no kWh/interval
    conversion here (see module docstring on `TARGET_INTERVAL_RESOLVED`).
    """
    columns = [
        site_column, device_column, circuit_column, time_column,
        "net_power_w", "net_import_w", "net_export_w",
    ]
    if interval_table is None or not len(interval_table):
        return pd.DataFrame(columns=columns)

    load_side = interval_table[interval_table[type_column] == load_type].copy()
    if not len(load_side):
        return pd.DataFrame(columns=columns)

    load_side["net_power_w"] = load_side[power_column]
    load_side["net_import_w"] = load_side["net_power_w"].clip(lower=0.0)
    load_side["net_export_w"] = (-load_side["net_power_w"]).clip(lower=0.0)

    return (
        load_side[columns]
        .sort_values([site_column, circuit_column, time_column])
        .reset_index(drop=True)
    )


def write_month_table(
    frame: pd.DataFrame, store_dir, year: int, month: int, *,
    table_name: str,
    partition_key: str = "dt_month",
    compression: str = "zstd",
) -> Path | None:
    """
    Write one month's already-built output table to
    `<store_dir>/<table_name>/<partition_key>=YYYY-MM/part-0000.parquet`.

    Returns the path written, or None for an empty frame (nothing is
    written -- an empty file would otherwise silently inflate a manifest's
    file count for no reason). One file per month is enough here -- unlike
    `ami_extract`'s landing chunks, `ami_raw`/`ami_meter` are already the
    small, derived, per-month aggregate, not the raw multi-chunk pull.
    """
    if frame is None or not len(frame):
        return None
    partition_dir = Path(store_dir) / table_name / f"{partition_key}={year}-{month:02d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    path = partition_dir / "part-0000.parquet"
    frame.to_parquet(path, compression=compression, index=False)
    return path


def run_build(
    month_frames, resolution: pd.DataFrame, circuit_polarity: pd.DataFrame,
    ami_raw_dir, ami_meter_dir, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    device_column: str = "device_id",
    time_column: str = "t_stamp",
    partition_key: str = "dt_month",
) -> pd.DataFrame:
    """
    Orchestrate the full build: for every (year, month, frame) in
    `month_frames` (e.g. `ami_revalidate.iter_month_partitions` over the
    landed extract), build that month's interval table from the FINAL
    resolution (post Phase 4 + Section 7b + full-year revalidation),
    derive `ami_raw` and `ami_meter` for that month, and write both
    immediately -- never holding more than one month's tables in memory.

    Returns a provenance DataFrame, one row per month: `year, month,
    n_raw_rows, n_meter_rows, raw_path, meter_path`. A month that lands no
    rows in either table still gets a row here (with 0 counts and `None`
    paths), so a silent gap in the final dataset is visible in this
    manifest rather than only inferable from a missing partition directory.
    """
    provenance_rows = []
    for year, month, frame in month_frames:
        interval_table = Resolution.build_interval_table(
            frame, resolution,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, device_column=device_column,
            time_column=time_column,
        )
        ami_raw = build_ami_raw(
            interval_table, circuit_polarity,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, time_column=time_column,
        )
        ami_meter = build_ami_meter(
            interval_table,
            site_column=site_column, circuit_column=circuit_column,
            device_column=device_column, type_column=type_column,
            time_column=time_column,
        )
        raw_path = write_month_table(
            ami_raw, ami_raw_dir, year, month,
            table_name="ami_raw", partition_key=partition_key,
        )
        meter_path = write_month_table(
            ami_meter, ami_meter_dir, year, month,
            table_name="ami_meter", partition_key=partition_key,
        )
        provenance_rows.append({
            "year": year, "month": month,
            "n_raw_rows": len(ami_raw), "n_meter_rows": len(ami_meter),
            "raw_path": raw_path, "meter_path": meter_path,
        })
        del interval_table, ami_raw, ami_meter

    return pd.DataFrame(provenance_rows)
