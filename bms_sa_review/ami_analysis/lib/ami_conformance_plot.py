from __future__ import annotations

import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats as _stats

from bms_sa_review.ami_analysis.lib import ami_conformance as Conformance
from bms_sa_review.ami_analysis.lib import ami_pv_baseline as PvBaseline
from shared.ciccada_config import AS4777

__all__ = [
    "fetch_site_day_conformance", 
    "plot_site_day_conformance",
    "fetch_circuit_scope_conformance", 
    "plot_pv_circuit_vvar_scatter",
    "plot_assessable_vvar_window",
    "plot_assessable_vvar_scatter3d",
    "compute_assessable_vvar_fit", 
    "fetch_cohort_assessable_vvar_slopes", 
    "plot_cohort_assessable_vvar_slopes",
    "add_offset_pct_columns",
    "add_slope_pass_fail_asymmetric",
    "add_offset_pass_fail",
    "STATUS_ORDER", 
    "STATUS_COLORS", 
    "STATUS_LABELS",
    "SIGNED_GATE_VVAR_VARIANTS",
    "GATE_ABS",
    "GATE_SIGNED",
    "apply_signed_export_gate",
    "check_export_sign",
    "fetch_cohort_vvar_slopes_gate_comparison",
    "summarise_gate_comparison",
    "plot_site_gate_comparison",
    "plot_cohort_gate_slope_error",
]

_VVAR = AS4777["VVAR"]
_VW = AS4777["VW"]
_QCAP = AS4777["QCAP"]
_TOL_FRAC = AS4777["TOL_FRAC"]


# ---------------------------------------------------------------------------
# Level/offset test -- does the OLS line sit close to the required line, not
# just have the right slope? Evaluated at three reference voltages on the
# Volt-VAr ramp itself (V3=240 -> V4=258), NOT the narrower assessable
# window (V3 -> VW.V1=253) used to fit the line in the first place.
# ---------------------------------------------------------------------------
_VVAR_V3 = _VVAR["V3"]                      # 240 -- start of the absorbing ramp
_VVAR_V4 = _VVAR["V4"]                      # 258 -- end of the absorbing ramp (saturation)
_VVAR_VMID = (_VVAR_V3 + _VVAR_V4) / 2.0    # 249 -- midpoint of the V3-V4 ramp


# Fixed slope of the absorbing ramp (V3->V4), independent of actual output
# power -- the reference for plot_assessable_vvar_window's comparison.
_VVAR_ABSORB_SLOPE_PCT_PER_V = -_VVAR["Q4"] / (_VVAR["V4"] - _VVAR["V3"]) * 100

#: `voltvar_status` values `ami_conformance.build_voltvar_sql` can produce,
#: worst-first for scatter z-order (each drawn as its own `ax.scatter` call,
#: later calls on top -- conformant points, the most numerous by far on a
#: healthy site, are drawn LAST so they don't visually bury the rarer
#: nonconformant ones), plus matching colours/legend labels. Same status
#: vocabulary as `ami_conformance.build_voltvar_sql`'s docstring -- this is
#: NOT the `dnsp_analysis/notebooks/voltvar_voltwatt_lib.py` original's
#: status names (`conforming`/`not_assessable`, no `Q_` prefix); mapped
#: 1:1 in meaning, renamed to match this project's own SQL output exactly.
STATUS_ORDER = [
    "Q_adverse", "Q_inactive", "Q_significant_shortfall",
    "Q_not_exposed", "Q_near_conformant", "Q_major_surplus", "Q_not_assessable", "Q_conformant",
]
STATUS_COLORS = {
    "Q_conformant": "#1565c0", "Q_near_conformant": "#2e7d32",
    "Q_significant_shortfall": "#ef6c00", "Q_inactive": "#c62828",
    "Q_adverse": "#7f1d1d", "Q_major_surplus": "#6a1b9a",
    "Q_not_exposed": "#9e9e9e",
    "Q_not_assessable": "#9e9e9e",
}
STATUS_LABELS = {
    "Q_conformant": "Conformant (within band)",
    "Q_near_conformant": "Near-conformant (>110% of required, close)",
    "Q_significant_shortfall": "Significant shortfall (10-90% of required)",
    "Q_inactive": "Inactive (no response, wrong side of required)",
    "Q_adverse": "Adverse (wrong direction entirely)",
    "Q_major_surplus": "Major surplus (>>110% of required)",
    "Q_not_exposed": "Within deadband (220~240V)",
    "Q_not_assessable": f"Not assessable (|P| < {_QCAP['P_MIN']*100:.0f}% rating)",
}


def _aest_day_bounds(date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """AEST calendar day -> [start, end) in the raw `t_stamp` storage range
    (`t_stamp + 10h = AEST`, fixed offset, no DST -- same convention used
    everywhere else in this notebook)."""
    day = datetime.date.fromisoformat(date) if isinstance(date, str) else date
    start_utc = pd.Timestamp(day) - pd.Timedelta(hours=10)
    return start_utc, start_utc + pd.Timedelta(days=1)


def _normalize_circuit_ids(pv_circuit_id) -> tuple[int, ...]:
    """
    `pv_circuit_id`, in any of the forms described in the module docstring,
    normalized to a non-empty tuple of plain ints. The one place multi-phase
    PV support enters this module -- every public function below calls this
    first and works from the tuple it returns from then on.
    """
    if isinstance(pv_circuit_id, str):
        parts = [p for p in pv_circuit_id.split("|") if p]
        if not parts:
            raise ValueError(f"pv_circuit_id={pv_circuit_id!r} has no circuit ids in it.")
        return tuple(int(p) for p in parts)
    try:
        return (int(pv_circuit_id),)
    except TypeError:
        ids = tuple(int(c) for c in pv_circuit_id)
        if not ids:
            raise ValueError("pv_circuit_id was an empty sequence.")
        return ids


def _circuit_id_label(pv_circuit_ids: tuple[int, ...]) -> str:
    """Display label for titles/axis text -- `"circuit 123"`, or
    `"circuits 123+456+789 (summed)"` for a multi-phase PV inference."""
    if len(pv_circuit_ids) == 1:
        return f"circuit {pv_circuit_ids[0]}"
    return f"circuits {'+'.join(str(c) for c in pv_circuit_ids)} (summed)"


def _pv_rows_sql(source_table: str, site_id, pv_circuit_ids: tuple[int, ...], extra_where: str) -> str:
    """
    SQL selecting `pv_circuit_ids`' reading at `site_id`, restricted by
    `extra_where` (any predicate on `source_table`'s own raw columns -- a
    day's t_stamp bounds, a scope filter, or `"TRUE"`).

    A single circuit_id (the common, single-phase case) is a plain
    passthrough: `SELECT *`, every raw column kept, unmodified. More than
    one circuit_id (a multi-phase PV inference) sums `P_kw`/`Q_kvar` and
    averages `V` across them per `(site_id, t_stamp)` -- see the module
    docstring for why. Only `site_id, t_stamp, circuit_id, V, P_kw, Q_kvar`
    survive that combination; the representative `circuit_id` in the
    combined case is just the first id given (a label, not a lookup key
    from here on) -- nothing downstream of this needs any other raw column
    (e.g. `S_kva`) for `pv_circuit_ids`' own combined reading, since panels
    1/2/4/6 already draw every circuit's own raw row straight from
    `source_table`/`all_circuits`, untouched by this function entirely.
    """
    site_id = int(site_id)
    if len(pv_circuit_ids) == 1:
        return f"""
            SELECT * FROM {source_table}
            WHERE site_id = {site_id} AND circuit_id = {pv_circuit_ids[0]} AND ({extra_where})
        """
    ids_sql = ", ".join(str(c) for c in pv_circuit_ids)
    return f"""
        SELECT site_id, t_stamp,
               {pv_circuit_ids[0]} AS circuit_id,
               avg(V) AS V,
               sum(P_kw) AS P_kw,
               sum(Q_kvar) AS Q_kvar
        FROM {source_table}
        WHERE site_id = {site_id} AND circuit_id IN ({ids_sql}) AND ({extra_where})
        GROUP BY site_id, t_stamp
    """


def _rating_kw_for_site(con, site_id, *, rating_column, metadata_table) -> float:
    rating_row = con.sql(f"""
        SELECT {rating_column} AS rating
        FROM {metadata_table}
        WHERE site_id = {int(site_id)}
    """).df()
    if rating_row.empty or pd.isna(rating_row["rating"].iloc[0]) or rating_row["rating"].iloc[0] <= 0:
        raise ValueError(
            f"No usable {rating_column!r} for site_id={site_id} in {metadata_table} "
            "-- check the site_id and rating_column."
        )
    return float(rating_row["rating"].iloc[0])


def _baseline_correct_and_classify(
    con, pv_rows_raw, *,
    site_id, pv_circuit_ids, rating_kw, capability_profile,
    source_table, use_baseline_correction, baseline_where_clause, nighttime_hours,
    not_found_message: str,
) -> pd.DataFrame:
    """
    Shared second half of both `fetch_site_day_conformance` and
    `fetch_circuit_scope_conformance`: nighttime-baseline-correct
    `pv_rows_raw` (`pv_circuit_ids`' already-fetched, already-combined rows
    -- see `_pv_rows_sql` -- for any date range), then run it through
    `ami_conformance.build_voltvar_sql`/`build_voltwatt_sql`. See
    `fetch_site_day_conformance`'s docstring for the full column list this
    adds.
    """
    if pv_rows_raw.empty:
        raise ValueError(not_found_message)

    if use_baseline_correction:
        # `_pv_rows_sql` here (not a plain circuit_id match) so a multi
        # -phase PV's nighttime baseline is computed from the SAME
        # summed-P/Q, averaged-V combination as the daytime rows being
        # corrected -- never from each phase's own separate nighttime
        # median, which would be a different (and wrong) baseline for a
        # signal that no longer exists per-phase after combination.
        baseline_source = _pv_rows_sql(source_table, site_id, pv_circuit_ids, baseline_where_clause)
        baseline_sql = PvBaseline.build_nighttime_baseline_sql(
            f"({baseline_source})", where_clause="TRUE", nighttime_hours=nighttime_hours,
        )
        baseline_row = con.sql(baseline_sql).df()
        if (
            baseline_row.empty
            or pd.isna(baseline_row["nighttime_median_p_kw"].iloc[0])
            or pd.isna(baseline_row["nighttime_median_q_kvar"].iloc[0])
        ):
            raise ValueError(
                f"No nighttime ({nighttime_hours[0]}-{nighttime_hours[1]}h AEST) data to "
                f"build a baseline for site_id={site_id}, circuit_id(s)={pv_circuit_ids} "
                f"under baseline_where_clause={baseline_where_clause!r} -- widen the scope, "
                "or pass use_baseline_correction=False to compare the raw net reading instead."
            )
        baseline_p_kw = float(baseline_row["nighttime_median_p_kw"].iloc[0])
        baseline_q_kvar = float(baseline_row["nighttime_median_q_kvar"].iloc[0])
        pv_rows_estimated = PvBaseline.estimate_pv_signal(
            pv_rows_raw, baseline_p_kw, baseline_q_kvar,
        )
        power_column, reactive_column = "P_kw_pv_est", "Q_kvar_pv_est"
    else:
        baseline_p_kw = baseline_q_kvar = float("nan")
        pv_rows_estimated = pv_rows_raw.copy()
        power_column, reactive_column = "P_kw", "Q_kvar"

    pv_rows_estimated["rating_kw"] = rating_kw
    con.register("_pv_rows_estimated", pv_rows_estimated)
    try:
        voltvar_sql = Conformance.build_voltvar_sql(
            "_pv_rows_estimated", rating_column="rating_kw",
            power_column=power_column, reactive_column=reactive_column,
            capability_profile=capability_profile,
        )
        both_sql = Conformance.build_voltwatt_sql(
            f"({voltvar_sql})", rating_column="rating_kw", power_column=power_column,
        )
        pv_classified = con.sql(both_sql).df().sort_values("t_stamp").reset_index(drop=True)
    finally:
        con.unregister("_pv_rows_estimated")

    pv_classified["nighttime_baseline_p_kw"] = baseline_p_kw
    pv_classified["nighttime_baseline_q_kvar"] = baseline_q_kvar
    return pv_classified


def fetch_site_day_conformance(
    con, site_id, date, pv_circuit_id, *,
    rating_column: str = "ac_capacity_kw",
    capability_profile: str = "review_corrected",
    source_table: str = "ami_meter_residential",
    metadata_table: str = "ami_site_metadata_residential",
    use_baseline_correction: bool = True,
    baseline_where_clause: str = "TRUE",
    nighttime_hours: tuple[int, int] = (0, 4),
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Query side of `plot_site_day_conformance` -- split out so the SQL wiring
    can be tested without matplotlib. `pv_circuit_id`: single id, `"|"`-
    joined multi-id string, or list/tuple -- see the module docstring for
    the multi-phase combination this applies. Returns:

      - `all_circuits`: one row per (circuit_id, t_stamp) for every circuit
        at `site_id` on the AEST day of `date`, straight from `source_table`
        (raw `V`, `P_kw`, `Q_kvar`, `S_kva`, ...) -- always the RAW net
        reading, regardless of `use_baseline_correction`.
      - `pv_classified`: the same day's rows for `pv_circuit_id` ONLY, with
        `P_kw_pv_est`/`Q_kvar_pv_est` added (see below), then passed through
        `ami_conformance.build_voltvar_sql` (nested inside
        `build_voltwatt_sql`, so both sets of derived columns come back in
        one frame) -- adds `Q_voltvar`, `Q_cap_absorbing`,
        `capability_assessable`, `Q_min_final`, `Q_max_final`, `Q_impact`,
        `voltvar_status`, `max_P_volt_watt`, `nonconformance_voltwatt`. When
        `use_baseline_correction` is True (the default), these are computed
        from `P_kw_pv_est`/`Q_kvar_pv_est`, not the raw net `P_kw`/`Q_kvar`
        -- see `ami_pv_baseline.py`'s module docstring for why a raw net
        comparison can look qualitatively wrong for reactive power even
        when nothing is broken. `pv_classified` also carries
        `nighttime_baseline_p_kw`/`nighttime_baseline_q_kvar` (constant
        across the frame) so the correction actually applied is inspectable.
      - `rating_kw`: the single scalar `rating_column` value used for both
        (raises `ValueError` if `site_id` has no metadata row, or that
        column is null/non-positive there).

    `baseline_where_clause` scopes which rows `ami_pv_baseline
    .build_nighttime_baseline_sql` aggregates over to estimate the
    nighttime baseline (default: the circuit's entire history in
    `source_table`) -- pass e.g. a scope-months filter to match whatever
    window the rest of the notebook uses, for consistency. A wider,
    multi-month baseline is more stable than this one day's own overnight
    reading; set `use_baseline_correction=False` to compare the raw net
    reading directly instead (the original, uncorrected behaviour).

    Raises `ValueError` if there is no `source_table` data for `pv_circuit_id`
    on that day (a wrong circuit_id, or a day outside this site's monitoring
    window, are the two likely causes), or if `use_baseline_correction` is
    True and no nighttime baseline could be computed for this circuit.
    """
    pv_circuit_ids = _normalize_circuit_ids(pv_circuit_id)
    start_utc, end_utc = _aest_day_bounds(date)
    rating_kw = _rating_kw_for_site(
        con, site_id, rating_column=rating_column, metadata_table=metadata_table,
    )

    all_circuits = con.sql(f"""
        SELECT *
        FROM {source_table}
        WHERE site_id = {int(site_id)}
          AND t_stamp >= TIMESTAMP '{start_utc}'
          AND t_stamp <  TIMESTAMP '{end_utc}'
        ORDER BY circuit_id, t_stamp
    """).df()

    day_where = f"t_stamp >= TIMESTAMP '{start_utc}' AND t_stamp < TIMESTAMP '{end_utc}'"
    pv_day_raw = con.sql(_pv_rows_sql(source_table, site_id, pv_circuit_ids, day_where)).df()
    if not pv_day_raw.empty:
        pv_day_raw = pv_day_raw.sort_values("t_stamp").reset_index(drop=True)

    pv_classified = _baseline_correct_and_classify(
        con, pv_day_raw,
        site_id=site_id, pv_circuit_ids=pv_circuit_ids, rating_kw=rating_kw,
        capability_profile=capability_profile, source_table=source_table,
        use_baseline_correction=use_baseline_correction,
        baseline_where_clause=baseline_where_clause, nighttime_hours=nighttime_hours,
        not_found_message=(
            f"No {source_table} rows for site_id={site_id}, circuit_id(s)={pv_circuit_ids} "
            f"on {date} (AEST) -- check the circuit_id(s) and date are within this site's "
            "monitoring window."
        ),
    )
    return all_circuits, pv_classified, rating_kw


def fetch_circuit_scope_conformance(
    con, site_id, pv_circuit_id, *,
    scope_where: str = "TRUE",
    rating_column: str = "ac_capacity_kw",
    capability_profile: str = "review_corrected",
    source_table: str = "ami_meter_residential",
    metadata_table: str = "ami_site_metadata_residential",
    use_baseline_correction: bool = True,
    baseline_where_clause: str = "TRUE",
    nighttime_hours: tuple[int, int] = (0, 4),
) -> tuple[pd.DataFrame, float]:
    """
    Like `fetch_site_day_conformance`, but classifies EVERY interval of
    `pv_circuit_id` (same accepted forms -- see the module docstring)
    matched by `scope_where` (e.g. a whole `scope_sql(...)`
    months filter), not just one AEST day -- feeds
    `plot_pv_circuit_vvar_scatter`'s multi-interval scatter, the same way
    `fetch_site_day_conformance` feeds the single-day 6-panel plot. Returns
    `(pv_classified, rating_kw)` -- see `fetch_site_day_conformance`'s
    docstring for the full column list `pv_classified` carries (this reuses
    the exact same baseline-correction and classification logic, just over
    a wider row selection).

    Raises `ValueError` if `scope_where` matches no rows for this circuit,
    or (when `use_baseline_correction` is True) no nighttime baseline data.
    """
    pv_circuit_ids = _normalize_circuit_ids(pv_circuit_id)
    rating_kw = _rating_kw_for_site(
        con, site_id, rating_column=rating_column, metadata_table=metadata_table,
    )
    pv_rows_raw = con.sql(_pv_rows_sql(source_table, site_id, pv_circuit_ids, scope_where)).df()
    if not pv_rows_raw.empty:
        pv_rows_raw = pv_rows_raw.sort_values("t_stamp").reset_index(drop=True)

    pv_classified = _baseline_correct_and_classify(
        con, pv_rows_raw,
        site_id=site_id, pv_circuit_ids=pv_circuit_ids, rating_kw=rating_kw,
        capability_profile=capability_profile, source_table=source_table,
        use_baseline_correction=use_baseline_correction,
        baseline_where_clause=baseline_where_clause, nighttime_hours=nighttime_hours,
        not_found_message=(
            f"No {source_table} rows for site_id={site_id}, circuit_id(s)={pv_circuit_ids} "
            f"matching scope_where={scope_where!r}."
        ),
    )
    return pv_classified, rating_kw


def plot_site_day_conformance(
    con, site_id, date, pv_circuit_id, *,
    rating_column: str = "ac_capacity_kw",
    capability_profile: str = "review_corrected",
    source_table: str = "ami_meter_residential",
    metadata_table: str = "ami_site_metadata_residential",
    use_baseline_correction: bool = True,
    baseline_where_clause: str = "TRUE",
    nighttime_hours: tuple[int, int] = (0, 4),
    figsize: tuple[float, float] = (12, 24),
):
    """
    6-stacked-subplot AS/NZS 4777.2 diagnostic for one site/day, sharing the
    x-axis:

      1. Voltage -- one line per circuit, plus the cross-circuit average
         (when there's more than one circuit), with the VVAR/VW threshold
         voltages marked.
      2. Reactive power (`Q_kvar`) -- one line per circuit, raw net reading
         (unmodified -- panel context only, not the corrected signal used
         in panel 3).
      3. `pv_circuit_id`'s reactive power vs. the Volt-VAr band, as
         %-of-`rating_column`. When `use_baseline_correction` is True (the
         default), the "Measured" line and the band/required-Q comparison
         all use the nighttime-baseline-corrected estimate
         (`Q_kvar_pv_est`, see `ami_pv_baseline.py`) rather than the raw net
         `Q_kvar` -- a blind, uncorrected net reading can look qualitatively
         wrong (see module docstring) even when nothing is broken. The
         baseline actually subtracted is annotated on the panel. Also
         includes the Volt-VAr required curve, its `Q_min_final`/
         `Q_max_final` band (Figure 2.1 capability-clamped, from
         `build_voltvar_sql` -- IDENTICAL math to Section 4), and the "not
         assessable" region (`capability_assessable == False`, i.e. below
         `QCAP.P_MIN`) shaded.
      4. Active power (`P_kw`) -- one line per circuit, raw net reading
         (unmodified, same caveat as panel 2).
      5. `pv_circuit_id`'s active power vs. the Volt-Watt ceiling, as
         %-of-`rating_column` -- same baseline-correction treatment as
         panel 3 (using `P_kw_pv_est` when `use_baseline_correction` is
         True), with the Volt-Watt ceiling (+ tolerance) from
         `build_voltwatt_sql`, and the "not exposed" region
         (`V <= VW.V1`, i.e. `nonconformance_voltwatt` is NULL) shaded.
      6. Total apparent power (`S_kva`, summed across every circuit at the
         site) -- one line, raw.

    Returns `(fig, all_circuits, pv_classified)` -- the figure plus the two
    frames `fetch_site_day_conformance` returned, for further inspection
    (e.g. `pv_classified.voltvar_status.value_counts()`).

    See the module docstring, and `ami_pv_baseline.py`'s, for why panels
    3/5 compare a baseline-corrected estimate rather than the raw net
    `P_kw`/`Q_kvar` used (unmodified) everywhere else in this plot, unlike
    the `dnsp_analysis/notebooks/draft.ipynb` original this was ported
    from (which flipped sign on a genuine single-meter inverter signal --
    a different problem to the one solved here).
    """
    pv_circuit_ids = _normalize_circuit_ids(pv_circuit_id)
    pv_circuit_id_set = set(pv_circuit_ids)
    pv_label = _circuit_id_label(pv_circuit_ids)

    all_circuits, pv_classified, rating_kw = fetch_site_day_conformance(
        con, site_id, date, pv_circuit_ids,
        rating_column=rating_column, capability_profile=capability_profile,
        source_table=source_table, metadata_table=metadata_table,
        use_baseline_correction=use_baseline_correction,
        baseline_where_clause=baseline_where_clause,
        nighttime_hours=nighttime_hours,
    )
    start_utc, end_utc = _aest_day_bounds(date)
    day = datetime.date.fromisoformat(date) if isinstance(date, str) else date

    circuit_ids = sorted(all_circuits["circuit_id"].unique())
    palette = plt.cm.tab10.colors
    circuit_colors = {c: palette[i % len(palette)] for i, c in enumerate(circuit_ids)}
    # Representative colour for panels 3/5's single combined PV line -- the
    # first PV-bearing circuit's own colour, for visual continuity with the
    # highlighted line(s) in panels 1/2/4.
    pv_color = circuit_colors[pv_circuit_ids[0]]

    pivot_v = all_circuits.pivot_table(index="t_stamp", columns="circuit_id", values="V", aggfunc="mean").sort_index()
    pivot_p = all_circuits.pivot_table(index="t_stamp", columns="circuit_id", values="P_kw", aggfunc="mean").sort_index()
    pivot_q = all_circuits.pivot_table(index="t_stamp", columns="circuit_id", values="Q_kvar", aggfunc="mean").sort_index()
    if "S_kva" in all_circuits.columns:
        total_s = all_circuits.pivot_table(
            index="t_stamp", columns="circuit_id", values="S_kva", aggfunc="mean"
        ).sort_index().sum(axis=1, min_count=1)
    else:
        total_s = np.hypot(pivot_p, pivot_q).sum(axis=1, min_count=1)

    plot_times = pv_classified["t_stamp"].to_numpy()
    tol_pct = 100.0 * _TOL_FRAC

    def _draw_all_circuits(ax, pivoted, title, ylabel):
        for circuit_id in pivoted.columns:
            is_pv = circuit_id in pv_circuit_id_set
            style = dict(
                color=circuit_colors[circuit_id],
                linewidth=2.0 if is_pv else 1.3,
                label=f"circuit {circuit_id}" + (" (PV)" if is_pv else ""),
            )
            ax.plot(pivoted.index.to_numpy(), pivoted[circuit_id].to_numpy(), **style)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlim(start_utc.to_pydatetime(), end_utc.to_pydatetime())
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)

    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        f"site_id={site_id}  |  {pv_label} (PV)  |  {day.isoformat()} (AEST)  |  "
        f"{rating_column}={rating_kw:.2f} kW  |  capability_profile={capability_profile}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # --- 1. Voltage, all circuits + average, with VVAR/VW threshold lines --
    voltage_vals = pivot_v.to_numpy(dtype=float)
    v_min = min(235.0, np.nanmin(voltage_vals) - 2) if voltage_vals.size else 235.0
    v_max = max(262.0, np.nanmax(voltage_vals) + 2) if voltage_vals.size else 262.0
    # No shaded band below V3 -- matching the `draft.ipynb` original, which
    # only colours the three bands above the Volt-VAr deadband (green
    # V3-VW.V1, orange VW.V1-V4, red V4+); below V3 nothing is required, so
    # nothing is shaded there.
    axes[0].axhspan(_VVAR["V3"], _VW["V1"], color="green", alpha=0.10)
    axes[0].axhspan(_VW["V1"], _VVAR["V4"], color="orange", alpha=0.12)
    axes[0].axhspan(_VVAR["V4"], v_max, color="red", alpha=0.10)
    _draw_all_circuits(axes[0], pivot_v, "1. Voltage by circuit", "Volts")
    if len(circuit_ids) > 1:
        avg_v = pivot_v.mean(axis=1)
        axes[0].plot(avg_v.index.to_numpy(), avg_v.to_numpy(), color="black", linewidth=2, label="Average")
    for voltage, color, label in [
        (_VVAR["V3"], "green", f"{_VVAR['V3']:.0f} V -- Volt-VAr ramp starts"),
        (_VW["V1"], "orange", f"{_VW['V1']:.0f} V -- Volt-Watt starts"),
        (_VVAR["V4"], "red", f"{_VVAR['V4']:.0f} V -- max VAr absorption"),
        (_VW["V2"], "purple", f"{_VW['V2']:.0f} V -- Volt-Watt reaches {_VW['P2']*100:.0f}%"),
    ]:
        axes[0].axhline(voltage, color=color, linestyle="--", linewidth=1.2, label=label)
    axes[0].set_ylim(v_min, v_max)
    axes[0].legend(fontsize=7, ncol=2)

    # --- 2. Reactive power, all circuits, raw --------------------------- #
    _draw_all_circuits(axes[1], pivot_q, "2. Reactive power (Q_kvar) by circuit", "Q_kvar (net)")
    axes[1].axhline(0, color="black", linewidth=0.7, alpha=0.5)

    # --- 3. pv_circuit_id: Q as % of rating, with Volt-VAr band --------- #
    q_measured_column = "Q_kvar_pv_est" if use_baseline_correction else "Q_kvar"
    measured_q_label = (
        f"Baseline-corrected {pv_label} Q (est.)"
        if use_baseline_correction else f"Measured {pv_label} Q (raw net)"
    )
    vv_not_required = (~pv_classified["capability_assessable"].astype(bool)).to_numpy()
    q_required_pct = 100.0 * pv_classified["Q_voltvar"].to_numpy() / rating_kw
    q_min_pct = 100.0 * pv_classified["Q_min_final"].to_numpy() / rating_kw
    q_max_pct = 100.0 * pv_classified["Q_max_final"].to_numpy() / rating_kw
    # In the "not assessable" region (|P| below QCAP.P_MIN), `build_voltvar_
    # sql`'s capability-clamp relaxes the band outward to whatever the
    # (near-zero-P) capability floor computes to -- which the standard
    # doesn't actually constrain there, and which can blow up to an
    # implausibly wide band with a baseline-corrected P estimate hovering
    # near zero. That region is already shaded grey as "not required" --
    # draw a simple, bounded +/-tolerance band there instead of the
    # clamped one, matching how `draft.ipynb`'s own `calculate_voltvar_band`
    # falls back to a plain `theoretical +/- tolerance` band whenever a
    # response isn't required, rather than running the capability-clamp
    # logic (which is only meaningful where a response IS required).
    q_min_pct = np.where(vv_not_required, q_required_pct - tol_pct, q_min_pct)
    q_max_pct = np.where(vv_not_required, q_required_pct + tol_pct, q_max_pct)
    q_measured_pct = 100.0 * pv_classified[q_measured_column].to_numpy() / rating_kw

    axes[2].fill_between(
        plot_times, 0, 1, where=vv_not_required, transform=axes[2].get_xaxis_transform(),
        color="grey", alpha=0.15, linewidth=0,
        label=f"Not assessable: |P| < {_QCAP['P_MIN']*100:.0f}% rating",
    )
    axes[2].fill_between(
        plot_times, q_min_pct, q_max_pct, color="darkorange", alpha=0.27, linewidth=0,
        label=f"Volt-VAr band (capability-clamped, +/-{tol_pct:.0f}%)",
    )
    axes[2].plot(plot_times, q_required_pct, color="darkorange", linewidth=1.6, linestyle="--",
                 label="Theoretical Volt-VAr required Q")
    axes[2].plot(plot_times, q_measured_pct, color=pv_color, linewidth=1.5,
                 label=measured_q_label)
    axes[2].axhline(0, color="black", linewidth=0.7, alpha=0.6)
    panel3_title = "3. PV circuit reactive power vs. AS/NZS 4777.2 Volt-VAr band"
    if use_baseline_correction:
        baseline_q = pv_classified["nighttime_baseline_q_kvar"].iloc[0]
        panel3_title += f"  (baseline-corrected; nighttime Q baseline={baseline_q:.3f} kvar)"
    axes[2].set_title(panel3_title, fontsize=10)
    axes[2].set_ylabel(f"% of {rating_column} ({rating_kw:.2f} kW)")
    axes[2].set_xlim(start_utc.to_pydatetime(), end_utc.to_pydatetime())
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=7)

    # --- 4. Active power, all circuits, raw ------------------------------ #
    _draw_all_circuits(axes[3], pivot_p, "4. Active power (P_kw) by circuit", "P_kw (net)")
    axes[3].axhline(0, color="black", linewidth=0.7, alpha=0.5)

    # --- 5. pv_circuit_id: P as % of rating, with Volt-Watt ceiling ------ #
    p_measured_column = "P_kw_pv_est" if use_baseline_correction else "P_kw"
    measured_p_label = (
        f"Baseline-corrected {pv_label} P (est.)"
        if use_baseline_correction else f"Measured {pv_label} P (raw net)"
    )
    vw_not_required = pv_classified["nonconformance_voltwatt"].isna().to_numpy()
    ceiling_pct = 100.0 * (pv_classified["max_P_volt_watt"].to_numpy()) / rating_kw
    p_measured_pct = 100.0 * pv_classified[p_measured_column].to_numpy() / rating_kw

    axes[4].fill_between(
        plot_times, 0, 1, where=vw_not_required, transform=axes[4].get_xaxis_transform(),
        color="grey", alpha=0.15, linewidth=0,
        label=f"Not exposed: V <= {_VW['V1']:.0f} V",
    )
    axes[4].plot(plot_times, ceiling_pct, color="darkorange", linewidth=1.8, linestyle="--",
                 label="Volt-Watt ceiling (+tolerance)")
    axes[4].plot(plot_times, p_measured_pct, color=pv_color, linewidth=1.5,
                 label=measured_p_label)
    axes[4].axhline(100, color="black", linestyle=":", linewidth=1, alpha=0.6, label="100% rating")
    axes[4].axhline(0, color="black", linewidth=0.7, alpha=0.5)
    panel5_title = "5. PV circuit active power vs. AS/NZS 4777.2 Volt-Watt ceiling"
    if use_baseline_correction:
        baseline_p = pv_classified["nighttime_baseline_p_kw"].iloc[0]
        panel5_title += f"  (baseline-corrected; nighttime P baseline={baseline_p:.3f} kW)"
    axes[4].set_title(panel5_title, fontsize=10)
    axes[4].set_ylabel(f"% of {rating_column} ({rating_kw:.2f} kW)")
    axes[4].set_xlim(start_utc.to_pydatetime(), end_utc.to_pydatetime())
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(fontsize=7)

    # --- 6. Total apparent power, all circuits summed -------------------- #
    axes[5].plot(total_s.index.to_numpy(), total_s.to_numpy(), color="black", linewidth=2, label="Total S (all circuits)")
    axes[5].set_title("6. Total apparent power (S_kva), all circuits summed")
    axes[5].set_ylabel("S_kva")
    axes[5].set_xlabel("Timestamp (raw t_stamp -- add 10h for AEST)")
    axes[5].set_xlim(start_utc.to_pydatetime(), end_utc.to_pydatetime())
    axes[5].grid(True, alpha=0.25)
    axes[5].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    plt.show()

    return fig, all_circuits, pv_classified


def _required_q_reference_curve(con, rating_kw, *, capability_profile, v_lo=200.0, v_hi=280.0, n=400):
    """
    A smooth Q-vs-V reference curve for the scatter's background: required
    Q (and its +/-tolerance band) at every V in `[v_lo, v_hi]`, AT FULL
    ACTIVE-POWER OUTPUT (`P_kw = rating_kw`, i.e. 100% -- always capability
    -assessable, so the capability clamp never narrows this reference).
    Built by running the one grid table through `build_voltvar_sql` itself
    (not a separately re-derived curve formula), so it's guaranteed
    consistent with what actually classified each scattered point.
    """
    grid = pd.DataFrame({
        "V": np.linspace(v_lo, v_hi, n),
        "P_kw": rating_kw,
        "Q_kvar": 0.0,
        "rating_kw": rating_kw,
    })
    con.register("_vvar_reference_grid", grid)
    try:
        curve = con.sql(
            Conformance.build_voltvar_sql(
                "_vvar_reference_grid", rating_column="rating_kw", capability_profile=capability_profile,
            )
        ).df()
    finally:
        con.unregister("_vvar_reference_grid")
    return curve.sort_values("V")


def plot_pv_circuit_vvar_scatter(
    con, site_id, pv_circuit_id, *,
    scope_where: str = "TRUE",
    period_label: str = "",
    rating_column: str = "ac_capacity_kw",
    capability_profile: str = "review_corrected",
    source_table: str = "ami_meter_residential",
    metadata_table: str = "ami_site_metadata_residential",
    use_baseline_correction: bool = True,
    baseline_where_clause: str = "TRUE",
    nighttime_hours: tuple[int, int] = (0, 4),
    v_range: tuple[float, float] = (230.0, 262.0),
    q_range_pct: tuple[float, float] = (-60.0, 40.0),
    figsize: tuple[float, float] = (11, 8),
):
    """
    Q-vs-V scatter for `pv_circuit_id`, one point per interval matched by
    `scope_where` (e.g. `scope_sql(SCOPE_MONTHS)`, to cover the same window
    as the rest of this notebook), coloured by `voltvar_status`. Ported in
    SHAPE from `dnsp_analysis/notebooks/voltvar_voltwatt_lib.py`'s
    `plot_vvar_month_scatter` -- see the module docstring for what's
    deliberately different (classification comes from this project's own
    `build_voltvar_sql`/baseline-correction pipeline, not a separate
    pandas re-implementation).

    Like panels 3/5 of `plot_site_day_conformance`, uses the nighttime
    -baseline-corrected `Q_kvar_pv_est` (not raw net `Q_kvar`) when
    `use_baseline_correction` is True (the default) -- see that function's
    and `ami_pv_baseline.py`'s docstrings for why.

    Returns `(fig, pv_classified, rating_kw)` -- `pv_classified` has one row
    per scattered interval, for further inspection (e.g.
    `pv_classified.voltvar_status.value_counts()`).
    """
    pv_label = _circuit_id_label(_normalize_circuit_ids(pv_circuit_id))
    pv_classified, rating_kw = fetch_circuit_scope_conformance(
        con, site_id, pv_circuit_id,
        scope_where=scope_where, rating_column=rating_column,
        capability_profile=capability_profile, source_table=source_table,
        metadata_table=metadata_table, use_baseline_correction=use_baseline_correction,
        baseline_where_clause=baseline_where_clause, nighttime_hours=nighttime_hours,
    )
    q_measured_column = "Q_kvar_pv_est" if use_baseline_correction else "Q_kvar"
    measured_label = "baseline-corrected estimate" if use_baseline_correction else "raw net reading"

    d = pv_classified.copy()
    d["Q_pct"] = 100.0 * d[q_measured_column] / rating_kw
    n_total = len(d)
    counts = d["voltvar_status"].value_counts()

    reference = _required_q_reference_curve(con, rating_kw, capability_profile=capability_profile)
    q_required_pct = 100.0 * reference["Q_voltvar"].to_numpy() / rating_kw
    q_min_pct = 100.0 * reference["Q_min_final"].to_numpy() / rating_kw
    q_max_pct = 100.0 * reference["Q_max_final"].to_numpy() / rating_kw

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    for status in STATUS_ORDER:
        sub = d.loc[d["voltvar_status"] == status]
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["V"], sub["Q_pct"], s=5, alpha=0.35 if status == "Q_conformant" else 0.55,
            color=STATUS_COLORS[status], zorder=3,
            label=f"{STATUS_LABELS[status]} ({n:,}, {100*n/n_total:.1f}%)",
        )
    ax.plot(reference["V"].to_numpy(), q_required_pct, color="#f59e0b", linewidth=1.8, zorder=5,
             label="Required Q (AS/NZS 4777.2 curve, at 100% rating)")
    ax.fill_between(
        reference["V"].to_numpy(), q_min_pct, q_max_pct, color="#f59e0b", alpha=0.15, linewidth=0, zorder=2,
        label=f"Capability-clamped band (+/-{_TOL_FRAC*100:.0f}%, at 100% rating)",
    )
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    for voltage, label in [
        (_VVAR["V3"], f"V3 {_VVAR['V3']:.0f}"), (_VW["V1"], f"V1 {_VW['V1']:.0f}"),
        (_VVAR["V4"], f"V4 {_VVAR['V4']:.0f}"),
    ]:
        if v_range[0] <= voltage <= v_range[1]:
            ax.axvline(voltage, color="grey", linewidth=0.6, linestyle=":", zorder=1)
            ax.text(voltage + 0.4, q_range_pct[1] * 0.95, label, fontsize=6, color="grey", va="top", ha="left")

    ax.set_xlim(*v_range)
    ax.set_ylim(*q_range_pct)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
    ax.set_xlabel("Voltage (V)", fontsize=9)
    ax.set_ylabel(
        f"Reactive power ({measured_label}, % of {rating_column})\n"
        "+ = supplying, - = absorbing (generator convention)", fontsize=9,
    )
    ax.set_title(
        f"site_id={site_id}  |  {pv_label} (PV)  |  {period_label}\n"
        f"Volt-VAr response vs AS/NZS 4777.2 ({capability_profile})  |  "
        f"{rating_column}={rating_kw:.2f} kW  |  {n_total:,} intervals",
        fontsize=9.5, fontweight="bold", loc="left",
    )
    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.92, edgecolor="#cccccc", ncol=1)
    ax.grid(color="#ebebeb", linewidth=0.5)
    fig.tight_layout()
    plt.show()

    print(f"Status breakdown ({n_total:,} intervals total):")
    for status in STATUS_ORDER:
        n = int(counts.get(status, 0))
        print(f"  {status:<26} {n:>7,}  ({100*n/n_total:5.1f}%)")

    return fig, pv_classified, rating_kw

###########################################################

DEFAULT_VVAR_VARIANTS = [
    {"name": "Polarity-corrected only", "source_table": "ami_meter_residential_polarity_corrected",
     "circuit_col": "load_circuit_id", "q_col": "Q_kvar", "use_baseline_correction": False},
    {"name": "Sign-flip only", "source_table": "ami_meter_residential_sign_flipped",
     "circuit_col": "load_circuit_id", "q_col": "Q_kvar", "use_baseline_correction": False},
    {"name": "Polarity-corrected + baseline estimate", "source_table": "ami_meter_residential_polarity_corrected",
     "circuit_col": "load_circuit_id", "q_col": "Q_kvar_pv_est", "use_baseline_correction": True},
    {"name": "Ground truth (ami_raw_phaseseparate)", "source_table": "ami_raw_phaseseparate_pv_only",
     "circuit_col": "pv_circuit_id", "q_col": "Q_kvar", "use_baseline_correction": False},
]


def _fit_assessable_window(
        classified_df, 
        rating_kw, q_col, 
        *, 
        daytime_hours=None,
        v_bin_width=1.0, 
        v_min=None, 
        v_max=None,
        power_col=None
        ):
    """
    Shared filter + fit logic behind both plot_assessable_vvar_window and
    compute_assessable_vvar_fit -- one implementation, so the two never
    drift apart. Returns (window, binned, slope, intercept, fit_stats);
    raises ValueError if fewer than 2 assessable points survive the filter.

    `v_min`/`v_max` narrow the fitted voltage range beyond the default
    [V3, VW.V1) assessable window -- e.g. v_min=242 to drop the bottom of
    the ramp where measurement noise near V3=240 dominates. Defaults to the
    full [V3, VW.V1) window when not given. The required-line ANCHOR stays
    at the real V3=240 regardless (the standard's own ramp start) -- only
    the fitted/plotted range changes.
    """

    d = classified_df.copy()
    d["Q_pct"] = 100.0 * d[q_col] / rating_kw

    v3, v4 = _VVAR["V3"], _VVAR["V4"]
    vw_v1 = _VW["V1"]
    lo = v3 if v_min is None else v_min
    hi = vw_v1 if v_max is None else v_max

    mask = (d["V"] >= lo) & (d["V"] < hi) & (d["capability_assessable"])
    # Signed net-export gate: keep only intervals EXPORTING >= 20% of rating.
    # abs(P) also admits heavy household import on a net-metered circuit.
    if power_col is not None:
        mask &= d[power_col] >= _QCAP["P_MIN"] * rating_kw
    if daytime_hours is not None:
        hour_start, hour_end = daytime_hours
        hour_aest = (d["t_stamp"].dt.hour + 10) % 24
        mask &= (hour_aest >= hour_start) & (hour_aest < hour_end)

    window = d.loc[mask].copy()
    if len(window) < 2:
        raise ValueError(
            f"Only {len(window)} assessable points in [{lo}, {hi}) V "
            f"(n_total={len(d)}) -- not enough to fit a slope."
        )

    lin = _stats.linregress(window["V"], window["Q_pct"])
    slope_emp, intercept_emp = lin.slope, lin.intercept
    fit_stats = {
        "n_total": len(d), "n_window": len(window),
        "r_squared": lin.rvalue ** 2, "slope_stderr": lin.stderr,
        "slope_ci95": (slope_emp - 1.96 * lin.stderr, slope_emp + 1.96 * lin.stderr),
        "v_min": lo, "v_max": hi,
    }

    window["v_bin"] = (window["V"] // v_bin_width) * v_bin_width
    binned = window.groupby("v_bin")["Q_pct"].agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="size",
    ).reset_index()

    return window, binned, slope_emp, intercept_emp, fit_stats

def plot_assessable_vvar_window(
    classified_df, rating_kw, q_col, *,
    title_suffix="", ax=None, daytime_hours: tuple[int, int] | None = None,
    v_bin_width: float = 1.0, v_min=None, v_max=None, power_col=None,
):
    """
    Scatter of assessable Volt-VAr intervals (V vs Q as % of rating) in the
    [V3, VW.V1) window, with a binned median + IQR band, the required ramp,
    and the OLS fit. Filter + fit are shared with compute_assessable_vvar_fit
    via _fit_assessable_window, so the plot and the cohort numbers can never
    drift apart.

    `daytime_hours`: optional AEST (start, end) hour filter.
    `v_min`/`v_max`: narrow the fitted/plotted voltage range (default: the
        full [V3, VW.V1) window).
    `power_col`: e.g. "P_kw" (sign-flip / ground truth) or "P_kw_pv_est"
        (baseline variant) -- applies the signed net-export gate, keeping only
        intervals with power_col >= 20% of rating. Default None = the original
        abs(P) gate only. Don't use it on "Polarity-corrected only", where
        export is negative P.

    Returns (slope_emp, intercept_emp, window, fit_stats).
    """
    v3 = _VVAR["V3"]
    window, binned, slope_emp, intercept_emp, fit_stats = _fit_assessable_window(
        classified_df, rating_kw, q_col, daytime_hours=daytime_hours,
        v_bin_width=v_bin_width, v_min=v_min, v_max=v_max, power_col=power_col,
    )
    lo, hi = fit_stats["v_min"], fit_stats["v_max"]
    n_total, n_window = fit_stats["n_total"], fit_stats["n_window"]

    time_label = f", {daytime_hours[0]:02d}:00-{daytime_hours[1]:02d}:00 AEST" if daytime_hours else ""
    gate_label = f", signed gate ({power_col} ≥ {_QCAP['P_MIN']*100:.0f}% rating)" if power_col else ""
    print(f"{n_window:,} / {n_total:,} intervals ({100*n_window/n_total:.1f}%) fall in "
          f"the assessable [{lo}, {hi}) V window{time_label}{gate_label}.")

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=130)

    ax.scatter(window["V"], window["Q_pct"], s=5, alpha=0.25, color="#2a78d6", zorder=2,
               label=f"Assessable intervals (n={n_window:,})")
    ax.fill_between(binned["v_bin"], binned["q25"], binned["q75"], color="#1baf7a", alpha=0.25, zorder=3,
                    label=f"IQR (25th-75th pct, per {v_bin_width:g}V bin)")
    ax.plot(binned["v_bin"], binned["median"], color="#1baf7a", linewidth=2, zorder=4,
            label=f"Median (per {v_bin_width:g}V bin)")

    v_line = np.linspace(lo, hi, 50)
    ax.plot(v_line, _VVAR_ABSORB_SLOPE_PCT_PER_V * (v_line - v3), color="#f59e0b", linewidth=2, zorder=5,
            label=f"Required: slope = {_VVAR_ABSORB_SLOPE_PCT_PER_V:.3f} %/V")
    ax.plot(v_line, slope_emp * v_line + intercept_emp, color="#eb6834", linewidth=2, linestyle="--", zorder=5,
            label=f"OLS fit: slope = {slope_emp:.3f} ± {fit_stats['slope_stderr']:.3f} %/V, "
                  f"R²={fit_stats['r_squared']:.2f}")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Voltage (V)", fontsize=9)
    ax.set_ylabel("Reactive power (% of rating)\n+ = supplying, - = absorbing", fontsize=9)
    ax.set_title(
        f"Assessable Volt-VAr window [{lo}, {hi}) V{time_label}{gate_label}{title_suffix}\n"
        f"Required {_VVAR_ABSORB_SLOPE_PCT_PER_V:.3f} %/V vs OLS {slope_emp:.3f} %/V "
        f"(R²={fit_stats['r_squared']:.2f}, ratio {slope_emp/_VVAR_ABSORB_SLOPE_PCT_PER_V:.2f}x)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(color="#ebebeb", linewidth=0.5)
    if own_ax:
        fig.tight_layout()
        plt.show()

    return slope_emp, intercept_emp, window, fit_stats


def compute_assessable_vvar_fit(classified_df, rating_kw, q_col, *, daytime_hours=None, power_col=None):
    """
    Non-plotting version of plot_assessable_vvar_window's fit -- for
    looping over many sites/variants without rendering a figure per call.
    Returns a flat dict: n_total, n_window, slope, intercept, r_squared,
    slope_stderr, slope_ci_low, slope_ci_high.
    """
    _, _, slope_emp, intercept_emp, fit_stats = _fit_assessable_window(
        classified_df, rating_kw, q_col, daytime_hours=daytime_hours, power_col=power_col
    )
    return {
        "n_total": fit_stats["n_total"], "n_window": fit_stats["n_window"],
        "slope": slope_emp, "intercept": intercept_emp,
        "r_squared": fit_stats["r_squared"], "slope_stderr": fit_stats["slope_stderr"],
        "slope_ci_low": fit_stats["slope_ci95"][0], "slope_ci_high": fit_stats["slope_ci95"][1],
    }


def fetch_cohort_assessable_vvar_slopes(
    con, site_circuit_map, *,
    scope_where, rating_column="ac_capacity_kw",
    capability_profile="review_corrected",
    metadata_table="ami_site_metadata_residential",
    daytime_hours=None,
    signed_gate=False,
    nighttime_hours=(0, 4),
    variants=None,
) -> pd.DataFrame:
    """
    Runs compute_assessable_vvar_fit for every (site, variant) pair in
    `site_circuit_map` (columns: site_id, and one circuit-id column per
    variant's `circuit_col`, e.g. load_circuit_id/pv_circuit_id -- NaN
    means that site has no circuit for that variant and it's skipped,
    not silently dropped -- its row still appears with status set).
    A site/variant combination that raises (missing nighttime data for
    baseline correction, too few assessable points, etc.) is recorded
    with its error message rather than aborting the whole run.
    """
    if variants is None:
        variants = DEFAULT_VVAR_VARIANTS

    rows = []
    for _, site_row in site_circuit_map.iterrows():
        site_id = int(site_row["site_id"])
        for variant in variants:
            circuit_id = site_row.get(variant["circuit_col"])
            if pd.isna(circuit_id):
                rows.append({"site_id": site_id, "variant": variant["name"], "status": "missing_circuit"})
                continue
            circuit_id = int(circuit_id)
            try:
                classified_df, rating_kw = fetch_circuit_scope_conformance(
                    con, site_id, circuit_id,
                    scope_where=scope_where, rating_column=rating_column,
                    capability_profile=capability_profile,
                    source_table=variant["source_table"], metadata_table=metadata_table,
                    use_baseline_correction=variant["use_baseline_correction"],
                    nighttime_hours=nighttime_hours,
                )
                # Baseline variant exports as positive P_kw_pv_est; the others as positive P_kw.
                power_col = None
                if signed_gate:
                    power_col = "P_kw_pv_est" if variant["use_baseline_correction"] else "P_kw"
                fit = compute_assessable_vvar_fit(
                    classified_df, rating_kw, variant["q_col"], daytime_hours=daytime_hours,
                    power_col=power_col,
                )
            except ValueError as exc:
                rows.append({"site_id": site_id, "circuit_id": circuit_id, "variant": variant["name"],
                             "status": f"error: {exc}"})
                continue

            rows.append({
                "site_id": site_id, "circuit_id": circuit_id, "variant": variant["name"], "status": "ok",
                "rating_kw": rating_kw, **fit,
            })

    return pd.DataFrame(rows)


def plot_cohort_assessable_vvar_slopes(
    cohort_fits, *, min_n_window: int = 20, variant_order=None, palette=None, figsize=(11, 7),
):
    """
    Box + strip plot of per-site empirical slopes (%rating/V), grouped by
    variant, against the standard's fixed required slope. Rows with
    status != 'ok', or n_window < min_n_window (too few points to trust
    the fit), are excluded and reported, not silently dropped.
    """
    if variant_order is None:
        variant_order = [v["name"] for v in DEFAULT_VVAR_VARIANTS]
    if palette is None:
        # Project's own validated categorical palette (see 04_filtered_data_description.ipynb).
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

    ok = cohort_fits.loc[cohort_fits["status"] == "ok"].copy()
    excluded_status = cohort_fits.loc[cohort_fits["status"] != "ok"]
    thin = ok.loc[ok["n_window"] < min_n_window]
    plotted = ok.loc[ok["n_window"] >= min_n_window]

    print(f"{len(cohort_fits):,} (site, variant) pairs total.")
    print(f"  excluded (missing circuit / fit error): {len(excluded_status):,}")
    if len(excluded_status):
        print(excluded_status["status"].value_counts())
    print(f"  excluded (n_window < {min_n_window}): {len(thin):,}")
    print(f"  plotted: {len(plotted):,}")

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    data_by_variant = [plotted.loc[plotted["variant"] == v, "slope"].values for v in variant_order]

    bp = ax.boxplot(
        data_by_variant, positions=range(len(variant_order)), widths=0.5,
        patch_artist=True, showfliers=False, zorder=3,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(42)
    for i, (values, color) in enumerate(zip(data_by_variant, palette)):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=14, alpha=0.5, color=color, zorder=4)

    ax.axhline(_VVAR_ABSORB_SLOPE_PCT_PER_V, color="#f59e0b", linewidth=2, linestyle="--", zorder=5,
               label=f"Required slope = {_VVAR_ABSORB_SLOPE_PCT_PER_V:.3f} %/V")
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)

    ax.set_xticks(range(len(variant_order)))
    ax.set_xticklabels(variant_order, fontsize=8.5, rotation=12, ha="right")
    ax.set_ylabel("Empirical slope (%rating/V)", fontsize=9)
    ax.set_title(
        f"Cohort-wide assessable Volt-VAr slope by variant (n={plotted['site_id'].nunique():,} sites)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(color="#ebebeb", linewidth=0.5, axis="y")
    fig.tight_layout()
    plt.show()

    return plotted


def plot_cohort_ols_lines(
    fits, *, variant_order=None, palette=None, figsize=(24, 6), line_alpha=0.25,
):
    """
    One panel per variant, each showing every site's individual OLS fit
    line (reconstructed from its already-computed slope/intercept -- no
    re-querying) overlaid on the same axes, against the required slope
    reference line and the mean of that variant's site-level fits.

    `fits` is the filtered DataFrame plot_cohort_assessable_vvar_slopes
    returns (status == 'ok' and n_window >= min_n_window already applied),
    or an equivalent subset of fetch_cohort_assessable_vvar_slopes's
    output -- pass the same one you plotted the boxplot from, so the two
    views describe the same underlying set of sites.
    """
    if variant_order is None:
        variant_order = [v["name"] for v in DEFAULT_VVAR_VARIANTS]
    if palette is None:
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

    v3, vw_v1 = _VVAR["V3"], _VW["V1"]
    v_line = np.linspace(v3, vw_v1, 50)

    fig, axes = plt.subplots(1, len(variant_order), figsize=figsize, dpi=130, sharey=True)
    if len(variant_order) == 1:
        axes = [axes]

    for ax, variant_name, color in zip(axes, variant_order, palette):
        sub = fits.loc[fits["variant"] == variant_name]
        for _, row in sub.iterrows():
            ax.plot(v_line, row["slope"] * v_line + row["intercept"],
                     color=color, linewidth=1, alpha=line_alpha, zorder=2)

        if len(sub):
            mean_slope = sub["slope"].mean()
            mean_intercept = sub["intercept"].mean()
            ax.plot(v_line, mean_slope * v_line + mean_intercept, color=color, linewidth=2.5, zorder=4,
                    label=f"Mean of site fits: {mean_slope:.3f} %/V")

        ax.plot(v_line, _VVAR_ABSORB_SLOPE_PCT_PER_V * (v_line - v3), color="#f59e0b", linewidth=2,
                linestyle="--", zorder=5, label=f"Required: {_VVAR_ABSORB_SLOPE_PCT_PER_V:.3f} %/V")
        ax.axhline(0, color="black", linewidth=0.5, zorder=1)
        ax.set_title(f"{variant_name}\n(n={len(sub):,} sites)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Voltage (V)", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(color="#ebebeb", linewidth=0.5)

    axes[0].set_ylabel("Reactive power (% of rating)", fontsize=9)
    fig.suptitle("Cohort site-level OLS fit lines by variant", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return fig

def add_slope_pass_fail(fits, tolerance_pct: float) -> pd.DataFrame:
    """
    Adds `pct_error` (signed % deviation of `slope` from the required
    slope, relative to the required slope's MAGNITUDE) and `passed`
    (True when |pct_error| <= tolerance_pct) to a copy of `fits`.

    Symmetric by construction: a site whose slope is too shallow (closer
    to 0), too steep (overshoots), or the wrong sign entirely all get a
    pct_error magnitude that reflects how far off they are -- there's no
    special-casing for the required slope being negative.
    """
    out = fits.copy()
    required = _VVAR_ABSORB_SLOPE_PCT_PER_V
    out["pct_error"] = 100.0 * (out["slope"] - required) / abs(required)
    out["passed"] = out["pct_error"].abs() <= tolerance_pct
    return out


def plot_cohort_ols_lines_pass_fail(
    fits, tolerance_pct: float, *, variant_order=None, palette=None,
    figsize=(24, 11), line_alpha=0.3,
):
    """
    2x4 grid: columns are the 4 variants, row 1 is sites whose empirical
    slope falls within `tolerance_pct`% of the required slope (PASSED),
    row 2 is everyone else (FAILED) -- both rows come from the SAME
    tolerance test (add_slope_pass_fail applied once, here), so pass/fail
    membership can't silently disagree between rows. Each panel overlays
    that group's individual OLS lines (from slope/intercept, no
    re-querying), the required reference line, and that group's mean fit.

    Returns the tolerance-flagged DataFrame (fits + pct_error + passed).
    """
    if variant_order is None:
        variant_order = [v["name"] for v in DEFAULT_VVAR_VARIANTS]
    if palette is None:
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

    flagged = add_slope_pass_fail(fits, tolerance_pct)
    v3, vw_v1 = _VVAR["V3"], _VW["V1"]
    v_line = np.linspace(v3, vw_v1, 50)

    fig, axes = plt.subplots(2, len(variant_order), figsize=figsize, dpi=130, sharey=True, sharex=True)
    row_specs = [(True, "PASSED", "#1b7f4c"), (False, "FAILED", "#b91c1c")]

    for row_idx, (want_passed, row_label, row_color) in enumerate(row_specs):
        for col_idx, (variant_name, color) in enumerate(zip(variant_order, palette)):
            ax = axes[row_idx, col_idx]
            sub = flagged.loc[(flagged["variant"] == variant_name) & (flagged["passed"] == want_passed)]

            for _, row in sub.iterrows():
                ax.plot(v_line, row["slope"] * v_line + row["intercept"],
                         color=color, linewidth=1, alpha=line_alpha, zorder=2)

            if len(sub):
                mean_slope = sub["slope"].mean()
                mean_intercept = sub["intercept"].mean()
                ax.plot(v_line, mean_slope * v_line + mean_intercept, color=color, linewidth=2.5, zorder=4,
                        label=f"Mean: {mean_slope:.3f} %/V")

            ax.plot(v_line, _VVAR_ABSORB_SLOPE_PCT_PER_V * (v_line - v3), color="#f59e0b", linewidth=2,
                    linestyle="--", zorder=5, label=f"Required: {_VVAR_ABSORB_SLOPE_PCT_PER_V:.3f} %/V")
            ax.axhline(0, color="black", linewidth=0.5, zorder=1)

            if row_idx == 0:
                ax.set_title(variant_name, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{row_label} (±{tolerance_pct:.0f}%)\nReactive power (% of rating)",
                               fontsize=8.5, color=row_color, fontweight="bold")
            ax.text(0.03, 0.05, f"n={len(sub):,}", transform=ax.transAxes, fontsize=8,
                     color=row_color, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Voltage (V)", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper right")
            ax.grid(color="#ebebeb", linewidth=0.5)

    fig.suptitle(
        f"Cohort site-level OLS fits, split by slope tolerance "
        f"(±{tolerance_pct:.0f}% of {_VVAR_ABSORB_SLOPE_PCT_PER_V:.3f} %/V required)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return flagged


def _required_vvar_pct(v_ref):
    """Required Volt-VAr absorbing %, per the standard's own ramp definition
    (anchored at V3=240, fixed slope `_VVAR_ABSORB_SLOPE_PCT_PER_V`),
    evaluated at an arbitrary reference voltage `v_ref`."""
    return _VVAR_ABSORB_SLOPE_PCT_PER_V * (v_ref - _VVAR_V3)


def add_offset_pct_columns(fits: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `offset_pct_v3`, `offset_pct_v4`, `offset_pct_mid` to a copy of
    `fits`: the fitted OLS line's LEVEL minus the required line's level,
    each evaluated at a fixed reference voltage (V3=240, V4=258, and their
    midpoint=249) -- not at V=0 (the raw `intercept`, which sits far
    outside the observed window and hugely amplifies tiny slope
    differences through extrapolation).

    Sign convention: positive = fitted line sits ABOVE the required line at
    that voltage (under-absorbing / too weak a response); negative = below
    (over-absorbing / too strong a response). All three are computed
    unconditionally so they're available to query later (e.g. "which sites
    are fine at V3 but drift badly by V4"), even though only
    `offset_pct_mid` is used to gate pass/fail for now.

    Requires `fits` to already have `slope` and `intercept` columns (i.e.
    call this after `fetch_cohort_assessable_vvar_slopes` /
    `plot_cohort_assessable_vvar_slopes` / `add_slope_pass_fail`).
    """
    out = fits.copy()
    for label, v_ref in (("v3", _VVAR_V3), ("v4", _VVAR_V4), ("mid", _VVAR_VMID)):
        fitted_pct = out["slope"] * v_ref + out["intercept"]
        out[f"offset_pct_{label}"] = fitted_pct - _required_vvar_pct(v_ref)
    return out


def add_slope_pass_fail_asymmetric(
    fits: pd.DataFrame, *, neg_tolerance_pct: float, pos_tolerance_pct: float,
) -> pd.DataFrame:
    """
    Asymmetric-band companion to `add_slope_pass_fail`. Adds
    `slope_band_low`, `slope_band_high`, `slope_passed_asym` to a copy of
    `fits`, WITHOUT touching `pct_error`/`passed` (those stay whatever the
    symmetric `add_slope_pass_fail`/`plot_cohort_ols_lines_pass_fail` last
    set them to).

    `neg_tolerance_pct`/`pos_tolerance_pct` are % changes to the required
    slope's MAGNITUDE (required = -3.3333 %/V), not raw offsets:

        edge = required_slope * (1 + tolerance_pct / 100)

    Example (your worked numbers): neg_tolerance_pct=-10,
    pos_tolerance_pct=100 on required=-3.3333 gives
    edge_a = -3.3333*(1-0.10) = -3.000 %/V (allowed to under-absorb this much)
    edge_b = -3.3333*(1+1.00) = -6.667 %/V (allowed to over-absorb this much)
    -> band = [-6.667, -3.000], and `slope_passed_asym` is True when `slope`
    falls inside it.
    """
    required = _VVAR_ABSORB_SLOPE_PCT_PER_V
    edge_a = required * (1 + neg_tolerance_pct / 100.0)
    edge_b = required * (1 + pos_tolerance_pct / 100.0)
    lo, hi = min(edge_a, edge_b), max(edge_a, edge_b)

    out = fits.copy()
    out["slope_band_low"] = lo
    out["slope_band_high"] = hi
    out["slope_passed_asym"] = out["slope"].between(lo, hi)
    return out


def add_offset_pass_fail(
    fits: pd.DataFrame, *, neg_tolerance_pp: float = -5.0, pos_tolerance_pp: float = 5.0,
    offset_col: str = "offset_pct_mid",
) -> pd.DataFrame:
    """
    Level/offset pass-fail test. Adds `offset_band_low`, `offset_band_high`,
    `offset_passed` to a copy of `fits`.

    Unlike the slope test, the required value of an offset is always 0 by
    construction (fitted line == required line means offset == 0), so the
    tolerance here is a plain +/- band in PERCENTAGE POINTS around 0, not a
    multiplier on a nonzero required value. Defaults to +/-5pp -- tune to
    taste. Pass asymmetric values if you want more slack on one side (e.g.
    tolerate under-absorbing more than over-absorbing):

        add_offset_pass_fail(fits, neg_tolerance_pp=-8, pos_tolerance_pp=4)

    `offset_col` defaults to the midpoint (249V) offset, per your current
    "one test drives pass/fail for now" choice -- pass "offset_pct_v3" or
    "offset_pct_v4" to gate on one of the other two reference points
    instead. Requires `add_offset_pct_columns(fits)` to have been called
    first (raises KeyError with a clear message otherwise).
    """
    if offset_col not in fits.columns:
        raise KeyError(
            f"{offset_col!r} not found in fits -- call add_offset_pct_columns(fits) first"
        )
    out = fits.copy()
    out["offset_band_low"] = neg_tolerance_pp
    out["offset_band_high"] = pos_tolerance_pp
    out["offset_passed"] = out[offset_col].between(neg_tolerance_pp, pos_tolerance_pp)
    return out


def plot_assessable_vvar_scatter3d(
    classified_df, rating_kw, q_col, *,
    title_suffix="", daytime_hours: tuple[int, int] | None = None,
    ols_grouping: str | None = "scope",
    min_group_n: int = 20,
    show_required_plane: bool = True,
    show_colorbar: bool = True,   # <-- add this
    elev: float = 22, azim: float = -60,
    figsize: tuple[float, float] = (12, 10),
    box_aspect: tuple[float, float, float] = (1.0, 4.0, 1.0),
):
    """
    (Same behavior/docstring as before -- see prior version for the full
    parameter reference.) New in this version:
      - `show_required_plane`: set False to omit the orange AS/NZS 4777.2
        required-ramp plane and show only the scatter + whatever
        `ols_grouping` draws.
      - Lighter 3D pane/grid styling, muted-ink text, and (for
        ols_grouping="surface") a colorbar for the height-to-color mapping.
    """
    import matplotlib.dates as mdates
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
    from scipy import stats as _stats

    window, _, _, _, fit_stats = _fit_assessable_window(
        classified_df, rating_kw, q_col, daytime_hours=daytime_hours,
    )

    v3, v4 = _VVAR["V3"], _VVAR["V4"]
    vw_v1 = _VW["V1"]

    window = window.copy()
    window["t_aest"] = window["t_stamp"] + pd.Timedelta(hours=10)
    window["time_num"] = mdates.date2num(window["t_aest"])

    time_label = f", {daytime_hours[0]:02d}:00-{daytime_hours[1]:02d}:00 AEST" if daytime_hours else ""

    fig = plt.figure(figsize=figsize, dpi=130)
    ax = fig.add_subplot(111, projection="3d")

    for status in STATUS_ORDER:
        sub = window.loc[window["voltvar_status"] == status]
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["V"], sub["time_num"], sub["Q_pct"],
            s=6, alpha=0.35 if status == "Q_conformant" else 0.55,
            color=STATUS_COLORS[status], depthshade=False,
            label=f"{STATUS_LABELS[status]} ({n:,}, {100*n/fit_stats['n_window']:.1f}%)",
        )

    v_grid = np.linspace(v3, vw_v1, 30)
    t_min, t_max = window["time_num"].min(), window["time_num"].max()
    t_span = np.linspace(t_min, t_max, 2) if t_max > t_min else np.array([t_min, t_min + 1])
    V_mesh, T_mesh = np.meshgrid(v_grid, t_span)

    if show_required_plane:
        Q_required_mesh = _VVAR_ABSORB_SLOPE_PCT_PER_V * (V_mesh - v3)
        ax.plot_surface(V_mesh, T_mesh, Q_required_mesh, color="#f59e0b", alpha=0.25, linewidth=0, shade=False)

    ols_label_lines = []
    surf_mappable = None
    if ols_grouping == "scope":
        lin = _stats.linregress(window["V"], window["Q_pct"])
        Q_fit_mesh = lin.slope * V_mesh + lin.intercept
        ax.plot_surface(V_mesh, T_mesh, Q_fit_mesh, color="#2a78d6", alpha=0.18, linewidth=0, shade=False)
        ols_label_lines.append(f"Whole-scope OLS: {lin.slope:.3f} %/V (R²={lin.rvalue**2:.2f}, n={len(window):,})")

    elif ols_grouping in ("day", "week"):
        block_key = (
            window["t_aest"].dt.floor("D") if ols_grouping == "day"
            else window["t_aest"].dt.to_period("W").dt.start_time
        )
        n_blocks_drawn = 0
        for _, sub in window.groupby(block_key):
            if len(sub) < min_group_n:
                continue
            lin = _stats.linregress(sub["V"], sub["Q_pct"])
            t_lo, t_hi = sub["time_num"].min(), sub["time_num"].max()
            t_block = np.linspace(t_lo, t_hi, 2) if t_hi > t_lo else np.array([t_lo, t_lo + 0.3])
            Vb, Tb = np.meshgrid(v_grid, t_block)
            Qb = lin.slope * Vb + lin.intercept
            ax.plot_surface(Vb, Tb, Qb, color="#2a78d6", alpha=0.35, linewidth=0, shade=False)
            n_blocks_drawn += 1
        ols_label_lines.append(f"OLS per {ols_grouping} ({n_blocks_drawn} block(s) with >= {min_group_n} pts)")

    elif ols_grouping == "weekday_weekend":
        is_weekday = window["t_aest"].dt.dayofweek < 5
        for label, mask, color in [("Weekday", is_weekday, "#2a78d6"), ("Weekend", ~is_weekday, "#eb6834")]:
            sub = window.loc[mask]
            if len(sub) < min_group_n:
                continue
            lin = _stats.linregress(sub["V"], sub["Q_pct"])
            Qg = lin.slope * V_mesh + lin.intercept
            ax.plot_surface(V_mesh, T_mesh, Qg, color=color, alpha=0.20, linewidth=0, shade=False)
            ols_label_lines.append(f"{label} OLS: {lin.slope:.3f} %/V (R²={lin.rvalue**2:.2f}, n={len(sub):,})")

    elif ols_grouping == "surface":
        norm = TwoSlopeNorm(vcenter=0, vmin=window["Q_pct"].min(), vmax=max(window["Q_pct"].max(), 0.1))
        surf_mappable = ax.plot_trisurf(
            window["V"], window["time_num"], window["Q_pct"],
            cmap="RdBu_r", norm=norm, alpha=0.65, linewidth=0.05, edgecolor="none",
        )
        ols_label_lines.append(f"Triangulated surface through all {len(window):,} assessable points (no fitting)")

    elif ols_grouping is not None:
        raise ValueError(f"Unknown ols_grouping={ols_grouping!r}")

    # --- lighter panes/grid, muted-ink text (instead of mplot3d's default
    # heavy grey panes + pure black text) -- _axinfo is private API, so this
    # may need adjusting on a future matplotlib version if it stops applying.
    ink, muted_ink, grid_color = "#333333", "#555555", (0, 0, 0, 0.08)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor((0, 0, 0, 0.15))
        axis._axinfo["grid"]["color"] = grid_color
        axis.label.set_color(ink)
    ax.tick_params(colors=muted_ink, labelsize=7.5)

    ax.set_xlabel("Voltage (V)", fontsize=9, labelpad=8)
    ax.set_ylabel("Time (AEST)", fontsize=9, labelpad=28)
    ax.set_zlabel("Reactive power (% of rating)\n+ = supplying, - = absorbing", fontsize=8, labelpad=8)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_yticklabels():
        label.set_rotation(60)
        label.set_ha("right")
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda z, _: f"{z:+.0f}%"))
    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=elev, azim=azim)

    if surf_mappable is not None and show_colorbar:
        cbar = fig.colorbar(surf_mappable, ax=ax, shrink=0.55, pad=0.12)
        cbar.set_label("Reactive power (% of rating)", fontsize=8, color=ink)
        cbar.ax.tick_params(labelsize=7, colors=muted_ink)

    subtitle = "  |  ".join(ols_label_lines) if ols_label_lines else "no reference surface (ols_grouping=None)"
    plane_note = "" if show_required_plane else "  (required-ramp plane hidden)"
    ax.set_title(
        f"Assessable Volt-VAr window [{v3}, {vw_v1}) V{time_label}{title_suffix}{plane_note}\n"
        f"n={fit_stats['n_window']:,} assessable intervals\n{subtitle}",
        fontsize=9, fontweight="bold", color=ink,
    )
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.92, edgecolor="#cccccc")
    fig.tight_layout()
    plt.show()

    return fig, ax, window

def plot_assessable_vvar_scatter3d_interactive(
    classified_df, rating_kw, q_col, *,
    title_suffix="", daytime_hours: tuple[int, int] | None = None,
    ols_grouping: str | None = "day",  # None | "scope" | "day" | "week" | "weekday_weekend" | "surface"
    min_group_n: int = 20,
    show_required_plane: bool = True,
    show_colorbar: bool = False,
    aspect_ratio: tuple[float, float, float] = (1.0, 3.0, 1.0),
    height: int = 750,
):
    """
    Interactive (drag-to-rotate/zoom/pan) counterpart to
    `plot_assessable_vvar_scatter3d`, built on plotly instead of matplotlib
    (`pip install plotly nbformat`). Same assessable-window filter, same
    axes (x=Voltage, y=Time (AEST), z=Reactive power % of rating), same
    `ols_grouping` options and `show_required_plane`/`show_colorbar`
    toggles -- see that function's docstring for what each one draws.

    Differences from the matplotlib version: no `elev`/`azim` (rotate by
    dragging in the rendered widget instead); `aspect_ratio` maps to
    plotly's `scene.aspectratio` the same way `box_aspect` did; hovering a
    point shows its exact V/time/Q% values.

    Returns the plotly Figure -- call `.show()` on it to render (or just
    make it the last expression in a Jupyter cell).
    """
    import plotly.graph_objects as go
    from scipy import stats as _stats
    from scipy.spatial import Delaunay

    window, _, _, _, fit_stats = _fit_assessable_window(
        classified_df, rating_kw, q_col, daytime_hours=daytime_hours,
    )

    v3, v4 = _VVAR["V3"], _VVAR["V4"]
    vw_v1 = _VW["V1"]

    window = window.copy()
    window["t_aest"] = window["t_stamp"] + pd.Timedelta(hours=10)
    time_label = f", {daytime_hours[0]:02d}:00-{daytime_hours[1]:02d}:00 AEST" if daytime_hours else ""

    fig = go.Figure()

    # --- scatter: every assessable point, colored by status ------------- #
    for status in STATUS_ORDER:
        sub = window.loc[window["voltvar_status"] == status]
        if sub.empty:
            continue
        n = len(sub)
        fig.add_trace(go.Scatter3d(
            x=sub["V"], y=sub["t_aest"], z=sub["Q_pct"],
            mode="markers",
            marker=dict(size=3, color=STATUS_COLORS[status], opacity=0.6),
            name=f"{STATUS_LABELS[status]} ({n:,}, {100*n/fit_stats['n_window']:.1f}%)",
            hovertemplate="V=%{x:.1f}<br>Time=%{y}<br>Q=%{z:+.1f}%<extra></extra>",
        ))

    v_edge = [v3, vw_v1]
    t_min, t_max = window["t_aest"].min(), window["t_aest"].max()

    def _flat_surface(v_range, t_range, slope, intercept, color, name, opacity):
        V_mesh, T_mesh = np.meshgrid(v_range, t_range)
        Q_mesh = slope * (V_mesh - v3) + intercept if intercept is None else slope * V_mesh + intercept
        fig.add_trace(go.Surface(
            x=V_mesh, y=T_mesh, z=Q_mesh,
            colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity,
            name=name, hoverinfo="skip",
        ))

    if show_required_plane:
        V_mesh, T_mesh = np.meshgrid(v_edge, [t_min, t_max])
        Q_req = _VVAR_ABSORB_SLOPE_PCT_PER_V * (V_mesh - v3)
        fig.add_trace(go.Surface(
            x=V_mesh, y=T_mesh, z=Q_req, colorscale=[[0, "#f59e0b"], [1, "#f59e0b"]],
            showscale=False, opacity=0.25, name="Required ramp", hoverinfo="skip",
        ))

    ols_label_lines = []
    if ols_grouping == "scope":
        lin = _stats.linregress(window["V"], window["Q_pct"])
        _flat_surface(v_edge, [t_min, t_max], lin.slope, lin.intercept, "#2a78d6", "Whole-scope OLS", 0.25)
        ols_label_lines.append(f"Whole-scope OLS: {lin.slope:.3f} %/V (R²={lin.rvalue**2:.2f}, n={len(window):,})")

    elif ols_grouping in ("day", "week"):
        block_key = (
            window["t_aest"].dt.floor("D") if ols_grouping == "day"
            else window["t_aest"].dt.to_period("W").dt.start_time
        )
        n_blocks_drawn = 0
        for _, sub in window.groupby(block_key):
            if len(sub) < min_group_n:
                continue
            lin = _stats.linregress(sub["V"], sub["Q_pct"])
            t_lo, t_hi = sub["t_aest"].min(), sub["t_aest"].max()
            if t_hi == t_lo:
                t_hi = t_lo + pd.Timedelta(hours=7)
            _flat_surface(v_edge, [t_lo, t_hi], lin.slope, lin.intercept, "#2a78d6", None, 0.45)
            n_blocks_drawn += 1
        ols_label_lines.append(f"OLS per {ols_grouping} ({n_blocks_drawn} block(s) with >= {min_group_n} pts)")

    elif ols_grouping == "weekday_weekend":
        is_weekday = window["t_aest"].dt.dayofweek < 5
        for label, mask, color in [("Weekday", is_weekday, "#2a78d6"), ("Weekend", ~is_weekday, "#eb6834")]:
            sub = window.loc[mask]
            if len(sub) < min_group_n:
                continue
            lin = _stats.linregress(sub["V"], sub["Q_pct"])
            _flat_surface(v_edge, [t_min, t_max], lin.slope, lin.intercept, color, f"{label} OLS", 0.20)
            ols_label_lines.append(f"{label} OLS: {lin.slope:.3f} %/V (R²={lin.rvalue**2:.2f}, n={len(sub):,})")

    elif ols_grouping == "surface":
        time_num_for_tri = window["t_aest"].astype("int64").to_numpy()  # ns since epoch, for triangulation only
        tri = Delaunay(np.column_stack([window["V"].to_numpy(), time_num_for_tri]))
        q_abs_max = max(abs(window["Q_pct"].min()), abs(window["Q_pct"].max()), 0.1)
        fig.add_trace(go.Mesh3d(
            x=window["V"], y=window["t_aest"], z=window["Q_pct"],
            i=tri.simplices[:, 0], j=tri.simplices[:, 1], k=tri.simplices[:, 2],
            intensity=window["Q_pct"], colorscale="RdBu", reversescale=True,
            cmin=-q_abs_max, cmax=q_abs_max, opacity=0.7,
            showscale=show_colorbar, name="Triangulated surface",
            hovertemplate="V=%{x:.1f}<br>Time=%{y}<br>Q=%{z:+.1f}%<extra></extra>",
        ))
        ols_label_lines.append(f"Triangulated surface through all {len(window):,} assessable points (no fitting)")

    elif ols_grouping is not None:
        raise ValueError(f"Unknown ols_grouping={ols_grouping!r}")

    subtitle = "  |  ".join(ols_label_lines) if ols_label_lines else "no reference surface (ols_grouping=None)"
    plane_note = "" if show_required_plane else " (required-ramp plane hidden)"

    fig.update_layout(
        title=dict(
            text=(
                f"Assessable Volt-VAr window [{v3}, {vw_v1}) V{time_label}{title_suffix}{plane_note}<br>"
                f"<sup>n={fit_stats['n_window']:,} assessable intervals  |  {subtitle}</sup>"
            ),
            font=dict(size=13),
        ),
        scene=dict(
            xaxis_title="Voltage (V)",
            yaxis_title="Time (AEST)",
            zaxis_title="Reactive power (% of rating)",
            aspectmode="manual",
            aspectratio=dict(x=aspect_ratio[0], y=aspect_ratio[1], z=aspect_ratio[2]),
        ),
        height=height,
        legend=dict(font=dict(size=9)),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    return fig


# ===========================================================================
# Signed net-export capability gate (net-metered Volt-VAr variants)
# ===========================================================================
#
# ami_conformance.build_voltvar_sql marks an interval assessable when abs(P) >= QCAP.P_MIN * rating. 
# On a PV-only circuit (ground truth) that is fine -- P is the inverter's own output. 
# On a NET-metered circuit, P is (PV - household load), so heavy household IMPORT (e.g. a 03:00 hot-water
# block, 18:00-20:00 evening peak) also passes abs(P) >= 20% rating while the PV inverter is producing ~nothing. 
# Those intervals form the flat Q~0 cluster in the net-metered scatter and tilt the OLS slope.
#
# FIX: 
# require NET EXPORT >= QCAP.P_MIN * rating instead. 
# With no storage on site (storage sites are already excluded upstream), 
# household load >= 0, so net export >= X  ==>  PV output >= X. 
# The signed gate is therefore a *sufficient* condition for the real inverter being assessable 
# (it can't admit a non-generating interval), 
# at the cost of recall (daytime intervals where PV is high but so is the load get dropped).
#
# applied POST-HOC to the already-classified frame from fetch_circuit_scope_conformance, not inside the SQL. 
# This is exact, not an approximation: 
# signed-export >= X implies abs(P) >= X, so the signed gate only ever turns capability_assessable TRUE -> FALSE, 
# never the reverse every row it keeps has exactly the voltvar_status the SQL already gave it.
# Doing it post-hoc also means ONE fetch per (site, variant) serves both gates, 
# so the cohort comparison below costs the same as the existing fetch_cohort_assessable_vvar_slopes run.
# ===========================================================================

GATE_ABS = "abs"                    # existing behaviour: abs(P) >= P_MIN * rating
GATE_SIGNED = "signed_export"       # new: export_sign * P >= P_MIN * rating
GATES = (GATE_ABS, GATE_SIGNED)

GROUND_TRUTH_VARIANT_NAME = "Ground truth (ami_raw_phaseseparate)"

# Per variant: which P column build_voltvar_sql gated on, and which sign
_EXPORT_GATE_SPEC = {
    "Polarity-corrected only": ("P_kw", -1.0),
    "Sign-flip only": ("P_kw", +1.0),
    "Polarity-corrected + baseline estimate": ("P_kw_pv_est", +1.0),
    GROUND_TRUTH_VARIANT_NAME: ("P_kw", +1.0),
}

#: DEFAULT_VVAR_VARIANTS plus the two keys the signed gate needs
SIGNED_GATE_VVAR_VARIANTS = [
    {**v, "power_col": _EXPORT_GATE_SPEC[v["name"]][0], "export_sign": _EXPORT_GATE_SPEC[v["name"]][1]}
    for v in DEFAULT_VVAR_VARIANTS
]


def apply_signed_export_gate(
    classified_df: pd.DataFrame, rating_kw: float, power_col: str, *,
    export_sign: float = 1.0, p_min_frac: float | None = None,
) -> pd.DataFrame:
    """
    Returns a COPY of `classified_df` (output of fetch_circuit_scope_conformance
    / fetch_site_day_conformance, `review_corrected` profile) with the
    capability gate tightened from abs(P) to signed net export:

        capability_assessable := capability_assessable_abs
                                 AND export_sign * power_col >= p_min_frac * rating_kw

    Adds:
      - `capability_assessable_abs`: the original abs(P) gate, kept for audit.
      - `dropped_by_signed_gate`: TRUE where the abs gate passed but the
        signed gate did not (i.e. |P| was large only because of import).
    Updates, on dropped rows only: `voltvar_status` -> 'Q_not_assessable'
    and `Q_impact` -> NaN (same as any other not-assessable interval).

    `export_sign` is +1 if PV export is POSITIVE in `power_col`, -1 if
    NEGATIVE -- see `_EXPORT_GATE_SPEC` / `check_export_sign`. A NaN P fails
    the gate (comparison with NaN is False). `p_min_frac` defaults to the
    standard's own QCAP.P_MIN (0.20), the same threshold build_voltvar_sql uses.
    """
    if export_sign not in (1.0, -1.0, 1, -1):
        raise ValueError(f"export_sign must be +1 or -1, got {export_sign!r}")
    if power_col not in classified_df.columns:
        raise KeyError(
            f"{power_col!r} not in classified_df -- baseline variants gate on "
            "'P_kw_pv_est', raw/sign-flip/ground-truth variants on 'P_kw'."
        )
    p_min_frac = _QCAP["P_MIN"] if p_min_frac is None else p_min_frac

    out = classified_df.copy()
    abs_gate = out["capability_assessable"].fillna(False).astype(bool)
    signed_gate = (export_sign * out[power_col]) >= p_min_frac * rating_kw
    new_gate = abs_gate & signed_gate          # subset of the abs gate, by construction
    dropped = abs_gate & ~new_gate

    out["capability_assessable_abs"] = abs_gate
    out["capability_assessable"] = new_gate
    out["dropped_by_signed_gate"] = dropped
    out.loc[dropped, "voltvar_status"] = "Q_not_assessable"
    if "Q_impact" in out.columns:
        out.loc[dropped, "Q_impact"] = np.nan
    return out


def check_export_sign(
    classified_df: pd.DataFrame, power_col: str, export_sign: float, *,
    midday_hours: tuple[int, int] = (11, 14),
) -> dict:
    """
    Sanity check for `export_sign`: over the AEST `midday_hours` window (when
    a PV site should mostly be exporting), returns the median of
    export_sign * P and the fraction of intervals where it is > 0. Both
    should be clearly positive; a negative median means `export_sign` is
    backwards for this variant (the signed gate would then keep IMPORT and
    drop export -- the opposite of the intent).
    """
    hour_aest = (classified_df["t_stamp"].dt.hour + 10) % 24
    mid = export_sign * classified_df.loc[
        (hour_aest >= midday_hours[0]) & (hour_aest < midday_hours[1]), power_col
    ]
    return {
        "n_midday": int(mid.notna().sum()),
        "median_signed_p_kw": float(mid.median()) if len(mid) else float("nan"),
        "frac_signed_p_positive": float((mid > 0).mean()) if len(mid) else float("nan"),
    }


def _window_fit_with_timestamps(classified_df, rating_kw, q_col, *,
                                daytime_hours=None, v_min=None, v_max=None):
    """
    Same fit as compute_assessable_vvar_fit (via _fit_assessable_window, so
    the filter can never drift), plus the window's t_stamps -- needed to
    measure how well a net-metered variant's assessable set matches ground
    truth's. Raises ValueError (from _fit_assessable_window) if < 2 points.
    """
    window, _, slope, intercept, fs = _fit_assessable_window(
        classified_df, rating_kw, q_col,
        daytime_hours=daytime_hours, v_min=v_min, v_max=v_max,
    )
    fit = {
        "n_total": fs["n_total"], "n_window": fs["n_window"],
        "slope": slope, "intercept": intercept,
        "r_squared": fs["r_squared"], "slope_stderr": fs["slope_stderr"],
        "slope_ci_low": fs["slope_ci95"][0], "slope_ci_high": fs["slope_ci95"][1],
        "v_min": fs["v_min"], "v_max": fs["v_max"],
    }
    return fit, pd.Index(window["t_stamp"].unique())


def fetch_cohort_vvar_slopes_gate_comparison(
    con, site_circuit_map, *,
    scope_where, rating_column="ac_capacity_kw",
    capability_profile="review_corrected",
    metadata_table="ami_site_metadata_residential",
    daytime_hours=None, v_min=None, v_max=None,
    nighttime_hours=(0, 4),
    variants=None,
    ground_truth_variant=GROUND_TRUTH_VARIANT_NAME,
    progress_every: int | None = 10,
) -> pd.DataFrame:
    """
    Like fetch_cohort_assessable_vvar_slopes, but fits every (site, variant)
    under BOTH gates (`gate` column: 'abs' / 'signed_export') from a single
    fetch each. Output is long: one row per (site_id, variant, gate), with the
    same fit columns as fetch_cohort_assessable_vvar_slopes (so filtering to
    one gate gives a frame the existing plot/pass-fail helpers accept), plus:

      - power_col, export_sign            -- what the signed gate used
      - n_dropped_by_gate                 -- (signed rows) abs-window points the
                                             signed gate removed
      - gt_slope                          -- ground truth's slope (abs gate, i.e.
                                             ground truth's own definition)
      - slope_err_vs_gt                   -- slope - gt_slope (%rating/V)
      - n_window_gt, n_overlap_gt         -- ground-truth window size, and how
                                             many of this row's window t_stamps
                                             are also in it
      - precision_vs_gt, recall_vs_gt     -- overlap / n_window, overlap / n_window_gt

    The ground-truth comparison columns are NaN where the site has no usable
    ground-truth fit. `daytime_hours`, `v_min`, `v_max` are passed straight to
    _fit_assessable_window (defaults = the existing notebook's behaviour).
    Errors are recorded per row in `status`, never abort the run.
    """
    if capability_profile != "review_corrected":
        raise ValueError(
            "The signed gate tightens the review_corrected abs(P) gate; under "
            f"{capability_profile!r} every interval is assessable, so there is "
            "nothing equivalent to compare."
        )
    if variants is None:
        variants = SIGNED_GATE_VVAR_VARIANTS
    for v in variants:
        missing = {"power_col", "export_sign"} - set(v)
        if missing:
            raise KeyError(f"variant {v['name']!r} is missing {sorted(missing)} -- use SIGNED_GATE_VVAR_VARIANTS")

    rows = []
    n_sites = len(site_circuit_map)
    for i, (_, site_row) in enumerate(site_circuit_map.iterrows(), start=1):
        site_id = int(site_row["site_id"])
        site_rows, windows = [], {}

        for variant in variants:
            base = {"site_id": site_id, "variant": variant["name"],
                    "power_col": variant["power_col"], "export_sign": variant["export_sign"]}
            circuit_id = site_row.get(variant["circuit_col"])
            if pd.isna(circuit_id):
                site_rows += [{**base, "gate": g, "status": "missing_circuit"} for g in GATES]
                continue
            circuit_id = int(circuit_id)
            base["circuit_id"] = circuit_id

            try:
                classified_df, rating_kw = fetch_circuit_scope_conformance(
                    con, site_id, circuit_id,
                    scope_where=scope_where, rating_column=rating_column,
                    capability_profile=capability_profile,
                    source_table=variant["source_table"], metadata_table=metadata_table,
                    use_baseline_correction=variant["use_baseline_correction"],
                    nighttime_hours=nighttime_hours,
                )
            except ValueError as exc:
                site_rows += [{**base, "gate": g, "status": f"error: {exc}"} for g in GATES]
                continue

            gated = {
                GATE_ABS: classified_df,
                GATE_SIGNED: apply_signed_export_gate(
                    classified_df, rating_kw, variant["power_col"], export_sign=variant["export_sign"],
                ),
            }
            for gate, df in gated.items():
                row = {**base, "gate": gate, "rating_kw": rating_kw}
                try:
                    fit, window_ts = _window_fit_with_timestamps(
                        df, rating_kw, variant["q_col"],
                        daytime_hours=daytime_hours, v_min=v_min, v_max=v_max,
                    )
                except ValueError as exc:
                    site_rows.append({**row, "status": f"error: {exc}"})
                    continue
                windows[(variant["name"], gate)] = window_ts
                site_rows.append({**row, "status": "ok", **fit})

        # --- how many abs-window points did the signed gate remove? ---
        for r in site_rows:
            if r["gate"] == GATE_SIGNED and r["status"] == "ok" and (r["variant"], GATE_ABS) in windows:
                r["n_dropped_by_gate"] = len(windows[(r["variant"], GATE_ABS)]) - len(windows[(r["variant"], GATE_SIGNED)])

        # --- compare every ok row against ground truth's own (abs-gate) window ---
        gt_key = (ground_truth_variant, GATE_ABS)
        gt_row = next((r for r in site_rows if r["variant"] == ground_truth_variant
                       and r["gate"] == GATE_ABS and r["status"] == "ok"), None)
        if gt_row is not None:
            w_gt = windows[gt_key]
            for r in site_rows:
                if r["status"] != "ok":
                    continue
                w = windows[(r["variant"], r["gate"])]
                overlap = len(w.intersection(w_gt))
                r.update({
                    "gt_slope": gt_row["slope"],
                    "slope_err_vs_gt": r["slope"] - gt_row["slope"],
                    "n_window_gt": len(w_gt),
                    "n_overlap_gt": overlap,
                    "precision_vs_gt": overlap / len(w) if len(w) else np.nan,
                    "recall_vs_gt": overlap / len(w_gt) if len(w_gt) else np.nan,
                })

        rows.extend(site_rows)
        if progress_every and (i % progress_every == 0 or i == n_sites):
            print(f"  {i}/{n_sites} sites done")

    return pd.DataFrame(rows)


def summarise_gate_comparison(
    fits: pd.DataFrame, *, min_n_window: int = 20, precision_threshold: float = 0.95,
    paired_only: bool = True,
) -> pd.DataFrame:
    """
    One row per (variant, gate) summarising fetch_cohort_vvar_slopes_gate_comparison:
    site count, median window size, median slope / R^2, median precision and
    recall vs ground truth, number of sites whose precision falls below
    `precision_threshold` (i.e. > 5% of their window isn't assessable by
    ground truth's own definition), and median / 90th-percentile
    |slope error vs ground truth|.

    Rows with status != 'ok' or n_window < min_n_window are excluded. With
    `paired_only` (default), a (site, variant) is kept only if BOTH gates
    survive that filter, so the abs vs signed comparison is over the same
    sites (the signed gate can push a thin site below min_n_window).
    """
    ok = fits.loc[(fits["status"] == "ok") & (fits["n_window"] >= min_n_window)].copy()
    if paired_only:
        n_gates = ok.groupby(["site_id", "variant"])["gate"].transform("nunique")
        ok = ok.loc[n_gates == len(GATES)]
    ok["abs_slope_err_vs_gt"] = ok["slope_err_vs_gt"].abs()

    below_col = f"n_sites_precision_below_{precision_threshold:.2f}"
    summary = (
        ok.groupby(["variant", "gate"], sort=False)
        .agg(
            n_sites=("site_id", "nunique"),
            median_n_window=("n_window", "median"),
            median_slope=("slope", "median"),
            median_r_squared=("r_squared", "median"),
            median_precision_vs_gt=("precision_vs_gt", "median"),
            **{below_col: ("precision_vs_gt", lambda s: int((s < precision_threshold).sum()))},
            median_recall_vs_gt=("recall_vs_gt", "median"),
            median_abs_slope_err=("abs_slope_err_vs_gt", "median"),
            p90_abs_slope_err=("abs_slope_err_vs_gt", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )
    return summary


def plot_site_gate_comparison(
    panels, *, daytime_hours=None, v_min=None, v_max=None, suptitle="", figsize=None,
):
    """
    One panel per variant for ONE site: the abs-gate assessable window,
    split into points the signed gate KEEPS (blue dots) and DROPS (orange
    crosses), with the required ramp, the abs-gate OLS line (grey dashed,
    what the notebook currently fits) and the signed-gate OLS line (green).

    `panels`: list of dicts with keys
        label, classified_df, rating_kw, q_col, power_col, export_sign
    Returns (fig, results) -- `results` is a DataFrame with each panel's
    n / slope / R^2 under both gates.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.5 * n, 6), dpi=130, sharey=True, squeeze=False)
    v3 = _VVAR["V3"]
    results = []

    for ax, p in zip(axes[0], panels):
        rating = p["rating_kw"]
        gated = apply_signed_export_gate(
            p["classified_df"], rating, p["power_col"], export_sign=p["export_sign"],
        )
        res = {"variant": p["label"]}
        try:
            w_abs, _, s_abs, i_abs, fs_abs = _fit_assessable_window(
                p["classified_df"], rating, p["q_col"],
                daytime_hours=daytime_hours, v_min=v_min, v_max=v_max,
            )
        except ValueError as exc:
            ax.set_title(f"{p['label']}\n{exc}", fontsize=9)
            results.append(res)
            continue
        res.update(n_abs=fs_abs["n_window"], slope_abs=s_abs, r2_abs=fs_abs["r_squared"])

        kept = gated.loc[w_abs.index, "capability_assessable"].to_numpy()
        ax.scatter(w_abs.loc[kept, "V"], w_abs.loc[kept, "Q_pct"], s=6, alpha=0.3, color="#2a78d6",
                   zorder=2, label=f"Kept by signed gate (n={kept.sum():,})")
        ax.scatter(w_abs.loc[~kept, "V"], w_abs.loc[~kept, "Q_pct"], s=12, alpha=0.5, marker="x",
                   linewidths=0.8, color="#eb6834", zorder=3,
                   label=f"Dropped: |P| from import only (n={(~kept).sum():,})")

        lo, hi = fs_abs["v_min"], fs_abs["v_max"]
        v_line = np.linspace(lo, hi, 50)
        ax.plot(v_line, _VVAR_ABSORB_SLOPE_PCT_PER_V * (v_line - v3), color="#f59e0b", linewidth=2, zorder=4,
                label=f"Required: {_VVAR_ABSORB_SLOPE_PCT_PER_V:.2f} %/V")
        ax.plot(v_line, s_abs * v_line + i_abs, color="#6b7280", linewidth=1.8, linestyle="--", zorder=5,
                label=f"OLS |P| gate: {s_abs:.2f} %/V, R²={fs_abs['r_squared']:.2f}")

        try:
            _, _, s_sig, i_sig, fs_sig = _fit_assessable_window(
                gated, rating, p["q_col"], daytime_hours=daytime_hours, v_min=v_min, v_max=v_max,
            )
            res.update(n_signed=fs_sig["n_window"], slope_signed=s_sig, r2_signed=fs_sig["r_squared"])
            ax.plot(v_line, s_sig * v_line + i_sig, color="#1baf7a", linewidth=2, zorder=6,
                    label=f"OLS signed gate: {s_sig:.2f} %/V, R²={fs_sig['r_squared']:.2f}")
        except ValueError as exc:
            res.update(n_signed=0)
            ax.text(0.02, 0.98, f"Signed gate: {exc}", transform=ax.transAxes, fontsize=7, va="top")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(p["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Voltage (V)", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(color="#ebebeb", linewidth=0.5)
        results.append(res)

    axes[0][0].set_ylabel("Reactive power (% of rating)\n+ = supplying, - = absorbing", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()
    return fig, pd.DataFrame(results)


def plot_cohort_gate_slope_error(
    fits: pd.DataFrame, *, min_n_window: int = 20, variant_order=None,
    ylim: tuple[float, float] | None = None, figsize=None,
) -> pd.DataFrame:
    """
    Paired before/after plot, one panel per net-metered variant: each site's
    slope error vs ground truth (%rating/V) under the abs gate (left, orange)
    and the signed gate (right, blue), joined by a grey line; black diamonds
    = medians. Points nearer 0 = closer to the real inverter's slope. Only
    sites where both gates have n_window >= min_n_window are drawn. `ylim`
    clips the view (the count of points outside it is printed in the title).
    Returns the paired (site_id, variant, abs, signed_export) frame.
    """
    ok = fits.loc[(fits["status"] == "ok") & (fits["n_window"] >= min_n_window)
                  & fits["slope_err_vs_gt"].notna()]
    paired = (ok.pivot_table(index=["site_id", "variant"], columns="gate", values="slope_err_vs_gt")
                .dropna(subset=list(GATES)).reset_index())
    if variant_order is None:
        variant_order = [v["name"] for v in SIGNED_GATE_VVAR_VARIANTS
                         if v["name"] != GROUND_TRUTH_VARIANT_NAME and v["name"] in set(paired["variant"])]

    n = len(variant_order)
    fig, axes = plt.subplots(1, n, figsize=figsize or (4.5 * n, 6), dpi=130, sharey=True, squeeze=False)
    for ax, variant in zip(axes[0], variant_order):
        d = paired.loc[paired["variant"] == variant]
        for _, r in d.iterrows():
            ax.plot([0, 1], [r[GATE_ABS], r[GATE_SIGNED]], color="#c8c8c8", linewidth=0.8, zorder=2)
        ax.scatter(np.zeros(len(d)), d[GATE_ABS], s=18, color="#eb6834", zorder=3, label="|P| gate (current)")
        ax.scatter(np.ones(len(d)), d[GATE_SIGNED], s=18, color="#2a78d6", zorder=3, label="Signed export gate")
        med = [d[GATE_ABS].median(), d[GATE_SIGNED].median()]
        ax.plot([0, 1], med, color="black", linewidth=2, marker="D", markersize=7, zorder=4, label="Median")
        ax.axhline(0, color="black", linewidth=0.6, zorder=1)

        n_out = 0
        if ylim is not None:
            vals = d[list(GATES)].to_numpy().ravel()
            n_out = int(((vals < ylim[0]) | (vals > ylim[1])).sum())
        med_abs_err = (d[GATE_ABS].abs().median(), d[GATE_SIGNED].abs().median())
        ax.set_title(
            f"{variant}\nn={len(d)} sites; median |err| {med_abs_err[0]:.2f} \u2192 {med_abs_err[1]:.2f} %/V"
            + (f"\n({n_out} points outside view)" if n_out else ""),
            fontsize=9, fontweight="bold",
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["|P| gate\n(current)", "Signed export\ngate"], fontsize=8.5)
        ax.set_xlim(-0.4, 1.4)
        ax.grid(color="#ebebeb", linewidth=0.5, axis="y")
    axes[0][0].set_ylabel("Slope error vs ground truth (%rating/V)\n0 = matches the real inverter", fontsize=9)
    axes[0][0].legend(fontsize=7.5, loc="lower left")
    if ylim is not None:
        axes[0][0].set_ylim(*ylim)
    fig.tight_layout()
    plt.show()
    return paired