"""
Sensitivity analysis.
=====================

Deliverable D15.

Every headline number in this study rests on choices that were forced by the data
rather than chosen on merit. This module sweeps them, so each result ships with a
range instead of a point estimate.

The axes, in rough order of how much they are likely to matter
---------------------------------------------------------------
``reactive_orientation``
    The unresolved one. 213 sites fit as-delivered, 106 fit flipped. Anything
    direction-based (``Q_adverse``, Method A's ``Q < 0`` gate) moves with it.

``rating_basis`` / ``empirical_limit_basis``
    ``s_99`` vs ``s_95`` vs ``s_max``. No nameplate exists, so the whole
    AS/NZS 4777.2 curve is scaled by an OBSERVED quantile. A site that never
    approached its inverter limit gets a low ``s_99``, a smaller required Q, and a
    flattering verdict -- while simultaneously tripping Method A's apparent-limit
    test more readily. The two biases run in opposite directions for the two
    methods, which is worth showing rather than asserting.

``voltage_aggregation``
    ``mean`` (default, and correct for three-phase) vs ``max``. Worth ~5
    percentage points of site conformance.

``tolerance_fraction``
    The +/-4% band, re-anchored to ``s_99`` because there is no nameplate.

``night_anomaly_selection``
    Whether the 20 night-generation sites (5 likely storage, 15 stray timestamps)
    are in or out.

``peak_hour_end``
    Method A's window. The legacy query used ``BETWEEN start AND end``, inclusive,
    so reproducing it needs ``peak_hour_end=15`` against this half-open version.

What a sweep can and cannot tell you
------------------------------------
A number that barely moves across a sweep is robust to that choice. A number that
moves a lot is CONDITIONAL on it, and must be reported as such -- not averaged
across the sweep, which would invent a value no defensible configuration produces.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from oem_analysis.config import se_config as C
from oem_analysis.lib import se_params
# One percentage helper for the whole project: numpy NaN on a zero
# denominator, never pd.NA (which promotes int columns to object and then
# breaks .round()).
from oem_analysis.lib.se_conformance import _pct

__all__ = [
    "sweep_conformance",
    "sweep_method_a",
    "tornado",
    "sweep_min_intervals",
    "min_interval_exposure_profile",
    "MIN_INTERVAL_SWEEP",
    "DEFAULT_SWEEPS",
]

#: The standard axes. Each entry maps a label to the config changes it applies.
DEFAULT_SWEEPS: dict[str, dict] = {
    "baseline":                       {},
    "reactive sign: flipped":         {"reactive_orientation": "flipped"},
    "voltage: max of phases":         {"voltage_aggregation": "max"},
    "capacity basis: s_95":           {"rating_basis": "s_95", "empirical_limit_basis": "s_95",
                                       "tolerance_basis": "s_95"},
    "capacity basis: s_max":          {"rating_basis": "s_max", "empirical_limit_basis": "s_max",
                                       "tolerance_basis": "s_max"},
    "tolerance: 2%":                  {"tolerance_fraction": 0.02},
    "tolerance: 6%":                  {"tolerance_fraction": 0.06},
    "include night-anomaly sites":    {"night_anomaly_selection": "include"},
    "exclude derating intervals":     {"derating_selection": "exclude"},
    "sites with >= 300 days":         {"min_days_observed": 300},
    "single-phase only":              {"phase_cohort": "single"},
    "three-phase only":               {"phase_cohort": "three"},
}


def sweep_conformance(
    con: duckdb.DuckDBPyConnection,
    sweeps: dict[str, dict] | None = None,
    config=None, params=None, verbose: bool = True,
) -> pd.DataFrame:
    """
    Volt-VAr conformance under each configuration.

    One full interval-level rescore per row, so this is the expensive notebook.
    Roughly a minute per sweep on a laptop; the default set is twelve.

    Returns fleet reduced non-conformance, the site conformance rate, and the
    assessable denominator -- the last because a sweep that moves the rate by
    moving the DENOMINATOR is telling you something quite different from one that
    moves the numerator.
    """
    from oem_analysis.lib import se_conformance as cf

    base = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    sweeps = sweeps or DEFAULT_SWEEPS

    rows = []
    for label, changes in sweeps.items():
        cfg = base.with_changes(**changes) if changes else base
        site_day = cf.voltvar_site_day(con, cfg, params)
        summary = cf.voltvar_summary(site_day, by_cohort=False).iloc[0]
        verdicts = cf.voltvar_site_verdicts(site_day, cfg)
        assessable = verdicts.assessable > 0
        rows.append({
            "scenario": label,
            "changes": ", ".join(f"{k}={v}" for k, v in changes.items()) or "(defaults)",
            "n_sites": int(verdicts.site_alias.nunique()),
            "assessable_intervals": int(summary.capability_assessable_intervals),
            "reduced_nonconf_pct": round(summary.reduced_nonconf_pct, 2),
            "Q_adverse_pct": summary.Q_adverse_pct,
            "Q_significant_shortfall_pct": summary.Q_significant_shortfall_pct,
            "pct_sites_conformant": round(
                100 * (verdicts.loc[assessable, "verdict"] == "conformant").mean(), 2),
        })
        if verbose:
            print(f"  {label:<34} reduced NC {rows[-1]['reduced_nonconf_pct']:>6.2f}%  "
                  f"sites conformant {rows[-1]['pct_sites_conformant']:>6.2f}%", flush=True)
    return pd.DataFrame(rows)


def sweep_method_a(
    con: duckdb.DuckDBPyConnection,
    sweeps: dict[str, dict] | None = None,
    config=None, params=None,
    adverse: pd.DataFrame | None = None,
    param_sweeps: dict[str, dict] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Method A under each configuration.

    ``param_sweeps`` additionally varies ``SEVoltVarParams`` -- the detection
    window rather than the cohort. Useful for the peak-hour and voltage-band
    choices, which are detection parameters, not cohort definitions.

    Watch ``energy_kWh`` against ``eligible_intervals``. The capacity basis moves
    both: a lower ``s_limit`` makes the apparent-limit test fire more often
    (raising the count) while shrinking the headroom displacement per interval
    (lowering the energy). Reporting only one would mislead.
    """
    from oem_analysis.lib import se_curtailment as cu

    base_cfg = (config or se_params.CONFIG).validate()
    base_par = (params or se_params.PARAMS).validate()
    sweeps = sweeps or {k: v for k, v in DEFAULT_SWEEPS.items()
                        if k not in ("include night-anomaly sites",)}
    param_sweeps = param_sweeps or {
        "peak window 11-14 (default)": {},
        "peak window 11-15 (legacy inclusive)": {"peak_hour_end": 15},
        "peak window 10-16 (wide)": {"peak_hour_start": 10, "peak_hour_end": 16},
        "no symptom gate": {"require_apparent_limit_symptom": False},
    }

    rows = []
    for label, changes in sweeps.items():
        cfg = base_cfg.with_changes(**changes) if changes else base_cfg
        frame = cu.method_a_site_year(con, cfg, base_par, adverse=adverse)
        summary = cu.method_a_summary(frame, cfg, by_cohort=False).iloc[0]
        rows.append(_method_a_row(label, changes, summary))
        if verbose:
            print(f"  [cohort] {label:<34} {rows[-1]['energy_kWh']:>10,.0f} kWh", flush=True)

    for label, changes in param_sweeps.items():
        par = base_par.with_changes(**changes) if changes else base_par
        frame = cu.method_a_site_year(con, base_cfg, par, adverse=adverse)
        summary = cu.method_a_summary(frame, base_cfg, by_cohort=False).iloc[0]
        rows.append(_method_a_row(label, changes, summary))
        if verbose:
            print(f"  [params] {label:<34} {rows[-1]['energy_kWh']:>10,.0f} kWh", flush=True)

    return pd.DataFrame(rows)


def _method_a_row(label, changes, summary) -> dict:
    return {
        "scenario": label,
        "changes": ", ".join(f"{k}={v}" for k, v in changes.items()) or "(defaults)",
        "eligible_sites": int(summary.eligible_sites),
        "symptom_sites": int(summary.symptom_sites),
        "eligible_intervals": int(summary.eligible_intervals),
        "symptom_intervals": int(summary.symptom_intervals),
        "symptom_pct_of_eligible": summary.symptom_pct_of_eligible,
        "energy_kWh": summary.headroom_displacement_kWh,
    }


def tornado(sweep: pd.DataFrame, metric: str, baseline_label: str = "baseline") -> pd.DataFrame:
    """
    Rank scenarios by how far they move a metric from baseline.

    The output to read before writing any number down. Anything near the top is a
    choice the result is conditional on, and belongs in the limitations rather
    than in a footnote.
    """
    if baseline_label not in set(sweep.scenario):
        raise ValueError(f"no row labelled {baseline_label!r} in the sweep")
    base = float(sweep.loc[sweep.scenario == baseline_label, metric].iloc[0])

    out = sweep[["scenario", "changes", metric]].copy()
    out["baseline"] = base
    out["delta"] = out[metric] - base
    out["pct_change"] = (100 * out.delta / base).round(2) if base else None
    out["abs_delta"] = out.delta.abs()
    return (out[out.scenario != baseline_label]
            .sort_values("abs_delta", ascending=False)
            .drop(columns="abs_delta")
            .reset_index(drop=True))


#: Default sweep for the site-level minimum-interval rule. Starts at 1 (the
#: original's behaviour, and ours) and runs to 20.
MIN_INTERVAL_SWEEP = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50)


def sweep_min_intervals(site_day: pd.DataFrame, mode: str = "voltwatt",
                        config=None, minimums=MIN_INTERVAL_SWEEP) -> pd.DataFrame:
    """
    How much does the site conformance rate depend on the minimum-interval rule?

    A site with one assessable interval can only score 0% or 100%, so the 10%
    rule has no meaning there -- the verdict is decided by a single reading. This
    sweeps the threshold and reports what the fleet rate does.

    Neither Hossein's original nor ``bms_sa_review`` applies a minimum: the
    original has no count filter anywhere for Volt-Watt (the only ``total_count >
    5`` in that repository is in ``sustained_operation.ipynb``, a different
    analysis), and the review sets ``min_site_intervals = 1`` while providing
    ``minimum_interval_sensitivity`` and plotting it. So the defensible position
    is not a chosen threshold -- it is this curve, reported.

    ``pct_dropped_conformant`` is the column that shows the DIRECTION of the bias.
    If the sites being removed were mostly conformant, raising the minimum pushes
    the rate DOWN, and a lower headline is an artefact of the filter rather than
    worse behaviour. If they were mostly non-conformant, the reverse.

    ``mode`` selects the denominator: ``"voltwatt"`` uses voltage-exposed
    intervals, ``"voltvar"`` capability-assessable ones.
    """
    from oem_analysis.lib import se_conformance as cf

    config = (config or se_params.CONFIG).validate()
    if mode not in ("voltwatt", "voltvar"):
        raise ValueError(f"mode must be 'voltwatt' or 'voltvar', got {mode!r}")

    if mode == "voltwatt":
        verdicts = cf.voltwatt_site_verdicts(site_day, config)
        count_col, denom_label = "exposed", "voltage-exposed intervals (V > 253 V)"
    else:
        verdicts = cf.voltvar_site_verdicts(site_day, config)
        count_col, denom_label = "assessable", "capability-assessable intervals"

    verdicts = verdicts.copy()
    verdicts["cohort"] = verdicts.is_three_phase.map(
        {True: "three-phase", False: "single-phase"})
    # Only sites with something to assess can move under this rule.
    tested = verdicts[verdicts[count_col].fillna(0) > 0]
    n_tested = len(tested)

    rows = []
    for m in minimums:
        kept = tested[tested[count_col] >= m]
        dropped = tested[tested[count_col] < m]
        n_conf = int((kept.verdict == "conformant").sum())
        rows.append({
            "min_intervals": m,
            "denominator": denom_label,
            "n_sites": len(kept),
            "n_dropped": len(dropped),
            "pct_sites_dropped": _pct(len(dropped), n_tested),
            "pct_conformant": _pct(n_conf, len(kept)),
            "pct_nonconformant": _pct(len(kept) - n_conf, len(kept)),
            # Which way does the filter push the headline?
            "pct_dropped_conformant": _pct(
                (dropped.verdict == "conformant").sum(), len(dropped)),
            "median_nonconf_frac": round(float(kept.nonconf_fraction.median()), 4)
                                   if len(kept) else float("nan"),
        })
    return pd.DataFrame(rows)


def min_interval_exposure_profile(site_day: pd.DataFrame, mode: str = "voltwatt",
                                  config=None) -> pd.DataFrame:
    """
    How thin is the evidence, site by site? The distribution behind the sweep.

    Also reports how many NON-CONFORMANT verdicts rest on very few intervals,
    which is the specific worry -- a site called non-conformant on one reading.
    """
    from oem_analysis.lib import se_conformance as cf

    config = (config or se_params.CONFIG).validate()
    verdicts = (cf.voltwatt_site_verdicts(site_day, config) if mode == "voltwatt"
                else cf.voltvar_site_verdicts(site_day, config))
    count_col = "exposed" if mode == "voltwatt" else "assessable"

    tested = verdicts[verdicts[count_col].fillna(0) > 0]
    nonconf = tested[tested.verdict == "non-conformant"]

    rows = []
    for threshold in (1, 2, 3, 5, 10, 20, 50, 100):
        at_or_below = tested[tested[count_col] <= threshold]
        nc_below = nonconf[nonconf[count_col] <= threshold]
        rows.append({
            "intervals_at_or_below": threshold,
            "n_sites": len(at_or_below),
            "pct_of_tested_sites": _pct(len(at_or_below), len(tested)),
            "n_nonconformant": len(nc_below),
            "pct_of_all_nonconformant": _pct(len(nc_below), len(nonconf)),
        })
    print(f"{len(tested):,} sites with >= 1 {count_col} interval | "
          f"median {tested[count_col].median():,.0f} | "
          f"{len(nonconf):,} non-conformant")
    return pd.DataFrame(rows)
