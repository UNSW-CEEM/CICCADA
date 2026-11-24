"""
Volt-VAr conformance metrics shared by every pipeline (pandas).
=============================================================================

Considering the zones:

# Zones (P as fraction of S_rated; voltage boundaries are the curve's own):
#   A  response required                    |  P >= 20%,  V < 220 or V > 240
#   B  sufficient power, voltage deadband   |  P >= 20%,  220 <= V <= 240
#   C  low power, voltage exposed           |  P <  20%,  V < 220 or V > 240   (incl. night)
#   D  low power, voltage deadband          |  P <  20%,  220 <= V <= 240      (incl. night)
# Descriptive names in as4777_curves.ZONE_LABELS.

Three selectable outputs:

1. Non-conformance frequency ("which sites to fix first")
    failures in zones A+B+C+D / all valid timestamps (day and night),
    reported next to voltage exposure = zone A intervals / all valid timestamps.
    Decomposition (all over n_valid):
       frequency = exposure * (1 - zone A pass rate) + fail_B + fail_C + fail_D
2. Remote conformance rate ("which inverters respond when needed")
    denominator = zone A intervals + zone C intervals that fail.
    Voltage deadband (B, D) excluded.
    "not assessable" below the minimum
       (CONFORMANCE MIN_ZONE_A_INTERVALS over at least MIN_ZONE_A_DAYS days).
3. Legacy conformance rate (back-comparison): P >= 20% at any voltage, failures =
   adverse + inactive + significant shortfall from the existing Stage 2
   nearest-edge Q_impact buckets (deadband deviations score as passes there).

Workflow, identical in every pipeline:
    scored = score_intervals(df, tol=0.04)        # one row per interval
    counts = zone_counts(scored, by=["site_id", "month"])
    out    = conformance_metrics(counts, failure_definition="reduced")

`zone_counts` stores interval counts per zone x category,
so every metric is a sum of columns and all columns are additive across months
(also n_days_A, because days do not overlap between months).
The failure definition is applied at metric time, so one count table serves both.
The tolerance is baked into the categories and recorded in a `tol` column,
never add up count tables with different `tol`.

Valid interval: V, P, Q and S_rated present, S_rated > 0, 0 < V < 300 V.
Invalid intervals are dropped from every denominator and counted by reason.
"""

import calendar

import numpy as np
import pandas as pd
from shared.as4777_curves import (
    VVAR_CATEGORIES,
    ZONES,
    q_conformance_floor_absorbing,
    q_impact_nearest_edge,
    vvar_category,
    vvar_required_q_np,
    vvar_zone,
)
from shared.ciccada_config import AS4777, CONFORMANCE

_QIMP = AS4777["QIMP"]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _check_tol(tol, name="tol"):
    if not (0.0 <= float(tol) <= 0.2):
        raise ValueError(
            f"{name} must be a fraction of S_rated in [0, 0.2], got {tol!r}"
        )
    return float(tol)


def failure_categories(failure_definition):
    """Categories counted as failures for 'reduced' or 'strict'."""
    try:
        cats = CONFORMANCE["FAILURE_DEFINITIONS"][failure_definition]
        unknown = set(cats) - set(VVAR_CATEGORIES)
        if unknown:
            raise ValueError(
                f"FAILURE_DEFINITIONS[{failure_definition!r}] names unknown "
                f"categories {sorted(unknown)}; valid: {VVAR_CATEGORIES}"
            )
        return cats
    except KeyError as exc:
        raise ValueError(
            "failure_definition must be one of "
            f"{sorted(CONFORMANCE['FAILURE_DEFINITIONS'])}, got {failure_definition!r}"
        ) from exc


def expected_intervals(year, month, interval_minutes=CONFORMANCE["INTERVAL_MINUTES"]):
    """Intervals in a calendar month."""
    days = calendar.monthrange(int(year), int(month))[1]
    return days * 24 * 60 // interval_minutes


# ---------------------------------------------------------------------------
# Legacy (Stage 2 / OEM 'review_corrected') definition, for back-comparison
# ---------------------------------------------------------------------------
def legacy_category(q, v, p, s_rated, tol=CONFORMANCE["TOL_DEFAULT"]):
    """
    Existing Stage 2 bucket per interval
    (build_conformance_voltvar.py, capability_profile='review_corrected'): clamped band, nearest-edge Q_impact, then QIMP thresholds.
    Returns (assessable, category) arrays.
    Category is None when Q is inside the band or the interval is below 20% P.
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    p = np.asarray(p, dtype=float)
    s = np.asarray(s_rated, dtype=float)
    t = tol * s
    curve = vvar_required_q_np(v, s)
    q_cap_abs = np.vectorize(q_conformance_floor_absorbing, otypes=[float])(p, s)
    vmax, vmin = curve + t, curve - t
    q_max = np.where(vmax < 0, np.maximum(vmax, q_cap_abs + t), vmax)
    q_min = np.where(vmin > 0, np.minimum(vmin, -q_cap_abs - t), vmin)
    with np.errstate(invalid="ignore"):
        assessable = np.abs(p) >= AS4777["QCAP"]["P_MIN"] * s
        outside = (q < q_min) | (q > q_max)
    impact = np.vectorize(q_impact_nearest_edge, otypes=[float])(q, q_min, q_max)
    cat = np.select(
        [
            ~(assessable & outside),
            impact < _QIMP["thr1"],
            impact <= _QIMP["thr2"],
            impact < _QIMP["thr3"],
            impact <= _QIMP["thr4"],
        ],
        [None, "adverse", "inactive", "significant_shortfall", "near_conformant"],
        "surplus",
    ).astype(object)
    return assessable, cat


LEGACY_FAILURES = ("adverse", "inactive", "significant_shortfall")


# ---------------------------------------------------------------------------
# 1. Score every interval
# ---------------------------------------------------------------------------
def score_intervals(
    df,
    *,
    v="V",
    p="P_kW",
    q="Q_kvar",
    s="s_rated",
    tol=CONFORMANCE["TOL_DEFAULT"],
    p_tol=CONFORMANCE["P_TOL_DEFAULT"],
    apply_p_tol_vvar=CONFORMANCE["APPLY_P_TOL_VVAR"],
    key=None,
    with_legacy=True,
):
    """
    Add validity flags, zone, category (and the legacy-metric bucket) to a one-row-per-interval DataFrame.
    Column names are parameters so every pipeline can pass its own.
    `key` (e.g. ["site_id", "t_stamp"]) drops duplicate timestamps; the number dropped is returned in `attrs`.
    """
    tol = _check_tol(tol)
    p_tol = _check_tol(p_tol, "p_tol")
    p_tol_vvar = p_tol if apply_p_tol_vvar else 0.0

    out = df.copy()
    n_dup = 0
    if key is not None:
        dup = out.duplicated(subset=list(key), keep="first")
        n_dup = int(dup.sum())
        out = out.loc[~dup]

    V = pd.to_numeric(out[v], errors="coerce").to_numpy(dtype=float)
    P = pd.to_numeric(out[p], errors="coerce").to_numpy(dtype=float)
    Q = pd.to_numeric(out[q], errors="coerce").to_numpy(dtype=float)
    S = pd.to_numeric(out[s], errors="coerce").to_numpy(dtype=float)

    lo, hi = CONFORMANCE["VALID_V_OPEN"]
    with np.errstate(invalid="ignore"):
        out["missing_v"] = np.isnan(V)
        out["v_out_of_range"] = ~np.isnan(V) & ~((V > lo) & (V < hi))
        out["missing_p"] = np.isnan(P)
        out["missing_q"] = np.isnan(Q)
        out["missing_rating"] = np.isnan(S) | ~(S > 0)
    out["valid"] = ~(
        out["missing_v"]
        | out["v_out_of_range"]
        | out["missing_p"]
        | out["missing_q"]
        | out["missing_rating"]
    )

    valid = out["valid"].to_numpy()
    zone = np.full(len(out), None, dtype=object)
    cat = np.full(len(out), None, dtype=object)
    if valid.any():
        zone[valid] = vvar_zone(V[valid], P[valid], S[valid], p_tol_vvar)
        cat[valid] = vvar_category(
            Q[valid], V[valid], P[valid], S[valid], tol, p_tol_vvar
        )
    out["zone"] = zone
    out["category"] = cat

    if with_legacy:
        cur_ass = np.zeros(len(out), dtype=bool)
        cur_cat = np.full(len(out), None, dtype=object)
        if valid.any():
            a, c = legacy_category(Q[valid], V[valid], P[valid], S[valid], tol)
            cur_ass[valid] = a
            cur_cat[valid] = c
        out["legacy_assessable"] = cur_ass
        out["legacy_category"] = cur_cat

    out["tol"] = tol
    out["p_tol_vvar"] = p_tol_vvar
    out.attrs["n_duplicates_dropped"] = n_dup
    return out


# ---------------------------------------------------------------------------
# 2. Additive counts per group (e.g. site x month)
# ---------------------------------------------------------------------------
def count_columns():
    """Names of the zone x category count columns, in a fixed order."""
    return [f"n_{z}_{c}" for z in ZONES for c in VVAR_CATEGORIES]


def zone_counts(scored, by=("site_id", "month"), day_col="day"):
    """
    Counts per group.

    All columns are additive across groups with the same `tol` (n_days_A too, as long as groups do not share days).

    Columns:
    n_rows, n_valid, n_missing_v, n_v_out_of_range, n_missing_p,
    n_missing_q, n_missing_rating, n_A..n_D, n_{zone}_{category},
    n_days_A, n_legacy_total, n_legacy_fail, tol, p_tol_vvar.
    """
    by = list(by)
    if day_col not in scored.columns:
        raise KeyError(
            f"zone_counts needs a local-day column ({day_col!r}) to count "
            "distinct zone A days; pass day_col= or add one."
        )
    for col in ("tol", "p_tol_vvar"):
        if scored[col].nunique() > 1:
            raise ValueError(f"scored frame mixes several {col} values")

    flags = [
        "missing_v",
        "v_out_of_range",
        "missing_p",
        "missing_q",
        "missing_rating",
        "valid",
    ]
    base = scored.groupby(by, dropna=False)[flags].sum()
    base.columns = ["n_" + c for c in flags]
    base.insert(0, "n_rows", scored.groupby(by, dropna=False).size())

    valid = scored.loc[scored["valid"]]
    zc = valid.groupby(by + ["zone", "category"], dropna=False).size()
    if len(zc):
        zc = zc.unstack(["zone", "category"], fill_value=0)
        zc.columns = [f"n_{z}_{c}" for z, c in zc.columns]
    else:
        zc = pd.DataFrame(index=base.index)
    zc = zc.reindex(index=base.index, columns=count_columns(), fill_value=0).fillna(0)

    out = base.join(zc.astype("int64"))
    for z in ZONES:
        out[f"n_{z}"] = out[[f"n_{z}_{c}" for c in VVAR_CATEGORIES]].sum(axis=1)

    days = valid.loc[valid["zone"] == "A"].groupby(by, dropna=False)[day_col].nunique()
    out["n_days_A"] = days.reindex(out.index, fill_value=0).astype("int64")

    if "legacy_assessable" in scored.columns:
        cur = valid.loc[valid["legacy_assessable"]]
        out["n_legacy_total"] = (
            cur.groupby(by, dropna=False).size().reindex(out.index, fill_value=0)
        )
        out["n_legacy_fail"] = (
            cur.loc[cur["legacy_category"].isin(LEGACY_FAILURES)]
            .groupby(by, dropna=False)
            .size()
            .reindex(out.index, fill_value=0)
        )

    out["tol"] = scored["tol"].iloc[0] if len(scored) else np.nan
    out["p_tol_vvar"] = scored["p_tol_vvar"].iloc[0] if len(scored) else np.nan
    return out.reset_index()


def sum_counts(counts, by=("site_id",)):
    """Add zone_counts() rows up to coarser groups (e.g. site over all months)."""
    by = list(by)
    for col in ("tol", "p_tol_vvar"):
        if counts[col].nunique() > 1:
            raise ValueError(f"cannot add counts with different {col} values")
    num = [c for c in counts.columns if c.startswith("n_")]
    out = counts.groupby(by, dropna=False)[num].sum().reset_index()
    out["tol"] = counts["tol"].iloc[0]
    out["p_tol_vvar"] = counts["p_tol_vvar"].iloc[0]
    return out


# ---------------------------------------------------------------------------
# 3. Metrics from counts
# ---------------------------------------------------------------------------
def _div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


#: Column names per metric (explicit, so tables and plots never guess).
METRIC_COLUMNS = {
    "nonconformance_frequency": {
        "share": "nonconformance_frequency",
        "assessable": "nonconformance_frequency_assessable",
        "verdict": "nonconformance_frequency_verdict",
    },
    "remote_conformance_rate": {
        "share": "remote_nonconformance_rate",
        "assessable": "remote_conformance_rate_assessable",
        "verdict": "remote_conformance_rate_verdict",
    },
    "legacy_conformance_rate": {
        "share": "legacy_nonconformance_rate",
        "assessable": "legacy_conformance_rate_assessable",
        "verdict": "legacy_conformance_rate_verdict",
    },
}


def metric_columns(metric):
    """Share / assessable / verdict column names for one metric name."""
    try:
        return METRIC_COLUMNS[metric]
    except KeyError as exc:
        raise ValueError(
            f"metric must be one of {sorted(METRIC_COLUMNS)}, got {metric!r}"
        ) from exc


def _verdict(share, assessable, threshold):
    return np.select(
        [~np.asarray(assessable, dtype=bool), np.asarray(share) <= threshold],
        ["not assessable", "conformant"],
        "non-conformant",
    )


def conformance_metrics(
    counts,
    failure_definition=CONFORMANCE["FAILURE_DEFINITION_DEFAULT"],
    min_zone_a_intervals=CONFORMANCE["MIN_ZONE_A_INTERVALS"],
    min_zone_a_days=CONFORMANCE["MIN_ZONE_A_DAYS"],
    frequency_threshold=CONFORMANCE["FREQUENCY_THRESHOLD"],
    remote_threshold=CONFORMANCE["REMOTE_THRESHOLD"],
    legacy_threshold=CONFORMANCE["LEGACY_THRESHOLD"],
    frequency_min_valid_intervals=CONFORMANCE["FREQUENCY_MIN_VALID_INTERVALS"],
    legacy_min_intervals=CONFORMANCE["LEGACY_MIN_INTERVALS"],
):
    """
    Add all three metrics, and one site tag per metric, to a zone_counts()
    table or to sum_counts() of one (e.g. whole-period site results).

    Tags: 'conformant' if the non-conforming share <= that metric's threshold,
    'non-conformant' otherwise, 'not assessable' below that metric's minimum.
    Column names per metric: METRIC_COLUMNS / metric_columns().
    """
    fails = failure_categories(failure_definition)
    out = counts.copy()
    for z in ZONES:
        out[f"fail_{z}"] = out[[f"n_{z}_{c}" for c in fails]].sum(axis=1)
        out[f"pass_{z}"] = out[f"n_{z}"] - out[f"fail_{z}"]
    n_valid = out["n_valid"]

    # 1. Non-conformance frequency, voltage exposure and decomposition
    total_fail = sum(out[f"fail_{z}"] for z in ZONES)
    out["nonconformance_frequency"] = _div(total_fail, n_valid)
    out["voltage_exposure"] = _div(out["n_A"], n_valid)
    out["zone_A_pass_rate"] = _div(out["pass_A"], out["n_A"])
    for z in ZONES:
        out[f"frequency_term_{z}"] = _div(out[f"fail_{z}"], n_valid)
    ok = (n_valid > 0) & (n_valid >= frequency_min_valid_intervals)
    out["nonconformance_frequency_assessable"] = ok
    out["nonconformance_frequency_verdict"] = _verdict(
        out["nonconformance_frequency"], ok, frequency_threshold
    )

    # 2. Remote conformance rate (option 3)
    denom = out["n_A"] + out["fail_C"]
    out["remote_denominator"] = denom
    ok = (out["n_A"] >= min_zone_a_intervals) & (out["n_days_A"] >= min_zone_a_days)
    ok = ok & (denom > 0)
    out["remote_conformance_rate_assessable"] = ok
    rate = _div(out["fail_A"] + out["fail_C"], denom)
    out["remote_nonconformance_rate"] = np.where(ok, rate, np.nan)
    out["remote_conformance_rate"] = 1.0 - out["remote_nonconformance_rate"]
    out["remote_conformance_rate_verdict"] = _verdict(
        out["remote_nonconformance_rate"], ok, remote_threshold
    )

    # 3. Legacy conformance rate (back-comparison)
    if "n_legacy_total" in out.columns:
        ok = (out["n_legacy_total"] > 0) & (
            out["n_legacy_total"] >= legacy_min_intervals
        )
        out["legacy_conformance_rate_assessable"] = ok
        out["legacy_nonconformance_rate"] = _div(
            out["n_legacy_fail"], out["n_legacy_total"]
        )
        out["legacy_conformance_rate"] = 1.0 - out["legacy_nonconformance_rate"]
        out["legacy_conformance_rate_verdict"] = _verdict(
            out["legacy_nonconformance_rate"], ok, legacy_threshold
        )

    # Coverage, if the caller added n_expected (see expected_intervals()).
    if "n_expected" in out.columns:
        out["coverage"] = _div(n_valid, out["n_expected"])

    out["failure_definition"] = failure_definition
    out["min_zone_a_intervals"] = min_zone_a_intervals
    out["min_zone_a_days"] = min_zone_a_days
    out["frequency_threshold"] = frequency_threshold
    out["remote_threshold"] = remote_threshold
    out["legacy_threshold"] = legacy_threshold
    return out
