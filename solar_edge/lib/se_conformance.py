"""
Volt-VAr and Volt-Watt conformance scoring.
===========================================

Deliverables D9 (Volt-VAr) and D10 (Volt-Watt). Ports of
``build_conformance_voltvar.py`` and ``build_conformance_voltwatt.py`` to DuckDB
over the local store.

The AS/NZS 4777.2 curves themselves are NOT reimplemented here. Every normative
expression comes from ``bms_sa_review.shared.as4777_curves`` -- ``vvar_required_q_sql``,
``q_conformance_floor_absorbing_sql``, ``q_impact_nearest_edge_sql``,
``vw_max_p_sql``, ``tol_kw_sql`` -- and runs unchanged in DuckDB. That is what
makes these results comparable to the Solar Analytics ones by construction rather
than by inspection.

Two departures from the original, both forced and both recorded in the manifest
--------------------------------------------------------------------------------
1. **Capacity basis.** The original scales the required-Q curve, the +/-4% band,
   the 20% assessability rule and the Figure 2.1 capability floor by provider
   ``ac_capacity_kw``. This delivery has no nameplate, so all of them are scaled
   by ``s_99`` instead. ``s_99`` is an *observed* p99 of apparent power, so a site
   that never approached its inverter limit gets a low ``s_99``, which makes its
   required Q smaller and its conformance look better. The bias has a direction
   and it is stated; D15 sweeps the quantile.

2. **Cohorts are scored separately by default.** D6 found single- and three-phase
   sites moving in opposite reactive directions with near mirror-image deadband
   shapes. Until that is resolved, pooling them averages a response against its
   inverse. ``voltvar_summary`` therefore reports by cohort, and pooling is an
   explicit opt-in.

A latent bug in the original, deliberately not reproduced
---------------------------------------------------------
``build_conformance_voltwatt.py`` (around line 160) builds the exposure predicate
as ``vw_exposed = "round(V, 6) > 253.0"`` and then writes
``CASE WHEN V > {vw_exposed} THEN ...``, which expands to
``CASE WHEN V > round(V, 6) > 253.0``. That is a chained comparison, not the
intended test. The GHI variant of the same file uses ``WHEN {vw_exposed}``
correctly. Worth checking against the published Volt-Watt basic rates.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from solar_edge.config import se_config as C
from solar_edge.lib import se_contract as contract
from solar_edge.lib import se_params

C.bootstrap_sys_path()
from bms_sa_review.shared.as4777_curves import (  # noqa: E402
    q_cap_absorbing_sql,
    q_conformance_floor_absorbing_sql,
    q_impact_nearest_edge_sql,
    tol_kw_sql,
    vvar_required_q_sql,
    vw_max_p_sql,
)

__all__ = [
    "voltvar_interval_sql",
    "voltvar_site_day",
    "voltvar_summary",
    "voltvar_site_verdicts",
    "voltvar_impact_distribution",
    "voltwatt_site_day",
    "voltwatt_summary",
    "voltwatt_site_verdicts",
    "conformance_funnel",
]

_A = C.as4777()
_QIMP = _A["QIMP"]
CAPABILITY_PROFILES = ("review_corrected", "hossein_m3")

#: The five Q_impact categories, in order. Names follow the corrected scheme in
#: build_conformance_voltvar.py -- the two middle bands were swapped in Milestone 3.
Q_CATEGORIES = (
    "Q_adverse",                 # wrong direction
    "Q_inactive",                # no response
    "Q_significant_shortfall",   # responded, but far short
    "Q_near_conformant",         # 90-110% of required
    "Q_major_surplus",           # over-response
)

#: reduced non-conformance = adverse + inactive + significant shortfall.
#: Q_near_conformant is EXCLUDED: those inverters deliver 90-110% of required Q.
REDUCED_NONCONF = ("Q_adverse", "Q_inactive", "Q_significant_shortfall")


# ═══════════════════════════════════════════════════════════════════════════
# D9. VOLT-VAR
# ═══════════════════════════════════════════════════════════════════════════

def voltvar_interval_sql(config, params, capability_profile: str = "review_corrected") -> str:
    """
    Interval-level Volt-VAr scoring, as a SQL relation.

    Mirrors the CTE chain of ``build_conformance_voltvar.build_sql``:
    ``required_q`` -> ``tol_band`` -> ``clamped`` -> ``q_impact`` -> ``classified``.

    ``capability_profile``:
      * ``review_corrected`` -- reactive-power priority above 0.8 S, and intervals
        below 0.2 S are marked unassessable rather than scored. This is the
        defensible reading: Figure 2.1 sets no quantified minimum below 20%.
      * ``hossein_m3`` -- the Milestone 3 behaviour, which uses the shrinking
        fixed-P circle above 0.8 S and assesses every interval. Kept for
        reconciliation only.
    """
    if capability_profile not in CAPABILITY_PROFILES:
        raise ValueError(f"capability_profile must be one of {CAPABILITY_PROFILES}")

    config = config.validate()
    params = params.validate()

    rating = contract.capacity_column(config.rating_basis)
    empirical = contract.capacity_column(config.empirical_limit_basis)
    v = contract.voltage_sql(config.voltage_aggregation, "i")

    q_required = vvar_required_q_sql("V", "rating_capacity")
    tol = tol_kw_sql("rating_capacity", config.tolerance_fraction)

    if capability_profile == "review_corrected":
        q_cap = q_conformance_floor_absorbing_sql("P_kW", "rating_capacity")
        assessable = (
            f"CASE WHEN abs(P_kW) >= {_A['QCAP']['P_MIN']} * rating_capacity "
            "THEN 1 ELSE 0 END"
        )
    else:
        q_cap = q_cap_absorbing_sql("P_kW", "rating_capacity")
        assessable = "1"

    q_impact = q_impact_nearest_edge_sql(
        "Q_kvar", "Q_min_final", "Q_max_final", "capability_assessable"
    )
    dist = "least(abs(Q_kvar - Q_min_final), abs(Q_kvar - Q_max_final))"
    out_of_band = "(Q_kvar < Q_min_final OR Q_kvar > Q_max_final)"

    def bucket(name: str, condition: str) -> str:
        return (
            f"CASE WHEN capability_assessable = 1 AND {out_of_band} AND {condition} "
            f"THEN {dist} ELSE 0 END AS {name}"
        )

    buckets = ",\n            ".join([
        bucket("Q_adverse", f"Q_impact < {_QIMP['thr1']}"),
        bucket("Q_inactive", f"Q_impact >= {_QIMP['thr1']} AND Q_impact <= {_QIMP['thr2']}"),
        bucket("Q_significant_shortfall",
               f"Q_impact > {_QIMP['thr2']} AND Q_impact < {_QIMP['thr3']}"),
        bucket("Q_near_conformant",
               f"Q_impact >= {_QIMP['thr3']} AND Q_impact <= {_QIMP['thr4']}"),
        bucket("Q_major_surplus", f"Q_impact > {_QIMP['thr4']}"),
    ])

    return f"""
    WITH data AS (
        SELECT i.site_alias,
               i.ts_aest,
               i.P_kW,
               i.Q_kvar,
               {v}                          AS V,
               {rating}                     AS rating_capacity,
               {empirical}                  AS empirical_limit,
               i.derating_active,
               s.is_three_phase,
               s.state
        FROM se_interval i
        {contract.cohort_join_sql('i')}
        WHERE {contract.cohort_where_sql(config)}
          AND i.P_kW IS NOT NULL
          AND i.Q_kvar IS NOT NULL
          AND {v} IS NOT NULL
    ),
    required_q AS (
        SELECT *,
               ({q_required})   AS Q_voltvar,
               ({q_cap})        AS Q_cap_absorbing,
               ({assessable})   AS capability_assessable
        FROM data
    ),
    tol_band AS (
        SELECT *,
               -Q_cap_absorbing        AS Q_cap_supplying,
               Q_voltvar + {tol}       AS Q_voltvar_max,
               Q_voltvar - {tol}       AS Q_voltvar_min
        FROM required_q
    ),
    clamped AS (
        SELECT *,
               CASE WHEN Q_voltvar_max < 0
                    THEN greatest(Q_voltvar_max, Q_cap_absorbing + {tol})
                    ELSE Q_voltvar_max END AS Q_max_final,
               CASE WHEN Q_voltvar_min > 0
                    THEN least(Q_voltvar_min, Q_cap_supplying - {tol})
                    ELSE Q_voltvar_min END AS Q_min_final
        FROM tol_band
    ),
    q_impact AS (
        SELECT *, ({q_impact}) AS Q_impact FROM clamped
    ),
    classified AS (
        SELECT
            site_alias, ts_aest, P_kW, Q_kvar, V, is_three_phase, state,
            rating_capacity, empirical_limit, derating_active,
            capability_assessable, Q_impact, Q_min_final, Q_max_final,
            {buckets},
            -- Apparent-limit symptom (Method A seed). Deliberately driven by the
            -- empirical limit, not the rating basis, and confined to the band
            -- where Volt-VAr acts alone.
            CASE WHEN V > {params.v_low} AND V <= {params.v_high}
                  AND Q_kvar < 0
                  AND sqrt(power(Q_kvar, 2) + power(P_kW, 2)) >= empirical_limit
                 THEN 1 ELSE 0 END AS curtailment_eligible,
            CASE WHEN V > {_A['VVAR']['V3']} THEN 1 ELSE 0 END AS exposed
        FROM q_impact
    )
    SELECT * FROM classified
    """


def voltvar_site_day(
    con: duckdb.DuckDBPyConnection, config=None, params=None,
    capability_profile: str = "review_corrected",
) -> pd.DataFrame:
    """
    Aggregate Volt-VAr scoring to one row per (site, AEST date).

    The site-day grain matches ``conformance_voltvar_v2`` so the two are directly
    comparable, and it keeps the result small enough for pandas while preserving
    every denominator the site-level verdict needs.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()

    sums = ",\n               ".join(
        f"sum({c}) AS {c}_sum" for c in Q_CATEGORIES
    )
    counts = ",\n               ".join(
        f"count(*) FILTER (WHERE {c} > 0) AS {c}_count" for c in Q_CATEGORIES
    )

    return con.execute(
        f"""
        WITH scored AS ({voltvar_interval_sql(config, params, capability_profile)})
        SELECT site_alias,
               any_value(is_three_phase)                    AS is_three_phase,
               any_value(state)                             AS state,
               CAST(ts_aest AS DATE)                        AS day_aest,
               count(*)                                     AS all_intervals_count,
               sum(exposed)                                 AS exposed_count,
               sum(capability_assessable)                   AS total_count,
               {sums},
               {counts},
               sum(curtailment_eligible)                    AS curtailment_eligible_count,
               count(*) FILTER (WHERE derating_active)      AS derating_count,
               '{config.rating_basis}'                      AS rating_basis,
               '{capability_profile}'                       AS capability_profile
        FROM scored
        GROUP BY site_alias, CAST(ts_aest AS DATE)
        """
    ).df()


def voltvar_summary(site_day: pd.DataFrame, by_cohort: bool = True) -> pd.DataFrame:
    """
    Fleet Volt-VAr conformance rates.

    ``by_cohort=True`` (the default) splits single- from three-phase. D6 found the
    two moving in opposite reactive directions with mirror-image deadband shapes;
    until that is resolved, a pooled rate averages a response against its inverse
    and understates both. Pass ``by_cohort=False`` only with that caveat stated.

    ``reduced_nonconf`` = adverse + inactive + significant shortfall.
    ``Q_near_conformant`` is excluded: those inverters deliver 90-110% of required
    reactive power. Milestone 3 included that band and excluded the shortfall band,
    an artefact of the swapped names (R4), so these figures will not match it.
    """
    frame = site_day.copy()
    keys = ["is_three_phase"] if by_cohort else []

    reduced_cols = [f"{c}_count" for c in REDUCED_NONCONF]
    frame["reduced_nonconf_count"] = frame[reduced_cols].sum(axis=1)

    rows = []
    for key, group in (frame.groupby(keys) if keys else [((), frame)]):
        assessable = group.total_count.sum()
        rows.append({
            "cohort": ("three-phase" if key[0] else "single-phase") if keys else "all sites",
            "n_sites": group.site_alias.nunique(),
            "all_intervals": int(group.all_intervals_count.sum()),
            "exposed_intervals": int(group.exposed_count.sum()),
            "capability_assessable_intervals": int(assessable),
            **{f"{c}_intervals": int(group[f"{c}_count"].sum()) for c in Q_CATEGORIES},
            "reduced_nonconf_intervals": int(group.reduced_nonconf_count.sum()),
            "reduced_nonconf_pct": (
                100 * group.reduced_nonconf_count.sum() / assessable if assessable else float("nan")
            ),
            "curtailment_eligible_intervals": int(group.curtailment_eligible_count.sum()),
        })
    return pd.DataFrame(rows)


def _safe_fraction(numerator, denominator):
    """Elementwise ratio with a float NaN (not pd.NA) where the denominator is zero."""
    import numpy as np

    return np.where(
        denominator.to_numpy() > 0,
        numerator.to_numpy() / np.maximum(denominator.to_numpy(), 1),
        np.nan,
    )


def _verdict(fraction, denominator, threshold: float, empty_label: str):
    """
    Site verdict from a non-conformance fraction.

    Written as explicit boolean logic rather than ``pd.cut``, which silently
    produces a nullable Categorical and then raises "boolean value of NA is
    ambiguous" the moment a site has an empty denominator.

    A site with no assessable or exposed intervals gets ``empty_label``, NOT
    "conformant". Scoring an unobserved site as passing is how a conformance rate
    quietly becomes a measure of coverage instead.
    """
    import numpy as np

    verdict = np.where(fraction <= threshold, "conformant", "non-conformant")
    return np.where(denominator.to_numpy() > 0, verdict, empty_label)


def voltvar_site_verdicts(site_day: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    One verdict per site: conformant if the reduced non-conformance fraction is at
    or below ``site_nonconf_threshold`` (10%), matching the Solar Analytics rule.

    Sites with no capability-assessable intervals get no verdict. That is not the
    same as conforming, and scoring them as conformant would inflate the rate --
    exactly the conservative-by-accident path the legacy NULL handling took.
    """
    config = (config or se_params.CONFIG).validate()
    reduced_cols = [f"{c}_count" for c in REDUCED_NONCONF]

    site = site_day.groupby("site_alias", as_index=False).agg(
        is_three_phase=("is_three_phase", "first"),
        state=("state", "first"),
        assessable=("total_count", "sum"),
        exposed=("exposed_count", "sum"),
        **{c: (c, "sum") for c in reduced_cols},
    )
    site["reduced_nonconf"] = site[reduced_cols].sum(axis=1)
    site["nonconf_fraction"] = _safe_fraction(site.reduced_nonconf, site.assessable)
    site["verdict"] = _verdict(
        site.nonconf_fraction, site.assessable, config.site_nonconf_threshold,
        empty_label="not assessable",
    )
    return site


def voltvar_impact_distribution(
    con: duckdb.DuckDBPyConnection, config=None, params=None, bins: int = 40
) -> pd.DataFrame:
    """Histogram of the signed Q_impact ratio, by cohort. 1.0 = at the required edge."""
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    return con.execute(
        f"""
        WITH scored AS ({voltvar_interval_sql(config, params)})
        SELECT CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               round(least(greatest(Q_impact, -2.0), 3.0) * {bins} / 5.0) * 5.0 / {bins}
                                                            AS q_impact_bin,
               count(*)                                     AS n_intervals
        FROM scored
        WHERE capability_assessable = 1 AND Q_impact IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()


# ═══════════════════════════════════════════════════════════════════════════
# D10. VOLT-WATT
# ═══════════════════════════════════════════════════════════════════════════

def voltwatt_site_day(con: duckdb.DuckDBPyConnection, config=None) -> pd.DataFrame:
    """
    Volt-Watt conformance, one row per (site, AEST date).

    A site is *exposed* above 253 V and non-conformant when measured P exceeds the
    Volt-Watt ceiling plus the 4% tolerance.

    Without a counterfactual this is the "basic" variant only: it detects
    generating ABOVE the ceiling, but cannot distinguish an inverter that
    correctly curtailed from one that simply had no sun. The GHI variant
    (``conformance_voltwattghi``) needs D12 irradiance and is not runnable yet.

    Note also from D6 that only ~0.96% of intervals exceed 253 V, so the exposed
    population is thin and site-level rates will be noisy. Report counts alongside
    percentages.
    """
    config = (config or se_params.CONFIG).validate()
    rating = contract.capacity_column(config.rating_basis)
    v = contract.voltage_sql(config.voltage_aggregation, "i")

    # Round before the strict boundary test, as the original does, so that
    # floating-point noise cannot flip an interval across 253.0 V.
    v_scored = "round(V, 6)"
    exposed = f"{v_scored} > {_A['VW']['V1']}"
    max_p = vw_max_p_sql(v_scored, "rating_capacity")
    tol = tol_kw_sql("rating_capacity", config.tolerance_fraction)

    return con.execute(
        f"""
        WITH data AS (
            SELECT i.site_alias, i.ts_aest, i.P_kW, {v} AS V,
                   {rating} AS rating_capacity,
                   i.derating_active, s.is_three_phase, s.state
            FROM se_interval i
            {contract.cohort_join_sql('i')}
            WHERE {contract.cohort_where_sql(config)}
              AND i.P_kW IS NOT NULL AND {v} IS NOT NULL
        ),
        limits AS (
            SELECT *, ({max_p}) + {tol} AS max_P_volt_watt FROM data
        ),
        scored AS (
            SELECT *,
                   CASE WHEN {exposed}
                        THEN greatest(0, P_kW - max_P_volt_watt)
                        ELSE NULL END AS nonconformance_voltwatt
            FROM limits
        )
        SELECT site_alias,
               any_value(is_three_phase)                        AS is_three_phase,
               any_value(state)                                 AS state,
               CAST(ts_aest AS DATE)                            AS day_aest,
               sum(P_kW)                                        AS P_kW_sum,
               sum(nonconformance_voltwatt)                     AS nonconformance_sum,
               count(*) FILTER (WHERE nonconformance_voltwatt > 0)
                                                                AS nonconformance_count,
               count(*)                                         AS all_intervals_count,
               count(*) FILTER (WHERE {exposed})                AS total_count,
               count(*) FILTER (WHERE {exposed} AND derating_active)
                                                                AS exposed_derating_count,
               '{config.rating_basis}'                          AS rating_basis
        FROM scored
        GROUP BY site_alias, CAST(ts_aest AS DATE)
        """
    ).df()


def voltwatt_summary(site_day: pd.DataFrame, by_cohort: bool = True) -> pd.DataFrame:
    """Fleet Volt-Watt rates. ``exposed`` is the denominator, not all intervals."""
    keys = ["is_three_phase"] if by_cohort else []
    rows = []
    for key, group in (site_day.groupby(keys) if keys else [((), site_day)]):
        exposed = group.total_count.sum()
        nonconf = group.nonconformance_count.sum()
        rows.append({
            "cohort": ("three-phase" if key[0] else "single-phase") if keys else "all sites",
            "n_sites": group.site_alias.nunique(),
            "sites_exposed": group.loc[group.total_count > 0, "site_alias"].nunique(),
            "all_intervals": int(group.all_intervals_count.sum()),
            "exposed_intervals": int(exposed),
            "nonconformant_intervals": int(nonconf),
            "nonconformant_pct_of_exposed": (
                100 * nonconf / exposed if exposed else float("nan")
            ),
            "excess_kWh": group.nonconformance_sum.sum() * C.INTERVAL_H,
            "exposed_with_derating_flag": int(group.exposed_derating_count.sum()),
        })
    return pd.DataFrame(rows)


def voltwatt_site_verdicts(site_day: pd.DataFrame, config=None) -> pd.DataFrame:
    """One Volt-Watt verdict per site, on the same 10% rule as Volt-VAr."""
    config = (config or se_params.CONFIG).validate()
    site = site_day.groupby("site_alias", as_index=False).agg(
        is_three_phase=("is_three_phase", "first"),
        state=("state", "first"),
        exposed=("total_count", "sum"),
        nonconf=("nonconformance_count", "sum"),
        excess_kW_sum=("nonconformance_sum", "sum"),
    )
    site["nonconf_fraction"] = _safe_fraction(site.nonconf, site.exposed)
    site["verdict"] = _verdict(
        site.nonconf_fraction, site.exposed, config.site_nonconf_threshold,
        empty_label="not exposed",
    )
    return site


def conformance_funnel(vvar_site_day: pd.DataFrame, vwatt_site_day: pd.DataFrame) -> pd.DataFrame:
    """
    Population attrition for both response modes, side by side.

    The point is the gap between *exposed* and *assessable*. A site can sit in the
    Volt-VAr voltage band all year and still never be assessable, because below
    20% of rated power Figure 2.1 sets no quantified minimum capability. Reporting
    a conformance rate without this table invites the reader to assume the
    denominator is the whole fleet.
    """
    rows = [
        {"mode": "Volt-VAr", "stage": "1. all cohort intervals",
         "n_intervals": int(vvar_site_day.all_intervals_count.sum()),
         "n_sites": vvar_site_day.site_alias.nunique()},
        {"mode": "Volt-VAr", "stage": "2. voltage-exposed (V > 240)",
         "n_intervals": int(vvar_site_day.exposed_count.sum()),
         "n_sites": vvar_site_day.loc[vvar_site_day.exposed_count > 0, "site_alias"].nunique()},
        {"mode": "Volt-VAr", "stage": "3. capability-assessable (P >= 0.2 S)",
         "n_intervals": int(vvar_site_day.total_count.sum()),
         "n_sites": vvar_site_day.loc[vvar_site_day.total_count > 0, "site_alias"].nunique()},
        {"mode": "Volt-Watt", "stage": "1. all cohort intervals",
         "n_intervals": int(vwatt_site_day.all_intervals_count.sum()),
         "n_sites": vwatt_site_day.site_alias.nunique()},
        {"mode": "Volt-Watt", "stage": "2. voltage-exposed (V > 253)",
         "n_intervals": int(vwatt_site_day.total_count.sum()),
         "n_sites": vwatt_site_day.loc[vwatt_site_day.total_count > 0, "site_alias"].nunique()},
    ]
    return pd.DataFrame(rows)
