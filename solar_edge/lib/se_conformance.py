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
    "voltvar_measures",
    "site_verdict_measures",
    "VERDICT_MEASURES",
    "MEASURES",
    "voltvar_site_verdicts",
    "voltvar_impact_distribution",
    "voltwatt_site_day",
    "voltwatt_summary",
    "voltwatt_site_verdicts",
    "voltwatt_verdict_measures",
    "voltwatt_ghi_site_day",
    "voltwatt_ghi_summary",
    "voltwatt_ghi_site_verdicts",
    "VW_VERDICT_MEASURES",
    "conformance_funnel",
    "enrich_site_day",
    "voltvar_site_table",
    "voltwatt_site_table",
    "voltvar_breakdown",
    "voltwatt_breakdown",
    "CAPACITY_BANDS",
    "CAPACITY_LABELS",
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
    q = contract.q_expr(config, "i")

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
               -- Oriented per config.reactive_orientation, never read raw. The
               -- store holds SolarEdge's own sign; the analysis decides what to
               -- do with it, and that decision is in the manifest.
               {q}                          AS Q_kvar,
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
    # Per-unit shortfall: the same kvar distance divided by the site's rating, so
    # a 5 kVA and a 10 kVA site contribute on the same scale. This is what makes
    # the kVArh/kW/h rate meaningful across a fleet of mixed sizes.
    pu_sums = ",\n               ".join(
        f"sum({c} / nullif(rating_capacity, 0)) AS {c}_pu_sum" for c in Q_CATEGORIES
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
               {pu_sums},
               any_value(rating_capacity)                   AS rating_kva,
               sum(curtailment_eligible)                    AS curtailment_eligible_count,
               count(*) FILTER (WHERE derating_active)      AS derating_count,
               '{config.rating_basis}'                      AS rating_basis,
               '{capability_profile}'                       AS capability_profile
        FROM scored
        GROUP BY site_alias, CAST(ts_aest AS DATE)
        """
    ).df()


#: Capacity bands used for every "by system size" breakdown, so the bins cannot
#: drift between tables.
CAPACITY_BANDS = [0, 3, 4, 5, 6, 8, 10, 1000]
CAPACITY_LABELS = ["<3", "3-4", "4-5", "5-6", "6-8", "8-10", ">10"]


def _pct(numerator, denominator, decimals: int = 3):
    """
    Percentage with NaN where the denominator is zero.

    Uses numpy NaN, never ``pd.NA``. ``series.replace(0, pd.NA)`` silently promotes
    an integer column to object dtype, and the next ``.round()`` raises
    "Expected numeric dtype, got object instead". That has bitten this module
    three times; route every rate through here.
    """
    import numpy as np

    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, 100.0 * num / den, np.nan)
    return np.round(out, decimals)


def enrich_site_day(con: duckdb.DuckDBPyConnection, site_day: pd.DataFrame) -> pd.DataFrame:
    """
    Attach site dimension and capacity to a scored site-day frame.

    Adds ``postcode``, ``s_99`` and a banded ``capacity_band``, which is what every
    breakdown groups on. Kept separate from the scoring so the expensive interval
    pass is not repeated once per breakdown dimension.
    """
    meta = con.execute(
        """
        SELECT s.site_alias, s.postcode, s.state AS site_state,
               s.is_three_phase AS site_three_phase, c.s_99
        FROM se_site s LEFT JOIN se_site_capacity c USING (site_alias)
        """
    ).df()
    out = site_day.merge(meta, on="site_alias", how="left")
    # Ordered categorical, NOT object: keeps <3 .. >10 in size order when grouped
    # or sorted, instead of falling back to alphabetical ("<3" after ">10").
    out["capacity_band"] = pd.Categorical(
        pd.cut(out.s_99, bins=CAPACITY_BANDS, labels=CAPACITY_LABELS, right=False),
        categories=CAPACITY_LABELS, ordered=True,
    )
    return out


def _site_share(verdicts: pd.DataFrame, by: str) -> pd.DataFrame:
    """Percentage of SITES in each verdict, per group. Uses the 10% site rule."""
    counts = (
        verdicts.groupby([by, "verdict"], observed=True)
        .size().unstack(fill_value=0)
    )
    pct = (100 * counts.div(counts.sum(axis=1), axis=0)).round(2)
    pct.columns = [f"pct_sites_{c.replace(' ', '_').replace('-', '_')}" for c in pct.columns]
    out = pct.reset_index()
    out.insert(1, "n_sites", counts.sum(axis=1).values)
    return out


def voltvar_site_table(
    con: duckdb.DuckDBPyConnection, site_day: pd.DataFrame, config=None,
    adverse: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    One row per site: category percentages, verdict, and everything needed to pick
    sites to inspect in ``05_site_explorer.ipynb``.

    This is the bridge between the fleet result and the per-site investigation.
    Sort or filter it, take the ``site_alias`` values, and feed them to
    ``se_explore.explore(site)``.

    Percentages are against that site's own capability-assessable intervals, so a
    site with 200 assessable intervals and one with 200,000 are on the same scale
    -- but ``assessable_intervals`` is carried alongside precisely so a 100% rate
    on a handful of intervals is not mistaken for a finding.

    If ``adverse`` (from ``se_adverse.classify_adverse_sites``) is supplied, the
    polarity triage is joined on, so sites flagged as ``polarity_suspect`` can be
    excluded or inspected first.
    """
    config = (config or se_params.CONFIG).validate()
    enriched = enrich_site_day(con, site_day)

    cat_cols = [f"{c}_count" for c in Q_CATEGORIES]
    per_site = enriched.groupby("site_alias", as_index=False).agg(
        state=("site_state", "first"),
        postcode=("postcode", "first"),
        is_three_phase=("site_three_phase", "first"),
        s_99=("s_99", "first"),
        capacity_band=("capacity_band", "first"),
        n_days=("day_aest", "nunique"),
        all_intervals=("all_intervals_count", "sum"),
        exposed_intervals=("exposed_count", "sum"),
        assessable_intervals=("total_count", "sum"),
        curtailment_eligible=("curtailment_eligible_count", "sum"),
        derating_intervals=("derating_count", "sum"),
        **{c: (c, "sum") for c in cat_cols},
    )

    for cat in Q_CATEGORIES:
        per_site[f"pct_{cat}"] = _pct(per_site[f"{cat}_count"],
                                      per_site.assessable_intervals)
    per_site["reduced_nonconf_count"] = per_site[
        [f"{c}_count" for c in REDUCED_NONCONF]].sum(axis=1)
    per_site["pct_reduced_nonconf"] = _pct(per_site.reduced_nonconf_count,
                                           per_site.assessable_intervals)

    per_site["verdict"] = _verdict(
        per_site.pct_reduced_nonconf / 100.0,
        per_site.assessable_intervals,
        config.site_nonconf_threshold,
        empty_label="not assessable",
    )
    per_site["cohort"] = per_site.is_three_phase.map(
        {True: "three-phase", False: "single-phase"})

    if adverse is not None:
        per_site = per_site.merge(
            adverse[["site_alias", "adverse_class", "ratio_to_required"]],
            on="site_alias", how="left")

    cols = ["site_alias", "cohort", "state", "postcode", "s_99", "capacity_band",
            "n_days", "all_intervals", "exposed_intervals", "assessable_intervals",
            "verdict", "pct_reduced_nonconf"] + \
           [f"pct_{c}" for c in Q_CATEGORIES] + \
           ["curtailment_eligible", "derating_intervals"]
    if adverse is not None:
        cols += ["adverse_class", "ratio_to_required"]

    return per_site[cols].sort_values("pct_reduced_nonconf", ascending=False)


def voltwatt_site_table(
    con: duckdb.DuckDBPyConnection, site_day: pd.DataFrame, config=None
) -> pd.DataFrame:
    """One row per site for Volt-Watt: exposure, non-conformance rate and severity."""
    config = (config or se_params.CONFIG).validate()
    enriched = enrich_site_day(con, site_day)

    per_site = enriched.groupby("site_alias", as_index=False).agg(
        state=("site_state", "first"),
        postcode=("postcode", "first"),
        is_three_phase=("site_three_phase", "first"),
        s_99=("s_99", "first"),
        capacity_band=("capacity_band", "first"),
        all_intervals=("all_intervals_count", "sum"),
        exposed_intervals=("total_count", "sum"),
        nonconformant_intervals=("nonconformance_count", "sum"),
        excess_kW_sum=("nonconformance_sum", "sum"),
        exposed_with_derating=("exposed_derating_count", "sum"),
    )
    per_site["pct_nonconformant_of_exposed"] = _pct(
        per_site.nonconformant_intervals, per_site.exposed_intervals)
    per_site["excess_Wh"] = per_site.excess_kW_sum * C.INTERVAL_H * 1000.0
    per_site["severity_Wh_per_kVA_per_exposed"] = _pct(
        per_site.excess_Wh, per_site.s_99 * per_site.exposed_intervals, decimals=6) / 100.0
    per_site["verdict"] = _verdict(
        per_site.pct_nonconformant_of_exposed / 100.0,
        per_site.exposed_intervals,
        config.site_nonconf_threshold,
        empty_label="not exposed",
    )
    per_site["cohort"] = per_site.is_three_phase.map(
        {True: "three-phase", False: "single-phase"})
    return per_site.sort_values("pct_nonconformant_of_exposed", ascending=False)


def voltvar_breakdown(
    con: duckdb.DuckDBPyConnection, site_day: pd.DataFrame, by: str = "site_state",
    config=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Volt-VAr results split by a fleet dimension.

    ``by``: ``site_state``, ``capacity_band``, ``postcode`` or ``cohort``.

    Returns two frames, deliberately kept apart because they answer different
    questions and have different denominators:

    ``site_pct``
        Percentage of SITES conformant / non-conformant / not assessable, on the
        10% rule. Equal weight per site.
    ``interval_pct``
        Percentage of capability-assessable INTERVALS in each of the five
        Q_impact categories. Weighted by how much each site was observed.

    A group can look bad on one and fine on the other -- a handful of heavily
    observed sites can dominate the interval view while barely moving the site
    view. Reporting only one of them hides that.
    """
    config = (config or se_params.CONFIG).validate()
    enriched = enrich_site_day(con, site_day)
    if by == "cohort":
        enriched["cohort"] = enriched.is_three_phase.map(
            {True: "three-phase", False: "single-phase"})

    verdicts = voltvar_site_verdicts(site_day, config).merge(
        enriched[["site_alias", by]].drop_duplicates("site_alias"),
        on="site_alias", how="left")
    site_pct = _site_share(verdicts, by)

    cat_cols = [f"{c}_count" for c in Q_CATEGORIES]
    grouped = enriched.groupby(by, observed=True).agg(
        n_sites=("site_alias", "nunique"),
        assessable_intervals=("total_count", "sum"),
        **{c: (c, "sum") for c in cat_cols},
    ).reset_index()

    for cat in Q_CATEGORIES:
        grouped[f"pct_{cat}"] = _pct(grouped[f"{cat}_count"], grouped.assessable_intervals)
    grouped["pct_reduced_nonconf"] = _pct(
        grouped[[f"{c}_count" for c in REDUCED_NONCONF]].sum(axis=1),
        grouped.assessable_intervals,
    )

    keep = [by, "n_sites", "assessable_intervals"] + \
           [f"pct_{c}" for c in Q_CATEGORIES] + ["pct_reduced_nonconf"]
    out = grouped[keep]
    # Ordered dimensions (capacity_band) sort by the dimension itself; unordered
    # ones by exposure, so the biggest populations lead.
    sort_key = by if isinstance(out[by].dtype, pd.CategoricalDtype) else "assessable_intervals"
    return (site_pct.sort_values(by) if sort_key == by else site_pct,
            out.sort_values(sort_key, ascending=(sort_key == by)))


def voltwatt_breakdown(
    con: duckdb.DuckDBPyConnection, site_day: pd.DataFrame, by: str = "site_state",
    config=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Volt-Watt results split by a fleet dimension.

    Same two-frame structure as ``voltvar_breakdown``. The interval denominator
    here is EXPOSED intervals (V > 253 V), not all intervals -- a site that never
    saw high voltage was never tested, and including it would dilute the rate with
    sites that had no opportunity to fail.

    ``severity_Wh_per_kVA_per_exposed`` is the normalised magnitude metric from the
    Solar Analytics notebook: nonconformance Wh divided by (capacity x exposed
    intervals). Frequency says how often; this says how far over.
    """
    config = (config or se_params.CONFIG).validate()
    enriched = enrich_site_day(con, site_day)
    if by == "cohort":
        enriched["cohort"] = enriched.is_three_phase.map(
            {True: "three-phase", False: "single-phase"})

    verdicts = voltwatt_site_verdicts(site_day, config).merge(
        enriched[["site_alias", by]].drop_duplicates("site_alias"),
        on="site_alias", how="left")
    site_pct = _site_share(verdicts, by)

    per_site = enriched.groupby(["site_alias", by], observed=True, as_index=False).agg(
        exposed=("total_count", "sum"),
        nonconf=("nonconformance_count", "sum"),
        excess_kW=("nonconformance_sum", "sum"),
        s_99=("s_99", "first"),
    )
    per_site["excess_Wh"] = per_site.excess_kW * C.INTERVAL_H * 1000.0
    per_site["capacity_exposure"] = per_site.s_99 * per_site.exposed

    grouped = per_site.groupby(by, observed=True).agg(
        n_sites=("site_alias", "nunique"),
        sites_exposed=("exposed", lambda x: int((x > 0).sum())),
        exposed_intervals=("exposed", "sum"),
        nonconformant_intervals=("nonconf", "sum"),
        excess_Wh=("excess_Wh", "sum"),
        capacity_exposure=("capacity_exposure", "sum"),
    ).reset_index()

    grouped["pct_intervals_nonconformant"] = _pct(
        grouped.nonconformant_intervals, grouped.exposed_intervals)
    grouped["severity_Wh_per_kVA_per_exposed"] = (
        _pct(grouped.excess_Wh, grouped.capacity_exposure, decimals=6) / 100.0)

    sort_key = by if isinstance(grouped[by].dtype, pd.CategoricalDtype) else "exposed_intervals"
    return (site_pct.sort_values(by) if sort_key == by else site_pct,
            grouped.sort_values(sort_key, ascending=(sort_key == by)))


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
        assessable = int(group.total_count.sum())
        all_int = int(group.all_intervals_count.sum())
        exposed = int(group.exposed_count.sum())

        def pct(n, denom):
            return round(100 * n / denom, 3) if denom else float("nan")

        row = {
            "cohort": ("three-phase" if key[0] else "single-phase") if keys else "all sites",
            "n_sites": group.site_alias.nunique(),
            "all_intervals": all_int,
            "exposed_intervals": exposed,
            "pct_exposed_of_all": pct(exposed, all_int),
            "capability_assessable_intervals": assessable,
            "pct_assessable_of_exposed": pct(assessable, exposed),
        }
        # Counts AND percentages side by side. The percentage denominator is the
        # capability-assessable population, NOT all intervals -- below 0.2 S the
        # standard sets no quantified requirement, so those intervals cannot be
        # scored and must not sit in the denominator.
        for c in Q_CATEGORIES:
            n = int(group[f"{c}_count"].sum())
            row[f"{c}_intervals"] = n
            row[f"{c}_pct"] = pct(n, assessable)
        reduced = int(group.reduced_nonconf_count.sum())
        row["reduced_nonconf_intervals"] = reduced
        row["reduced_nonconf_pct"] = pct(reduced, assessable)
        row["curtailment_eligible_intervals"] = int(group.curtailment_eligible_count.sum())
        rows.append(row)
    return pd.DataFrame(rows)


VERDICT_MEASURES = {
    "pct_sites": ("% of sites", "one vote each"),
    "pct_intervals": ("% of intervals", "volume-weighted"),
    "kvarh": ("kVArh shortfall", "absolute"),
    "kvarh_per_kw_per_h": ("kVArh / kW / h", "normalised"),
}


def site_verdict_measures(site_day: pd.DataFrame, config=None,
                          by_cohort: bool = True) -> pd.DataFrame:
    """
    The 10% site verdict expressed in the same four currencies as
    ``voltvar_measures``, so the two figures can be read against each other.

    Every site gets one verdict; this then asks, for each verdict, what share of
    SITES, of assessable INTERVALS, of shortfall kVArh, and what normalised rate
    it accounts for. Reported per cohort by default, for the D6 reason.

    Only reduced non-conformance (adverse + inactive + significant shortfall)
    counts toward the kVArh attributed to a verdict -- the same three categories
    the verdict itself is computed from, so the energy column and the pass/fail
    decision cannot disagree about what "non-conformance" means.
    """
    import numpy as np

    config = (config or se_params.CONFIG).validate()
    verdicts = voltvar_site_verdicts(site_day, config)[
        ["site_alias", "verdict", "is_three_phase"]]

    energy_cols = [f"{c}_sum" for c in REDUCED_NONCONF]
    pu_cols = [f"{c}_pu_sum" for c in REDUCED_NONCONF]
    per_site = site_day.groupby("site_alias", as_index=False).agg(
        assessable=("total_count", "sum"),
        **{c: (c, "sum") for c in energy_cols + pu_cols},
    )
    per_site["kvar_sum"] = per_site[energy_cols].sum(axis=1)
    per_site["pu_sum"] = per_site[pu_cols].sum(axis=1)
    frame = verdicts.merge(per_site, on="site_alias", how="left")
    frame["cohort"] = frame.is_three_phase.map(
        {True: "three-phase", False: "single-phase"})

    keys = ["cohort"] if by_cohort else []
    rows = []
    for key, group in (frame.groupby(keys) if keys else [((), frame)]):
        cohort = key[0] if keys else "all sites"
        n_sites = len(group)
        assessable = float(group.assessable.sum())
        for verdict, sub in group.groupby("verdict"):
            sub_assessable = float(sub.assessable.sum())
            rows.append({
                "cohort": cohort,
                "verdict": verdict,
                "n_sites": len(sub),
                "pct_sites": _pct(len(sub), n_sites),
                "pct_intervals": _pct(sub_assessable, assessable),
                "kvarh": round(float(sub.kvar_sum.sum()) * C.INTERVAL_H, 2),
                # Normalised by THIS verdict's own assessable time, so it reads
                # as "how badly did these sites behave while we watched them",
                # not "how much of the fleet's shortfall did they cause".
                "kvarh_per_kw_per_h": round(
                    float(sub.pu_sum.sum()) / sub_assessable, 5
                ) if sub_assessable > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def voltvar_measures(site_day: pd.DataFrame, config=None, by_cohort: bool = True) -> pd.DataFrame:
    """
    The same Volt-VAr result expressed four ways, as one tidy frame.

    Milestone 3 reported Volt-VAr four times over because no single denominator
    answers every question, and the four disagree in informative ways. Returned
    long, one row per (cohort, category, measure):

    ``pct_intervals``
        Share of capability-assessable intervals in each category. Counts every
        interval equally, so a site reporting all year outweighs one reporting a
        week. Answers "how often does this happen".

    ``pct_sites``
        Share of sites for which that category alone exceeds
        ``config.site_nonconf_threshold`` (10%) of their assessable intervals --
        the same rule the site verdict uses, applied per category. One vote per
        site regardless of data volume. Answers "how many inverters".

    ``kvarh``
        Total reactive-energy shortfall, sum over intervals of
        ``least(|Q - Q_min|, |Q - Q_max|)`` x ``INTERVAL_H``. This is Hossein's
        ``sum(Q_adverse)`` etc. converted from kvar-intervals into kVArh. It is
        size- and volume-weighted: big sites and long records dominate. Answers
        "how much reactive support was actually missing".

    ``kvarh_per_kw_per_h``
        The same energy normalised by site rating and by assessable time. Because
        both the numerator and the denominator carry ``INTERVAL_H``, it reduces to
        the MEAN PER-UNIT SHORTFALL over assessable intervals -- dimensionless,
        comparable across fleet sizes and reporting periods. 0.10 means the fleet
        sat, on average, a tenth of a per-unit kvar outside its allowed band.
        Answers "how badly, per inverter, per hour".

    Read them together. A category that is large in ``pct_intervals`` but small in
    ``kvarh`` is a frequent near-miss; large in ``kvarh`` but small in
    ``pct_intervals`` is a rare, severe failure concentrated in big systems.
    """
    import numpy as np

    config = (config or se_params.CONFIG).validate()
    frame = site_day.copy()
    keys = ["is_three_phase"] if by_cohort else []

    per_site = frame.groupby(
        (["is_three_phase"] if by_cohort else []) + ["site_alias"], as_index=False
    ).agg({
        "total_count": "sum",
        **{f"{c}_count": "sum" for c in Q_CATEGORIES},
        **{f"{c}_sum": "sum" for c in Q_CATEGORIES},
        **{f"{c}_pu_sum": "sum" for c in Q_CATEGORIES},
    })

    rows = []
    groups = (per_site.groupby(keys) if keys else [((), per_site)])
    for key, g in groups:
        cohort = ("three-phase" if key[0] else "single-phase") if keys else "all sites"
        assessable = float(g.total_count.sum())
        # Only sites with something to assess get a vote; a site with no
        # assessable intervals is absent, not conformant.
        votes = g[g.total_count > 0]
        n_votes = len(votes)

        for cat in Q_CATEGORIES:
            site_frac = votes[f"{cat}_count"] / votes.total_count
            rows.append({
                "cohort": cohort,
                "category": cat,
                "n_sites": n_votes,
                "assessable_intervals": int(assessable),
                "pct_intervals": _pct(g[f"{cat}_count"].sum(), assessable),
                "pct_sites": _pct((site_frac > config.site_nonconf_threshold).sum(),
                                  n_votes),
                "kvarh": round(float(g[f"{cat}_sum"].sum()) * C.INTERVAL_H, 2),
                "kvarh_per_kw_per_h": round(
                    float(g[f"{cat}_pu_sum"].sum()) / assessable, 5
                ) if assessable > 0 else np.nan,
            })
    return pd.DataFrame(rows)


MEASURES = {
    "pct_intervals": ("% of assessable intervals",
                      "every interval counts once"),
    "pct_sites": ("% of sites over the 10% rule",
                  "every site counts once"),
    "kvarh": ("Reactive energy shortfall (kVArh)",
              "size- and volume-weighted total"),
    "kvarh_per_kw_per_h": ("kVArh / kW / h",
                           "mean per-unit shortfall, size-normalised"),
}


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

    # <= threshold, matching the ORIGINAL SolA2024 analysis
    # (OEM_installDate_confrate.ipynb):
    #     conf_data    = ... where nonconf_ratio <= .1
    #     nonconf_data = ... where nonconf_ratio >  .1
    # A site sitting exactly on 0.10 is CONFORMANT.
    #
    # Note bms_sa_review.conformance_metrics.aggregate_sites uses strict `<`
    # instead. The two differ only for a ratio of exactly 0.10, which no site in
    # this fleet hits, but the original is followed here for comparability.
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
               any_value(rating_capacity)                       AS rating_kva,
               sum(P_kW)                                        AS P_kW_sum,
               sum(nonconformance_voltwatt)                     AS nonconformance_sum,
               -- Excess divided by the site's own rating, so a 5 kVA and a 15 kVA
               -- system contribute on the same scale to the normalised metric.
               sum(nonconformance_voltwatt / nullif(rating_capacity, 0))
                                                                AS nonconformance_pu_sum,
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


VW_VERDICT_MEASURES = {
    "pct_sites": ("% of sites", "one vote each"),
    "pct_intervals": ("% of exposed intervals", "volume-weighted"),
    "kwh": ("kWh over the ceiling", "absolute"),
    "kwh_per_kw_per_h": ("kWh / kW / h", "normalised"),
}


def voltwatt_verdict_measures(site_day: pd.DataFrame, config=None,
                              by_cohort: bool = True) -> pd.DataFrame:
    """
    The Volt-Watt 10% verdict in the same four currencies as the Volt-VAr one.

    Deliberately parallel to ``site_verdict_measures`` so the two figures can be
    read side by side, with two differences that follow from the standard:

    * the denominator is **exposed** intervals (V > 253 V), not
      capability-assessable ones -- Volt-Watt has no 20%-of-rating floor, but a
      site that never saw high voltage was never tested;
    * the energy is **kWh of active power above the ceiling**, not kVArh of
      missing reactive power. Volt-Watt non-conformance means generating too much,
      so the quantity is real energy that should not have been exported.

    Sites with no exposed intervals carry the verdict ``not exposed``. They are
    not conformant; they were never asked the question.
    """
    import numpy as np

    config = (config or se_params.CONFIG).validate()
    verdicts = voltwatt_site_verdicts(site_day, config)[
        ["site_alias", "verdict", "is_three_phase"]]

    per_site = site_day.groupby("site_alias", as_index=False).agg(
        exposed=("total_count", "sum"),
        excess_kW_sum=("nonconformance_sum", "sum"),
        excess_pu_sum=("nonconformance_pu_sum", "sum"),
    )
    frame = verdicts.merge(per_site, on="site_alias", how="left")
    frame["cohort"] = frame.is_three_phase.map(
        {True: "three-phase", False: "single-phase"})

    keys = ["cohort"] if by_cohort else []
    rows = []
    for key, group in (frame.groupby(keys) if keys else [((), frame)]):
        cohort = key[0] if keys else "all sites"
        n_sites = len(group)
        exposed_total = float(group.exposed.sum())
        for verdict, sub in group.groupby("verdict"):
            sub_exposed = float(sub.exposed.sum())
            rows.append({
                "cohort": cohort,
                "verdict": verdict,
                "n_sites": len(sub),
                "pct_sites": _pct(len(sub), n_sites),
                "pct_intervals": _pct(sub_exposed, exposed_total),
                "kwh": round(float(sub.excess_kW_sum.sum()) * C.INTERVAL_H, 2),
                "kwh_per_kw_per_h": round(
                    float(sub.excess_pu_sum.sum()) / sub_exposed, 5
                ) if sub_exposed > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def voltwatt_ghi_site_day(con: duckdb.DuckDBPyConnection, config=None) -> pd.DataFrame:
    """
    D10b. Volt-Watt **response-supported** conformance -- section 5b.

    Port of ``conformance_voltwattghi`` from ``Volt-Watt-ghi.ipynb``. Where
    ``voltwatt_site_day`` asks only "did P exceed the ceiling", this first asks
    "was there enough sun to find out", using the D12 counterfactual.

    An exposed interval is **response-supported** -- i.e. it counts in the
    denominator -- when::

        uncurtailed_P > max_P_volt_watt   OR   uncurtailed_P IS NULL

    The first branch is the real test: available power exceeded the permitted
    ceiling, so an inverter that stayed below it demonstrably curtailed rather
    than merely lacking sun. The second branch is Hossein's, reproduced
    deliberately: where the counterfactual has no prediction the interval falls
    back to the basic maximum-output test rather than being dropped. That keeps
    the two variants comparable, but it means a poorly covered fleet quietly
    reverts toward 5a -- so ``supported_by_model_count`` is reported alongside,
    and any headline from this table should be read next to it.

    Two outputs, and they are not the same thing:

    ``nonconformance`` -- kW generated above the ceiling. A violation.
    ``curtailment``     -- kW the counterfactual says were available but not
                           generated, while staying below the ceiling.
                           **Evidence of correct response**, not of a fault.

    Requires ``se_uncurtailedpv``. Notebook 04 section 5 builds it; the BOM
    extract alone is not enough.
    """
    config = (config or se_params.CONFIG).validate()
    if not C.store_path("se_uncurtailedpv").exists():
        raise FileNotFoundError(
            "se_uncurtailedpv not found -- the response-supported Volt-Watt test "
            "needs the D12 counterfactual.\n"
            f"  expected: {C.store_path('se_uncurtailedpv')}\n"
            "  Notebook 04 section 5 builds it (build_structured -> fit_ghi_model\n"
            "  -> build_uncurtailedpv). Running the BOM extract alone is not enough:\n"
            "  that lands irradiance, it does not fit the per-site PV model.\n"
            "  Section 5a (voltwatt_site_day) runs without it."
        )

    rating = contract.capacity_column(config.rating_basis)
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    v_scored = "round(V, 6)"
    exposed = f"{v_scored} > {_A['VW']['V1']}"
    max_p = vw_max_p_sql(v_scored, "rating_capacity")
    tol = tol_kw_sql("rating_capacity", config.tolerance_fraction)

    return con.execute(
        f"""
        WITH data AS (
            SELECT i.site_alias, i.ts_aest, i.P_kW, {v} AS V,
                   {rating} AS rating_capacity,
                   s.is_three_phase, s.state
            FROM se_interval i
            {contract.cohort_join_sql('i')}
            WHERE {contract.cohort_where_sql(config)}
              AND i.P_kW IS NOT NULL AND {v} IS NOT NULL
        ),
        exposed_only AS (
            -- Filter to the exposed population FIRST, as the original does with
            -- HAVING avg(voltage) > 253. Below 253 V Volt-Watt imposes no limit,
            -- so joining the counterfactual there would be wasted work.
            SELECT * FROM data WHERE {exposed}
        ),
        joined AS (
            SELECT e.*, u.uncurtailed_P,
                   ({max_p}) + {tol} AS max_P_volt_watt
            FROM exposed_only e
            LEFT JOIN se_uncurtailedpv u
                   ON e.site_alias = u.site_alias AND e.ts_aest = u.ts_aest
        ),
        scored AS (
            SELECT *,
                   CASE WHEN uncurtailed_P > max_P_volt_watt OR uncurtailed_P IS NULL
                        THEN greatest(0, P_kW - max_P_volt_watt)
                        ELSE NULL END AS nonconformance_vwghi,
                   CASE WHEN uncurtailed_P > max_P_volt_watt AND P_kW < max_P_volt_watt
                        THEN uncurtailed_P - P_kW
                        ELSE NULL END AS curtailment_vwghi
            FROM joined
        )
        SELECT site_alias,
               any_value(is_three_phase)                         AS is_three_phase,
               any_value(state)                                  AS state,
               CAST(ts_aest AS DATE)                             AS day_aest,
               any_value(rating_capacity)                        AS rating_kva,
               sum(P_kW)                                         AS P_kW_sum,
               count(*)                                          AS exposed_count,
               count(uncurtailed_P)                              AS covered_count,
               count(*) FILTER (WHERE uncurtailed_P > max_P_volt_watt)
                                                                 AS supported_by_model_count,
               -- Denominator: response-supported intervals, model-backed or fallback.
               count(nonconformance_vwghi)                       AS total_count,
               sum(nonconformance_vwghi)                         AS nonconformance_sum,
               sum(nonconformance_vwghi / nullif(rating_capacity, 0))
                                                                 AS nonconformance_pu_sum,
               count(*) FILTER (WHERE nonconformance_vwghi > 0)  AS nonconformance_count,
               sum(curtailment_vwghi)                            AS curtailment_sum,
               count(*) FILTER (WHERE curtailment_vwghi > 0)     AS curtailment_count,
               '{config.rating_basis}'                           AS rating_basis
        FROM scored
        GROUP BY site_alias, CAST(ts_aest AS DATE)
        """
    ).df()


def voltwatt_ghi_summary(site_day: pd.DataFrame, by_cohort: bool = True) -> pd.DataFrame:
    """
    Fleet rates for the response-supported test, with its coverage exposed.

    ``pct_supported_by_model`` is the number to read first. It is the share of
    exposed intervals where the counterfactual actually established that power
    was available. The remainder reached the denominator through the
    ``uncurtailed_P IS NULL`` fallback and carries only 5a's evidence -- if that
    share is large, this table is mostly 5a wearing a different name.
    """
    keys = ["is_three_phase"] if by_cohort else []
    rows = []
    for key, group in (site_day.groupby(keys) if keys else [((), site_day)]):
        exposed = int(group.exposed_count.sum())
        supported = int(group.total_count.sum())
        model_backed = int(group.supported_by_model_count.sum())
        nonconf = int(group.nonconformance_count.sum())
        rows.append({
            "cohort": ("three-phase" if key[0] else "single-phase") if keys else "all sites",
            "n_sites": group.site_alias.nunique(),
            "exposed_intervals": exposed,
            "counterfactual_covered": int(group.covered_count.sum()),
            "pct_covered": _pct(group.covered_count.sum(), exposed),
            "response_supported_intervals": supported,
            "supported_by_model": model_backed,
            "pct_supported_by_model": _pct(model_backed, exposed),
            "nonconformant_intervals": nonconf,
            "pct_nonconformant_of_supported": _pct(nonconf, supported),
            "excess_kWh": round(float(group.nonconformance_sum.sum()) * C.INTERVAL_H, 2),
            "demonstrated_curtailment_intervals": int(group.curtailment_count.sum()),
            "demonstrated_curtailment_kWh": round(
                float(group.curtailment_sum.sum()) * C.INTERVAL_H, 2),
        })
    return pd.DataFrame(rows)


def voltwatt_ghi_site_verdicts(site_day: pd.DataFrame, config=None) -> pd.DataFrame:
    """Same 10% rule, denominator = response-supported intervals."""
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
        empty_label="not supported",
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
