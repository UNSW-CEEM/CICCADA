"""
Volt-VAr-induced active power curtailment: Methods A, B and C.
==============================================================

Deliverables D11 (Method A), D13 (Method B) and D14 (Method C).

Port of ``data_query/lib/voltvar_queries.py`` + ``voltvar_metrics.py``.

The three methods bracket the same quantity rather than agreeing
---------------------------------------------------------------
**Method A -- apparent-limit symptom scan (D11).** Model-independent. Flags an
interval when the inverter is absorbing reactive power AND sitting on its
apparent-power circle::

    Q_kvar < 0
    AND sqrt(P^2 + Q^2) >= s_limit - tol * capacity

The proxy energy is the *headroom displacement*, ``s_limit - sqrt(s_limit^2 - Q^2)``:
the kW of circle room that Q consumed. It assumes every such interval was
sun-limited, so it is a **loose upper bound**.

**Method B -- counterfactual attribution (D13).** Joins the clear-sky GHI
counterfactual and counts only generation the sun could actually have delivered::

    pmax_measured_q = sqrt(s_limit^2 - Q^2)
    attributed_kW   = greatest(0, uncurtailed_P - greatest(P_meas, pmax_measured_q))

Because the GHI model is trained on data that may itself contain curtailment, and
cloud enhancement is capped, this is a **lower bound**. A and B are reported as a
range and are NOT reconciled.

**Method C -- derating-flag corroboration (D14).** SolarEdge reports
``derating_active`` per interval. It has no Solar Analytics counterpart and is not
a third estimate -- it is an *independent label* against which A and B can be
tested on the same fleet.

The sign caveat that constrains all three
------------------------------------------
Method A's gate is ``Q_kvar < 0`` -- absorbing. That is a DIRECTION test, and the
reactive sign is not fully resolved (213 sites fit as-delivered, 106 fit flipped).
Roughly 106 sites will therefore be gated wrongly.

``exclude_polarity_suspect=True`` (the default) drops the sites that
``se_adverse`` classifies as ``polarity_suspect`` before scanning. That is a
defensible cohort restriction, not a correction -- it removes sites whose
direction cannot be trusted rather than silently flipping them.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from solar_edge.config import se_config as C
from solar_edge.lib import se_contract as contract
from solar_edge.lib import se_params

C.bootstrap_sys_path()
from bms_sa_review.shared.as4777_curves import (  # noqa: E402
    tol_kw_sql,
    vvar_required_q_sql,
    vw_max_p_sql,
)

__all__ = [
    "method_a_site_year",
    "method_a_summary",
    "method_b_site_year",
    "method_b_summary",
    "method_c_confusion",
    "method_c_by_voltage",
    "evidence_tiers",
    "eligible_context",
    "method_comparison",
    "fleet_statement",
    "fleet_narrative",
    "curtailment_concentration",
    "voltwatt_curtailment_site_year",
    "voltwatt_curtailment_summary",
    "voltwatt_curtailment_note",
    "cohort_site_count",
    "fleet_curtailment_report",
    "curtailment_concentration",
    "fleet_curtailment_narrative",
    "fleet_potential_generation",
    "voltwatt_fleet_report",
    "voltwatt_curtailment_narrative",
]

_A = C.as4777()


# ═══════════════════════════════════════════════════════════════════════════
# SHARED ELIGIBILITY
# ═══════════════════════════════════════════════════════════════════════════

def _eligible_cte(config, params, exclude_polarity_suspect: bool, con=None) -> str:
    """
    The eligible population for curtailment detection.

    Narrower than the conformance cohort, and deliberately so:

    * **240-253 V only.** Below 240 V no absorption is required; above 253 V
      Volt-Watt also acts, so curtailment there cannot be attributed to Volt-VAr
      alone. This band is what isolates the mechanism.
    * **Peak-solar hours.** Curtailment is only observable when the inverter
      wanted to produce more than it did.
    * **Clear-sky gate** when irradiance is available (D12). Without it a
      shortfall cannot be distinguished from cloud, which is precisely why
      Method A is only an upper bound.
    """
    config = config.validate()
    params = params.validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    q = contract.q_expr(config, "i")
    limit = contract.capacity_column(config.empirical_limit_basis)
    tol_basis = contract.capacity_column(config.tolerance_basis)

    extra = []
    if exclude_polarity_suspect:
        extra.append("coalesce(pol.adverse_class, 'x') <> 'polarity_suspect'")

    join_pol = (
        "LEFT JOIN _polarity pol ON i.site_alias = pol.site_alias"
        if exclude_polarity_suspect else ""
    )

    return f"""
        eligible AS (
            SELECT i.site_alias,
                   i.ts_aest,
                   year(i.ts_aest)              AS year,
                   i.P_kW,
                   {q}                          AS Q_kvar,
                   {v}                          AS V,
                   {limit}                      AS s_limit,
                   {tol_basis}                  AS tol_capacity,
                   {contract.capacity_column(config.rating_basis)} AS rating_capacity,
                   i.derating_active,
                   s.is_three_phase, s.state
            FROM se_interval i
            {contract.cohort_join_sql('i')}
            {join_pol}
            WHERE {contract.cohort_where_sql(config, extra=extra)}
              AND i.P_kW IS NOT NULL AND i.Q_kvar IS NOT NULL AND {v} IS NOT NULL
              AND {contract.v_band_sql(params, v)}
              AND {contract.peak_hours_sql(params, 'i.ts_aest')}
              AND {limit} > 0
        )
    """


def _register_polarity(con, adverse: pd.DataFrame | None):
    """Register the polarity triage so the eligibility CTE can exclude on it."""
    if adverse is None:
        con.execute(
            "CREATE OR REPLACE TEMP VIEW _polarity AS "
            "SELECT NULL::VARCHAR AS site_alias, NULL::VARCHAR AS adverse_class WHERE 1=0"
        )
    else:
        con.register("_polarity_src", adverse[["site_alias", "adverse_class"]])
        con.execute("CREATE OR REPLACE TEMP VIEW _polarity AS SELECT * FROM _polarity_src")


# ═══════════════════════════════════════════════════════════════════════════
# D11. METHOD A -- APPARENT-LIMIT SYMPTOM SCAN
# ═══════════════════════════════════════════════════════════════════════════

def method_a_site_year(
    con: duckdb.DuckDBPyConnection, config=None, params=None,
    adverse: pd.DataFrame | None = None, exclude_polarity_suspect: bool = True,
) -> pd.DataFrame:
    """
    Method A, one row per (site, year). Mirrors ``fetch_method_a_site_year``.

    ``headroom_displacement_kw`` is the kW of apparent-power circle that reactive
    absorption consumed::

        s_limit - sqrt(s_limit^2 - Q^2)

    Summing it over flagged intervals gives the Method A proxy energy. It is an
    UPPER BOUND on Volt-VAr-induced curtailment: it assumes the inverter would
    have used that headroom, which is only true when the sun was available.

    A second, structural upward bias applies here that did not apply to Solar
    Analytics. ``s_limit`` is ``s_99``, an OBSERVED p99 of apparent power. A site
    that never approached its true inverter limit gets a low ``s_limit``, which
    makes ``sqrt(P^2+Q^2) >= s_limit - tol`` fire more readily. Both biases push
    the same way, so Method A here is a looser upper bound than the original.
    D15 sweeps the quantile.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    _register_polarity(con, adverse if exclude_polarity_suspect else None)
    tol = params_tol(config)

    return con.execute(
        f"""
        WITH {_eligible_cte(config, params, exclude_polarity_suspect)},
        scored AS (
            SELECT *,
                   sqrt(P_kW * P_kW + Q_kvar * Q_kvar)                  AS S_apparent,
                   s_limit - sqrt(greatest(s_limit * s_limit - Q_kvar * Q_kvar, 0))
                                                                        AS headroom_displacement_kw,
                   CASE WHEN Q_kvar < 0
                         AND sqrt(P_kW * P_kW + Q_kvar * Q_kvar)
                             >= s_limit - {tol}
                        THEN 1 ELSE 0 END                               AS symptom
            FROM eligible
        )
        SELECT site_alias, year,
               any_value(is_three_phase)                    AS is_three_phase,
               any_value(state)                             AS state,
               count(*)                                     AS eligible_count,
               count(*) FILTER (WHERE Q_kvar < 0)           AS absorbing_q_count,
               sum(symptom)                                 AS symptom_count,
               sum(CASE WHEN symptom = 1 THEN headroom_displacement_kw ELSE 0 END)
                                                            AS headroom_displacement_kw_sum,
               avg(CASE WHEN symptom = 1 THEN V END)        AS avg_symptom_voltage,
               avg(CASE WHEN symptom = 1 THEN P_kW END)     AS avg_symptom_p_kw,
               avg(CASE WHEN symptom = 1 THEN Q_kvar END)   AS avg_symptom_q_kvar,
               count(*) FILTER (WHERE symptom = 1 AND derating_active)
                                                            AS symptom_with_derating
        FROM scored
        GROUP BY site_alias, year
        """
    ).df()


def params_tol(config) -> str:
    """The ±4% tolerance term, anchored to the configured basis."""
    return f"{config.tolerance_fraction} * tol_capacity"


def method_a_summary(site_year: pd.DataFrame, config=None, by_cohort: bool = True) -> pd.DataFrame:
    """Fleet Method A totals. Energy = headroom displacement x interval hours."""
    config = (config or se_params.CONFIG).validate()
    frame = site_year.copy()
    frame["headroom_displacement_kwh"] = (
        frame.headroom_displacement_kw_sum * config.interval_h)
    frame["affected"] = frame.symptom_count > 0

    keys = ["is_three_phase"] if by_cohort else []
    rows = []
    for key, group in (frame.groupby(keys) if keys else [((), frame)]):
        eligible = int(group.eligible_count.sum())
        symptom = int(group.symptom_count.sum())
        rows.append({
            "cohort": ("three-phase" if key[0] else "single-phase") if keys else "all sites",
            "eligible_sites": group.site_alias.nunique(),
            "symptom_sites": int(group.loc[group.affected, "site_alias"].nunique()),
            "eligible_intervals": eligible,
            "absorbing_intervals": int(group.absorbing_q_count.sum()),
            "symptom_intervals": symptom,
            "symptom_pct_of_eligible": round(100 * symptom / eligible, 4) if eligible else None,
            "headroom_displacement_kWh": round(group.headroom_displacement_kwh.sum(), 1),
            "symptom_with_derating_flag": int(group.symptom_with_derating.sum()),
        })
    return pd.DataFrame(rows)


def eligible_context(
    con: duckdb.DuckDBPyConnection, config=None, params=None,
    adverse: pd.DataFrame | None = None, exclude_polarity_suspect: bool = True,
) -> pd.DataFrame:
    """
    The denominator: eligible intervals and their potential generation.

    Reported alongside every Method A figure. A curtailment total without the
    population it came from invites the reader to assume the denominator is the
    whole fleet-year, which it never is.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    _register_polarity(con, adverse if exclude_polarity_suspect else None)

    return con.execute(
        f"""
        WITH {_eligible_cte(config, params, exclude_polarity_suspect)}
        SELECT year, site_alias,
               count(*)                     AS n_eligible_intervals,
               round(sum(P_kW) * {config.interval_h}, 3) AS measured_kWh
        FROM eligible GROUP BY year, site_alias
        """
    ).df()


# ═══════════════════════════════════════════════════════════════════════════
# D13. METHOD B -- COUNTERFACTUAL ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def method_b_site_year(
    con: duckdb.DuckDBPyConnection, config=None, params=None,
    adverse: pd.DataFrame | None = None, exclude_polarity_suspect: bool = True,
) -> pd.DataFrame:
    """
    Method B with the four evidence tiers. Mirrors ``fetch_method_b_site_year``.

    REQUIRES ``se_uncurtailedpv`` from D12. Raises a clear error if it is absent
    rather than silently returning zeros -- a curtailment total of zero because a
    LEFT JOIN found nothing is indistinguishable from a real finding, and that is
    exactly the failure the legacy pipeline hit (the NULL path scoring sites
    conservatively).

    Tiers, each strictly narrower than the last::

        1  absorbing Q                                   Q < 0
        2  + apparent-limit symptom                      on the S-circle
        3  + counterfactual above measured-Q headroom    uncurtailed_P > pmax
        4  + attributable displacement > 0               the reportable number

    ``required_q_scenario_kW`` re-runs the same arithmetic against the
    standard-REQUIRED Q instead of the measured Q -- the "what if every inverter
    conformed" counterfactual.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()

    if not C.store_path("se_uncurtailedpv").exists():
        raise FileNotFoundError(
            "se_uncurtailedpv not found -- Method B needs the D12 counterfactual.\n"
            f"  expected: {C.store_path('se_uncurtailedpv')}\n"
            "  build it with notebook 04 (BOM extract) then se_counterfactual.\n"
            "  Method A (method_a_site_year) runs without it."
        )

    _register_polarity(con, adverse if exclude_polarity_suspect else None)
    tol = params_tol(config)
    q_required = vvar_required_q_sql("V", "rating_capacity")
    gate = "symptom = 1 AND" if params.require_apparent_limit_symptom else ""

    return con.execute(
        f"""
        WITH {_eligible_cte(config, params, exclude_polarity_suspect)},
        joined AS (
            SELECT e.*, u.uncurtailed_P
            FROM eligible e
            LEFT JOIN se_uncurtailedpv u
              ON e.site_alias = u.site_alias AND e.ts_aest = u.ts_aest
        ),
        limits AS (
            SELECT *,
                   ({q_required}) AS required_q_kvar,
                   sqrt(greatest(s_limit * s_limit - Q_kvar * Q_kvar, 0))
                       AS pmax_measured_q_kw,
                   sqrt(greatest(s_limit * s_limit - power(({q_required}), 2), 0))
                       AS pmax_required_q_kw,
                   CASE WHEN Q_kvar < 0
                         AND sqrt(P_kW * P_kW + Q_kvar * Q_kvar) >= s_limit - {tol}
                        THEN 1 ELSE 0 END AS symptom
            FROM joined
        ),
        tiers AS (
            SELECT *,
                   CASE WHEN Q_kvar < 0 THEN 1 ELSE 0 END AS tier1_absorbing,
                   symptom                                AS tier2_symptom,
                   CASE WHEN Q_kvar < 0 AND uncurtailed_P > pmax_measured_q_kw
                        THEN 1 ELSE 0 END                 AS tier3_cf_above_headroom,
                   CASE WHEN {gate} Q_kvar < 0 AND uncurtailed_P > pmax_measured_q_kw
                        THEN greatest(0, uncurtailed_P - greatest(P_kW, pmax_measured_q_kw))
                        ELSE 0 END                        AS attributed_kw,
                   CASE WHEN uncurtailed_P > pmax_required_q_kw
                        THEN greatest(0, uncurtailed_P - greatest(P_kW, pmax_required_q_kw))
                        ELSE 0 END                        AS required_q_scenario_kw
            FROM limits
        )
        SELECT site_alias, year,
               any_value(is_three_phase)                        AS is_three_phase,
               any_value(state)                                 AS state,
               count(*)                                         AS eligible_count,
               count(uncurtailed_P)                             AS counterfactual_covered_count,
               sum(tier1_absorbing)                             AS tier1_absorbing_count,
               sum(tier2_symptom)                               AS tier2_symptom_count,
               sum(tier3_cf_above_headroom)                     AS tier3_count,
               count(*) FILTER (WHERE attributed_kw > 0)        AS tier4_attributed_count,
               sum(attributed_kw)                               AS attributed_kw_sum,
               sum(required_q_scenario_kw)                      AS required_q_scenario_kw_sum,
               sum(coalesce(uncurtailed_P, 0))                  AS covered_potential_kw_sum,
               sum(CASE WHEN uncurtailed_P IS NOT NULL THEN P_kW ELSE 0 END)
                                                                AS covered_measured_kw_sum
        FROM tiers GROUP BY site_alias, year
        """
    ).df()


def method_b_summary(site_year: pd.DataFrame, config=None, by_cohort: bool = True) -> pd.DataFrame:
    """Fleet Method B totals, with counterfactual coverage stated up front."""
    config = (config or se_params.CONFIG).validate()
    frame = site_year.copy()
    for col in ("attributed_kw_sum", "required_q_scenario_kw_sum",
                "covered_potential_kw_sum"):
        frame[col.replace("_kw_sum", "_kwh")] = frame[col] * config.interval_h

    keys = ["is_three_phase"] if by_cohort else []
    rows = []
    for key, group in (frame.groupby(keys) if keys else [((), frame)]):
        potential = group.covered_potential_kwh.sum()
        attributed = group.attributed_kwh.sum()
        eligible = int(group.eligible_count.sum())
        covered = int(group.counterfactual_covered_count.sum())
        rows.append({
            "cohort": ("three-phase" if key[0] else "single-phase") if keys else "all sites",
            "eligible_sites": group.site_alias.nunique(),
            "counterfactual_covered_sites": int(
                group.loc[group.counterfactual_covered_count > 0, "site_alias"].nunique()),
            "eligible_intervals": eligible,
            "counterfactual_covered_intervals": covered,
            "counterfactual_coverage_pct": round(100 * covered / eligible, 2) if eligible else None,
            "tier1_absorbing": int(group.tier1_absorbing_count.sum()),
            "tier2_symptom": int(group.tier2_symptom_count.sum()),
            "tier3_cf_above_headroom": int(group.tier3_count.sum()),
            "tier4_attributed": int(group.tier4_attributed_count.sum()),
            "attributed_kWh": round(attributed, 1),
            "required_q_scenario_kWh": round(group.required_q_scenario_kwh.sum(), 1),
            "covered_potential_kWh": round(potential, 1),
            "attributed_pct_of_covered_potential": (
                round(100 * attributed / potential, 4) if potential else None),
        })
    return pd.DataFrame(rows)


def evidence_tiers(site_year: pd.DataFrame) -> pd.DataFrame:
    """
    The attrition funnel from "something happened" to "we can attribute lost kWh".

    Each tier is strictly narrower than the last. This is what makes the claim
    defensible -- the reader sees how much evidence is discarded at each step
    rather than being handed a single number.
    """
    labels = [
        ("Tier 1: absorbing Q", "tier1_absorbing_count"),
        ("Tier 2: + apparent-limit symptom", "tier2_symptom_count"),
        ("Tier 3: + counterfactual above measured-Q headroom", "tier3_count"),
        ("Tier 4: + attributable displacement", "tier4_attributed_count"),
    ]
    rows = []
    base = None
    for label, col in labels:
        n = int(site_year[col].sum())
        base = base if base is not None else n
        rows.append({
            "evidence_tier": label,
            "n_intervals": n,
            "n_sites": int(site_year.loc[site_year[col] > 0, "site_alias"].nunique()),
            "pct_of_tier1": round(100 * n / base, 2) if base else None,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# FLEET-LEVEL REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def cohort_site_count(con: duckdb.DuckDBPyConnection, config=None) -> int:
    """
    Sites in the residential-scale cohort, whether or not they were ever eligible.

    This is the *outer* denominator. The published report quotes symptom
    prevalence against the full cohort (7,218 of 15,526), not against the subset
    that happened to see 240-253 V, so reproducing that statistic needs this
    number and not ``eligible_sites``.
    """
    config = (config or se_params.CONFIG).validate()
    return int(con.execute(
        f"""
        SELECT count(DISTINCT i.site_alias)
        FROM se_interval i
        {contract.cohort_join_sql('i')}
        WHERE {contract.cohort_where_sql(config)}
        """
    ).fetchone()[0])


def fleet_potential_generation(con: duckdb.DuckDBPyConnection, config=None) -> pd.DataFrame:
    """
    The whole-fleet generation denominator, for "% of ALL potential generation".

    The eligible denominator in :func:`fleet_curtailment_report` covers only the
    240-253 V peak-solar band -- a few percent of the fleet's energy. Dividing by
    it answers "of the generation that was *at risk*, how much was lost?". This
    function supplies the other denominator: all cohort generation, everywhere,
    which answers "of everything the fleet made, how much did Volt-VAr cost?".
    Both are legitimate and they differ by orders of magnitude, so which one a
    figure uses has to be stated.

    Three columns, because the honest answer depends on what you trust:

    ``measured_MWh``
        Metered energy. Needs no model, so it is the one figure here that cannot
        be wrong. **This is the recommended denominator.**
    ``covered_potential_MWh``
        ``sum(uncurtailed_P)`` where a counterfactual exists, with
        ``covered_measured_MWh`` alongside it so the uplift is visible.
    ``potential_MWh``
        ``sum(coalesce(uncurtailed_P, P_kW))`` -- counterfactual where available,
        measured elsewhere.

    **Read ``pct_intervals_covered`` before using ``potential_MWh``.** Fleet-wide
    counterfactual coverage is far lower than it is inside the eligible band
    (~12% versus ~94%), because the GHI model only fits where clear-sky training
    days exist. On the uncovered majority the coalesce falls back to measured,
    which is a *lower* bound on potential -- so ``potential_MWh`` is understated,
    and any percentage using it is correspondingly overstated. The bias is small
    only because curtailment itself is small.
    """
    config = (config or se_params.CONFIG).validate()
    h = config.interval_h
    has_cf = C.store_path("se_uncurtailedpv").exists()
    if has_cf:
        from solar_edge.lib import se_store
        se_store.register_store_views(con)

    join = ("LEFT JOIN se_uncurtailedpv u "
            "ON u.site_alias = i.site_alias AND u.ts_aest = i.ts_aest") if has_cf else ""
    cf = "u.uncurtailed_P" if has_cf else "CAST(NULL AS DOUBLE)"

    frame = con.execute(
        f"""
        SELECT count(*)                                        AS n_intervals,
               count({cf})                                     AS n_covered,
               sum(i.P_kW)                                     AS measured_kw_sum,
               sum(coalesce({cf}, i.P_kW))                     AS potential_kw_sum,
               sum(CASE WHEN {cf} IS NOT NULL THEN {cf} END)   AS covered_potential_kw_sum,
               sum(CASE WHEN {cf} IS NOT NULL THEN i.P_kW END) AS covered_measured_kw_sum,
               count(DISTINCT i.site_alias)                    AS n_sites
        FROM se_interval i
        {contract.cohort_join_sql('i')}
        {join}
        WHERE {contract.cohort_where_sql(config)}
          AND i.P_kW IS NOT NULL
        """
    ).df()

    out = pd.DataFrame([{
        "n_sites": int(frame.n_sites.iloc[0]),
        "n_intervals": int(frame.n_intervals.iloc[0]),
        "n_covered": int(frame.n_covered.iloc[0]),
        "pct_intervals_covered": round(
            100 * frame.n_covered.iloc[0] / frame.n_intervals.iloc[0], 2)
            if frame.n_intervals.iloc[0] else None,
        "measured_MWh": round(float(frame.measured_kw_sum.iloc[0]) * h / 1000, 1),
        "potential_MWh": round(float(frame.potential_kw_sum.iloc[0]) * h / 1000, 1),
        "covered_measured_MWh": round(
            float(frame.covered_measured_kw_sum.fillna(0).iloc[0]) * h / 1000, 1),
        "covered_potential_MWh": round(
            float(frame.covered_potential_kw_sum.fillna(0).iloc[0]) * h / 1000, 1),
        "counterfactual_available": has_cf,
    }])
    out.attrs["measured_kWh"] = float(frame.measured_kw_sum.iloc[0]) * h
    out.attrs["potential_kWh"] = float(frame.potential_kw_sum.iloc[0]) * h
    return out


def fleet_curtailment_report(
    site_year: pd.DataFrame, config=None, cohort_sites: int | None = None,
    fleet_generation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    The four statistics the published report leads with, in its own terms.

    Mirrors: "*Of these, N sites (X%) showed the V-VAr curtailment symptom [...]
    The total estimated V-VAr-induced curtailment was E MWh, equivalent to Y% of
    eligible modelled PV generation over the screened intervals, or Z% of the
    eligible timestamps.*"

    Mapping, so the numbers are auditable rather than merely quotable::

        symptom sites              tier 2  (absorbing AND on the S-circle)
        curtailment energy         tier 4  attributed_kW x interval_h
        % of modelled generation   attributed / sum(uncurtailed_P) on covered intervals
        % of eligible timestamps   tier-4 intervals / eligible intervals

    Two site percentages are returned, not one. ``pct_of_cohort`` matches the
    report's denominator; ``pct_of_eligible`` is the share of sites that could
    have shown the symptom at all. They differ by however much of the fleet never
    reaches 240 V, and quoting the first without the second overstates how much of
    the fleet was actually examined.
    """
    config = (config or se_params.CONFIG).validate()

    eligible_intervals = int(site_year.eligible_count.sum())
    covered_intervals = int(site_year.counterfactual_covered_count.sum())
    symptom_intervals = int(site_year.tier2_symptom_count.sum())
    attributed_intervals = int(site_year.tier4_attributed_count.sum())

    attributed_kwh = site_year.attributed_kw_sum.sum() * config.interval_h
    potential_kwh = site_year.covered_potential_kw_sum.sum() * config.interval_h

    eligible_sites = site_year.site_alias.nunique()
    symptom_sites = int(
        site_year.loc[site_year.tier2_symptom_count > 0, "site_alias"].nunique())
    loss_sites = int(
        site_year.loc[site_year.attributed_kw_sum > 0, "site_alias"].nunique())
    outer = cohort_sites if cohort_sites else eligible_sites

    def pct(num, den):
        return round(100 * num / den, 3) if den else None

    rows = [
        ("cohort sites (outer denominator)", outer, ""),
        ("sites with any eligible interval", eligible_sites,
         f"{pct(eligible_sites, outer)}% of cohort"),
        ("sites showing the V-VAr symptom", symptom_sites,
         f"{pct(symptom_sites, outer)}% of cohort | "
         f"{pct(symptom_sites, eligible_sites)}% of eligible"),
        ("sites with effective energy loss", loss_sites,
         f"{pct(loss_sites, outer)}% of cohort | "
         f"{pct(loss_sites, eligible_sites)}% of eligible"),
        ("eligible timestamps (240-253 V, peak, clear-sky)", eligible_intervals, ""),
        ("counterfactual-covered timestamps", covered_intervals,
         f"{pct(covered_intervals, eligible_intervals)}% coverage"),
        ("symptom timestamps", symptom_intervals,
         f"{pct(symptom_intervals, eligible_intervals)}% of eligible"),
        ("curtailed timestamps (attributed > 0)", attributed_intervals,
         f"{pct(attributed_intervals, eligible_intervals)}% of eligible"),
        ("estimated V-VAr curtailment (kWh)", round(attributed_kwh, 1), ""),
        ("estimated V-VAr curtailment (MWh)", round(attributed_kwh / 1000, 3), ""),
        ("eligible modelled PV generation (MWh)", round(potential_kwh / 1000, 1),
         "sum of uncurtailed_P on covered intervals"),
        ("curtailment as % of ELIGIBLE potential generation",
         pct(attributed_kwh, potential_kwh),
         "of the generation that was at risk, how much was lost"),
    ]

    # Second denominator: all cohort generation, not just the screened band. Both
    # are legitimate and they differ by orders of magnitude, so a bare percentage
    # is meaningless without saying which one it used.
    if fleet_generation is not None and len(fleet_generation):
        g = fleet_generation.iloc[0]
        rows += [
            ("ALL fleet measured generation (MWh)", g.measured_MWh,
             f"{g.n_sites:,} sites, every interval"),
            ("ALL fleet potential generation (MWh)", g.potential_MWh,
             f"counterfactual on {g.pct_intervals_covered}% of intervals, "
             f"measured elsewhere"),
            ("curtailment as % of ALL measured generation",
             pct(attributed_kwh, g.measured_MWh * 1000),
             "model-free denominator -- recommended"),
            ("curtailment as % of ALL potential generation",
             pct(attributed_kwh, g.potential_MWh * 1000),
             "denominator understated where coverage is thin"),
        ]
    frame = pd.DataFrame(rows, columns=["statistic", "value", "note"])
    # Counts and percentages share this column, so leave it as object rather than
    # letting pandas upcast every site count to a float and print "15526.000".
    # Explicit object dtype: `.map` re-infers, so a mixed int/float column silently
    # upcasts back to float64 and prints every site count as "15526.000".
    frame["value"] = pd.Series(
        [int(v) if isinstance(v, float) and v.is_integer() else v
         for v in frame["value"]], dtype=object)
    return frame


#: Per-response energy column on the site-level frame. Volt-VAr Method B reports
#: attributed displacement; Volt-Watt reports shed energy directly.
_ENERGY_COL = {"voltvar": "attributed_kw_sum", "voltwatt": "curtailed_kw_sum"}


def curtailment_concentration(
    site_year: pd.DataFrame, config=None,
    quantiles: tuple[float, ...] = (1, 5, 10, 20, 50),
    mode: str = "voltvar",
) -> pd.DataFrame:
    """
    How concentrated the loss is: share of total curtailment held by the top X%.

    The report's Figure 37. Computed over sites with **non-zero** curtailment,
    which is the denominator that makes "the bottom half contribute 1%" a
    statement about affected sites rather than about the whole fleet.

    ``mode`` selects the energy column: ``voltvar`` uses Method B's attributed
    displacement, ``voltwatt`` uses shed energy. The shape of the curve is the
    comparable quantity -- the two totals are not, since they measure different
    mechanisms over different voltage bands.
    """
    config = (config or se_params.CONFIG).validate()
    col = _ENERGY_COL.get(mode)
    if col is None:
        raise ValueError(f"mode must be one of {sorted(_ENERGY_COL)}, got {mode!r}")
    if col not in site_year.columns:
        raise KeyError(
            f"{col!r} not in the frame -- mode={mode!r} expects the "
            f"{'Method B' if mode == 'voltvar' else 'Volt-Watt curtailment'} "
            f"site frame. Columns present: {sorted(site_year.columns)[:8]}..."
        )
    per_site = site_year.groupby("site_alias")[col].sum() * config.interval_h
    affected = per_site[per_site > 0].sort_values(ascending=False)
    total = affected.sum()
    n = len(affected)
    if not n or not total:
        return pd.DataFrame(columns=["top_pct", "n_sites", "kWh", "pct_of_total"])

    rows = []
    for q in quantiles:
        k = max(1, int(round(n * q / 100)))
        rows.append({
            "top_pct": q,
            "n_sites": k,
            "kWh": round(affected.iloc[:k].sum(), 1),
            "pct_of_total": round(100 * affected.iloc[:k].sum() / total, 1),
        })
    rows.append({
        "top_pct": "bottom 50%",
        "n_sites": n - max(1, int(round(n * 0.5))),
        "kWh": round(affected.iloc[max(1, int(round(n * 0.5))):].sum(), 1),
        "pct_of_total": round(
            100 * affected.iloc[max(1, int(round(n * 0.5))):].sum() / total, 1),
    })
    frame = pd.DataFrame(rows)
    frame.attrs["n_affected_sites"] = n
    frame.attrs["total_kWh"] = round(total, 1)
    frame.attrs["mode"] = mode
    frame.attrs["shares"] = (affected / total).to_numpy()
    return frame


def fleet_curtailment_narrative(
    site_year: pd.DataFrame, con: duckdb.DuckDBPyConnection | None = None,
    config=None, params=None, cohort_sites: int | None = None,
    concentration: pd.DataFrame | None = None,
    fleet_generation: pd.DataFrame | None = None,
) -> str:
    """
    The report paragraph, generated from the frame rather than transcribed.

    Every number is substituted from ``site_year`` so the prose cannot drift from
    the analysis -- the failure mode this is written to prevent is a figure being
    updated while the sentence around it is not.

    Reads as a **lower bound**, because Method B only counts what the clear-sky
    counterfactual confirms. Method A's total is the upper bound and belongs in
    the same paragraph if the range is being reported honestly.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    if cohort_sites is None and con is not None:
        cohort_sites = cohort_site_count(con, config)

    rep = fleet_curtailment_report(
        site_year, config, cohort_sites, fleet_generation).set_index("statistic")

    def val(key):
        return rep.loc[key, "value"]

    # Percentages are derived here, not read back out of the report table. That
    # table stores counts in the same column as percentages, and reading the wrong
    # one produced "202256% of the eligible timestamps" -- a number so wrong it was
    # obvious, which is the only reason it was caught.
    eligible_intervals = int(site_year.eligible_count.sum())
    covered_intervals = int(site_year.counterfactual_covered_count.sum())
    pct_ts = 100 * site_year.tier4_attributed_count.sum() / eligible_intervals
    pct_cov = 100 * covered_intervals / eligible_intervals

    eligible_sites = int(val("sites with any eligible interval"))
    outer = int(val("cohort sites (outer denominator)"))
    symptom = int(val("sites showing the V-VAr symptom"))
    loss = int(val("sites with effective energy loss"))
    mwh = val("estimated V-VAr curtailment (MWh)")
    pct_gen = val("curtailment as % of ELIGIBLE potential generation")

    years = sorted(site_year.year.unique())
    span = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
    lo, hi = params.v_low, params.v_high

    text = (
        f"At fleet level, the same logic was applied across the residential-scale "
        f"cohort defined above (n={outer:,}), across {span}. Of these, "
        f"{eligible_sites:,} sites ({100 * eligible_sites / outer:.1f}%) recorded at "
        f"least one eligible interval, and {symptom:,} sites "
        f"({100 * symptom / outer:.1f}% of the cohort; "
        f"{100 * symptom / eligible_sites:.1f}% of those eligible) showed the V-VAr "
        f"curtailment symptom, defined as reactive absorption during high-voltage, "
        f"clear-sky solar periods while operating near the apparent-power limit. The "
        f"total estimated V-VAr-induced curtailment across the analysed cohort was "
        f"{mwh:.3f} MWh, equivalent to {pct_gen:.3f}% of eligible modelled PV "
        f"generation over the screened intervals, or {pct_ts:.3f}% of the eligible "
        f"timestamps. Of the sites showing the symptom, {loss:,} presented an "
        f"effective loss of energy."
    )

    # The fleet-wide denominator, only if it was supplied. Quoting the
    # exposure-conditional rate alone invites the reader to hear it as a share of
    # all generation, which it is not -- the two differ by orders of magnitude.
    all_meas = rep["value"].get("curtailment as % of ALL measured generation")
    if all_meas is not None:
        text += (f" Against all generation recorded by the cohort, rather than only the "
                 f"screened intervals, the same loss is {all_meas:.4f}%.")

    if concentration is not None and len(concentration):
        c = concentration.set_index("top_pct")["pct_of_total"]
        parts = [f"the top {q}% account for {c[q]:.0f}%"
                 for q in (1, 5, 10, 20) if q in c.index]
        if parts:
            text += (" The distribution of estimated curtailment is highly concentrated "
                     "among a small subset of sites: " + ", ".join(parts[:-1])
                     + f", and {parts[-1]} of the total.")
        if "bottom 50%" in c.index:
            text += (f" The bottom half of affected sites collectively contribute "
                     f"{c['bottom 50%']:.1f}%.")

    text += (
        f" These results isolate curtailment associated with V-VAr reactive absorption "
        f"and intentionally exclude intervals above {hi:g} V where Volt-Watt and V-VAr "
        f"responses may overlap, as well as intervals below {lo:g} V where no "
        f"absorption is required. Reported fleet values should be read as conservative "
        f"lower-bound estimates over the selected cohort: Method B counts only the "
        f"shortfall the clear-sky counterfactual confirms, and the counterfactual "
        f"covers {val('counterfactual-covered timestamps') / val('eligible timestamps (240-253 V, peak, clear-sky)') * 100:.1f}% "
        f"of eligible intervals."
    )
    return text


# ═══════════════════════════════════════════════════════════════════════════
# D14. METHOD C -- DERATING-FLAG CORROBORATION
# ═══════════════════════════════════════════════════════════════════════════

def method_c_confusion(
    con: duckdb.DuckDBPyConnection, config=None, params=None,
    adverse: pd.DataFrame | None = None, exclude_polarity_suspect: bool = True,
) -> pd.DataFrame:
    """
    The Method A symptom against SolarEdge's own ``derating_active`` flag.

    A genuine confusion matrix over eligible intervals, which no Solar Analytics
    dataset could produce: an inverter-reported label for the thing Method A is
    trying to infer.

    **Read precision, not recall.** The raw flag is ``1.0`` or NULL, never ``0.0``,
    so "not derating" and "not reported" are indistinguishable. Of the intervals
    the inverter says it was derating, what fraction did Method A catch? -- sound.
    Of the intervals Method A missed, how many were really derating? -- unanswerable.

    Note also that the flag is not Volt-VAr specific. It fires for thermal limits,
    DC clipping and export control too, so agreement is corroboration of "something
    limited output", not proof of the mechanism.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    _register_polarity(con, adverse if exclude_polarity_suspect else None)
    tol = params_tol(config)

    frame = con.execute(
        f"""
        WITH {_eligible_cte(config, params, exclude_polarity_suspect)},
        scored AS (
            SELECT derating_active,
                   CASE WHEN Q_kvar < 0
                         AND sqrt(P_kW * P_kW + Q_kvar * Q_kvar) >= s_limit - {tol}
                        THEN TRUE ELSE FALSE END AS method_a_symptom
            FROM eligible
        )
        SELECT method_a_symptom, derating_active, count(*) AS n_intervals
        FROM scored GROUP BY 1, 2 ORDER BY 1 DESC, 2 DESC
        """
    ).df()

    total = frame.n_intervals.sum()
    frame["pct_of_eligible"] = (100 * frame.n_intervals / total).round(3)

    def cell(sym, der):
        m = frame[(frame.method_a_symptom == sym) & (frame.derating_active == der)]
        return int(m.n_intervals.iloc[0]) if len(m) else 0

    tp, fp = cell(True, True), cell(True, False)
    fn, tn = cell(False, True), cell(False, False)
    frame.attrs["precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
    frame.attrs["recall_uninterpretable"] = tp / (tp + fn) if (tp + fn) else float("nan")
    frame.attrs["counts"] = dict(tp=tp, fp=fp, fn=fn, tn=tn)
    return frame


def method_c_by_voltage(
    con: duckdb.DuckDBPyConnection, config=None, bin_v: float = 1.0,
) -> pd.DataFrame:
    """
    Derating rate against voltage across the whole 240-260 V range.

    The discriminator for whether the flag can corroborate a *Volt-VAr* claim at
    all. If the rate only climbs above 253 V it is tracking Volt-Watt, and says
    nothing about the 240-253 V band where Method A operates.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    return con.execute(
        f"""
        SELECT floor({v} / {bin_v}) * {bin_v}                 AS v_bin,
               count(*)                                        AS n_intervals,
               count(*) FILTER (WHERE i.derating_active)       AS n_derating,
               round(100.0 * count(*) FILTER (WHERE i.derating_active) / count(*), 3)
                                                               AS pct_derating,
               round(avg(i.P_kW), 3)                           AS mean_P_kW,
               round(median(i.Q_kvar), 4)                      AS median_Q_kvar
        FROM se_interval i
        {contract.cohort_join_sql('i')}
        WHERE {contract.cohort_where_sql(config)}
          AND {v} BETWEEN 235 AND 262 AND i.P_kW > 0.1
        GROUP BY 1 HAVING count(*) >= 500 ORDER BY 1
        """
    ).df()


# ═══════════════════════════════════════════════════════════════════════════
# A vs B vs C
# ═══════════════════════════════════════════════════════════════════════════

def method_comparison(
    method_a: pd.DataFrame, method_b: pd.DataFrame | None,
    confusion: pd.DataFrame | None, config=None,
) -> pd.DataFrame:
    """
    The three methods side by side, as a range rather than a reconciliation.

    A and B are NOT expected to agree. A assumes every symptom interval was
    sun-limited (upper bound); B counts only what the counterfactual confirms
    (lower bound). Reporting a single number between them would be a fabrication.
    """
    config = (config or se_params.CONFIG).validate()
    rows = [{
        "method": "A -- apparent-limit symptom scan",
        "bound": "upper",
        "sites_flagged": int(method_a.loc[method_a.symptom_count > 0, "site_alias"].nunique()),
        "intervals_flagged": int(method_a.symptom_count.sum()),
        "energy_kWh": round(method_a.headroom_displacement_kw_sum.sum() * config.interval_h, 1),
        "note": "assumes every symptom interval was sun-limited",
    }]
    if method_b is not None:
        rows.append({
            "method": "B -- counterfactual attribution",
            "bound": "lower",
            "sites_flagged": int(
                method_b.loc[method_b.tier4_attributed_count > 0, "site_alias"].nunique()),
            "intervals_flagged": int(method_b.tier4_attributed_count.sum()),
            "energy_kWh": round(method_b.attributed_kw_sum.sum() * config.interval_h, 1),
            "note": "counterfactual-confirmed only; contaminated training biases it low",
        })
    else:
        rows.append({
            "method": "B -- counterfactual attribution",
            "bound": "lower", "sites_flagged": None, "intervals_flagged": None,
            "energy_kWh": None, "note": "NOT RUNNABLE -- needs the D12 GHI counterfactual",
        })
    if confusion is not None:
        counts = confusion.attrs.get("counts", {})
        rows.append({
            "method": "C -- derating-flag corroboration",
            "bound": "label, not an estimate",
            "sites_flagged": None,
            "intervals_flagged": counts.get("tp", 0) + counts.get("fn", 0),
            "energy_kWh": None,
            "note": f"precision of Method A against the flag: "
                    f"{confusion.attrs.get('precision', float('nan')):.3f}",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# VOLT-WATT CURTAILMENT
# ═══════════════════════════════════════════════════════════════════════════

def voltwatt_curtailment_site_year(
    con: duckdb.DuckDBPyConnection, config=None, params=None
) -> pd.DataFrame:
    """
    Energy not generated because Volt-Watt reduced the permitted maximum.

    Port of ``curtailment_voltwattghi`` (``curtailment_voltwattghi.ipynb``).
    Conceptually simpler than the Volt-VAr methods: Volt-Watt curtailment is
    directly observable once you know what the site could have produced, because
    the standard states the ceiling explicitly. There is no Method A/B/C split --
    there is one calculation, and it needs the counterfactual.

    Per exposed interval (V > 253 V), with ``ceiling = vw_max_p(V, S) + 4%``:

    ==========================  ====================================================
    ``curtailed_kW``            ``uncurtailed_P - P_kW``   (Hossein's definition)
                                counted only when ``uncurtailed_P > ceiling``
                                AND ``P_kW < ceiling`` -- i.e. power was available
                                above the ceiling and the site stayed below it.
    ``mandated_kW``             ``uncurtailed_P - ceiling``
                                the part the standard REQUIRED to be shed.
    ``over_reduction_kW``       ``ceiling - P_kW``
                                the part shed BEYOND the requirement.
    ==========================  ====================================================

    ``curtailed_kW = mandated_kW + over_reduction_kW`` by construction, and the
    split matters: mandated energy is the designed cost of the standard, while a
    large over-reduction means the inverter backed off further than asked --
    a settings or control-loop question, not a compliance one. Reporting only the
    total conflates a working standard with a badly tuned inverter.

    The counterfactual is floored at measured P
    (``uncurtailed_P = greatest(prediction, P_kW)``), so ``curtailed_kW >= 0``
    always and the estimate can never claim a site produced less than observed.

    **This is a LOWER bound, and the reason is coverage, not conservatism in the
    formula.** Intervals with no ``uncurtailed_P`` contribute zero curtailment
    rather than an unknown, so every gap in the counterfactual pushes the total
    down. ``counterfactual_covered_count`` and ``exposed_count`` are returned
    together for exactly this reason -- read the ratio before quoting the energy.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()

    if not C.store_path("se_uncurtailedpv").exists():
        raise FileNotFoundError(
            "se_uncurtailedpv not found -- Volt-Watt curtailment needs the D12 "
            "counterfactual.\n"
            f"  expected: {C.store_path('se_uncurtailedpv')}\n"
            "  Build it with notebook 04 section 5.\n"
            "  Unlike Volt-VAr, there is no counterfactual-free variant: without\n"
            "  it you cannot tell curtailment from a cloud."
        )
    from solar_edge.lib import se_store

    se_store.register_store_views(con)

    rating = contract.capacity_column(config.rating_basis)
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    v_scored = "round(V, 6)"
    max_p = vw_max_p_sql(v_scored, "rating_kva")
    tol = tol_kw_sql("rating_kva", config.tolerance_fraction)
    exposed = f"{v_scored} > {_A['VW']['V1']}"

    return con.execute(
        f"""
        WITH base AS (
            SELECT i.site_alias, i.ts_aest, i.P_kW,
                   {v}        AS V,
                   {rating}   AS rating_kva,
                   s.is_three_phase, s.state,
                   u.uncurtailed_P
            FROM se_interval i
            {contract.cohort_join_sql('i')}
            LEFT JOIN se_uncurtailedpv u
                   ON u.site_alias = i.site_alias AND u.ts_aest = i.ts_aest
            WHERE {contract.cohort_where_sql(config)}
              AND i.P_kW IS NOT NULL AND {v} IS NOT NULL
        ),
        ceil AS (
            SELECT *, ({max_p}) + {tol} AS ceiling_kW FROM base WHERE {exposed}
        ),
        scored AS (
            SELECT *,
                   -- The response-opportunity gate: available power above the
                   -- ceiling AND the site sitting below it. Without the first
                   -- clause a cloudy interval would read as curtailment.
                   (uncurtailed_P > ceiling_kW AND P_kW < ceiling_kW) AS is_curtailed,
                   CASE WHEN uncurtailed_P > ceiling_kW AND P_kW < ceiling_kW
                        THEN uncurtailed_P - P_kW ELSE 0 END       AS curtailed_kW,
                   CASE WHEN uncurtailed_P > ceiling_kW AND P_kW < ceiling_kW
                        THEN uncurtailed_P - ceiling_kW ELSE 0 END AS mandated_kW,
                   CASE WHEN uncurtailed_P > ceiling_kW AND P_kW < ceiling_kW
                        THEN ceiling_kW - P_kW ELSE 0 END          AS over_reduction_kW
            FROM ceil
        )
        SELECT site_alias,
               any_value(is_three_phase)                  AS is_three_phase,
               any_value(state)                           AS state,
               any_value(rating_kva)                      AS rating_kva,
               count(*)                                   AS exposed_count,
               count(uncurtailed_P)                       AS counterfactual_covered_count,
               count(*) FILTER (WHERE uncurtailed_P > ceiling_kW)
                                                          AS response_opportunity_count,
               count(*) FILTER (WHERE is_curtailed)       AS curtailed_count,
               sum(curtailed_kW)                          AS curtailed_kw_sum,
               sum(mandated_kW)                           AS mandated_kw_sum,
               sum(over_reduction_kW)                     AS over_reduction_kw_sum,
               sum(P_kW)                                  AS measured_kw_sum,
               sum(coalesce(uncurtailed_P, P_kW))         AS potential_kw_sum
        FROM scored
        GROUP BY site_alias
        """
    ).df()


def voltwatt_curtailment_summary(site_year: pd.DataFrame, config=None,
                                 by_cohort: bool = True) -> pd.DataFrame:
    """
    Fleet Volt-Watt curtailment energy, with its coverage attached.

    ``pct_exposed_covered`` is the first number to read. Curtailment is only
    detectable where a counterfactual exists; the rest of the exposed intervals
    contribute zero, so the energy total scales with coverage and comparing two
    fleets (or two runs) without it is meaningless.

    ``pct_over_reduction`` splits the total: how much of the shed energy the
    standard actually required versus how much the inverters gave up beyond it.
    """
    import numpy as np

    config = (config or se_params.CONFIG).validate()
    h = C.INTERVAL_H
    keys = ["is_three_phase"] if by_cohort else []

    rows = []
    for key, g in (site_year.groupby(keys) if keys else [((), site_year)]):
        exposed = float(g.exposed_count.sum())
        covered = float(g.counterfactual_covered_count.sum())
        curtailed_kwh = float(g.curtailed_kw_sum.sum()) * h
        mandated_kwh = float(g.mandated_kw_sum.sum()) * h
        over_kwh = float(g.over_reduction_kw_sum.sum()) * h
        potential_kwh = float(g.potential_kw_sum.sum()) * h
        rows.append({
            "cohort": ("three-phase" if key[0] else "single-phase") if keys else "all sites",
            "n_sites": g.site_alias.nunique(),
            "sites_with_curtailment": int((g.curtailed_count > 0).sum()),
            "exposed_intervals": int(exposed),
            "counterfactual_covered": int(covered),
            "pct_exposed_covered": round(100 * covered / exposed, 2) if exposed else np.nan,
            "response_opportunity_intervals": int(g.response_opportunity_count.sum()),
            "curtailed_intervals": int(g.curtailed_count.sum()),
            "curtailed_kWh": round(curtailed_kwh, 2),
            "mandated_kWh": round(mandated_kwh, 2),
            "over_reduction_kWh": round(over_kwh, 2),
            "pct_over_reduction": round(100 * over_kwh / curtailed_kwh, 2)
                                  if curtailed_kwh else np.nan,
            "pct_of_exposed_potential": round(100 * curtailed_kwh / potential_kwh, 3)
                                        if potential_kwh else np.nan,
        })
    return pd.DataFrame(rows)


def voltwatt_fleet_report(
    site_frame: pd.DataFrame, config=None, cohort_sites: int | None = None,
) -> pd.DataFrame:
    """
    The Volt-Watt analogue of :func:`fleet_curtailment_report`.

    Deliberately parallel in shape so the two responses can be read side by side,
    but three of the rows mean something different and the labels say so:

    * **"exposed", not "eligible".** The Volt-VAr population is the 240-253 V
      band; the Volt-Watt population is everything above 253 V. They do not
      overlap by construction -- that is what keeps the two mechanisms separable.
    * **No symptom tier.** Volt-Watt needs no proxy: the standard states the
      ceiling, so "the sun could have delivered more than the ceiling allowed"
      is directly checkable. ``response_opportunity`` replaces the symptom count.
    * **The total splits.** ``mandated`` is the designed cost of the standard;
      ``over_reduction`` is what inverters shed beyond what was asked. Reporting
      only the sum conflates a working standard with a badly tuned inverter.
    """
    config = (config or se_params.CONFIG).validate()
    h = config.interval_h

    exposed = int(site_frame.exposed_count.sum())
    covered = int(site_frame.counterfactual_covered_count.sum())
    opportunity = int(site_frame.response_opportunity_count.sum())
    curtailed_n = int(site_frame.curtailed_count.sum())

    curtailed_kwh = float(site_frame.curtailed_kw_sum.sum()) * h
    mandated_kwh = float(site_frame.mandated_kw_sum.sum()) * h
    over_kwh = float(site_frame.over_reduction_kw_sum.sum()) * h
    potential_kwh = float(site_frame.potential_kw_sum.sum()) * h

    exposed_sites = site_frame.site_alias.nunique()
    opportunity_sites = int(
        site_frame.loc[site_frame.response_opportunity_count > 0, "site_alias"].nunique())
    loss_sites = int(
        site_frame.loc[site_frame.curtailed_kw_sum > 0, "site_alias"].nunique())
    outer = cohort_sites if cohort_sites else exposed_sites

    def pct(num, den):
        return round(100 * num / den, 3) if den else None

    rows = [
        ("cohort sites (outer denominator)", outer, ""),
        ("sites exposed to Volt-Watt (V > 253 V)", exposed_sites,
         f"{pct(exposed_sites, outer)}% of cohort"),
        ("sites with a response opportunity", opportunity_sites,
         f"{pct(opportunity_sites, outer)}% of cohort | "
         f"{pct(opportunity_sites, exposed_sites)}% of exposed"),
        ("sites with effective energy loss", loss_sites,
         f"{pct(loss_sites, outer)}% of cohort | "
         f"{pct(loss_sites, exposed_sites)}% of exposed"),
        ("exposed timestamps (V > 253 V)", exposed, ""),
        ("counterfactual-covered timestamps", covered,
         f"{pct(covered, exposed)}% coverage"),
        ("response-opportunity timestamps", opportunity,
         f"{pct(opportunity, exposed)}% of exposed"),
        ("curtailed timestamps", curtailed_n, f"{pct(curtailed_n, exposed)}% of exposed"),
        ("estimated Volt-Watt curtailment (kWh)", round(curtailed_kwh, 1), ""),
        ("estimated Volt-Watt curtailment (MWh)", round(curtailed_kwh / 1000, 3), ""),
        ("  of which mandated by the standard (MWh)", round(mandated_kwh / 1000, 3),
         f"{pct(mandated_kwh, curtailed_kwh)}% of total"),
        ("  of which over-reduction (MWh)", round(over_kwh / 1000, 3),
         f"{pct(over_kwh, curtailed_kwh)}% of total"),
        ("exposed potential PV generation (MWh)", round(potential_kwh / 1000, 1),
         "sum of uncurtailed_P over exposed intervals"),
        ("curtailment as % of exposed potential", pct(curtailed_kwh, potential_kwh), ""),
    ]
    frame = pd.DataFrame(rows, columns=["statistic", "value", "note"])
    frame["value"] = pd.Series(
        [int(v) if isinstance(v, float) and v.is_integer() else v
         for v in frame["value"]], dtype=object)
    return frame


def voltwatt_curtailment_narrative(
    site_frame: pd.DataFrame, con: duckdb.DuckDBPyConnection | None = None,
    config=None, cohort_sites: int | None = None,
    concentration: pd.DataFrame | None = None,
) -> str:
    """
    The Volt-Watt reporting paragraph, generated from the frame.

    Companion to :func:`fleet_curtailment_narrative`. States the mandated /
    over-reduction split explicitly, because for Volt-Watt the headline total is
    not by itself a compliance finding -- most of it is the standard working as
    designed.
    """
    config = (config or se_params.CONFIG).validate()
    if cohort_sites is None and con is not None:
        cohort_sites = cohort_site_count(con, config)

    h = config.interval_h
    exposed = int(site_frame.exposed_count.sum())
    covered = int(site_frame.counterfactual_covered_count.sum())
    curtailed_n = int(site_frame.curtailed_count.sum())
    curtailed_kwh = float(site_frame.curtailed_kw_sum.sum()) * h
    mandated_kwh = float(site_frame.mandated_kw_sum.sum()) * h
    over_kwh = float(site_frame.over_reduction_kw_sum.sum()) * h
    potential_kwh = float(site_frame.potential_kw_sum.sum()) * h

    exposed_sites = site_frame.site_alias.nunique()
    loss_sites = int(
        site_frame.loc[site_frame.curtailed_kw_sum > 0, "site_alias"].nunique())
    outer = cohort_sites or exposed_sites
    v1 = _A["VW"]["V1"]

    text = (
        f"The same counterfactual was applied to Volt-Watt across the residential-scale "
        f"cohort (n={outer:,}). Of these, {exposed_sites:,} sites "
        f"({100 * exposed_sites / outer:.1f}%) were exposed to Volt-Watt at least once, "
        f"defined as an average site voltage above {v1:g} V, and {loss_sites:,} sites "
        f"({100 * loss_sites / outer:.1f}% of the cohort; "
        f"{100 * loss_sites / exposed_sites:.1f}% of those exposed) showed an effective "
        f"loss of energy. The total estimated Volt-Watt curtailment was "
        f"{curtailed_kwh / 1000:.3f} MWh, equivalent to "
        f"{100 * curtailed_kwh / potential_kwh:.3f}% of modelled PV generation over the "
        f"exposed intervals, or {100 * curtailed_n / exposed:.3f}% of the exposed "
        f"timestamps."
    )

    if curtailed_kwh:
        text += (
            f" Of that total, {100 * mandated_kwh / curtailed_kwh:.1f}% "
            f"({mandated_kwh / 1000:.3f} MWh) was mandated by the standard's power "
            f"ceiling and {100 * over_kwh / curtailed_kwh:.1f}% "
            f"({over_kwh / 1000:.3f} MWh) was reduction beyond that requirement. Only "
            f"the second component represents inverter behaviour that the standard did "
            f"not ask for."
        )

    if concentration is not None and len(concentration):
        c = concentration.set_index("top_pct")["pct_of_total"]
        parts = [f"the top {q}% account for {c[q]:.0f}%"
                 for q in (1, 5, 10, 20) if q in c.index]
        if parts:
            text += (" The loss is similarly concentrated: " + ", ".join(parts[:-1])
                     + f", and {parts[-1]} of the total.")

    text += (
        f" Unlike the Volt-VAr figures, this population is disjoint by construction: "
        f"Volt-VAr is assessed over 240-{v1:g} V and Volt-Watt only above {v1:g} V, so "
        f"the two totals are additive rather than overlapping. This is also a lower "
        f"bound -- intervals without a counterfactual contribute zero rather than an "
        f"unknown, and the counterfactual covers {100 * covered / exposed:.1f}% of "
        f"exposed intervals."
    )
    return text


def voltwatt_curtailment_note(summary: pd.DataFrame) -> None:
    """Print the reading of a Volt-Watt curtailment summary, caveats included."""
    row = summary.iloc[-1] if len(summary) == 1 else summary.sum(numeric_only=True)
    covered = summary.counterfactual_covered.sum()
    exposed = summary.exposed_intervals.sum()
    curtailed = summary.curtailed_kWh.sum()
    mandated = summary.mandated_kWh.sum()
    over = summary.over_reduction_kWh.sum()

    print("Volt-Watt curtailment\n" + "=" * 60)
    print(f"  {curtailed:,.1f} kWh not generated while Volt-Watt was active")
    if curtailed:
        print(f"    {mandated:,.1f} kWh ({100 * mandated / curtailed:.1f}%) "
              "required by the standard")
        print(f"    {over:,.1f} kWh ({100 * over / curtailed:.1f}%) "
              "shed BEYOND the requirement")
    print(f"\n  Coverage: {covered:,} of {exposed:,} exposed intervals "
          f"({100 * covered / exposed:.1f}%) have a counterfactual.")
    print("  Uncovered intervals contribute ZERO, so this is a LOWER bound and it")
    print("  scales with coverage. Quote the two together or not at all.")
