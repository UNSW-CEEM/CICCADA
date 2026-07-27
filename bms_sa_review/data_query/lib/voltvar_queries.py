"""Builder-aligned Volt-VAr symptom and counterfactual attribution queries."""

import pandas as pd

from bms_sa_review.shared.as4777_curves import vvar_required_q_sql
from bms_sa_review.shared.ciccada_config import AS4777

from analysis_contract import flex_sql, years_sql


def _base_sql(config, params):
    sd = config.tables["structured_data"]
    rating = params.rating_basis
    empirical = params.empirical_limit_basis
    tolerance = params.tolerance_basis
    ghi = ""
    if params.apply_ghi_filter:
        ghi = f"AND sd.GHI_cs > 0 AND sd.GHI / sd.GHI_cs >= {params.ghi_cs_ratio_min}"
    return f"""
        site_meta AS (
            SELECT site_id,
                   bool_or(coalesce(flex_export_detected, False)) AS flex_export_detected
            FROM {config.metadata_table}
            WHERE is_pv = True GROUP BY site_id
        ),
        base AS (
            SELECT sd.site_id, sd.t_stamp, sd.actual_day,
                   year(sd.t_stamp + interval '10' hour) AS year,
                   sd.V,
                   sd.P_kw_norm * sd.normalization_capacity AS P_kW,
                   sd.Q_kvar_norm * sd.normalization_capacity AS Q_kvar,
                   sd.GHI, sd.GHI_cs,
                   sd.{rating} AS rating_capacity,
                   sd.{empirical} AS empirical_limit,
                   sd.{tolerance} AS tolerance_capacity
            FROM {sd} sd
            JOIN site_meta m ON sd.site_id = m.site_id
            WHERE {years_sql(params.years, "year(sd.t_stamp + interval '10' hour)")}
              AND {flex_sql(params.flex_selection, "m.flex_export_detected")}
              AND sd.ac_capacity_kw > 0
              AND sd.ac_capacity_kw <= {params.max_ac_capacity_kw}
              AND sd.{rating} > 0 AND sd.{empirical} > 0 AND sd.{tolerance} > 0
              AND sd.V > {params.v_low} AND sd.V < {params.v_high}
              AND hour(sd.t_stamp + interval '10' hour) >= {params.peak_hour_start}
              AND hour(sd.t_stamp + interval '10' hour) < {params.peak_hour_end}
              {ghi}
        )
    """


def fetch_input_coverage(aq, config, params):
    sd = config.tables["structured_data"]
    unc = config.tables["all_uncurtailedpv"]
    return aq(f"""
        SELECT dataset, year, n_rows, n_sites FROM (
            SELECT 'structured_data' AS dataset, year,
                   count(*) AS n_rows, count(DISTINCT site_id) AS n_sites
            FROM {sd} WHERE {years_sql(params.years)} GROUP BY year
            UNION ALL
            SELECT 'all_uncurtailedpv', year,
                   count(*), count(DISTINCT site_id)
            FROM {unc} WHERE {years_sql(params.years)} GROUP BY year
        ) ORDER BY dataset, year
    """, database=config.database)


def fetch_method_a_site_year(aq, config, params):
    """
    Method A symptom scan — reads the RAW ts table (legacy-equivalent).

    Reproduces the legacy fetch_method_a physics: aggregates power/reactive
    across a site's circuits per timestamp from ts, takes max(voltage), applies
    the s_99 apparent-limit test, and (optionally) the structured_data clear-sky
    GHI filter. Returns one row per (site_id, year) with the schema that
    method_a_summary expects.

    Columns returned:
        site_id, year, eligible_count, absorbing_q_count, symptom_count,
        headroom_displacement_kw_sum, avg_symptom_voltage, avg_symptom_p_kw,
        avg_symptom_q_kvar
    """
    sd        = config.tables["structured_data"]
    empirical = params.empirical_limit_basis      # 's_99' or 'ac_capacity_kw'
    tol       = params.tolerance_fraction
    years_in  = ", ".join(str(int(y)) for y in params.years)

    # Flex predicate on metadata (site-level), mirroring the base query.
    from analysis_contract import flex_sql
    flex_pred = flex_sql(params.flex_selection, "sm2.flex_export_detected")

    # Clear-sky GHI filter: join structured_data purely for the ghi/ghi_cs test,
    # exactly as the legacy _build_ghi_fragments did (P/Q/V still come from ts).
    ghi_join = ""
    ghi_filter = ""
    if params.apply_ghi_filter:
        ghi_join = (
            f"JOIN {sd} sd "
            f"ON si.site_id = sd.site_id AND si.t_stamp = sd.t_stamp"
        )
        ghi_filter = (
            f"AND sd.GHI_cs > 0 "
            f"AND sd.GHI / sd.GHI_cs >= {params.ghi_cs_ratio_min}"
        )

    return aq(f"""
        WITH site_meta AS (
            SELECT DISTINCT m.site_id, m.circuit_id, m.circuit_polarity,
                   m.ac_capacity_kw,
                   m.{empirical} AS s_limit
            FROM {config.metadata_table} m
            JOIN (
                SELECT site_id,
                       bool_or(coalesce(flex_export_detected, False)) AS flex_export_detected
                FROM {config.metadata_table}
                WHERE is_pv = True
                GROUP BY site_id
            ) sm2 ON m.site_id = sm2.site_id
            WHERE m.is_pv = True
              AND m.{empirical} > 0
              AND m.ac_capacity_kw > 0
              AND m.ac_capacity_kw <= {params.max_ac_capacity_kw}
              AND {flex_pred}
        ),
        d AS (
            SELECT
                sm.site_id,
                t.t_stamp,
                year(t.t_stamp + interval '10' hour) AS year,
                sm.s_limit,
                sm.ac_capacity_kw,
                t.voltage,
                t.power * sm.circuit_polarity / 1000.0                  AS P_kW,
                t.energy_reactive * sm.circuit_polarity / 1000.0 * 12   AS Q_kvar
            FROM ts t
            JOIN site_meta sm ON t.circuit_id = sm.circuit_id
            WHERE t.is_pv = True
              AND t.year IN ({years_in})
              AND t.voltage > {params.v_low} AND t.voltage < {params.v_high}
        ),
        site_interval AS (
            SELECT
                si.site_id,
                si.year,
                si.t_stamp,
                max(si.s_limit)        AS s_limit,
                max(si.ac_capacity_kw) AS ac_capacity_kw,
                max(si.voltage)        AS V_max,
                sum(si.P_kW)           AS P_kW,
                sum(si.Q_kvar)         AS Q_kvar
            FROM d si
            {ghi_join}
            WHERE hour(si.t_stamp + interval '10' hour)
                  BETWEEN {params.peak_hour_start} AND {params.peak_hour_end}
              {ghi_filter}
            GROUP BY si.site_id, si.year, si.t_stamp
        ),
        scored AS (
            SELECT
                site_id, year, s_limit, ac_capacity_kw, V_max, P_kW, Q_kvar,
                sqrt(P_kW*P_kW + Q_kvar*Q_kvar) AS S_apparent,
                s_limit - sqrt(greatest(s_limit*s_limit - Q_kvar*Q_kvar, 0))
                    AS headroom_displacement_kw,
                CASE WHEN Q_kvar < 0
                          AND sqrt(P_kW*P_kW + Q_kvar*Q_kvar)
                              >= s_limit - {tol} * ac_capacity_kw
                     THEN 1 ELSE 0 END AS apparent_limit_symptom
            FROM site_interval
        )
        SELECT
            site_id, year,
            count(*)                    AS eligible_count,
            count_if(Q_kvar < 0)        AS absorbing_q_count,
            sum(apparent_limit_symptom) AS symptom_count,
            sum(CASE WHEN apparent_limit_symptom = 1
                     THEN headroom_displacement_kw ELSE 0 END)
                AS headroom_displacement_kw_sum,
            avg(CASE WHEN apparent_limit_symptom = 1 THEN V_max   END) AS avg_symptom_voltage,
            avg(CASE WHEN apparent_limit_symptom = 1 THEN P_kW    END) AS avg_symptom_p_kw,
            avg(CASE WHEN apparent_limit_symptom = 1 THEN Q_kvar  END) AS avg_symptom_q_kvar
        FROM scored
        GROUP BY site_id, year
    """, database=config.database)


def fetch_method_b_site_year(aq, config, params):
    base = _base_sql(config, params)
    unc = config.tables["all_uncurtailedpv"]
    q_required = vvar_required_q_sql("V", "rating_capacity")
    tol = params.tolerance_fraction
    symptom_gate = "apparent_limit_symptom = 1 AND" if params.require_apparent_limit_symptom else ""
    return aq(f"""
        WITH {base}, joined AS (
            SELECT b.*, u.uncurtailed_P
            FROM base b
            LEFT JOIN {unc} u ON b.site_id=u.site_id AND b.t_stamp=u.t_stamp
        ), limits AS (
            SELECT *, ({q_required}) AS required_q_kvar,
                   sqrt(greatest(empirical_limit*empirical_limit-Q_kvar*Q_kvar,0))
                       AS pmax_measured_q_kw,
                   sqrt(greatest(empirical_limit*empirical_limit-power(({q_required}),2),0))
                       AS pmax_required_q_kw,
                   CASE WHEN Q_kvar < 0 AND sqrt(P_kW*P_kW+Q_kvar*Q_kvar)
                             >= empirical_limit-{tol}*tolerance_capacity
                        THEN 1 ELSE 0 END AS apparent_limit_symptom
            FROM joined
        ), tiers AS (
            SELECT *,
                   CASE WHEN Q_kvar < 0 THEN 1 ELSE 0 END AS tier1_absorbing,
                   apparent_limit_symptom AS tier2_symptom,
                   CASE WHEN Q_kvar < 0 AND uncurtailed_P > pmax_measured_q_kw THEN 1 ELSE 0 END
                       AS tier3_cf_above_measured_q_headroom,
                   CASE WHEN {symptom_gate} Q_kvar < 0
                                  AND uncurtailed_P > pmax_measured_q_kw
                        THEN greatest(0, uncurtailed_P-greatest(P_kW,pmax_measured_q_kw))
                        ELSE 0 END AS attributed_measured_q_kw,
                   CASE WHEN uncurtailed_P > pmax_required_q_kw
                        THEN greatest(0, uncurtailed_P-greatest(P_kW,pmax_required_q_kw))
                        ELSE 0 END AS required_q_scenario_kw
            FROM limits
        )
        SELECT site_id, year,
               count(*) AS eligible_count,
               count(uncurtailed_P) AS counterfactual_covered_count,
               sum(tier1_absorbing) AS tier1_absorbing_count,
               sum(tier2_symptom) AS tier2_symptom_count,
               sum(tier3_cf_above_measured_q_headroom) AS tier3_count,
               count_if(attributed_measured_q_kw > 0) AS tier4_attributed_count,
               sum(attributed_measured_q_kw) AS attributed_measured_q_kw_sum,
               sum(required_q_scenario_kw) AS required_q_scenario_kw_sum,
               sum(CASE WHEN uncurtailed_P IS NOT NULL THEN uncurtailed_P ELSE 0 END)
                   AS covered_potential_kw_sum,
               sum(CASE WHEN uncurtailed_P IS NOT NULL THEN P_kW ELSE 0 END)
                   AS covered_measured_kw_sum
        FROM tiers GROUP BY site_id, year
    """, database=config.database)


def fetch_method_b_intervals(aq, config, params, site_id, year):
    base = _base_sql(config, params)
    unc = config.tables["all_uncurtailedpv"]
    q_required = vvar_required_q_sql("V", "rating_capacity")
    tol = params.tolerance_fraction
    return aq(f"""
        WITH {base}, j AS (
            SELECT b.*, u.uncurtailed_P
            FROM base b LEFT JOIN {unc} u
              ON b.site_id=u.site_id AND b.t_stamp=u.t_stamp
            WHERE b.site_id={int(site_id)} AND b.year={int(year)}
        )
        SELECT *, sqrt(P_kW*P_kW+Q_kvar*Q_kvar) AS S_apparent,
               ({q_required}) AS required_q_kvar,
               sqrt(greatest(empirical_limit*empirical_limit-Q_kvar*Q_kvar,0))
                   AS pmax_measured_q_kw,
               sqrt(greatest(empirical_limit*empirical_limit-power(({q_required}),2),0))
                   AS pmax_required_q_kw,
               CASE WHEN Q_kvar < 0 AND sqrt(P_kW*P_kW+Q_kvar*Q_kvar)
                     >= empirical_limit-{tol}*tolerance_capacity THEN True ELSE False END
                   AS apparent_limit_symptom
        FROM j ORDER BY t_stamp
    """, database=config.database)


def fetch_stage2_vvar_baseline(aq, config, params):
    table = config.tables["conformance_voltvar"]
    return aq(f"""
        SELECT year,
               count(DISTINCT CASE WHEN total_count>0 THEN site_id END) AS assessable_sites,
               sum(total_count) AS capability_assessable_intervals,
               sum(Q_adverse_count+Q_inactive_count+Q_significant_shortfall_count)
                   AS reduced_nonconf_intervals,
               sum(curtailment_eligible_count) AS apparent_limit_symptom_intervals,
               sum(curtailment_voltvar_count) AS counterfactual_curtailment_intervals,
               sum(curtailment_voltvar_sum)*{AS4777['INTERVAL_H']} AS curtailed_kwh
        FROM {table} WHERE {years_sql(params.years)}
        GROUP BY year ORDER BY year
    """, database=config.database)

# ═════════════════════════════════════════════════════════════════════════
# Full-day telemetry fetch for the single-site day plot
# ═════════════════════════════════════════════════════════════════════════

def fetch_day_data(aq, config, params, site_id, date_str):
    """
    One full AEST day of telemetry + counterfactual for a single site.
    Queries raw ts (circuit-level), aggregates to site level, LEFT JOINs the
    counterfactual. Mirrors the legacy fetch_day_data exactly.
    """
    unc = config.tables["all_uncurtailedpv"]

    return aq(f"""
        WITH site_meta AS (
            SELECT DISTINCT circuit_id, circuit_polarity
            FROM {config.metadata_table}
            WHERE is_pv = True AND site_id = {int(site_id)}
        ),
        meas AS (
            SELECT
                t.t_stamp,
                max(t.voltage)                                             AS V,
                sum(t.power * sm.circuit_polarity / 1000.0)                AS P_kW,
                sum(t.energy_reactive * sm.circuit_polarity / 1000.0 * 12) AS Q_kvar
            FROM ts t
            JOIN site_meta sm ON t.circuit_id = sm.circuit_id
            WHERE t.is_pv = True
              AND date(t.t_stamp + interval '10' hour) = DATE '{date_str}'
            GROUP BY t.t_stamp
        )
        SELECT m.t_stamp, m.V, m.P_kW, m.Q_kvar,
               u.uncurtailed_P AS P_potential_kW
        FROM meas m
        LEFT JOIN (
                    SELECT site_id, t_stamp, avg(uncurtailed_P) AS uncurtailed_P
                    FROM {unc}
                    WHERE site_id = {int(site_id)}
                    GROUP BY site_id, t_stamp
                ) u
                ON u.site_id = {int(site_id)} AND u.t_stamp = m.t_stamp
        ORDER BY m.t_stamp
    """, database=config.database)

# ═════════════════════════════════════════════════════════════════════════
# Legacy-equivalent Method B: reads raw ts, single site, one row per Q<0 interval, 
# computes varcurt_kW exactly like the legacy fetch_method_b_site_year.
# ═════════════════════════════════════════════════════════════════════════

def fetch_method_b_legacy(aq, config, params, site_id, year):
    """
    Legacy-equivalent Method B for ONE site and ONE year.

    Mirrors the pre-evidence-tier fetch_method_b_site_year exactly:
      * raw ts, summed across the site's circuits, max(voltage)
      * s_limit from meta_up23c (empirical_limit_basis, default s_99)
      * one row per Q<0 interval in the 240-253 V peak-solar window
      * varcurt_kW = greatest(0, P_potential - sqrt(s_limit^2 - Q^2))
        (NO symptom gate, NO greatest(P, ...) floor — legacy behaviour)

    Returns interval-level rows:
        t_stamp, V_max, s_limit, ac_capacity_kw, P_meas_kW, Q_meas_kvar,
        P_potential_kW, P_max_given_Q, varcurt_kW
    """
    empirical = params.empirical_limit_basis        # 's_99' (legacy USE_S99=True)
    unc       = config.tables["all_uncurtailedpv"]

    return aq(f"""
        WITH site_meta AS (
            SELECT DISTINCT site_id, circuit_id, circuit_polarity,
                   ac_capacity_kw, {empirical} AS s_limit
            FROM {config.metadata_table}
            WHERE is_pv = True
              AND site_id = {int(site_id)}
              AND ac_capacity_kw > 0
              AND ac_capacity_kw <= {params.max_ac_capacity_kw}
        ),
        meas AS (
            SELECT
                sm.site_id, t.t_stamp,
                max(sm.s_limit)                                        AS s_limit,
                max(sm.ac_capacity_kw)                                 AS ac_capacity_kw,
                max(t.voltage)                                         AS V_max,
                sum(t.power * sm.circuit_polarity / 1000.0)            AS P_meas_kW,
                sum(t.energy_reactive * sm.circuit_polarity/1000.0*12) AS Q_meas_kvar
            FROM ts t
            JOIN site_meta sm ON t.circuit_id = sm.circuit_id
            WHERE t.year = {int(year)} AND t.is_pv = True
              AND t.voltage > {params.v_low} AND t.voltage < {params.v_high}
              AND hour(t.t_stamp + interval '10' hour)
                  BETWEEN {params.peak_hour_start} AND {params.peak_hour_end}
            GROUP BY sm.site_id, t.t_stamp
        ),
        pot AS (
            SELECT site_id, t_stamp, uncurtailed_P AS P_potential_kW
            FROM {unc}
            WHERE site_id = {int(site_id)}
        )
        SELECT
            m.t_stamp, m.V_max, m.s_limit, m.ac_capacity_kw,
            m.P_meas_kW, m.Q_meas_kvar,
            p.P_potential_kW,
            sqrt(greatest(m.s_limit*m.s_limit - m.Q_meas_kvar*m.Q_meas_kvar, 0))
                AS P_max_given_Q,
            greatest(0,
                p.P_potential_kW
                - sqrt(greatest(m.s_limit*m.s_limit - m.Q_meas_kvar*m.Q_meas_kvar, 0))
            ) AS varcurt_kW
        FROM meas m
        JOIN pot p ON m.t_stamp = p.t_stamp
        WHERE m.Q_meas_kvar < 0
        ORDER BY m.t_stamp
    """, database=config.database)