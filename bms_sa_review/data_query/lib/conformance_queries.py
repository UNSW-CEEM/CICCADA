"""Athena queries for Volt-Watt and Volt-VAr conformance and curtailment."""

import pandas as pd

from bms_sa_review.shared.as4777_curves import vw_max_p_sql
from bms_sa_review.shared.ciccada_config import AS4777

from analysis_contract import day_night_sql, flex_sql, years_sql


def _concat(frames):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_inventory(aq, config):
    frames = []
    for logical, table in config.tables.items():
        if logical not in {
            "structured_data", "all_uncurtailedpv", "conformance_voltvar",
            "conformance_voltwatt", "conformance_voltwattghi",
        }:
            continue
        frames.append(aq(f"""
            SELECT '{logical}' AS logical_name, '{table}' AS table_name,
                   count(*) AS n_rows, count(DISTINCT site_id) AS n_sites,
                   min(year) AS first_year, max(year) AS last_year
            FROM {table}
        """, database=config.database))
    return _concat(frames)


def fetch_stored_provenance(aq, config):
    specs = [
        (config.tables["structured_data"],
         "normalization_basis, voltage_aggregation, flex_selection"),
        (config.tables["all_uncurtailedpv"],
         "normalization_basis, counterfactual_cap_basis"),
        (config.tables["conformance_voltwatt"],
         "rating_basis, voltage_aggregation, flex_selection"),
        (config.tables["conformance_voltwattghi"],
         "rating_basis, voltage_aggregation, flex_selection"),
        (config.tables["conformance_voltvar"],
         "rating_basis, empirical_limit_basis, capability_profile, voltage_aggregation, flex_selection"),
    ]
    frames = []
    for table, cols in specs:
        group = ", ".join(c.strip() for c in cols.split(","))
        frames.append(aq(f"""
            SELECT '{table}' AS table_name, {group}, count(*) AS n_rows
            FROM {table}
            GROUP BY {group}
        """, database=config.database))
    return _concat(frames)


def fetch_metadata(aq, config):
    """One row per site plus conflict counters; never fan out a result merge."""
    return aq(f"""
        SELECT site_id,
               max(state) AS state,
               max(dnsp_name) AS dnsp,
               max(manufacturer) AS oem,
               max(year(pv_install_date)) AS install_year,
               max(ac_capacity_kw) AS ac_capacity_kw,
               max(s_99) AS s_99,
               bool_or(coalesce(flex_export_detected, False)) AS flex_export_detected,
               count(DISTINCT state) AS n_state_values,
               count(DISTINCT dnsp_name) AS n_dnsp_values,
               count(DISTINCT manufacturer) AS n_oem_values,
               count(DISTINCT ac_capacity_kw) AS n_ac_capacity_values,
               count(DISTINCT s_99) AS n_s99_values
        FROM {config.metadata_table}
        WHERE is_pv = True AND {flex_sql(config.flex_selection)}
        GROUP BY site_id
    """, database=config.database)


def fetch_population_funnel(aq, config):
    vw = config.tables["conformance_voltwatt"]
    ghi = config.tables["conformance_voltwattghi"]
    vv = config.tables["conformance_voltvar"]
    yw = years_sql(config.years)
    return aq(f"""
        WITH b AS (
            SELECT site_id,
                   sum(total_count) AS exposed,
                   sum(nonconformance_voltwatt_count) AS basic_nc
            FROM {vw} WHERE {yw} GROUP BY site_id
        ), g AS (
            SELECT site_id,
                   sum(total_count) AS exposed,
                   sum(assessable_count) AS response_supported,
                   sum(total_count - null_uncurtailed_P_count) AS counterfactual_present,
                   sum(null_uncurtailed_P_count) AS counterfactual_missing
            FROM {ghi} WHERE {yw} GROUP BY site_id
        ), v AS (
            SELECT site_id, sum(exposed_count) AS vv_exposed,
                   sum(total_count) AS vv_capability_assessable,
                   sum(curtailment_eligible_count) AS vv_symptom_intervals
            FROM {vv} WHERE {yw} GROUP BY site_id
        ), ids AS (
            SELECT site_id FROM b UNION SELECT site_id FROM g UNION SELECT site_id FROM v
        )
        SELECT
            count(*) AS sites_in_stage2,
            count_if(b.exposed > 0) AS vw_voltage_exposed_sites,
            count_if(g.counterfactual_present > 0) AS vw_counterfactual_present_sites,
            count_if(g.response_supported > 0) AS vw_response_supported_sites,
            count_if(g.counterfactual_missing > 0) AS vw_sites_with_missing_counterfactual,
            count_if(v.vv_exposed > 0) AS vv_voltage_exposed_sites,
            count_if(v.vv_capability_assessable > 0) AS vv_capability_assessable_sites,
            count_if(v.vv_symptom_intervals > 0) AS vv_apparent_limit_symptom_sites
        FROM ids
        LEFT JOIN b USING (site_id)
        LEFT JOIN g USING (site_id)
        LEFT JOIN v USING (site_id)
    """, database=config.database)


def fetch_vw_site_year(aq, config):
    table = config.tables["conformance_voltwatt"]
    return aq(f"""
        SELECT site_id, year,
               sum(nonconformance_voltwatt_count) AS nonconf_count,
               sum(nonconformance_voltwatt_sum) AS nonconf_kw_sum,
               sum(total_count) AS exposed_count,
               sum(all_intervals_count) AS all_intervals_count
        FROM {table}
        WHERE {years_sql(config.years)}
          AND {day_night_sql(config.day_night)}
        GROUP BY site_id, year
    """, database=config.database)


def fetch_vw_response_site_year(aq, config):
    table = config.tables["conformance_voltwattghi"]
    return aq(f"""
        SELECT site_id, year,
               sum(nonconformance_voltwattghi_count) AS nonconf_count,
               sum(nonconformance_voltwattghi_sum) AS nonconf_kw_sum,
               sum(curtailment_voltwattghi_count) AS curtailed_count,
               sum(curtailment_voltwattghi_sum) AS curtailed_kw_sum,
               sum(assessable_count) AS response_supported_count,
               sum(null_uncurtailed_P_count) AS missing_counterfactual_count,
               sum(total_count) AS exposed_count,
               sum(all_intervals_count) AS all_intervals_count
        FROM {table}
        WHERE {years_sql(config.years)}
          AND {day_night_sql(config.day_night)}
        GROUP BY site_id, year
    """, database=config.database)


def fetch_vvar_site_year(aq, config):
    table = config.tables["conformance_voltvar"]
    return aq(f"""
        SELECT site_id, year,
               sum(Q_adverse_count) AS adverse,
               sum(Q_inactive_count) AS inactive,
               sum(Q_significant_shortfall_count) AS shortfall,
               sum(Q_near_conformant_count) AS near_conformant,
               sum(Q_major_surplus_count) AS surplus,
               sum(nonconformance_voltvar_count) AS any_outside_band_count,
               sum(curtailment_voltvar_count) AS curtailed_count,
               sum(curtailment_voltvar_sum) AS curtailed_kw_sum,
               sum(curtailment_eligible_count) AS symptom_count,
               sum(null_uncurtailed_P_count) AS symptom_missing_counterfactual_count,
               sum(exposed_count) AS voltage_exposed_count,
               sum(low_power_exposed_count) AS low_power_exposed_count,
               sum(total_count) AS capability_assessable_count,
               sum(all_intervals_count) AS all_intervals_count
        FROM {table}
        WHERE {years_sql(config.years)}
          AND {day_night_sql(config.day_night)}
        GROUP BY site_id, year
    """, database=config.database)


def fetch_monthly_fleet(aq, config, mechanism):
    if mechanism == "vw":
        table = config.tables["conformance_voltwatt"]
        nc, den = "nonconformance_voltwatt_count", "total_count"
    elif mechanism == "vw_response":
        table = config.tables["conformance_voltwattghi"]
        nc, den = "nonconformance_voltwattghi_count", "assessable_count"
    elif mechanism == "vvar":
        table = config.tables["conformance_voltvar"]
        nc = "Q_adverse_count + Q_inactive_count + Q_significant_shortfall_count"
        den = "total_count"
    else:
        raise ValueError("mechanism must be vw, vw_response, or vvar")
    return aq(f"""
        SELECT year, month,
               count(DISTINCT CASE WHEN {den} > 0 THEN site_id END) AS n_sites,
               sum({nc}) AS nonconf_count,
               sum({den}) AS denominator_count
        FROM {table}
        WHERE {years_sql(config.years)}
          AND {day_night_sql(config.day_night)}
        GROUP BY year, month ORDER BY year, month
    """, database=config.database)


def fetch_vw_energy_site_year(aq, config):
    """Volt-Watt energy and population funnel on counterfactual-covered rows."""
    sd = config.tables["structured_data"]
    unc = config.tables["all_uncurtailedpv"]
    rating = config.rating_basis
    max_p = vw_max_p_sql("sd.V", f"sd.{rating}")
    tol = AS4777["TOL_FRAC"]
    return aq(f"""
        WITH joined AS (
            SELECT sd.site_id, year(sd.t_stamp + interval '10' hour) AS year,
                   sd.V,
                   sd.P_kw_norm * sd.normalization_capacity AS measured_p_kw,
                   u.uncurtailed_P,
                   ({max_p}) + {tol} * sd.{rating} AS permitted_p_kw
            FROM {sd} sd
            JOIN {unc} u ON sd.site_id = u.site_id AND sd.t_stamp = u.t_stamp
            WHERE {years_sql(config.years, "year(sd.t_stamp + interval '10' hour)")}
        ), scored AS (
            SELECT *,
                   CASE WHEN V > {AS4777['VW']['V1']}
                              AND uncurtailed_P > permitted_p_kw
                              AND measured_p_kw <= permitted_p_kw
                        THEN 1 ELSE 0 END AS curtailed,
                   CASE WHEN V > {AS4777['VW']['V1']}
                              AND uncurtailed_P > permitted_p_kw
                              AND measured_p_kw <= permitted_p_kw
                        THEN greatest(uncurtailed_P - measured_p_kw, 0)
                        ELSE 0 END AS lost_p_kw
            FROM joined
        )
        SELECT site_id, year,
               count(*) AS counterfactual_covered_count,
               count_if(V > {AS4777['VW']['V1']}) AS counterfactual_exposed_count,
               count_if(V > {AS4777['VW']['V1']} AND uncurtailed_P > permitted_p_kw)
                   AS response_opportunity_count,
               sum(curtailed) AS curtailed_count,
               sum(lost_p_kw) AS curtailed_kw_sum,
               sum(measured_p_kw) AS measured_kw_sum,
               sum(uncurtailed_P) AS potential_kw_sum,
               sum(CASE WHEN curtailed=1 THEN measured_p_kw ELSE 0 END)
                   AS curtailed_interval_measured_kw_sum,
               sum(CASE WHEN curtailed=1 THEN uncurtailed_P ELSE 0 END)
                   AS curtailed_interval_potential_kw_sum
        FROM scored GROUP BY site_id, year
    """, database=config.database)

def fetch_vw_legacy_energy_site_year(aq, config):
    """
    Reproduce the legacy Volt-Watt energy denominator.

    Numerator:
        Identified Volt-Watt curtailed energy.

    Denominator:
        Measured generation over the GHI table's voltage-exposed population
        plus identified Volt-Watt loss.

    This matches the structure of the colleague's reported percentage.
    """
    table = config.tables["conformance_voltwattghi"]

    return aq(f"""
        SELECT
            site_id,
            year,

            sum(P_kW_sum)
                AS exposed_measured_kw_sum,

            sum(coalesce(curtailment_voltwattghi_sum, 0))
                AS curtailed_kw_sum,

            sum(coalesce(curtailment_voltwattghi_count, 0))
                AS curtailed_count,

            sum(total_count)
                AS exposed_count

        FROM {table}

        WHERE {years_sql(config.years)}
          AND {day_night_sql(config.day_night)}

        GROUP BY site_id, year
    """, database=config.database)

def fetch_legacy_population_by_year(aq, config):
    basic = config.tables["conformance_voltwatt"]
    ghi = config.tables["conformance_voltwattghi"]
    return aq(f"""
        SELECT b.year,
               count(DISTINCT CASE WHEN b.exposed > 0 THEN b.site_id END) AS voltage_exposed_sites,
               count(DISTINCT CASE WHEN g.assessable > 0 THEN g.site_id END) AS review_corrected_sites,
               count(DISTINCT CASE WHEN g.counterfactual_present > 0 THEN g.site_id END) AS legacy_like_sites
        FROM (
            SELECT site_id, year, sum(total_count) AS exposed
            FROM {basic} WHERE {years_sql(config.years)} GROUP BY site_id, year
        ) b
        LEFT JOIN (
            SELECT site_id, year, sum(assessable_count) AS assessable,
                   sum(total_count - null_uncurtailed_P_count) AS counterfactual_present
            FROM {ghi} WHERE {years_sql(config.years)} GROUP BY site_id, year
        ) g ON b.site_id = g.site_id AND b.year = g.year
        GROUP BY b.year ORDER BY b.year
    """, database=config.database)


def fetch_original_v2_membership(aq, config,
                                 original_basic="conformance_voltwatt",
                                 original_ghi="conformance_voltwattghi"):
    new_basic = config.tables["conformance_voltwatt"]
    new_ghi = config.tables["conformance_voltwattghi"]
    return aq(f"""
        WITH cohorts AS (
            SELECT site_id, 'original_basic' AS cohort FROM {original_basic}
            UNION ALL SELECT site_id, 'original_ghi' FROM {original_ghi}
            UNION ALL SELECT site_id, 'v2_basic' FROM {new_basic}
            UNION ALL SELECT site_id, 'v2_ghi_response_supported' FROM {new_ghi}
                GROUP BY site_id HAVING sum(assessable_count) > 0
            UNION ALL SELECT site_id, 'v2_ghi_counterfactual_present' FROM {new_ghi}
                GROUP BY site_id HAVING sum(total_count-null_uncurtailed_P_count) > 0
        )
        SELECT cohort, count(DISTINCT site_id) AS n_sites
        FROM cohorts GROUP BY cohort ORDER BY cohort
    """, database=config.database)


def fetch_legacy_vw_site(aq, config, legacy_table="conformance_voltwattghi"):
    """
    Fetch per-site totals from the LEGACY ``conformance_voltwattghi``
    table for reconciliation against the prior ARENA report.

    Column semantics in the legacy table:
      total_count = count(nonconformance_voltwattghi)  — NON-NULL scored
                    intervals, including NULL-counterfactual rows scored
                    via the ``uncurtailed_P IS NULL`` path.
      nonconformance_voltwattghi_count = sum(scored > 0)
      nonconformance_voltwattghi_sum   = sum(scored value) in kW
    """
    return aq(f"""
        SELECT site_id,
               SUM(nonconformance_voltwattghi_count) AS nc_count,
               SUM(nonconformance_voltwattghi_sum)   AS nc_kw_sum,
               SUM(total_count)                      AS total_count
        FROM {legacy_table}
        WHERE {years_sql(config.years)}
        GROUP BY site_id
    """, database=config.database)
