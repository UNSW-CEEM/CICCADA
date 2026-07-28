"""Reusable diagnostics for 01b_fleet_eda.ipynb."""

import pandas as pd

from shared.as4777_curves import (
    q_cap_absorbing_sql,
    q_conformance_floor_absorbing_sql,
    q_impact_nearest_edge_sql,
    tol_kw_sql,
    vvar_required_q_sql,
    vw_max_p_sql,
)

from shared.pipeline_options import (
    capacity_column,
    flex_predicate,
    voltage_aggregate_sql,
)

from shared.ciccada_config import AS4777, TABLES

def fetch_stored_day_verdict(
    site_id,
    date,
    aq_func,
    database,
    table=None,
):
    """Return the stored Volt-VAr result for one site and date."""
    date = pd.Timestamp(date)
    table = table or TABLES["conformance_voltvar"]

    return aq_func(
        f"""
        SELECT
            day_night,
            Q_adverse_count,
            Q_inactive_count,
            Q_significant_shortfall_count,
            Q_near_conformant_count,
            Q_major_surplus_count,
            nonconformance_voltvar_count,
            total_count,
            low_power_count,
            low_power_exposed_count,
            all_intervals_count
        FROM {table}
        WHERE site_id = {site_id}
          AND year = {date.year}
          AND month = {date.month}
          AND day = {date.day}
        ORDER BY day_night
        """,
        database=database,
    )


def recompute_vvar_day(site_id, date, aq_func, database):
    """Recompute interval-level Volt-VAr bands for one site and date."""
    date = pd.Timestamp(date)
    date_sql = date.strftime("%Y-%m-%d")

    q_required = vvar_required_q_sql("V", "ac_capacity_kw")
    q_cap = q_conformance_floor_absorbing_sql(
        "P_kW",
        "ac_capacity_kw",
    )
    tol = tol_kw_sql("ac_capacity_kw")

    q_impact_expr = q_impact_nearest_edge_sql(
        "Q_kvar",
        "Q_min_final",
        "Q_max_final",
        "capability_assessable",
    )

    qcap_p_min = AS4777["QCAP"]["P_MIN"]

    return aq_func(
        f"""
        WITH data AS (
            SELECT
                m.site_id,
                ts.t_stamp,
                sum(ts.power * m.circuit_polarity) / 1000 AS P_kW,
                sum(ts.energy_reactive * m.circuit_polarity) / 1000 * 12
                    AS Q_kvar,
                max(ts.voltage) AS V,
                max(m.ac_capacity_kw) AS ac_capacity_kw,
                max(m.s_99) AS S_99
            FROM ts
            JOIN (
                SELECT DISTINCT
                    circuit_id,
                    site_id,
                    circuit_polarity,
                    ac_capacity_kw,
                    s_99
                FROM meta_up23c
                WHERE is_pv = True
                  AND site_id = {site_id}
            ) m
              ON ts.circuit_id = m.circuit_id
            WHERE ts.year = {date.year}
              AND ts.month = {date.month}
              AND ts.is_pv = True
              AND ts.voltage > 0
              AND ts.voltage < 300
            GROUP BY m.site_id, ts.t_stamp
        ),
        required_q AS (
            SELECT
                *,
                {q_required} AS Q_voltvar,
                {q_cap} AS Q_cap_absorbing,

                CASE
                    WHEN abs(P_kW) >= {qcap_p_min} * ac_capacity_kw
                    THEN 1
                    ELSE 0
                END AS capability_assessable
            FROM data
        ),
        tol_band AS (
            SELECT
                *,
                -Q_cap_absorbing AS Q_cap_supplying,
                Q_voltvar + {tol} AS Q_voltvar_max,
                Q_voltvar - {tol} AS Q_voltvar_min
            FROM required_q
        ),
        clamped AS (
            SELECT
                *,
                CASE
                    WHEN Q_voltvar_max < 0
                    THEN greatest(
                        Q_voltvar_max,
                        Q_cap_absorbing + {tol}
                    )
                    ELSE Q_voltvar_max
                END AS Q_max_final,
                CASE
                    WHEN Q_voltvar_min > 0
                    THEN least(
                        Q_voltvar_min,
                        Q_cap_supplying - {tol}
                    )
                    ELSE Q_voltvar_min
                END AS Q_min_final
            FROM tol_band
        ),
        q_impact AS (
            SELECT
                *,
                {q_impact_expr} AS Q_impact
            FROM clamped
        )
        SELECT
            t_stamp AS t_stamp_utc,
            t_stamp + interval '10' hour AS t_stamp_aest,

            hour(t_stamp + interval '10' hour) AS hour_aest,

            CASE
                WHEN hour(t_stamp + interval '10' hour)
                     BETWEEN 6 AND 17
                THEN 'day'
                ELSE 'night'
            END AS day_night,

            round(V, 1) AS V,
            round(P_kW, 3) AS P_kW,

            round(
                abs(P_kW) / nullif(ac_capacity_kw, 0),
                4
            ) AS P_fraction_of_rated_proxy,

            round(Q_kvar, 3) AS Q_kvar,
            round(Q_voltvar, 3) AS Q_voltvar,
            round(Q_cap_absorbing, 3) AS Q_cap_absorbing,
            round(Q_max_final, 3) AS Q_max_final,
            round(Q_min_final, 3) AS Q_min_final,

            capability_assessable,

            CASE
                WHEN capability_assessable = 1
                 AND (
                     Q_kvar < Q_min_final
                     OR Q_kvar > Q_max_final
                 )
                THEN true
                ELSE false
            END AS outside_band,

            round(Q_impact, 3) AS Q_impact,

            CASE
                WHEN capability_assessable = 0
                    THEN 'not_assessable_low_power'

                WHEN Q_kvar >= Q_min_final
                 AND Q_kvar <= Q_max_final
                    THEN 'conformant'

                WHEN Q_impact < -0.1
                    THEN 'adverse'

                WHEN Q_impact >= -0.1
                 AND Q_impact <= 0.1
                    THEN 'inactive'

                WHEN Q_impact > 0.1
                 AND Q_impact < 0.9
                    THEN 'significant_shortfall'

                WHEN Q_impact >= 0.9
                 AND Q_impact <= 1.1
                    THEN 'near_conformant'

                ELSE 'major_surplus'
            END AS status

        FROM q_impact
        WHERE date(t_stamp + interval '10' hour) = date '{date_sql}'
        ORDER BY t_stamp
        """,
        database=database,
    )

def fetch_stored_voltwatt_day_verdict(
    site_id,
    date,
    aq_func,
    database,
    table="conformance_voltwatt_v2",
):
    """Return the stored day-level basic Volt-Watt result."""
    date = pd.Timestamp(date)

    return aq_func(
        f"""
        SELECT
            day_night,
            P_kW_sum,
            nonconformance_voltwatt_sum,
            nonconformance_voltwatt_count,
            total_count,
            all_intervals_count,
            rating_basis,
            voltage_aggregation,
            flex_selection
        FROM {table}
        WHERE site_id = {int(site_id)}
          AND year = {date.year}
          AND month = {date.month}
          AND day = {date.day}
        ORDER BY day_night
        """,
        database=database,
    )


def recompute_voltwatt_day(
    site_id,
    date,
    aq_func,
    database,
    rating_basis="ac_capacity_kw",
    voltage_aggregation="avg",
    flex_selection="exclude",
):
    """Recompute the basic Volt-Watt verdict for every recorded interval."""
    site_id = int(site_id)
    date = pd.Timestamp(date).normalize()

    if date.tzinfo is not None:
        date = date.tz_localize(None)

    # The requested AEST day expressed as a half-open UTC window.
    start_utc = date - pd.Timedelta(hours=10)
    end_utc = start_utc + pd.Timedelta(days=1)

    touched = {
        (start_utc.year, start_utc.month),
        ((end_utc - pd.Timedelta(seconds=1)).year,
         (end_utc - pd.Timedelta(seconds=1)).month),
    }
    partition_sql = " OR ".join(
        f"(ts.year = {year} AND ts.month = {month})"
        for year, month in sorted(touched)
    )

    voltage_sql = voltage_aggregate_sql(
        voltage_aggregation,
        "ts.voltage",
    )
    rating_col = capacity_column(rating_basis)
    flex_sql = flex_predicate(flex_selection)

    p_curve = vw_max_p_sql("V", rating_col)
    tolerance = tol_kw_sql(rating_col)
    vw_v1 = AS4777["VW"]["V1"]

    return aq_func(
        f"""
        WITH data AS (
            SELECT
                m.site_id,
                ts.t_stamp,
                sum(ts.power * m.circuit_polarity) / 1000 AS P_kW,
                {voltage_sql} AS V,
                max(m.ac_capacity_kw) AS ac_capacity_kw,
                max(m.s_99) AS S_99
            FROM ts
            JOIN (
                SELECT
                    circuit_id,
                    max(site_id) AS site_id,
                    max(circuit_polarity) AS circuit_polarity,
                    max(ac_capacity_kw) AS ac_capacity_kw,
                    max(s_99) AS s_99
                FROM meta_up23c
                WHERE is_pv = True
                  AND site_id = {site_id}
                  AND ac_capacity_kw > 0
                  AND s_99 > 0
                  AND {flex_sql}
                GROUP BY circuit_id
            ) m
              ON ts.circuit_id = m.circuit_id
            WHERE ({partition_sql})
              AND ts.t_stamp >= TIMESTAMP '{start_utc:%Y-%m-%d %H:%M:%S}'
              AND ts.t_stamp < TIMESTAMP '{end_utc:%Y-%m-%d %H:%M:%S}'
              AND ts.is_pv = True
              AND ts.voltage > 0
              AND ts.voltage < 300
            GROUP BY m.site_id, ts.t_stamp
        ),
        limits AS (
            SELECT
                *,
                ({p_curve}) AS P_limit_curve_kW,
                {tolerance} AS tolerance_kW,
                ({p_curve}) + {tolerance}
                    AS P_limit_with_tolerance_kW
            FROM data
        ),
        scored AS (
            SELECT
                *,
                CASE
                    WHEN V > {vw_v1}
                    THEN greatest(
                        0.0,
                        P_kW - P_limit_with_tolerance_kW
                    )
                    ELSE NULL
                END AS P_excess_kW
            FROM limits
        )
        SELECT
            t_stamp AS t_stamp_utc,
            t_stamp + interval '10' hour AS t_stamp_aest,

            CASE
                WHEN hour(t_stamp + interval '10' hour)
                     BETWEEN 6 AND 17
                THEN 'day'
                ELSE 'night'
            END AS day_night,

            round(V, 3) AS V,
            round(P_kW, 6) AS P_kW,
            round({rating_col}, 6) AS rating_kW,
            round(P_limit_curve_kW, 6) AS P_limit_curve_kW,
            round(tolerance_kW, 6) AS tolerance_kW,
            round(P_limit_with_tolerance_kW, 6)
                AS P_limit_with_tolerance_kW,
            round(
                P_limit_with_tolerance_kW - P_kW,
                6
            ) AS margin_to_limit_kW,
            round(P_excess_kW, 6) AS P_excess_kW,

            CASE
                WHEN V > {vw_v1} THEN true
                ELSE false
            END AS exposed,

            CASE
                WHEN V <= {vw_v1}
                    THEN 'not_exposed'
                WHEN P_kW IS NULL
                  OR P_limit_with_tolerance_kW IS NULL
                    THEN 'not_assessable_missing_data'
                WHEN P_excess_kW > 0
                    THEN 'nonconformant'
                ELSE 'conformant'
            END AS status

        FROM scored
        ORDER BY t_stamp
        """,
        database=database,
    )

def summarise_recomputed_day(intervals):
    """Summarise interval-level Volt-VAr assessment statuses."""
    result = intervals.copy()

    interval_summary = (
        result.groupby("day_night")
        .agg(
            intervals=("status", "size"),
            assessable=("capability_assessable", "sum"),
            outside_band=("outside_band", "sum"),
        )
        .reset_index()
    )

    interval_summary["not_assessable"] = (
        interval_summary["intervals"]
        - interval_summary["assessable"]
    )

    status_summary = (
        result.groupby(
            ["day_night", "status"],
            dropna=False,
        )
        .size()
        .rename("intervals")
        .reset_index()
    )

    return interval_summary, status_summary


def fetch_fleet_day_night_summary(
    aq_func,
    database,
    table=None,
):
    table = table or TABLES["conformance_voltvar"]

    return aq_func(
        f"""
        SELECT
            day_night,
            sum(total_count) AS total,
            sum(nonconformance_voltvar_count) AS nonconforming,
            sum(Q_inactive_count) AS inactive,
            sum(Q_significant_shortfall_count) AS shortfall,
            sum(Q_near_conformant_count) AS near_conformant
        FROM {table}
        GROUP BY day_night
        ORDER BY day_night
        """,
        database=database,
    )


def fetch_low_power_high_voltage(
    site_id,
    year,
    month,
    aq_func,
    database,
):
    """Inspect high-voltage, low-active-power Volt-VAr intervals."""
    q_required = vvar_required_q_sql("V", "ac_capacity_kw")
    q_cap = q_cap_absorbing_sql("P_kW", "S_99")
    tol = tol_kw_sql("ac_capacity_kw")
    qcap_p_min = AS4777["QCAP"]["P_MIN"]

    return aq_func(
        f"""
        WITH data AS (
            SELECT
                m.site_id,
                ts.t_stamp,
                sum(ts.power * m.circuit_polarity) / 1000 AS P_kW,
                sum(ts.energy_reactive * m.circuit_polarity) / 1000 * 12
                    AS Q_kvar,
                max(ts.voltage) AS V,
                max(m.ac_capacity_kw) AS ac_capacity_kw,
                max(m.s_99) AS S_99
            FROM ts
            JOIN (
                SELECT DISTINCT
                    circuit_id,
                    site_id,
                    circuit_polarity,
                    ac_capacity_kw,
                    s_99
                FROM meta_up23c
                WHERE is_pv = True
                  AND site_id = {site_id}
            ) m
              ON ts.circuit_id = m.circuit_id
            WHERE ts.year = {year}
              AND ts.month = {month}
              AND ts.is_pv = True
              AND ts.voltage > 0
              AND ts.voltage < 300
            GROUP BY m.site_id, ts.t_stamp
        ),
        required_q AS (
            SELECT
                *,
                {q_required} AS Q_voltvar,
                {q_cap} AS Q_cap_absorbing,

                CASE
                    WHEN abs(P_kW) >= {qcap_p_min} * ac_capacity_kw
                    THEN 1
                    ELSE 0
                END AS capability_assessable
            FROM data
        ),
        tol_band AS (
            SELECT
                *,
                -Q_cap_absorbing AS Q_cap_supplying,
                Q_voltvar + {tol} AS Q_voltvar_max,
                Q_voltvar - {tol} AS Q_voltvar_min
            FROM required_q
        ),
        clamped AS (
            SELECT
                *,
                CASE
                    WHEN Q_voltvar_max < 0
                    THEN greatest(
                        Q_voltvar_max,
                        Q_cap_absorbing + {tol}
                    )
                    ELSE Q_voltvar_max
                END AS Q_max_final,
                CASE
                    WHEN Q_voltvar_min > 0
                    THEN least(
                        Q_voltvar_min,
                        Q_cap_supplying - {tol}
                    )
                    ELSE Q_voltvar_min
                END AS Q_min_final
            FROM tol_band
        )
        SELECT
            hour(t_stamp + interval '10' hour) AS hour_aest,
            round(V, 1) AS V,
            round(P_kW, 3) AS P_kW,
            round(Q_kvar, 3) AS Q_kvar,
            round(Q_voltvar, 3) AS Q_voltvar,
            round(Q_cap_absorbing, 3) AS Q_cap_absorbing,
            round(Q_max_final, 3) AS Q_max_final,
            round(Q_min_final, 3) AS Q_min_final
        FROM clamped
        WHERE V > 240
          AND P_kW < 0.1
        ORDER BY t_stamp
        LIMIT 20
        """,
        database=database,
    )