"""
Fleet exploratory queries.
==========================

Deliverable D6. SQL over the store; returns tidy frames and nothing else. No
plotting, no interpretation — those live in ``se_plots`` and the notebook.

Every query that defines a cohort takes an ``SEAnalysisConfig`` and builds its
predicates through ``se_contract``, so no notebook can quietly analyse a
different population from the one its manifest claims.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from solar_edge.config import se_config as C
from solar_edge.lib import se_contract as contract
from solar_edge.lib import se_params

__all__ = [
    "fleet_composition",
    "capacity_distribution",
    "voltage_distribution",
    "voltage_band_occupancy",
    "monthly_coverage",
    "diurnal_profile",
    "derating_by_voltage",
    "derating_by_cohort",
    "voltvar_signature",
    "phase_cohort_comparison",
    "cohort_funnel",
]


def _cohort_cte(config) -> str:
    """The cohort as a CTE, used by every query that reports on a population."""
    return f"""
        cohort AS (
            SELECT i.*, s.is_three_phase, s.state AS site_state,
                   c.s_99, c.p_99
            FROM se_interval i
            {contract.cohort_join_sql('i')}
            WHERE {contract.cohort_where_sql(config)}
        )
    """


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION AND COVERAGE
# ═══════════════════════════════════════════════════════════════════════════

def fleet_composition(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Sites by state and phase configuration. Reads the site dimension only."""
    return con.execute(
        """
        SELECT state,
               count(*)                                     AS n_sites,
               count(*) FILTER (WHERE NOT is_three_phase)   AS single_phase,
               count(*) FILTER (WHERE is_three_phase)       AS three_phase,
               round(100.0 * count(*) FILTER (WHERE is_three_phase) / count(*), 1)
                                                            AS pct_three_phase,
               count(DISTINCT postcode)                     AS n_postcodes,
               count(*) FILTER (WHERE has_night_generation_anomaly)
                                                            AS n_night_anomaly
        FROM se_site
        GROUP BY state
        ORDER BY n_sites DESC
        """
    ).df()


def capacity_distribution(con: duckdb.DuckDBPyConnection, bin_kva: float = 1.0) -> pd.DataFrame:
    """Histogram of the s_99 empirical apparent-power limit."""
    return con.execute(
        f"""
        SELECT floor(s_99 / {bin_kva}) * {bin_kva} AS s_99_bin,
               count(*)                            AS n_sites
        FROM se_site_capacity
        WHERE s_99 > 0
        GROUP BY 1 ORDER BY 1
        """
    ).df()


def monthly_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Rows and reporting sites per AEST month — the fleet's growth and continuity."""
    return con.execute(
        """
        SELECT dt_month,
               count(*)                    AS n_rows,
               count(DISTINCT site_alias)  AS n_sites,
               round(avg(P_kW), 3)         AS mean_P_kW
        FROM se_interval
        GROUP BY dt_month ORDER BY dt_month
        """
    ).df()


def diurnal_profile(con: duckdb.DuckDBPyConnection, by_state: bool = False) -> pd.DataFrame:
    """Mean power and voltage by hour in the AEST analysis frame."""
    group = "state, hour_aest" if by_state else "hour_aest"
    select = "state," if by_state else ""
    return con.execute(
        f"""
        SELECT {select}
               hour(ts_aest)          AS hour_aest,
               count(*)               AS n_rows,
               round(avg(P_kW), 3)    AS mean_P_kW,
               round(avg(Q_kvar), 4)  AS mean_Q_kvar,
               round(avg(V_max), 2)   AS mean_V
        FROM se_interval
        GROUP BY {group} ORDER BY {group}
        """
    ).df()


# ═══════════════════════════════════════════════════════════════════════════
# VOLTAGE
# ═══════════════════════════════════════════════════════════════════════════

def voltage_distribution(
    con: duckdb.DuckDBPyConnection, config=None, bin_v: float = 1.0
) -> pd.DataFrame:
    """Histogram of site voltage across the cohort."""
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    return con.execute(
        f"""
        WITH {_cohort_cte(config)}
        SELECT floor({v} / {bin_v}) * {bin_v} AS v_bin,
               count(*)                       AS n_intervals,
               round(avg(P_kW), 3)            AS mean_P_kW,
               round(avg(Q_kvar), 4)          AS mean_Q_kvar
        FROM cohort
        WHERE {v} BETWEEN 180 AND 280
        GROUP BY 1 ORDER BY 1
        """
    ).df()


def voltage_band_occupancy(con: duckdb.DuckDBPyConnection, config=None) -> pd.DataFrame:
    """
    How many intervals sit in each AS/NZS 4777.2 response band.

    This is the population sizing that decides what the study can say. The
    Volt-VAr band is where the curtailment analysis lives; the Volt-Watt bands
    above 253 V are thin in this fleet, and that has to be stated up front rather
    than discovered when a confidence interval turns out meaningless.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    a = C.as4777()
    bands = [
        ("below V1 (Volt-VAr supply region)", f"{v} < {a['VVAR']['V1']}"),
        (f"V1-V2 supply ramp ({a['VVAR']['V1']:.0f}-{a['VVAR']['V2']:.0f} V)",
         f"{v} >= {a['VVAR']['V1']} AND {v} < {a['VVAR']['V2']}"),
        (f"V2-V3 deadband ({a['VVAR']['V2']:.0f}-{a['VVAR']['V3']:.0f} V)",
         f"{v} >= {a['VVAR']['V2']} AND {v} < {a['VVAR']['V3']}"),
        (f"V3-VW1 Volt-VAr absorb only ({a['VVAR']['V3']:.0f}-{a['VW']['V1']:.0f} V)",
         f"{v} >= {a['VVAR']['V3']} AND {v} < {a['VW']['V1']}"),
        (f"VW1-VVAR4 overlap ({a['VW']['V1']:.0f}-{a['VVAR']['V4']:.0f} V)",
         f"{v} >= {a['VW']['V1']} AND {v} < {a['VVAR']['V4']}"),
        (f"above VVAR4 ({a['VVAR']['V4']:.0f} V+)", f"{v} >= {a['VVAR']['V4']}"),
    ]
    unions = "\n            UNION ALL\n".join(
        f"""            SELECT '{label}' AS band, {i} AS band_order,
                   count(*) AS n_intervals,
                   count(DISTINCT site_alias) AS n_sites
            FROM cohort WHERE {predicate}"""
        for i, (label, predicate) in enumerate(bands)
    )
    frame = con.execute(
        f"WITH {_cohort_cte(config)}\n{unions}\nORDER BY band_order"
    ).df()
    total = frame.n_intervals.sum()
    frame["pct_of_intervals"] = (100 * frame.n_intervals / total).round(3)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# DERATING FLAG
# ═══════════════════════════════════════════════════════════════════════════

def derating_by_voltage(
    con: duckdb.DuckDBPyConnection, config=None, bin_v: float = 2.5
) -> pd.DataFrame:
    """
    Derating rate against voltage — the characterisation that decides whether the
    flag can support Method C at D14.

    A flag that is purely voltage-driven corroborates the Volt-Watt / Volt-VAr
    story. A flat baseline at ordinary voltages is something else (thermal, DC
    clipping, export limiting) and must not be attributed to grid response.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    return con.execute(
        f"""
        WITH {_cohort_cte(config)}
        SELECT floor({v} / {bin_v}) * {bin_v}                    AS v_bin,
               count(*)                                          AS n_intervals,
               count(*) FILTER (WHERE derating_active)            AS n_derating,
               round(100.0 * count(*) FILTER (WHERE derating_active) / count(*), 3)
                                                                 AS pct_derating,
               round(avg(P_kW), 3)                               AS mean_P_kW,
               round(avg(P_kW) FILTER (WHERE derating_active), 3) AS mean_P_derating
        FROM cohort
        WHERE {v} BETWEEN 200 AND 275
        GROUP BY 1 HAVING count(*) >= 100
        ORDER BY 1
        """
    ).df()


def derating_by_cohort(con: duckdb.DuckDBPyConnection, config=None) -> pd.DataFrame:
    """Derating rate split by phase configuration and generation level."""
    config = (config or se_params.CONFIG).validate()
    return con.execute(
        f"""
        WITH {_cohort_cte(config)}
        SELECT CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               CASE WHEN P_kW <= 0.1 THEN 'not generating'
                    WHEN P_kW < 0.5 * p_99 THEN 'below 50% of p_99'
                    ELSE 'at or above 50% of p_99' END       AS generation_level,
               count(*)                                       AS n_intervals,
               count(*) FILTER (WHERE derating_active)        AS n_derating,
               round(100.0 * count(*) FILTER (WHERE derating_active) / count(*), 3)
                                                              AS pct_derating
        FROM cohort
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()


# ═══════════════════════════════════════════════════════════════════════════
# VOLT-VAR SIGNATURE
# ═══════════════════════════════════════════════════════════════════════════

def voltvar_signature(
    con: duckdb.DuckDBPyConnection, config=None, bin_v: float = 2.5,
    min_p_kw: float = 0.2,
) -> pd.DataFrame:
    """
    Median reactive power against voltage — the shape that should reproduce the
    AS/NZS 4777.2 Australia A Volt-VAr curve.

    In the CICCADA generator convention, a conforming fleet supplies reactive
    power (Q > 0) below 207 V and absorbs it (Q < 0) above 240 V, so the median
    should slope DOWNWARD with rising voltage. This is the plot that confirms the
    D2 sign flip was right, and it is worth reading before trusting anything in
    D9 or D13.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    return con.execute(
        f"""
        WITH {_cohort_cte(config)}
        SELECT floor({v} / {bin_v}) * {bin_v}     AS v_bin,
               count(*)                            AS n_intervals,
               count(DISTINCT site_alias)          AS n_sites,
               round(median(Q_kvar), 4)            AS median_Q_kvar,
               round(avg(Q_kvar), 4)               AS mean_Q_kvar,
               round(100.0 * count(*) FILTER (WHERE Q_kvar < 0) / count(*), 2)
                                                   AS pct_absorbing,
               round(avg(P_kW), 3)                 AS mean_P_kW
        FROM cohort
        WHERE {v} BETWEEN 200 AND 270 AND P_kW > {min_p_kw}
        GROUP BY 1 HAVING count(*) >= 100
        ORDER BY 1
        """
    ).df()


def phase_cohort_comparison(
    con: duckdb.DuckDBPyConnection, config=None, min_p_kw: float = 0.2
) -> pd.DataFrame:
    """
    Volt-VAr response of single- vs three-phase sites, side by side.

    Raised in the D2 sign-convention work: the single-phase cohort shows a clear
    Volt-VAr slope while the three-phase cohort appeared flat. Either three-phase
    installations have the function disabled, or their per-phase reactive
    reporting differs. Resolve this before the two are pooled in D9 — a flat
    cohort dragged into a fleet median would understate the fleet response.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    return con.execute(
        f"""
        WITH {_cohort_cte(config)}
        SELECT CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               CASE WHEN {v} < 235 THEN 'low (<235 V)'
                    WHEN {v} < 250 THEN 'mid (235-250 V)'
                    ELSE 'high (>250 V)' END        AS v_band,
               count(*)                              AS n_intervals,
               count(DISTINCT site_alias)            AS n_sites,
               round(median(Q_kvar), 4)              AS median_Q_kvar,
               round(100.0 * count(*) FILTER (WHERE Q_kvar < 0) / count(*), 2)
                                                     AS pct_absorbing
        FROM cohort
        WHERE {v} BETWEEN 200 AND 270 AND P_kW > {min_p_kw}
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()


def reactive_character(
    con: duckdb.DuckDBPyConnection, config=None, min_p_kw: float = 0.5,
    bin_v: float = 5.0,
) -> pd.DataFrame:
    """
    Is the reactive power a Volt-VAr response, or a fixed power factor?

    The distinction decides whether the fleet can be assessed against the
    AS/NZS 4777.2 Volt-VAr curve at all, and the two are easy to confuse: site
    voltage rises when the site exports more, so ANY reactive power proportional
    to P will trace out an apparent slope against voltage. That confound has to be
    ruled out before a Q-vs-V plot means anything.

    Read the two columns together:

    * ``med_Q_over_P`` constant across voltage  -> fixed power factor, no response
    * ``med_abs_Q`` rising with voltage         -> genuine Volt-VAr behaviour

    Observed on this fleet: median power factor is 0.995-0.997 in BOTH cohorts, so
    reactive power is small everywhere. The fleet-median Volt-VAr response is weak;
    the strong responders identified during the sign-convention work are a minority.
    That is a finding about the fleet, not a defect in the analysis, but it means
    conformance rates must be read against how little most inverters are doing.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    return con.execute(
        f"""
        WITH {_cohort_cte(config)}
        SELECT CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               floor({v} / {bin_v}) * {bin_v}          AS v_bin,
               count(*)                                 AS n_intervals,
               round(median(P_kW), 3)                   AS med_P_kW,
               round(median(Q_kvar), 4)                 AS med_Q_kvar,
               round(median(abs(Q_kvar)), 4)            AS med_abs_Q_kvar,
               round(median(Q_kvar / nullif(P_kW, 0)), 4) AS med_Q_over_P,
               round(median(P_kW / nullif(sqrt(P_kW * P_kW + Q_kvar * Q_kvar), 0)), 4)
                                                        AS med_power_factor
        FROM cohort
        WHERE P_kW > {min_p_kw} AND {v} BETWEEN 220 AND 262
        GROUP BY 1, 2 HAVING count(*) >= 1000
        ORDER BY 1, 2
        """
    ).df()


# ═══════════════════════════════════════════════════════════════════════════
# COHORT FUNNEL
# ═══════════════════════════════════════════════════════════════════════════

def cohort_funnel(con: duckdb.DuckDBPyConnection, config=None, params=None) -> pd.DataFrame:
    """
    Attrition from the whole store down to the Volt-VAr detection window.

    The same discipline as ``fetch_population_funnel`` in the Solar Analytics
    work: show the reader how few intervals survive, and why, rather than
    presenting a final count with no denominator.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    v_all = contract.voltage_sql(config.voltage_aggregation, "i")

    stages = [
        ("1. all stored intervals",
         f"SELECT count(*) n, count(DISTINCT i.site_alias) s FROM se_interval i"),
        ("2. cohort filters applied",
         f"WITH {_cohort_cte(config)} SELECT count(*) n, count(DISTINCT site_alias) s FROM cohort"),
        ("3. + peak-solar window",
         f"WITH {_cohort_cte(config)} SELECT count(*) n, count(DISTINCT site_alias) s "
         f"FROM cohort WHERE {contract.peak_hours_sql(params, 'ts_aest')}"),
        ("4. + Volt-VAr voltage band",
         f"WITH {_cohort_cte(config)} SELECT count(*) n, count(DISTINCT site_alias) s "
         f"FROM cohort WHERE {contract.peak_hours_sql(params, 'ts_aest')} "
         f"AND {contract.v_band_sql(params, v)}"),
        ("5. + absorbing reactive power (Q < 0)",
         f"WITH {_cohort_cte(config)} SELECT count(*) n, count(DISTINCT site_alias) s "
         f"FROM cohort WHERE {contract.peak_hours_sql(params, 'ts_aest')} "
         f"AND {contract.v_band_sql(params, v)} AND Q_kvar < 0"),
    ]
    rows = []
    for label, sql in stages:
        got = con.execute(sql).fetchone()
        rows.append({"stage": label, "n_intervals": int(got[0]), "n_sites": int(got[1])})

    frame = pd.DataFrame(rows)
    frame["pct_of_all"] = (100 * frame.n_intervals / frame.n_intervals.iloc[0]).round(4)
    return frame
