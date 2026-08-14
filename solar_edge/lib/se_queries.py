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
    "voltvar_site_spread",
    "voltvar_site_response",
    "phase_cohort_comparison",
    "cohort_funnel",
]


def _cohort_cte(config) -> str:
    """The cohort as a CTE, used by every query that reports on a population."""
    return f"""
        cohort AS (
            SELECT i.* EXCLUDE (Q_kvar),
                   {contract.q_expr(config, 'i')} AS Q_kvar,
                   s.is_three_phase, s.state AS site_state,
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
    min_p_kw: float = 0.2, by_cohort: bool = True,
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
    cohort_col = ("CASE WHEN is_three_phase THEN 'three-phase' "
                  "ELSE 'single-phase' END AS cohort,") if by_cohort else ""
    group_by = "1, 2" if by_cohort else "1"
    return con.execute(
        f"""
        WITH {_cohort_cte(config)}
        SELECT {cohort_col}
               floor({v} / {bin_v}) * {bin_v}     AS v_bin,
               count(*)                            AS n_intervals,
               count(DISTINCT site_alias)          AS n_sites,
               round(median(Q_kvar), 4)            AS median_Q_kvar,
               round(avg(Q_kvar), 4)               AS mean_Q_kvar,
               -- The GAP between mean and median is the signal, not noise: it
               -- measures how much of the fleet's reactive power comes from a
               -- minority of strong responders. Where they agree the fleet is
               -- homogeneous; where the mean collapses away from a stable median,
               -- a few sites are doing all the work.
               round(avg(Q_kvar) - median(Q_kvar), 4) AS mean_minus_median,
               round(quantile_cont(Q_kvar, 0.05), 3) AS p05_Q_kvar,
               round(quantile_cont(Q_kvar, 0.95), 3) AS p95_Q_kvar,
               round(100.0 * count(*) FILTER (WHERE Q_kvar < 0) / count(*), 2)
                                                   AS pct_absorbing,
               round(avg(P_kW), 3)                 AS mean_P_kW
        FROM cohort
        WHERE {v} BETWEEN 200 AND 270 AND P_kW > {min_p_kw}
        GROUP BY {group_by} HAVING count(*) >= 100
        ORDER BY {group_by}
        """
    ).df()


def voltvar_site_spread(
    con: duckdb.DuckDBPyConnection, config=None, bin_v: float = 2.5,
    min_p_kw: float = 0.2, min_intervals_per_bin: int = 20,
) -> pd.DataFrame:
    """
    Spread of reactive response ACROSS SITES, per voltage bin.

    Two-stage aggregation, and the order matters:

    1. median Q per (site, voltage bin) -- one number per site
    2. quantiles of THOSE medians across sites

    ``voltvar_signature`` pools every interval together, so its median is
    interval-weighted: a site reporting 100,000 intervals counts a hundred times a
    site reporting 1,000, and the resulting number describes no actual inverter.
    Worse, it collapses a bimodal fleet -- most sites near zero, a minority
    responding strongly -- into a single point that sits between the two groups and
    represents neither.

    This gives each site equal weight and returns the distribution, so the two
    populations stay visible.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    return con.execute(
        f"""
        WITH {_cohort_cte(config)},
        per_site_bin AS (
            SELECT site_alias,
                   CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
                   floor({v} / {bin_v}) * {bin_v} AS v_bin,
                   median(Q_kvar)                 AS site_median_q,
                   count(*)                       AS n
            FROM cohort
            WHERE P_kW > {min_p_kw} AND {v} BETWEEN 200 AND 262
            GROUP BY 1, 2, 3
            HAVING count(*) >= {min_intervals_per_bin}
        )
        SELECT cohort, v_bin,
               count(*)                                       AS n_sites,
               round(quantile_cont(site_median_q, 0.10), 4)    AS p10,
               round(quantile_cont(site_median_q, 0.25), 4)    AS p25,
               round(quantile_cont(site_median_q, 0.50), 4)    AS p50,
               round(quantile_cont(site_median_q, 0.75), 4)    AS p75,
               round(quantile_cont(site_median_q, 0.90), 4)    AS p90,
               round(100.0 * count(*) FILTER (WHERE site_median_q < -0.05) / count(*), 1)
                                                               AS pct_sites_absorbing
        FROM per_site_bin
        GROUP BY 1, 2 HAVING count(*) >= 20
        ORDER BY 1, 2
        """
    ).df()


def voltvar_site_response(
    con: duckdb.DuckDBPyConnection, config=None,
    low_v: float = 240.0, high_v: float = 250.0,
    min_p_kw: float = 0.2, min_samples: int = 100,
) -> pd.DataFrame:
    """
    One number per site: how much its reactive power moves between the deadband
    and the upper Volt-VAr ramp.

    ``delta_q_kvar = median Q above high_v - median Q below low_v``.

    In the generator convention a conforming inverter absorbs more as voltage
    rises, so a NEGATIVE delta is the expected direction. Plotting the
    distribution of this single number is what shows the fleet is bimodal --
    something no pooled median can do.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "cohort")
    return con.execute(
        f"""
        WITH {_cohort_cte(config)}
        SELECT site_alias,
               CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               any_value(site_state)                                     AS state,
               any_value(s_99)                                           AS s_99,
               round(median(Q_kvar) FILTER (WHERE {v} < {low_v}), 4)      AS q_deadband,
               round(median(Q_kvar) FILTER (WHERE {v} > {high_v}), 4)     AS q_high,
               round(median(Q_kvar) FILTER (WHERE {v} > {high_v})
                     - median(Q_kvar) FILTER (WHERE {v} < {low_v}), 4)    AS delta_q_kvar,
               count(*) FILTER (WHERE {v} < {low_v})                      AS n_deadband,
               count(*) FILTER (WHERE {v} > {high_v})                     AS n_high
        FROM cohort
        WHERE P_kW > {min_p_kw} AND {v} BETWEEN 200 AND 262
        GROUP BY 1, 2
        HAVING count(*) FILTER (WHERE {v} < {low_v}) >= {min_samples}
           AND count(*) FILTER (WHERE {v} > {high_v}) >= {min_samples}
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

    Three normalisations are returned, and they are NOT interchangeable:

    * ``med_abs_Q_kvar`` -- the raw magnitude. Rising with voltage -> genuine
      Volt-VAr behaviour.
    * ``med_power_factor`` -- cos(phi) = P / S = P / sqrt(P^2 + Q^2). The correct
      definition of power factor, bounded in [0, 1]. Constant across voltage ->
      fixed-power-factor mode, no Volt-VAr response.
    * ``med_tan_phi`` -- Q / P. This is tan(phi), NOT power factor; an earlier
      version of this module mislabelled it. Kept for reference, but read it
      knowing the confound: because P grows faster than |Q| as voltage rises, the
      ratio falls even while the response strengthens.

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
               -- Q/P is tan(phi), NOT power factor. Power factor is cos(phi) =
               -- P / S = P / sqrt(P^2 + Q^2), computed as med_power_factor below.
               --
               -- tan(phi) is also CONFOUNDED here and is kept only so the confound
               -- is visible: site voltage rises when export rises, so P grows in
               -- the denominator faster than |Q| grows in the numerator. A falling
               -- ratio does NOT mean a weakening response. Read med_abs_Q_kvar
               -- against med_P_kW, or med_power_factor.
               round(median(Q_kvar / nullif(P_kW, 0)), 4) AS med_tan_phi,
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
