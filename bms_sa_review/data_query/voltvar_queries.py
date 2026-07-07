"""
voltvar_queries.py — Athena SQL queries for Volt-VAr curtailment analysis.

All functions take explicit arguments for:
  - params:    the PARAMS dict (analysis-specific, changes per notebook run)
  - has_ghi:   bool — whether structured_data has ghi/ghi_cs columns
  - aq_func:   the query runner function (aq from aws_config)
  - database:  which Athena database to query (e.g. SAI)

Standard constants (AS4777, INTERVAL_H) are imported from ciccada_config.
"""

from ciccada_config import AS4777

# Helper: build phase-filter SQL fragments
def _build_phase_fragments(phase_filter):
    """
    Return (phase_cte, phase_join, phase_where) SQL fragments
    based on the PHASE_FILTER setting.
    """
    if phase_filter == "all":
        return "", "", ""

    op = "= 1" if phase_filter == "single" else "> 1"
    phase_cte = (
        ",\n        circuit_counts AS (\n"
        "            SELECT site_id, count(DISTINCT circuit_id) AS n_circuits\n"
        "            FROM meta_up23c WHERE is_pv = True\n"
        "            GROUP BY site_id\n"
        "        )"
    )
    phase_join  = "JOIN circuit_counts cc ON sm.site_id = cc.site_id"
    phase_where = f"AND cc.n_circuits {op}"
    return phase_cte, phase_join, phase_where


# ── Helper: build GHI clear-sky SQL fragments ────────────────
def _build_ghi_fragments(apply_ghi_filter, has_ghi, ghi_cs_ratio_min):
    """
    Return (ghi_join, ghi_filter) SQL fragments for the clear-sky filter.
    """
    if apply_ghi_filter and has_ghi:
        ghi_join = (
            "JOIN structured_data sd "
            "ON d.site_id = sd.site_id AND d.t_stamp = sd.t_stamp"
        )
        ghi_filter = (
            f"AND sd.ghi_cs > 0 "
            f"AND sd.ghi / sd.ghi_cs >= {ghi_cs_ratio_min}"
        )
        return ghi_join, ghi_filter
    return "", ""


# ═════════════════════════════════════════════════════════════
# Method A: Fleet-wide symptom scan
# ═════════════════════════════════════════════════════════════

def fetch_method_a(year, apply_ghi_filter, params, has_ghi, aq_func, database):
    """
    Run the Method A symptom scan for a single year.

    Parameters
    ----------
    year : int
        Calendar year to scan in ts.
    apply_ghi_filter : bool
        If True, join structured_data and require ghi/ghi_cs >= GHI_CS_RATIO_MIN.
        If False, no clear-sky filter (peak-hour-only proxy).
    params : dict
        The PARAMS dict from the notebook (V_LOW, V_HIGH, USE_S99, etc.).
    has_ghi : bool
        Whether structured_data has ghi/ghi_cs columns.
    aq_func : callable
        The Athena query function (aq from aws_config).
    database : str
        Athena database name (e.g. SAI).

    Returns
    -------
    pd.DataFrame
        One row per site: site_id, n_flagged_intervals, avg_V, avg_P_kW,
        avg_Q_kvar, avg_s_limit, est_curtailed_kWh.
    """
    P = params
    _limit_col = "s_99" if P["USE_S99"] else "ac_capacity_kw"

    phase_cte, phase_join, phase_where = _build_phase_fragments(P["PHASE_FILTER"])
    ghi_join, ghi_filter = _build_ghi_fragments(
        apply_ghi_filter, has_ghi, P["GHI_CS_RATIO_MIN"]
    )

    sql = f"""
        WITH site_meta AS (
            SELECT DISTINCT site_id, circuit_id, circuit_polarity,
                   ac_capacity_kw, {_limit_col} AS s_limit
            FROM meta_up23c
            WHERE is_pv = True
            AND {_limit_col} > 0
            AND ac_capacity_kw > 0
            AND ac_capacity_kw <= {P['MAX_AC_CAPACITY_KW']}
        ){phase_cte},
        d AS (
            SELECT
                t.circuit_id,
                t.t_stamp,
                sm.site_id,
                sm.s_limit,
                sm.ac_capacity_kw,
                t.voltage,
                t.power * sm.circuit_polarity / 1000.0                  AS P_kW,
                t.energy_reactive * sm.circuit_polarity / 1000.0 * 12   AS Q_kvar
            FROM ts t
            JOIN site_meta sm ON t.circuit_id = sm.circuit_id
            {phase_join}
            WHERE t.year = {year} AND t.is_pv = True
              AND t.voltage > {P['V_LOW']} AND t.voltage < {P['V_HIGH']}
              {phase_where}
        ),
        site_interval AS (
            SELECT
                d.site_id,
                d.t_stamp,
                max(d.s_limit)        AS s_limit,
                max(d.ac_capacity_kw) AS ac_capacity_kw,
                max(d.voltage)        AS V_max,
                sum(d.P_kW)           AS P_kW,
                sum(d.Q_kvar)         AS Q_kvar
            FROM d
            {ghi_join}
            WHERE hour(d.t_stamp + interval '10' hour)
                  BETWEEN {P['PEAK_HOUR_START']} AND {P['PEAK_HOUR_END']}
              {ghi_filter}
            GROUP BY d.site_id, d.t_stamp
        ),
        flagged AS (
            SELECT
                site_id, t_stamp, s_limit, ac_capacity_kw, V_max, P_kW, Q_kvar,
                sqrt(P_kW*P_kW + Q_kvar*Q_kvar) AS S_apparent,
                s_limit - sqrt(greatest(s_limit*s_limit - Q_kvar*Q_kvar, 0))
                    AS P_headroom_lost_kW
            FROM site_interval
            WHERE Q_kvar < 0
              AND sqrt(P_kW*P_kW + Q_kvar*Q_kvar)
                  >= s_limit - {P['S_TOL_FRAC']} * ac_capacity_kw
        )
        SELECT
            site_id,
            count(*)                              AS n_flagged_intervals,
            round(avg(V_max), 1)                  AS avg_V,
            round(avg(P_kW), 2)                   AS avg_P_kW,
            round(avg(Q_kvar), 2)                 AS avg_Q_kvar,
            round(avg(s_limit), 2)                AS avg_s_limit,
            round(
                sum(P_headroom_lost_kW) * {AS4777['INTERVAL_H']}, 2
            )                                     AS est_curtailed_kWh
        FROM flagged
        GROUP BY site_id
    """
    return aq_func(sql, database=database)