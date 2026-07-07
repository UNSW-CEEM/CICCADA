"""
Athena SQL queries for Volt-VAr curtailment analysis.

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


# Helper: build GHI clear-sky SQL fragments
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

# ═════════════════════════════════════════════════════════════
# Method B: Counterfactual quantification (single site, single year)
# ═════════════════════════════════════════════════════════════
 
def fetch_method_b_site_year(year, site_id, params, aq_func, database):
    """
    Run Method B counterfactual query for one site and one year.
 
    Returns interval-level rows with P_meas, Q_meas, P_potential,
    P_max_given_Q, and varcurt_kW.
 
    Parameters
    ----------
    year : int
    site_id : int
    params : dict
    aq_func : callable
    database : str
 
    Returns
    -------
    pd.DataFrame
        One row per qualifying 5-min interval.
    """
    P = params
    _limit_col = "s_99" if P["USE_S99"] else "ac_capacity_kw"
 
    sql = f"""
        WITH site_meta AS (
            SELECT DISTINCT site_id, circuit_id, circuit_polarity,
                   ac_capacity_kw, {_limit_col} AS s_limit
            FROM meta_up23c
            WHERE is_pv = True
            AND site_id = {site_id}
            AND ac_capacity_kw > 0
            AND ac_capacity_kw <= {P['MAX_AC_CAPACITY_KW']}
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
            WHERE t.year = {year} AND t.is_pv = True
              AND t.voltage > {P['V_LOW']} AND t.voltage < {P['V_HIGH']}
              AND hour(t.t_stamp + interval '10' hour)
                  BETWEEN {P['PEAK_HOUR_START']} AND {P['PEAK_HOUR_END']}
            GROUP BY sm.site_id, t.t_stamp
        ),
        pot AS (
            SELECT site_id, t_stamp, uncurtailed_P AS P_potential_kW
            FROM all_uncurtailedpv
            WHERE site_id = {site_id}
        )
        SELECT
            m.t_stamp, m.V_max, m.s_limit, m.ac_capacity_kw,
            m.P_meas_kW, m.Q_meas_kvar,
            p.P_potential_kW,
            sqrt(greatest(
                m.s_limit*m.s_limit - m.Q_meas_kvar*m.Q_meas_kvar, 0
            )) AS P_max_given_Q,
            greatest(0,
                p.P_potential_kW
                - sqrt(greatest(
                    m.s_limit*m.s_limit - m.Q_meas_kvar*m.Q_meas_kvar, 0
                ))
            ) AS varcurt_kW
        FROM meas m
        JOIN pot p ON m.t_stamp = p.t_stamp
        WHERE m.Q_meas_kvar < 0
        ORDER BY m.t_stamp
    """
    return aq_func(sql, database=database)
 
# ═════════════════════════════════════════════════════════════
# Day-level data fetch (for single-day plots)
# ═════════════════════════════════════════════════════════════
 
def fetch_day_data(site_id, date_str, aq_func, database):
    """
    Fetch one full day of telemetry + counterfactual for a single site.
 
    Parameters
    ----------
    site_id : int
    date_str : str
        Format 'YYYY-MM-DD'.
    aq_func : callable
    database : str
 
    Returns
    -------
    pd.DataFrame
        Columns: t_stamp, V, P_kW, Q_kvar, P_potential_kW.
        Empty DataFrame if no data found.
    """
    year  = int(date_str.split("-")[0])
    month = int(date_str.split("-")[1])
 
    sql = f"""
        WITH site_meta AS (
            SELECT DISTINCT circuit_id, circuit_polarity
            FROM meta_up23c WHERE is_pv = True AND site_id = {site_id}
        ),
        meas AS (
            SELECT
                t.t_stamp,
                max(t.voltage)                                           AS V,
                sum(t.power * sm.circuit_polarity / 1000.0)              AS P_kW,
                sum(t.energy_reactive * sm.circuit_polarity / 1000.0*12) AS Q_kvar
            FROM ts t
            JOIN site_meta sm ON t.circuit_id = sm.circuit_id
            WHERE t.year = {year} AND t.month = {month} AND t.is_pv = True
              AND date(t.t_stamp + interval '10' hour) = date '{date_str}'
            GROUP BY t.t_stamp
        )
        SELECT m.t_stamp, m.V, m.P_kW, m.Q_kvar,
               u.uncurtailed_P AS P_potential_kW
        FROM meas m
        LEFT JOIN all_uncurtailedpv u
          ON u.site_id = {site_id} AND u.t_stamp = m.t_stamp
        ORDER BY m.t_stamp
    """
    return aq_func(sql, database=database)

# ═════════════════════════════════════════════════════════════
# Fleet denominators: eligible and all-timestamp context
# ═════════════════════════════════════════════════════════════
 
def fetch_eligible_context_for_year(year, apply_ghi_filter, params, has_ghi,
                                    aq_func, database):
    """
    Eligible denominator for one year.
 
    Same eligibility conditions as Method A (voltage band, peak-hour,
    optional GHI clear-sky) but WITHOUT Q < 0 or apparent-power-at-limit
    filters.  Returns one row per site with interval count and potential
    generation kWh.
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
            SELECT t.circuit_id, t.t_stamp, sm.site_id, t.voltage
            FROM ts t
            JOIN site_meta sm ON t.circuit_id = sm.circuit_id
            {phase_join}
            WHERE t.year = {year}
              AND t.is_pv = True
              AND t.voltage > {P['V_LOW']}
              AND t.voltage < {P['V_HIGH']}
              {phase_where}
        ),
        site_interval AS (
            SELECT d.site_id, d.t_stamp, max(d.voltage) AS V_max
            FROM d
            {ghi_join}
            WHERE hour(d.t_stamp + interval '10' hour)
                  BETWEEN {P['PEAK_HOUR_START']} AND {P['PEAK_HOUR_END']}
              {ghi_filter}
            GROUP BY d.site_id, d.t_stamp
        )
        SELECT
            {year} AS year,
            si.site_id,
            count(*) AS n_eligible_intervals,
            round(
                sum(coalesce(u.uncurtailed_P, 0)) * {AS4777['INTERVAL_H']}, 3
            ) AS eligible_potential_kWh
        FROM site_interval si
        LEFT JOIN all_uncurtailedpv u
          ON u.site_id = si.site_id AND u.t_stamp = si.t_stamp
        GROUP BY si.site_id
    """
    return aq_func(sql, database=database)
 
 
def fetch_all_timestamp_context_for_year(year, params, aq_func, database):
    """
    All-timestamp denominator for one year.
 
    Same site/phase/limit filters as Method A but WITHOUT voltage band,
    peak-hour, GHI, Q, or apparent-power filters.  Represents all observed
    PV site-timestamps for the comparable fleet in a given year.
    """
    P = params
    _limit_col = "s_99" if P["USE_S99"] else "ac_capacity_kw"
 
    phase_cte, phase_join, phase_where = _build_phase_fragments(P["PHASE_FILTER"])
 
    sql = f"""
        WITH site_meta AS (
            SELECT DISTINCT site_id, circuit_id, circuit_polarity,
                   ac_capacity_kw, {_limit_col} AS s_limit
            FROM meta_up23c
            WHERE is_pv = True
              AND ac_capacity_kw > 0
              AND ac_capacity_kw <= {P['MAX_AC_CAPACITY_KW']}
              AND {_limit_col} > 0
        ){phase_cte},
        d AS (
            SELECT t.circuit_id, t.t_stamp, sm.site_id
            FROM ts t
            JOIN site_meta sm ON t.circuit_id = sm.circuit_id
            {phase_join}
            WHERE t.year = {year}
              AND t.is_pv = True
              {phase_where}
        ),
        site_interval AS (
            SELECT d.site_id, d.t_stamp
            FROM d
            GROUP BY d.site_id, d.t_stamp
        )
        SELECT
            {year} AS year,
            si.site_id,
            count(*) AS n_all_intervals,
            round(
                sum(coalesce(u.uncurtailed_P, 0)) * {AS4777['INTERVAL_H']}, 3
            ) AS all_potential_kWh
        FROM site_interval si
        LEFT JOIN all_uncurtailedpv u
          ON u.site_id = si.site_id AND u.t_stamp = si.t_stamp
        GROUP BY si.site_id
    """
    return aq_func(sql, database=database)
 