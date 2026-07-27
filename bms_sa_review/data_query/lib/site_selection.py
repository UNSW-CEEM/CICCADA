"""
Pick interesting sites and pull their telemetry
=====================================================================

The site-selection workflow from `01_data_exploration.ipynb` (cells 49-52),
lifted out so the notebook is an orchestrator.

    rank_sites(...)        Rank sites by non-conformance for a chosen mechanism.
                           -> ranked DataFrame you eyeball to pick a site.

    pull_site_telemetry(.) Given a site_id, fetch its metadata (incl. nameplate,
                           so you never type it), auto-resolve the best year/
                           month with actual telemetry, pull the circuit series,
                           and add t_stamp_aest / P_kW / Q_kvar. Returns
                           (df, site_info).

    suggest_days(...)      Rank days by how many intervals breached the
                           mechanism's voltage threshold -> which day to plot.

    voltage_diagnostic(.)  Compare what the conformance table saw (max V across
                           circuits) against raw circuit voltage. [was cell 52]

NOTE: WIRED TO THE _v2 SCHEMAS
------------------------
The per-mechanism ranking column and table are resolved in MODE_MAP below.
Two things changed vs the original:

  * Volt-VAr ranks on the RENAMED, correct columns. The original ranked on
    `nonconformance_voltvar_red_count`, whose STORED value is wrong in any table
    built before the R5 fix. Here the ranking metric is derived from the three
    failure bands (adverse + inactive + significant_shortfall), so it's correct
    regardless of when the table was built. The old q_minor_deviation /
    q_major_deficit breakdown columns are replaced by their _v2 names.

  * sust_op_3w and antiisland are NOT rebuilt (still legacy). MODE_MAP marks
    them, and `rank_sites` prints a warning when you pick one.
"""

import pandas as pd
import pytz

from shared.ciccada_config import SAI, AS4777, TABLES, REBUILT

FIXED_OFFSET = pytz.FixedOffset(600)


# ═════════════════════════════════════════════════════════════
# Mode map — per-mechanism table, ranking expression, rebuilt flag
# ═════════════════════════════════════════════════════════════
# `rank_expr` is summed in SQL as the non-conformance numerator. For Volt-VAr it
# is the R5-correct failure definition, built from the bucket columns so it does
# not depend on the (possibly stale) stored red column.
MODE_MAP = {
    "voltwatt": {
        "table":     TABLES["conformance_voltwatt"],
        "rank_expr": "nonconformance_voltwatt_count",
        "logical":   "conformance_voltwatt",
    },
    "voltvar": {
        "table":     TABLES["conformance_voltvar"],
        # R5: adverse + inactive + significant_shortfall
        "rank_expr": ("Q_adverse_count + Q_inactive_count "
                      "+ Q_significant_shortfall_count"),
        "logical":   "conformance_voltvar",
    },
    "sust_op_3w": {
        "table":     TABLES["conformance_sust_op_3w"],
        "rank_expr": "nonconformance_sust_op_3w_count",
        "logical":   "conformance_sust_op_3w",
    },
    "antiisland": {
        "table":     TABLES["conformance_antiisland"],
        "rank_expr": "nonconformance_antiisland_count",
        "logical":   "conformance_antiisland",
    },
}

# Voltage threshold that defines an "event" for each mechanism (day picker).
_V_THRESHOLD = {
    "voltwatt":   AS4777["VW"]["V1"],       # e.g. 253
    "voltvar":    AS4777["VVAR"]["V4"],     # e.g. 258
    "sust_op_3w": AS4777["SUSTOP_V_BAND"][1],
    "antiisland": AS4777["AI_V_BAND"][0],
}


# ═════════════════════════════════════════════════════════════
# 1. Rank sites
# ═════════════════════════════════════════════════════════════

def rank_sites(mode, aq_func, behaviour="nonconforming", year=2024, month=None,
               min_days=20, n_results=30):
    """
    Rank sites by non-conformance for one mechanism.

    Parameters
    ----------
    mode      : 'voltwatt' | 'voltvar' | 'sust_op_3w' | 'antiisland'
    behaviour : 'nonconforming' (worst first) | 'conforming' (best first)
    month     : None -> full year; int -> single month
    """
    if mode not in MODE_MAP:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODE_MAP)}")

    cfg   = MODE_MAP[mode]
    table = cfg["table"]
    rank  = cfg["rank_expr"]

    if cfg["logical"] not in REBUILT:
        print(f"[!] '{mode}' reads {table}, which was NOT rebuilt (legacy). "
              f"Its non-conformance still carries the UTC/avg-V/flex bugs "
              f"and is not comparable to the Volt-VAr/Volt-Watt rates.\n")

    order    = "DESC" if behaviour == "nonconforming" else "ASC"
    year_f   = f"year = {year}"
    month_f  = f" AND month = {month}" if month is not None else ""
    period   = f"{year}-{month:02d}" if month else str(year)

    # Volt-VAr: pull the five-band breakdown for context in the ranked table
    extra = ""
    if mode == "voltvar":
        extra = """
            ,sum(Q_adverse_count)               AS nc_adverse
            ,sum(Q_inactive_count)              AS nc_inactive
            ,sum(Q_significant_shortfall_count) AS nc_shortfall
            ,sum(Q_near_conformant_count)       AS nc_near_conformant
            ,sum(Q_major_surplus_count)         AS nc_surplus
            ,sum(curtailment_voltvar_count)     AS nc_curtailment
            ,round(100.0 * sum(nonconformance_voltvar_count)
                   / nullif(sum(total_count), 0), 2) AS all_nc_pct
        """

    ranked = aq_func(f"""
        SELECT
            site_id,
            sum({rank})                                    AS nonconf_intervals,
            sum(total_count)                               AS total_intervals,
            count(*)                                       AS n_days,
            round(100.0 * sum({rank})
                  / nullif(sum(total_count), 0), 2)        AS nonconf_pct
            {extra}
        FROM {table}
        WHERE {year_f}{month_f}
        GROUP BY site_id
        HAVING sum(total_count) > 0 AND count(*) >= {min_days}
        ORDER BY nonconf_pct {order}
        LIMIT {n_results}
    """, database=SAI)

    print(f"{behaviour.upper()} sites: {mode}  |  period: {period}  |  table: {table}")
    if mode == "voltvar":
        print("Ranking metric = adverse + inactive + significant_shortfall "
              "(R5-correct). all_nc_pct = all NC / total_count.")
    print(f"Returned {len(ranked)} sites. Top = most {behaviour}.")
    return ranked


# ═════════════════════════════════════════════════════════════
# 2. Pull one site's telemetry (nameplate comes from metadata)
# ═════════════════════════════════════════════════════════════

def pull_site_telemetry(site_id, mode, aq_func, year=2024, month=None,
                        behaviour="nonconforming",
                        force_year=None, force_month=None):
    """
    Fetch metadata + one circuit's telemetry for `site_id`, auto-resolving the
    year/month to one that actually has data. Returns (df, info).

    `df`   has t_stamp_aest, P_kW, Q_kvar ready for the plotters.
    `info` is a dict incl. `ac_capacity_kw` (nameplate) — so you never type it.

    The nameplate, polarity, manufacturer etc. all come from meta_up23c, exactly
    as the original notebook did.
    """
    # ── metadata ─────────────────────────────────────────────────────────────
    candidate = aq_func(f"""
        SELECT DISTINCT
            m.site_id, m.state, m.dnsp_name, m.ac_capacity_kw,
            m.manufacturer, m.model, m.min_time, m.max_time,
            c.circuit_id, c.circuit_polarity
        FROM meta_up23c m
        JOIN circuits c ON m.site_id = c.site_id
        WHERE c.is_pv = True AND m.site_id = {site_id}
        LIMIT 1
    """, database=SAI)

    if candidate.empty:
        raise ValueError(f"No PV metadata for site {site_id}.")

    info = {
        "site_id":        int(candidate["site_id"].iloc[0]),
        "circuit_id":     int(candidate["circuit_id"].iloc[0]),
        "circuit_polarity": int(candidate["circuit_polarity"].iloc[0]),
        "ac_capacity_kw": float(candidate["ac_capacity_kw"].iloc[0]),
        "manufacturer":   f"{candidate['manufacturer'].iloc[0]} {candidate['model'].iloc[0]}",
        "state":          candidate["state"].iloc[0],
        "dnsp":           candidate["dnsp_name"].iloc[0],
        "min_time":       pd.Timestamp(candidate["min_time"].iloc[0]),
        "max_time":       pd.Timestamp(candidate["max_time"].iloc[0]),
    }

    print(f"Site {info['site_id']}  |  {info['state']} / {info['dnsp']}  |  "
          f"{info['ac_capacity_kw']:.1f} kW  |  {info['manufacturer']}")
    print(f"Data: {info['min_time'].date()} -> {info['max_time'].date()}  |  "
          f"polarity {info['circuit_polarity']}")

    # ── resolve year ─────────────────────────────────────────────────────────
    plot_year = force_year if force_year is not None else year
    if force_year is None and not (info["min_time"].year <= plot_year <= info["max_time"].year):
        plot_year = info["max_time"].year
        print(f"[!] year {year} outside data range; using {plot_year}")

    # ── resolve month (best NC month that also has telemetry) ────────────────
    plot_month = force_month if force_month is not None else month
    if plot_month is None:
        cfg = MODE_MAP[mode]
        tele = aq_func(f"""
            SELECT DISTINCT month FROM ts
            JOIN (SELECT DISTINCT circuit_id FROM meta_up23c
                  WHERE is_pv = True AND site_id = {info['site_id']}) m
              USING (circuit_id)
            WHERE year = {plot_year} AND is_pv = True
            ORDER BY month
        """, database=SAI)
        available = set(tele["month"].tolist()) if not tele.empty else set()

        if available:
            order = "DESC" if behaviour == "nonconforming" else "ASC"
            monthly = aq_func(f"""
                SELECT month,
                       round(100.0 * sum({cfg['rank_expr']})
                             / nullif(sum(total_count), 0), 2) AS nc_pct
                FROM {cfg['table']}
                WHERE year = {plot_year} AND site_id = {info['site_id']}
                GROUP BY month
                HAVING sum(total_count) > 0
                ORDER BY nc_pct {order}
            """, database=SAI)
            hit = monthly[monthly["month"].isin(available)]
            plot_month = int(hit["month"].iloc[0]) if not hit.empty else sorted(available)[0]
            print(f"Auto-selected month {plot_month} (of available {sorted(available)})")
        else:
            print(f"[!] no telemetry for {plot_year}")

    info["plot_year"], info["plot_month"] = plot_year, plot_month

    # telemetry pull
    month_clause = f"AND month = {plot_month}" if plot_month else ""
    df = aq_func(f"""
        SELECT t_stamp, circuit_id, voltage, current, power, power_factor,
               energy_reactive
        FROM ts
        WHERE is_pv = True AND year = {plot_year} {month_clause}
          AND circuit_id = {info['circuit_id']}
        ORDER BY t_stamp
    """, database=SAI)

    if df.empty:
        print(f"[!] no telemetry rows (year={plot_year}, month={plot_month}, "
              f"circuit={info['circuit_id']})")
        return df, info

    pol = info["circuit_polarity"]
    df["t_stamp_aest"] = (pd.to_datetime(df["t_stamp"])
                            .dt.tz_localize("UTC").dt.tz_convert(FIXED_OFFSET))
    df["P_kW"]   = df["power"]           / 1000 * pol
    df["Q_kvar"] = df["energy_reactive"] / 1000 * 12 * pol
    # plot_operational/plot_protective read a 'voltage' column
    print(f"Rows: {len(df):,}  ({df['t_stamp_aest'].min().date()} -> "
          f"{df['t_stamp_aest'].max().date()})")
    return df, info


# ═════════════════════════════════════════════════════════════
# 3. Suggest which day to plot
# ═════════════════════════════════════════════════════════════

def suggest_days(df, mode, top=15):
    """
    Rank days by how many intervals breached the mechanism's voltage threshold.
    Returns a Series indexed by date. Pick one for OVERRIDE_DATE / ZOOM_DATE.
    """
    v_thr = _V_THRESHOLD.get(mode, 253)
    hits = (df[df["voltage"] > v_thr]
            .groupby(df["t_stamp_aest"].dt.date)["voltage"]
            .count().sort_values(ascending=False))

    if hits.empty:
        print(f"No days with V > {v_thr} V. "
              f"Voltage range: {df['voltage'].min():.1f}-{df['voltage'].max():.1f} V")
        if mode == "voltvar":  # fall back to the deadband-high threshold
            v_lo = AS4777["VVAR"]["V3"]
            hits = (df[df["voltage"] > v_lo]
                    .groupby(df["t_stamp_aest"].dt.date)["voltage"]
                    .count().sort_values(ascending=False))
            if not hits.empty:
                print(f"Falling back to V > {v_lo} V:")
    if not hits.empty:
        print(f"Days with V > {v_thr} V (most first):")
        print(hits.head(top).to_string())
        print(f"\n  -> best candidate: {hits.index[0]}  ({hits.iloc[0]} intervals)")
    return hits


# ═════════════════════════════════════════════════════════════
# 4. Voltage diagnostic (max-V-across-circuits vs raw)   [cell 52]
# ═════════════════════════════════════════════════════════════

def voltage_diagnostic(site_id, year, aq_func, v_floor=255):
    """
    Show what the conformance builder saw: max(voltage) across all of a site's
    circuits per timestamp, vs the per-circuit values. Explains why a site can
    look non-conformant when a single circuit looks fine (R1: max, not avg).
    """
    circuits = aq_func(f"""
        SELECT circuit_id, circuit_polarity, is_pv
        FROM meta_up23c WHERE site_id = {site_id} AND is_pv = True
    """, database=SAI)
    print(f"PV circuits at site {site_id}: {len(circuits)}")

    hi = aq_func(f"""
        SELECT t_stamp,
               max(voltage) AS max_V_all_circuits,
               avg(voltage) AS avg_V_all_circuits,
               count(*)     AS n_circuits
        FROM ts
        JOIN (SELECT DISTINCT circuit_id FROM meta_up23c
              WHERE site_id = {site_id} AND is_pv = True) c USING (circuit_id)
        WHERE year = {year} AND is_pv = True
        GROUP BY t_stamp
        HAVING max(voltage) > {v_floor}
        ORDER BY max_V_all_circuits DESC
        LIMIT 20
    """, database=SAI)
    print(f"Timestamps where max(V) across circuits > {v_floor} V:")
    return circuits, hi

# ═════════════════════════════════════════════════════════════
# 5. Pull a full month of site-level intervals for the Q-vs-V scatter
# ═════════════════════════════════════════════════════════════

def pull_month_scatter(site_id, year, month, aq_func):
    """
    Site-level per-interval aggregates for one month, for the Volt-VAr Q-vs-V
    scatter (explore_plots.plot_vvar_month_scatter).

    Returns a DataFrame with t_stamp, P_kW, Q_kvar, V (one row per 5-min
    interval, summed across the site's circuits).

    NOTE: uses avg(voltage) to match the original exploration plot. The
    conformance pipeline uses max(voltage) (R1); this is a visual diagnostic,
    not a conformance score, so the choice is cosmetic here — but if you want
    the scatter to match exactly what the table scored, change avg -> max.
    """
    df = aq_func(f"""
        WITH site_agg AS (
            SELECT
                t_stamp,
                sum(power           * m.circuit_polarity / 1000)      AS P_kW,
                sum(energy_reactive * m.circuit_polarity / 1000 * 12) AS Q_kvar,
                avg(voltage)                                          AS V
            FROM ts
            JOIN (SELECT DISTINCT circuit_id, circuit_polarity
                  FROM meta_up23c
                  WHERE is_pv = True AND site_id = {site_id}) m USING (circuit_id)
            WHERE year = {year} AND month = {month} AND is_pv = True
              AND voltage > 0 AND voltage < 300
            GROUP BY t_stamp
        )
        SELECT t_stamp, P_kW, Q_kvar, V FROM site_agg ORDER BY t_stamp
    """, database=SAI)
    print(f"Pulled {len(df):,} intervals for site {site_id}, {year}-{month:02d}.")
    return df
