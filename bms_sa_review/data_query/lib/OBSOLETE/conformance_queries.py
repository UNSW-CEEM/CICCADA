"""
conformance_queries.py  —  Athena SQL for the conformance analysis
===================================================================

All the SQL that lived inline in `02_conformance_curtailment_analysis.ipynb`,
lifted out so the notebook is a pure orchestrator.

Every function takes `aq_func` and `database` explicitly — no notebook globals.

WIRED TO THE _v2 SCHEMAS
------------------------
Column names come from the Stage 2 builders. The two renamed buckets:

    original                 _v2                        Q_impact   meaning
    ---------------------    -----------------------    --------   -------------------
    Q_minor_deviation_*      Q_significant_shortfall_*  0.1-0.9    responded, far short
    Q_major_deficit_*        Q_near_conformant_*        0.9-1.1    essentially conformant

DENOMINATORS (Stage 2, decision D4)
-----------------------------------
    total_count          intervals where the mechanism can act.
                         SAME meaning as the original -> tables stay comparable.
    all_intervals_count  every valid interval that site-day, any voltage. NEW.
    exposed_count        (Volt-VAr only) intervals with V > 240.
    assessable_count     (Volt-Watt GHI only) intervals the counterfactual can
                         actually adjudicate.

!! TWO TABLES ARE STILL LEGACY !!
---------------------------------
conformance_sust_op_3w and conformance_antiisland were NOT rebuilt in Stage 2.
They still carry the UTC date bug, avg(voltage), and no flex-export filter.
Conformance rates from them are NOT comparable to the Volt-VAr / Volt-Watt rates.
Call `table_provenance()` at the top of the notebook so this stays visible.
"""

import pandas as pd

from bms_sa_review.data_query.OBSOLETEciccada_config import AS4777, TABLES, REBUILT

# Resolve once, here, so no query module hard-codes a table name.
T_VVAR    = TABLES["conformance_voltvar"]
T_VW      = TABLES["conformance_voltwatt"]
T_VWGHI   = TABLES["conformance_voltwattghi"]
T_SUSTOP  = TABLES["conformance_sust_op_3w"]
T_ANTIISL = TABLES["conformance_antiisland"]


# ═════════════════════════════════════════════════════════════
# 0. Provenance — which tables are rebuilt, which are not
# ═════════════════════════════════════════════════════════════

def table_provenance():
    """
    Print which tables this analysis is reading and whether each was rebuilt.
    Call this FIRST in the notebook. The legacy rows are the ones to remember
    when writing anything up.
    """
    rows = []
    for key, name in TABLES.items():
        rows.append({
            "logical_name": key,
            "actual_table": name,
            "rebuilt": "YES" if key in REBUILT else "NO  <-- legacy",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print("Legacy tables still carry: UTC date extraction (R3/R9), avg(voltage)")
    print("(R1), and no flex-export filter (R2). Conformance rates from them are")
    print("NOT directly comparable to the rebuilt Volt-VAr / Volt-Watt rates.")
    return df


# ═════════════════════════════════════════════════════════════
# 1. Inventory / lineage sanity-check          [was notebook cell 8]
# ═════════════════════════════════════════════════════════════

def fetch_inventory(aq_func, database):
    """
    Row count, site count and year-month span for every conformance table.
    First thing to run: if a table is empty or a year is missing, everything
    downstream is quietly wrong.
    """
    rows = []
    for key, tbl in TABLES.items():
        if not key.startswith("conformance_"):
            continue
        try:
            r = aq_func(f"""
                SELECT '{tbl}'              AS tbl,
                       count(*)             AS n_rows,
                       count(DISTINCT site_id) AS n_sites,
                       min(year*100 + month)   AS ym_min,
                       max(year*100 + month)   AS ym_max
                FROM {tbl}
            """, database=database)
            r["rebuilt"] = "YES" if key in REBUILT else "NO"
            rows.append(r)
        except Exception as e:
            print(f"[!] {tbl}: {e}")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ═════════════════════════════════════════════════════════════
# 2. Per-site conformance roll-up              [was notebook cell 10]
# ═════════════════════════════════════════════════════════════

def fetch_site_conformance(table, nonconf_count_col, aq_func, database,
                           day_only=False, extra_where="",
                           denominator="total_count"):
    """
    Roll a daily site-grain conformance table up to per-site counts.

    Returns the raw counts only — the conformant/non-conformant VERDICT is
    applied in `conformance_metrics.site_conformance()`, because that's a
    thresholding decision, not a query.

    Parameters
    ----------
    table             : str  actual table name (use the T_* constants)
    nonconf_count_col : str  e.g. 'nonconformance_voltwatt_count'
    day_only          : bool restrict to day_night = 'day'
    extra_where       : str  e.g. "v_threshold = 258"
    denominator       : str  'total_count' (mechanism-exposed, comparable to the
                             original) or 'all_intervals_count' (every interval).
    """
    where = []
    if day_only:
        where.append("day_night = 'day'")
    if extra_where:
        where.append(extra_where)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    return aq_func(f"""
        SELECT site_id,
               SUM({nonconf_count_col}) AS nonconf_count,
               SUM({denominator})       AS total_count
        FROM {table}
        {where_sql}
        GROUP BY site_id
    """, database=database)


# ═════════════════════════════════════════════════════════════
# 3. Volt-VAr bucket components                [was notebook cells 12 + 16]
# ═════════════════════════════════════════════════════════════

def fetch_vvar_components(aq_func, database, day_only=True):
    """
    Per-site counts for each of the five Q_impact bands, from
    conformance_voltvar_v2.

    Returns the RAW bucket counts. The reduced non-conformance total is derived
    from these in `conformance_metrics.vvar_reduced()`:

        reduced = adverse + inactive + shortfall          (R5, resolved)

    The stored `nonconformance_voltvar_red_count` is ALSO selected, but only so
    `conformance_metrics.check_stored_red()` can verify it. Any table built
    before the R5 fix has the wrong value there (it summed near_conformant
    instead of shortfall), so nothing should read it directly.
    """
    where = "WHERE day_night = 'day'" if day_only else ""

    df = aq_func(f"""
        SELECT site_id,
               SUM(Q_adverse_count)               AS adverse,        -- < -0.1
               SUM(Q_inactive_count)              AS inactive,       -- -0.1..0.1
               SUM(Q_significant_shortfall_count) AS shortfall,      -- 0.1..0.9
               SUM(Q_near_conformant_count)       AS near_conformant,-- 0.9..1.1
               SUM(Q_major_surplus_count)         AS surplus,        -- > 1.1
               SUM(total_count)                   AS total_count,
               SUM(all_intervals_count)           AS all_intervals_count,
               SUM(exposed_count)                 AS exposed_count,
               -- verification only; never read as the answer
               SUM(nonconformance_voltvar_red_count) AS nonconformance_voltvar_red_count
        FROM {T_VVAR}
        {where}
        GROUP BY site_id
    """, database=database)

    return df[df["total_count"] > 0].copy()


def fetch_vvar_breakdown(aq_func, database, day_only=True):
    """
    Fleet-level (not per-site) interval counts in each Q_impact band, as a
    share of all evaluated intervals. This is the headline breakdown table.
    """
    where = "WHERE day_night = 'day'" if day_only else ""

    df = aq_func(f"""
        SELECT
            SUM(Q_adverse_count)               AS adverse__lt_m010,
            SUM(Q_inactive_count)              AS inactive__m010_to_010,
            SUM(Q_significant_shortfall_count) AS shortfall__010_to_090,
            SUM(Q_near_conformant_count)       AS near_conformant__090_to_110,
            SUM(Q_major_surplus_count)         AS surplus__gt_110,
            SUM(total_count)                   AS total_evaluated
        FROM {T_VVAR}
        {where}
    """, database=database).T.rename(columns={0: "interval_count"})

    total = df.loc["total_evaluated", "interval_count"]
    df["pct_of_evaluated"] = (df["interval_count"] / total * 100).round(2)
    return df


# ═════════════════════════════════════════════════════════════
# 4. Curtailment energy                        [was notebook cell 18]
# ═════════════════════════════════════════════════════════════

def fetch_curtailment_energy(table, sum_col, count_col, aq_func, database,
                             gen_col=None, day_only=False, year_filter=None):
    """
    Per-site curtailed energy from a conformance table.

    kW sums are converted to kWh in `conformance_metrics.curtailment_energy()`
    (multiply by INTERVAL_H) — kept out of SQL so the interval length lives in
    exactly one place.
    """
    conds = []
    if day_only:
        conds.append("day_night = 'day'")
    if year_filter:
        conds.append(f"year = {year_filter}")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    gen_sel = f"SUM({gen_col}) AS gen_kw_sum," if gen_col else ""

    return aq_func(f"""
        SELECT site_id,
               {gen_sel}
               SUM({sum_col})   AS curt_kw_sum,
               SUM({count_col}) AS curt_intervals
        FROM {table} {where}
        GROUP BY site_id
    """, database=database)


# ═════════════════════════════════════════════════════════════
# 5. Volt-Watt report metrics                  [was notebook cell 22]
# ═════════════════════════════════════════════════════════════

def fetch_vw_report(aq_func, database, year_filter=None):
    """
    The three per-site sub-metrics the milestone report uses, from the
    GHI-filtered table (conformance_voltwattghi_v2):

        nc_count      total 5-min non-conformance instances
        nc_sum_kw     summed NC magnitude (kW) -> becomes MWh in metrics
        gen_sum_kw    generation in eligible conditions -> normalisation base

    Uses the GHI table, not the basic one, because the report's Figure 8/9
    methodology excludes low-irradiance intervals where reduced power could just
    be cloud cover.

    Note the _v2 column names: nonconformance_voltwattghi_*, and `total_count`
    now means "intervals with V > 253" in BOTH Volt-Watt tables (decision D4),
    so it is a clean eligible denominator.
    """
    where = f"WHERE year = {year_filter}" if year_filter else ""

    return aq_func(f"""
        SELECT site_id,
               SUM(nonconformance_voltwattghi_sum)   AS nc_sum_kw,
               SUM(nonconformance_voltwattghi_count) AS nc_count,
               SUM(P_kW_sum)                         AS gen_sum_kw,
               SUM(total_count)                      AS total_count,
               SUM(all_intervals_count)              AS all_intervals_count,
               SUM(assessable_count)                 AS assessable_count,
               SUM(null_uncurtailed_P_count)         AS null_uncurtailed_P_count
        FROM {T_VWGHI}
        {where}
        GROUP BY site_id
    """, database=database)


# ═════════════════════════════════════════════════════════════
# 6. Site metadata                             [was notebook cell 24]
# ═════════════════════════════════════════════════════════════

def fetch_meta(aq_func, database, exclude_flex=True):
    """
    Site metadata for the state / DNSP / OEM breakdowns.

    exclude_flex defaults True to match the cohort the _v2 conformance tables
    were built on (R2/R18). If it didn't, the merge would attach metadata to
    sites that aren't in the conformance tables at all, and the breakdown
    denominators would silently disagree with the conformance denominators.
    """
    flex = "AND flex_export_detected = False" if exclude_flex else ""

    return aq_func(f"""
        SELECT DISTINCT site_id,
               state,
               dnsp_name             AS dnsp,
               manufacturer          AS oem,
               year(pv_install_date) AS install_year,
               ac_capacity_kw,
               s_99
        FROM meta_up23c
        WHERE is_pv = True
        {flex}
    """, database=database)


# ═════════════════════════════════════════════════════════════
# 7. Single-inverter cohort                    [was notebook cell 30]
# ═════════════════════════════════════════════════════════════

def fetch_single_inverter_cohort(aq_func, database, exclude_flex=True):
    """Sites with exactly one inverter (up to 3 circuits) — the established cohort."""
    flex = "AND flex_export_detected = False" if exclude_flex else ""

    return aq_func(f"""
        SELECT site_id, count(DISTINCT circuit_id) AS n_circuits
        FROM meta_up23c
        WHERE is_pv = True
        {flex}
        GROUP BY site_id
        HAVING count(DISTINCT circuit_id) <= 3
    """, database=database)


# ═════════════════════════════════════════════════════════════
# 8. Interval-level Volt-VAr pull              [was notebook cells 34 + 43]
# ═════════════════════════════════════════════════════════════

def fetch_vvar_intervals(site_ids, year, month, aq_func, database,
                         exclude_flex=True):
    """
    Raw Volt-VAr interval data for a cohort of sites in one month, for the
    scatter-vs-curve plots.

    Applies R1 (max voltage) and the AEST conversion, so what you plot matches
    what the _v2 conformance tables scored. The required-Q curve is NOT computed
    here — `conformance_plots` gets it from the as4777_curves keystone, so the
    curve on the plot is the same code that scored the table.

    Returns empty DataFrame if site_ids is empty.
    """
    if len(site_ids) == 0:
        return pd.DataFrame()

    ids  = ", ".join(str(s) for s in site_ids)
    flex = "AND flex_export_detected = False" if exclude_flex else ""

    return aq_func(f"""
        WITH site_meta AS (
            SELECT DISTINCT site_id, circuit_id, circuit_polarity,
                   ac_capacity_kw, s_99
            FROM meta_up23c
            WHERE is_pv = True
            {flex}
              AND site_id IN ({ids})
        )
        SELECT
            m.site_id,
            t.t_stamp,
            sum(t.power * m.circuit_polarity) / 1000            AS P_kW,
            sum(t.energy_reactive * m.circuit_polarity) / 1000 * 12 AS Q_kvar,
            max(t.voltage)       AS V,
            max(m.ac_capacity_kw) AS ac_capacity_kw,
            max(m.s_99)          AS s_99
        FROM ts t
        JOIN site_meta m ON t.circuit_id = m.circuit_id
        WHERE t.year = {year} AND t.month = {month}
          AND t.is_pv = True
          AND t.voltage > 0 AND t.voltage < 300
        GROUP BY m.site_id, t.t_stamp
    """, database=database)
