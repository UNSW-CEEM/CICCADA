"""
Data calc-write pipeline: Stage 2. Shared helpers
=================================================

WHAT IT PROVIDES
----------------
1. `aest_month_window(year, month)`
       The UTC half-open interval [start, end) that corresponds to one AEST
       calendar month, plus the list of (year, month) partitions of `ts` that
       interval touches.  This is the D2 fix: an AEST month is NOT a UTC month.

2. `site_agg_cte(...)`
       The `data` CTE: one row per site per 5-minute interval, with
         - P_kW   = sum(power * circuit_polarity) / 1000            (kW)
         - Q_kvar = sum(energy_reactive * circuit_polarity) / 1000 * 12  (kvar)
         - V      = AVG or MAX voltage across circuits, selected per run
         - ac_capacity_kw, S_99 from metadata
       restricted to is_pv sites with flex_export_detected = False  <- R2 fix
"""

from datetime import datetime, timedelta
from pipeline_options import flex_predicate, voltage_aggregate_sql

# ---------------------------------------------------------------------------
# Conventions shared by every Stage 2 table
# ---------------------------------------------------------------------------
TABLE_SUFFIX = "_v2"                            # never overwrite originals
WAREHOUSE    = "Trino-Warehouse/solar_analytics"
AEST         = "+ interval '10' hour"           # UTC -> AEST (no DST, per ciccada_config)

# The Stage 1 counterfactual table this stage joins against.
UNCURTAILED = f"all_uncurtailedpv{TABLE_SUFFIX}"

# Voltage sanity gate on the raw telemetry
V_MIN, V_MAX = 0, 300


# ===========================================================================
# 1. AEST MONTH WINDOWS
# ===========================================================================
def aest_month_window(year, month):
    """
    Return (utc_start, utc_end, partitions) for one AEST calendar month.

    An AEST month starts 10 hours BEFORE the UTC month of the same name, so it
    always straddles two UTC partitions of `ts`:

        AEST 2024-02  ==  UTC [2024-01-31 14:00, 2024-02-29 14:00)
                          touches ts partitions (2024,1) and (2024,2)

    If we sliced on UTC months and grouped by AEST dates (which R3 requires),
    the boundary day would be split across two INSERTs and we would end up with
    two rows sharing the same (year, month, day, site_id) key. Sums would still
    add up, but "one row per site-day" would silently stop being true. So we
    slice on AEST months and read the two UTC partitions they touch.

    Returns
    -------
    utc_start  : str  'YYYY-MM-DD HH:MM:SS'   (inclusive)
    utc_end    : str  'YYYY-MM-DD HH:MM:SS'   (exclusive)
    partitions : list[(int, int)]             ts (year, month) partitions to scan
    """
    start_aest = datetime(year, month, 1)
    end_aest   = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    start_utc = start_aest - timedelta(hours=10)
    end_utc   = end_aest   - timedelta(hours=10)

    # The half-open interval [start_utc, end_utc) can only ever touch the UTC
    # partition containing start_utc and the one containing (end_utc - 1s).
    last_utc = end_utc - timedelta(seconds=1)
    parts = sorted({(start_utc.year, start_utc.month), (last_utc.year, last_utc.month)})

    fmt = "%Y-%m-%d %H:%M:%S"
    return start_utc.strftime(fmt), end_utc.strftime(fmt), parts


def partition_predicate(partitions, alias="ts"):
    """SQL predicate that prunes `ts` to the given (year, month) partitions."""
    ors = " OR ".join(f"({alias}.year = {y} AND {alias}.month = {m})" for y, m in partitions)
    return f"({ors})"


def uncurtailed_partition_predicate(partitions, alias="a"):
    """
    Same pruning, applied to `all_uncurtailedpv_v2`.

    That table is partitioned on `year_p` / `month_p`, which are derived from
    the UTC t_stamp (see build_all_uncurtailedpv.py), so the SAME UTC partition
    list applies. Without this the LEFT JOIN would scan the whole table.
    """
    ors = " OR ".join(f"({alias}.year_p = {y} AND {alias}.month_p = {m})" for y, m in partitions)
    return f"({ors})"


# ===========================================================================
# 2. SITE-LEVEL AGGREGATE
# ===========================================================================
def meta_filter(part_filter, exclude_flex=None, flex_selection="exclude"):
    """
    WHERE clause for the `meta_up23c` subquery.

    flex_selection="exclude" -> keep explicitly unflagged sites only.
    flex_selection="include" -> include flagged, unflagged and NULL flags.
    flex_selection="only"    -> keep flagged sites only, for diagnostics.

    ``exclude_flex`` is retained for compatibility with the existing runner notebooks. 
    When supplied, True maps to ``exclude`` and False to ``include``.

    NOTE (deviation from original, deliberate):
    Also require `ac_capacity_kw > 0 AND s_99 > 0` so either labelled basis
    can be selected and directly compared on the same site population.
    The Q_impact denominator is `abs(Q_required) + 1e-9`
    A site with a zero/NULL selected rating gives a required-Q of 0, a
    denominator of 1e-9, and a Q_impact of ~1e9, which lands in
    Q_major_surplus and pollutes the fleet aggregate.
    `validate()` reports how many sites this drops.
    """
    if exclude_flex is not None:
        flex_selection = "exclude" if exclude_flex else "include"

    parts = [
        "is_pv = True",
        "ac_capacity_kw > 0",
        "s_99 > 0",
        flex_predicate(flex_selection),
    ]
    parts.append(part_filter)
    return " AND ".join(parts)


def site_agg_cte(partitions, utc_start, utc_end, part_filter,
                 exclude_flex=None, with_reactive=True,
                 flex_selection="exclude", voltage_aggregation="avg"):
    """
    Build the `data` CTE: one row per (site_id, t_stamp).
    `with_reactive=False` skips the energy_reactive column, which lets the
    Volt-Watt builders scan one less column off the fact table.
    """
    q_col = ("sum(ts.energy_reactive * m.circuit_polarity) / 1000 * 12 AS Q_kvar,"
             if with_reactive else "")
    v_col = voltage_aggregate_sql(voltage_aggregation, "ts.voltage")

    return f"""
    data AS (
        SELECT
            m.site_id,
            ts.t_stamp,
            sum(ts.power * m.circuit_polarity) / 1000 AS P_kW,
            {q_col}
            {v_col}                 AS V,
            max(m.ac_capacity_kw)   AS ac_capacity_kw,
            max(m.s_99)             AS S_99
        FROM ts
        JOIN (
            SELECT circuit_id,
                max(site_id)          AS site_id,
                max(circuit_polarity) AS circuit_polarity,
                max(ac_capacity_kw)   AS ac_capacity_kw,
                max(s_99)             AS s_99
            FROM meta_up23c
            WHERE {meta_filter(part_filter, exclude_flex, flex_selection)}
            GROUP BY circuit_id
        ) AS m ON ts.circuit_id = m.circuit_id
        WHERE {partition_predicate(partitions)}
          AND ts.t_stamp >= TIMESTAMP '{utc_start}'
          AND ts.t_stamp <  TIMESTAMP '{utc_end}'
          AND ts.is_pv = True
          AND ts.voltage > {V_MIN} AND ts.voltage < {V_MAX}
        GROUP BY m.site_id, ts.t_stamp
    )"""


# ===========================================================================
# 3. TEMPORAL EXTRACTION
# ===========================================================================
def temporal_cols(ts_col="t_stamp"):
    """
    AEST year / month / day / day_night. NEVER extract these from raw UTC.

    original was
        CASE WHEN hour(t_stamp) >= 20 OR hour(t_stamp) <= 7 THEN 'day' ELSE 'night'
    on the UTC timestamp. That is arithmetically the same window as AEST
    06:00-17:00, so his labels were correct. His DATE extraction (day/month/year straight off UTC t_stamp) was the
    actual bug: every interval before 10:00 AEST was booked to the previous day.
    """
    return f"""
        CAST(year({ts_col}  {AEST}) AS INT) AS year_aest,
        CAST(month({ts_col} {AEST}) AS INT) AS month_aest,
        CAST(day({ts_col}   {AEST}) AS INT) AS day_aest,
        CASE WHEN hour({ts_col} {AEST}) BETWEEN 6 AND 17
             THEN 'day' ELSE 'night' END AS day_night"""


# ===========================================================================
# 4. SLICE RUNNER  (shared loop mechanics)
# ===========================================================================
def run_months(aq, database, insert_sql_builder, year, months,
               n_parts=8, parts=None, exclude_flex=None, verbose=True,
               **builder_options):
    """
    Execute one INSERT per (AEST month x site-slice).

    `insert_sql_builder(partitions, utc_start, utc_end, part_filter, exclude_flex)`
    must return a complete INSERT statement. Site-slicing is `site_id % n_parts`
    """
    if parts is None:
        parts = list(range(n_parts))

    results = []
    for month in months:
        utc_start, utc_end, partitions = aest_month_window(year, month)
        for part in parts:
            part_filter = f"site_id % {n_parts} = {part}"
            sql = insert_sql_builder(
                partitions,
                utc_start,
                utc_end,
                part_filter,
                exclude_flex=exclude_flex,
                **builder_options,
            )
            aq(sql, database=database)
            msg = (f"loaded AEST {year}-{month:02d} part={part}/{n_parts} "
                   f"(UTC {utc_start} -> {utc_end}, partitions {partitions})")
            results.append(msg)
            if verbose:
                print(msg)
    return results
