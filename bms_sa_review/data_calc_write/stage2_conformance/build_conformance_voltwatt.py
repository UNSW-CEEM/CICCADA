"""
Data calc-write pipeline: Stage 2. Volt-Watt conformance
========================================================

    build_basic  -> conformance_voltwatt_v2
                    Did the site exceed the Volt-Watt P-limit? No counterfactual.

    build_ghi    -> conformance_voltwattghi_v2
                    Same, but the GHI counterfactual tells us whether the site
                    COULD have exceeded the limit. If it could not, the interval
                    is not evidence of conformance and must not be counted as
                    such (NULL). If it could have and didn't, that is Volt-Watt
                    curtailment.

COUNTS
-----------------
  total_count          intervals where Volt-Watt can act, i.e. V > 253.
                       Identical meaning in BOTH tables. This reproduces
                       Original basic-table denominator (his `HAVING
                       avg(voltage) > 253`), so the tables stay comparable.
  all_intervals_count  every valid interval for that site-day, any voltage.
                       The honest fleet-exposure denominator.
  assessable_count     (GHI table only) intervals where the counterfactual
                       permits a conformance verdict at all.

SAFE-BY-DEFAULT
---------------
Writes to `conformance_voltwatt_v2` / `conformance_voltwattghi_v2`.
"""

from shared.as4777_curves import vw_max_p_sql, tol_kw_sql
from stage2_common import (
    TABLE_SUFFIX, 
    WAREHOUSE, 
    UNCURTAILED,
    aest_month_window, 
    uncurtailed_partition_predicate,
    site_agg_cte, 
    temporal_cols, 
    run_months,
)

TARGET_BASIC = f"conformance_voltwatt{TABLE_SUFFIX}"
TARGET_GHI   = f"conformance_voltwattghi{TABLE_SUFFIX}"


from shared.ciccada_config import AS4777
# VW_V1 = 253.0

VW_V1 = AS4777["VW"]["V1"]
print(f"Using VW_V1 = {VW_V1} V (AS4777.2:2020 Australia A)")

# ===========================================================================
# DDL
# ===========================================================================
def create_table_basic(aq, database):
    aq(f"DROP TABLE IF EXISTS {TARGET_BASIC}", database=database)
    aq(f"""
        CREATE TABLE {TARGET_BASIC} (
            site_id                        BIGINT,
            day                            INT,
            day_night                      STRING,
            P_kW_sum                       DOUBLE,
            nonconformance_voltwatt_sum    DOUBLE,
            nonconformance_voltwatt_count  BIGINT,
            all_intervals_count            BIGINT,
            total_count                    BIGINT,
            year                           INT,
            month                          INT
        )
        PARTITIONED BY (year, month)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{TARGET_BASIC}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {TARGET_BASIC}"


def create_table_ghi(aq, database):
    aq(f"DROP TABLE IF EXISTS {TARGET_GHI}", database=database)
    aq(f"""
        CREATE TABLE {TARGET_GHI} (
            site_id                           BIGINT,
            day                               INT,
            day_night                         STRING,
            P_kW_sum                          DOUBLE,
            nonconformance_voltwattghi_sum    DOUBLE,
            curtailment_voltwattghi_sum       DOUBLE,
            nonconformance_voltwattghi_count  BIGINT,
            curtailment_voltwattghi_count     BIGINT,
            assessable_count                  BIGINT,
            null_uncurtailed_P_count          BIGINT,
            all_intervals_count               BIGINT,
            total_count                       BIGINT,
            year                              INT,
            month                             INT
        )
        PARTITIONED BY (year, month)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{TARGET_GHI}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {TARGET_GHI}"


def create_tables(aq, database):
    return [create_table_basic(aq, database), create_table_ghi(aq, database)]


# ===========================================================================
# INSERT -- basic
# ===========================================================================
def _insert_sql_basic(partitions, utc_start, utc_end, part_filter, exclude_flex=True):
    # Volt-Watt does not need reactive power -> one less column off the fact table.
    data = site_agg_cte(partitions, utc_start, utc_end, part_filter,
                        exclude_flex=exclude_flex, with_reactive=False)

    # curve from the keystone. Tolerance added explicitly, on nameplate.
    max_p = vw_max_p_sql("V", "ac_capacity_kw")
    tol   = tol_kw_sql("ac_capacity_kw")

    return f"""
    INSERT INTO {TARGET_BASIC}
    WITH
    {data},

    limits AS (
        SELECT site_id, t_stamp, P_kW, V, ac_capacity_kw,
               ({max_p}) + {tol} AS max_P_volt_watt
        FROM data
    ),

    -- NOTE: no `HAVING V > 253` on the data CTE. 
    -- Keeping every interval so that all_intervals_count is a real denominator, and instead
    -- gate the VERDICT on V > 253. 
    scored AS (
        SELECT site_id, P_kW, V,
               {temporal_cols('t_stamp')},
               CASE WHEN V > {VW_V1}
                    THEN greatest(0, P_kW - max_P_volt_watt)
                    ELSE NULL END AS nonconformance_voltwatt
        FROM limits
    )

    -- Column order MUST match create_table_basic().
    SELECT
        site_id,
        day_aest                                        AS day,
        day_night,
        sum(P_kW)                                       AS P_kW_sum,
        sum(nonconformance_voltwatt)                    AS nonconformance_voltwatt_sum,
        sum(CASE WHEN nonconformance_voltwatt > 0 THEN 1 ELSE 0 END)
                                                        AS nonconformance_voltwatt_count,
        count(*)                                        AS all_intervals_count,
        sum(CASE WHEN V > {VW_V1} THEN 1 ELSE 0 END)    AS total_count,   -- R13
        year_aest                                       AS year,
        month_aest                                      AS month
    FROM scored
    GROUP BY year_aest, month_aest, day_aest, day_night, site_id
    """


# ===========================================================================
# INSERT -- GHI-aware
# ===========================================================================
def _insert_sql_ghi(partitions, utc_start, utc_end, part_filter, exclude_flex=True):
    data = site_agg_cte(partitions, utc_start, utc_end, part_filter,
                        exclude_flex=exclude_flex, with_reactive=False)

    max_p    = vw_max_p_sql("V", "ac_capacity_kw")
    tol      = tol_kw_sql("ac_capacity_kw")
    unc_pred = uncurtailed_partition_predicate(partitions, alias="a")

    return f"""
    INSERT INTO {TARGET_GHI}
    WITH
    {data},

    -- Joins all_uncurtailedpv_v2
    limits AS (
        SELECT d.site_id, d.t_stamp, d.P_kW, d.V, d.ac_capacity_kw,
               ({max_p}) + {tol} AS max_P_volt_watt,
               a.uncurtailed_P
        FROM data d
        LEFT JOIN (
            SELECT a.site_id, a.t_stamp, a.uncurtailed_P
            FROM {UNCURTAILED} a
            WHERE {unc_pred}
        ) a ON d.site_id = a.site_id AND d.t_stamp = a.t_stamp
    ),

    scored AS (
        SELECT site_id, P_kW, V, uncurtailed_P, max_P_volt_watt,
               {temporal_cols('t_stamp')},

               -- Non-conformance is only ASSESSABLE when the site could plausibly
               -- have breached the limit (counterfactual above it), or when we
               -- have no counterfactual at all and must fall back on the raw
               -- reading. Otherwise NULL
               CASE WHEN V > {VW_V1}
                     AND (uncurtailed_P > max_P_volt_watt OR uncurtailed_P IS NULL)
                    THEN greatest(0, P_kW - max_P_volt_watt)
                    ELSE NULL END AS nonconformance_voltwattghi,

               -- Curtailment: it COULD have generated above the limit, and it
               -- didn't. NULL when absent, never 0.
               CASE WHEN V > {VW_V1}
                     AND uncurtailed_P > max_P_volt_watt
                     AND P_kW < max_P_volt_watt
                    THEN uncurtailed_P - P_kW
                    ELSE NULL END AS curtailment_voltwattghi,

               CASE WHEN V > {VW_V1}
                     AND (uncurtailed_P > max_P_volt_watt OR uncurtailed_P IS NULL)
                    THEN 1 ELSE 0 END AS assessable,

               -- Exposed interval with no counterfactual behind it
               CASE WHEN V > {VW_V1} AND uncurtailed_P IS NULL
                    THEN 1 ELSE 0 END AS null_uncurtailed_P
        FROM limits
    )

    -- Column order MUST match create_table_ghi().
    SELECT
        site_id,
        day_aest                                        AS day,
        day_night,
        sum(P_kW)                                       AS P_kW_sum,
        sum(nonconformance_voltwattghi)                 AS nonconformance_voltwattghi_sum,
        sum(curtailment_voltwattghi)                    AS curtailment_voltwattghi_sum,
        sum(CASE WHEN nonconformance_voltwattghi > 0 THEN 1 ELSE 0 END)
                                                        AS nonconformance_voltwattghi_count,
        sum(CASE WHEN curtailment_voltwattghi    > 0 THEN 1 ELSE 0 END)
                                                        AS curtailment_voltwattghi_count,
        sum(assessable)                                 AS assessable_count,
        sum(null_uncurtailed_P)                         AS null_uncurtailed_P_count,  -- R14
        count(*)                                        AS all_intervals_count,
        sum(CASE WHEN V > {VW_V1} THEN 1 ELSE 0 END)    AS total_count,               -- R13
        year_aest                                       AS year,
        month_aest                                      AS month
    FROM scored
    GROUP BY year_aest, month_aest, day_aest, day_night, site_id
    """


# ===========================================================================
# Runners
# ===========================================================================
def run_months_basic(aq, database, year, months, n_parts=8, parts=None,
                     exclude_flex=True):
    return run_months(aq, database, _insert_sql_basic, year, months,
                      n_parts=n_parts, parts=parts, exclude_flex=exclude_flex)


def run_months_ghi(aq, database, year, months, n_parts=8, parts=None,
                   exclude_flex=True):
    return run_months(aq, database, _insert_sql_ghi, year, months,
                      n_parts=n_parts, parts=parts, exclude_flex=exclude_flex)


def preview_sql(which="basic", year=2024, month=1, n_parts=8, part=0,
                exclude_flex=True):
    """Return the INSERT SQL for one slice WITHOUT running it."""
    utc_start, utc_end, partitions = aest_month_window(year, month)
    builder = _insert_sql_basic if which == "basic" else _insert_sql_ghi
    return builder(partitions, utc_start, utc_end,
                   f"site_id % {n_parts} = {part}", exclude_flex)


# ===========================================================================
# Validation
# ===========================================================================
def _shape(aq, database, table):
    return aq(f"""
        SELECT year, month, count(*) AS n_rows,
               count(DISTINCT site_id) AS n_sites,
               count(DISTINCT day) AS n_days
        FROM {table} GROUP BY year, month ORDER BY year, month
    """, database=database)


def _dupes(aq, database, table):
    return aq(f"""
        SELECT count(*) AS n_dupe_keys FROM (
            SELECT year, month, day, day_night, site_id
            FROM {table}
            GROUP BY year, month, day, day_night, site_id
            HAVING count(*) > 1
        )
    """, database=database)


def validate_basic(aq, database):
    shape = _shape(aq, database, TARGET_BASIC)
    dupes = _dupes(aq, database, TARGET_BASIC)
    coh = aq(f"""
        SELECT sum(CASE WHEN nonconformance_voltwatt_sum < 0 THEN 1 ELSE 0 END) AS neg_rows,
               sum(CASE WHEN total_count > all_intervals_count THEN 1 ELSE 0 END) AS count_inversion_rows,
               sum(CASE WHEN nonconformance_voltwatt_count > total_count THEN 1 ELSE 0 END) AS impossible_rows
        FROM {TARGET_BASIC}
    """, database=database)
    print("Rows / sites / days per AEST month:")
    print(shape.to_string(index=False))
    print(f"\nDuplicate keys (MUST be 0): {int(dupes['n_dupe_keys'].iloc[0])}")
    print("\nCoherence (all MUST be 0):")
    print(coh.to_string(index=False))
    return shape, dupes, coh


def validate_ghi(aq, database):
    shape = _shape(aq, database, TARGET_GHI)
    dupes = _dupes(aq, database, TARGET_GHI)
    coh = aq(f"""
        SELECT sum(CASE WHEN curtailment_voltwattghi_sum < 0 THEN 1 ELSE 0 END) AS neg_curtailment_rows,
               sum(CASE WHEN assessable_count > total_count THEN 1 ELSE 0 END)  AS impossible_rows,
               sum(CASE WHEN total_count > all_intervals_count THEN 1 ELSE 0 END) AS count_inversion_rows
        FROM {TARGET_GHI}
    """, database=database)
    cover = aq(f"""
        SELECT sum(total_count)              AS exposed_intervals,
               sum(null_uncurtailed_P_count) AS no_counterfactual,
               round(100.0 * sum(null_uncurtailed_P_count)
                     / nullif(sum(total_count), 0), 2) AS pct_missing
        FROM {TARGET_GHI}
    """, database=database)
    print("Rows / sites / days per AEST month:")
    print(shape.to_string(index=False))
    print(f"\nDuplicate keys (MUST be 0): {int(dupes['n_dupe_keys'].iloc[0])}")
    print("\nCoherence (all MUST be 0):")
    print(coh.to_string(index=False))
    print("\nCounterfactual coverage above 253 V (R14):")
    print(cover.to_string(index=False))
    return shape, dupes, coh, cover


def cross_check(aq, database):
    df = aq(f"""
        SELECT count(*) AS n_mismatched_site_days
        FROM {TARGET_BASIC} b
        JOIN {TARGET_GHI}   g
          ON b.site_id = g.site_id AND b.year = g.year
         AND b.month = g.month AND b.day = g.day AND b.day_night = g.day_night
        WHERE b.total_count <> g.total_count
    """, database=database)
    n = int(df["n_mismatched_site_days"].iloc[0])
    print(f"Site-days where basic.total_count != ghi.total_count (MUST be 0): {n}")
    return df
