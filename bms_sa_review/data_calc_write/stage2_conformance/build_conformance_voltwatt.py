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

from as4777_curves import vw_max_p_sql, tol_kw_sql
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
from ciccada_config import AS4777
from pipeline_options import capacity_column

TARGET_BASIC = f"conformance_voltwatt{TABLE_SUFFIX}"
TARGET_GHI   = f"conformance_voltwattghi{TABLE_SUFFIX}"

# VW_V1 = 253.0

VW_V1 = AS4777["VW"]["V1"]
print(f"Using VW_V1 = {VW_V1} V (AS4777.2:2020 Australia A)")

# ===========================================================================
# DDL
# ===========================================================================
def create_table_basic(aq, database, target=TARGET_BASIC):
    aq(f"DROP TABLE IF EXISTS {target}", database=database)
    aq(f"""
        CREATE TABLE {target} (
            site_id                        BIGINT,
            day                            INT,
            day_night                      STRING,
            P_kW_sum                       DOUBLE,
            nonconformance_voltwatt_sum    DOUBLE,
            nonconformance_voltwatt_count  BIGINT,
            all_intervals_count            BIGINT,
            total_count                    BIGINT,
            rating_basis                   STRING,
            voltage_aggregation            STRING,
            flex_selection                 STRING,
            year                           INT,
            month                          INT
        )
        PARTITIONED BY (year, month)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{target}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {target}"


def create_table_ghi(aq, database, target=TARGET_GHI):
    aq(f"DROP TABLE IF EXISTS {target}", database=database)
    aq(f"""
        CREATE TABLE {target} (
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
            rating_basis                      STRING,
            voltage_aggregation               STRING,
            flex_selection                    STRING,
            year                              INT,
            month                             INT
        )
        PARTITIONED BY (year, month)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{target}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {target}"


def create_tables(aq, database, target_basic=TARGET_BASIC, target_ghi=TARGET_GHI):
    return [
        create_table_basic(aq, database, target=target_basic),
        create_table_ghi(aq, database, target=target_ghi),
    ]


# ===========================================================================
# INSERT -- basic
# ===========================================================================
def _insert_sql_basic(
    partitions, utc_start, utc_end, part_filter, exclude_flex=None, *,
    target=TARGET_BASIC, rating_basis="ac_capacity_kw",
    voltage_aggregation="avg", flex_selection="exclude",
):
    # Volt-Watt does not need reactive power -> one less column off the fact table.
    data = site_agg_cte(
        partitions, utc_start, utc_end, part_filter,
        exclude_flex=exclude_flex, with_reactive=False,
        flex_selection=flex_selection,
        voltage_aggregation=voltage_aggregation,
    )

    rating_col = capacity_column(rating_basis)
    max_p = vw_max_p_sql("V", rating_col)
    tol   = tol_kw_sql(rating_col)

    return f"""
    INSERT INTO {target}
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
        '{rating_basis}'                                AS rating_basis,
        '{voltage_aggregation}'                         AS voltage_aggregation,
        '{flex_selection}'                              AS flex_selection,
        year_aest                                       AS year,
        month_aest                                      AS month
    FROM scored
    GROUP BY year_aest, month_aest, day_aest, day_night, site_id
    """


# ===========================================================================
# INSERT -- GHI-aware
# ===========================================================================
def _insert_sql_ghi(
    partitions, utc_start, utc_end, part_filter, exclude_flex=None, *,
    target=TARGET_GHI, uncurtailed=UNCURTAILED,
    rating_basis="ac_capacity_kw", voltage_aggregation="avg",
    flex_selection="exclude",
):
    data = site_agg_cte(
        partitions,
        utc_start,
        utc_end,
        part_filter,
        exclude_flex=exclude_flex,
        with_reactive=False,
        flex_selection=flex_selection,
        voltage_aggregation=voltage_aggregation,
    )

    rating_col = capacity_column(rating_basis)
    max_p = vw_max_p_sql("V", rating_col)
    tol = tol_kw_sql(rating_col)
    unc_pred = uncurtailed_partition_predicate(partitions, alias="a")

    return f"""
    INSERT INTO {target}
    WITH
    {data},

    -- Join the Stage 1 GHI counterfactual. This must be at most one row per
    -- (site_id, t_stamp); the notebook checks that dependency before Stage 2.
    limits AS (
        SELECT
            d.site_id,
            d.t_stamp,
            d.P_kW,
            d.V,
            d.ac_capacity_kw,
            ({max_p}) + {tol} AS max_P_volt_watt,
            a.uncurtailed_P
        FROM data d
        LEFT JOIN (
            SELECT
                a.site_id,
                a.t_stamp,
                a.uncurtailed_P
            FROM {uncurtailed} a
            WHERE {unc_pred}
        ) a
          ON d.site_id = a.site_id
         AND d.t_stamp = a.t_stamp
    ),

    scored AS (
        SELECT
            site_id,
            P_kW,
            V,
            uncurtailed_P,
            max_P_volt_watt,
            {temporal_cols('t_stamp')},

            -- A Volt-Watt verdict is assessable when:
            --
            --   1. actual P already exceeds the permitted limit, which is a
            --      definite violation even without a counterfactual; or
            --
            --   2. the GHI counterfactual exceeds the limit, showing that
            --      sufficient solar power was available to test the response.
            --
            -- Missing counterfactual plus actual P at/below the limit is
            -- unknown, not evidence of conformity.
            CASE
                WHEN V > {VW_V1}
                     AND P_kW IS NOT NULL
                     AND P_kW > max_P_volt_watt
                THEN P_kW - max_P_volt_watt

                WHEN V > {VW_V1}
                     AND P_kW IS NOT NULL
                     AND uncurtailed_P > max_P_volt_watt
                THEN 0

                ELSE NULL
            END AS nonconformance_voltwattghi,

            -- Volt-Watt-attributed curtailment:
            -- sufficient counterfactual power was available, while actual P
            -- remained at or below the permitted Volt-Watt limit.
            CASE
                WHEN V > {VW_V1}
                     AND P_kW IS NOT NULL
                     AND uncurtailed_P > max_P_volt_watt
                     AND P_kW <= max_P_volt_watt
                THEN uncurtailed_P - P_kW

                ELSE NULL
            END AS curtailment_voltwattghi,

            CASE
                WHEN V > {VW_V1}
                     AND P_kW IS NOT NULL
                     AND (
                         P_kW > max_P_volt_watt
                         OR uncurtailed_P > max_P_volt_watt
                     )
                THEN 1
                ELSE 0
            END AS assessable,

            -- Track every exposed interval for which Stage 1 supplied no
            -- counterfactual, including intervals that remain unassessable.
            CASE
                WHEN V > {VW_V1}
                     AND uncurtailed_P IS NULL
                THEN 1
                ELSE 0
            END AS null_uncurtailed_P

        FROM limits
    )

    -- Column order MUST match create_table_ghi().
    SELECT
        site_id,
        day_aest AS day,
        day_night,

        sum(P_kW) AS P_kW_sum,

        sum(nonconformance_voltwattghi)
            AS nonconformance_voltwattghi_sum,

        sum(curtailment_voltwattghi)
            AS curtailment_voltwattghi_sum,

        sum(
            CASE
                WHEN nonconformance_voltwattghi > 0 THEN 1
                ELSE 0
            END
        ) AS nonconformance_voltwattghi_count,

        sum(
            CASE
                WHEN curtailment_voltwattghi > 0 THEN 1
                ELSE 0
            END
        ) AS curtailment_voltwattghi_count,

        sum(assessable)
            AS assessable_count,

        sum(null_uncurtailed_P)
            AS null_uncurtailed_P_count,

        count(*)
            AS all_intervals_count,

        sum(
            CASE
                WHEN V > {VW_V1} THEN 1
                ELSE 0
            END
        ) AS total_count,

        '{rating_basis}' AS rating_basis,
        '{voltage_aggregation}' AS voltage_aggregation,
        '{flex_selection}' AS flex_selection,

        year_aest AS year,
        month_aest AS month

    FROM scored
    GROUP BY
        year_aest,
        month_aest,
        day_aest,
        day_night,
        site_id
    """


# ===========================================================================
# Runners
# ===========================================================================
def run_months_basic(aq, database, year, months, n_parts=8, parts=None,
                     exclude_flex=None, **run_options):
    return run_months(aq, database, _insert_sql_basic, year, months,
                      n_parts=n_parts, parts=parts, exclude_flex=exclude_flex,
                      **run_options)


def run_months_ghi(aq, database, year, months, n_parts=8, parts=None,
                   exclude_flex=None, **run_options):
    return run_months(aq, database, _insert_sql_ghi, year, months,
                      n_parts=n_parts, parts=parts, exclude_flex=exclude_flex,
                      **run_options)


def preview_sql(which="basic", year=2024, month=1, n_parts=8, part=0,
                exclude_flex=None, **run_options):
    """Return the INSERT SQL for one slice WITHOUT running it."""
    utc_start, utc_end, partitions = aest_month_window(year, month)
    builder = _insert_sql_basic if which == "basic" else _insert_sql_ghi
    return builder(
        partitions, utc_start, utc_end,
        f"site_id % {n_parts} = {part}",
        exclude_flex=exclude_flex,
        **run_options,
    )


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


def validate_basic(aq, database, target=TARGET_BASIC):
    shape = _shape(aq, database, target)
    dupes = _dupes(aq, database, target)
    coh = aq(f"""
        SELECT sum(CASE WHEN nonconformance_voltwatt_sum < 0 THEN 1 ELSE 0 END) AS neg_rows,
               sum(CASE WHEN total_count > all_intervals_count THEN 1 ELSE 0 END) AS count_inversion_rows,
               sum(CASE WHEN nonconformance_voltwatt_count > total_count THEN 1 ELSE 0 END) AS impossible_rows
        FROM {target}
    """, database=database)
    print("Rows / sites / days per AEST month:")
    print(shape.to_string(index=False))
    print(f"\nDuplicate keys (MUST be 0): {int(dupes['n_dupe_keys'].iloc[0])}")
    print("\nCoherence (all MUST be 0):")
    print(coh.to_string(index=False))
    return shape, dupes, coh


def validate_ghi(aq, database, target=TARGET_GHI):
    shape = _shape(aq, database, target)
    dupes = _dupes(aq, database, target)
    coh = aq(f"""
        SELECT sum(CASE WHEN curtailment_voltwattghi_sum < 0 THEN 1 ELSE 0 END) AS neg_curtailment_rows,
               sum(CASE WHEN assessable_count > total_count THEN 1 ELSE 0 END)  AS impossible_rows,
               sum(CASE WHEN total_count > all_intervals_count THEN 1 ELSE 0 END) AS count_inversion_rows
        FROM {target}
    """, database=database)
    cover = aq(f"""
        SELECT sum(total_count)              AS exposed_intervals,
               sum(null_uncurtailed_P_count) AS no_counterfactual,
               round(100.0 * sum(null_uncurtailed_P_count)
                     / nullif(sum(total_count), 0), 2) AS pct_missing
        FROM {target}
    """, database=database)
    print("Rows / sites / days per AEST month:")
    print(shape.to_string(index=False))
    print(f"\nDuplicate keys (MUST be 0): {int(dupes['n_dupe_keys'].iloc[0])}")
    print("\nCoherence (all MUST be 0):")
    print(coh.to_string(index=False))
    print("\nCounterfactual coverage above 253 V:")
    print(cover.to_string(index=False))
    return shape, dupes, coh, cover


def cross_check(aq, database, target_basic=TARGET_BASIC, target_ghi=TARGET_GHI):
    """
    Confirm that the basic and GHI-aware Volt-Watt tables contain the same
    site-day keys and were built from the same underlying telemetry intervals.
    """
    df = aq(f"""
        SELECT
            sum(CASE WHEN b.site_id IS NULL THEN 1 ELSE 0 END)
                AS keys_missing_from_basic,

            sum(CASE WHEN g.site_id IS NULL THEN 1 ELSE 0 END)
                AS keys_missing_from_ghi,

            sum(CASE
                    WHEN b.site_id IS NOT NULL
                     AND g.site_id IS NOT NULL
                     AND b.total_count <> g.total_count
                    THEN 1 ELSE 0
                END) AS total_count_mismatches,

            sum(CASE
                    WHEN b.site_id IS NOT NULL
                     AND g.site_id IS NOT NULL
                     AND b.all_intervals_count <> g.all_intervals_count
                    THEN 1 ELSE 0
                END) AS all_intervals_mismatches,

            sum(CASE
                    WHEN b.site_id IS NOT NULL
                     AND g.site_id IS NOT NULL
                     AND abs(b.P_kW_sum - g.P_kW_sum) > 1e-6
                    THEN 1 ELSE 0
                END) AS power_sum_mismatches

        FROM {target_basic} b
        FULL OUTER JOIN {target_ghi} g
          ON b.site_id = g.site_id
         AND b.year = g.year
         AND b.month = g.month
         AND b.day = g.day
         AND b.day_night = g.day_night
    """, database=database)

    print("Basic versus GHI Volt-Watt cross-check (all MUST be 0):")
    print(df.to_string(index=False))
    return df
