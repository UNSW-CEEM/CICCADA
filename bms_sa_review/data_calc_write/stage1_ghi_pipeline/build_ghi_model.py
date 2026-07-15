"""
Data calc-write pipeline: Stage 1. Step 3 of 4
===========================================

WHAT IT DOES
------------
Fits a per-site, per-time-of-day linear regression that predicts normalised
power from irradiance:

        P_norm / P_norm_cs  =  a  +  b * (GHI / GHI_cs)

with the constraint a = 1 - b (on a clear-sky day both ratios are 1, so the
line passes through (1, 1)). One (a, b, n) triple per (site_id, tod_bin).

Regressor is the RATIO GHI/GHI_cs, in 5-minute time-of-day bins. 

READS
-----
`structured_data{SUFFIX}` + `split_days{SUFFIX}` (train days only).

TRAINING FILTER
-----------------------------------------------------------------
    P_kw_norm_cs > 0.2   exclude dawn/dusk (reference too small)
    GHI > 50             exclude low-light noise
    P_kw_norm > 0.05     exclude near-zero generation
    P_kw_norm <= P_kw_norm_cs   quality gate
    V <= 253             exclude the Volt-Watt active zone
    (P_kw_norm >= 1 OR S_norm < 1.001)  exclude V-VAr-curtailed intervals
                         (inverter at full output OR not at the S-limit)

SAFE-BY-DEFAULT
---------------
Writes to `pv_ghi_norm_model{SUFFIX}` (default "_v2").
"""

from build_structured_data import TABLE_SUFFIX

SD    = f"structured_data{TABLE_SUFFIX}"
SPLIT = f"split_days{TABLE_SUFFIX}"
TARGET = f"pv_ghi_norm_model{TABLE_SUFFIX}"
WAREHOUSE = "Trino-Warehouse/solar_analytics"

TIME_BIN_MIN = 5   # 5-minute time-of-day bins (authoritative)


def create_table(aq, database):
    aq(f"DROP TABLE IF EXISTS {TARGET}", database=database)
    aq(f"""
        CREATE TABLE {TARGET} (
            site_id BIGINT,
            tod_bin STRING,
            a       DOUBLE,
            b       DOUBLE,
            n       BIGINT
        )
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{TARGET}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {TARGET}"


def run_year(aq, database, year, n_parts=1, parts=None):
    """
    Fit the model for one year.
    """
    if parts is None:
        parts = list(range(n_parts))

    results = []
    for part in parts:
        part_filter = f"site_id % {n_parts} = {part}"
        aq(f"""
            INSERT INTO {TARGET}
            WITH train_val_data AS (
                SELECT
                    site_id, actual_day, t_stamp,
                    CAST(CAST(date_trunc('minute', t_stamp + interval '10' hour)
                                        - interval '1' minute * (minute(t_stamp + interval '10' hour) % {TIME_BIN_MIN})
                                        AS TIME) AS VARCHAR) AS tod_bin,
                    GHI / GHI_cs AS x,
                    P_kw_norm / NULLIF(P_kw_norm_cs, 0.0) AS y
                FROM {SD}
                WHERE P_kw_norm_cs > 0.2 AND GHI > 50 AND P_kw_norm > 0.05
                  AND P_kw_norm <= P_kw_norm_cs
                  AND V <= 253 AND (P_kw_norm >= 1 OR S_norm < 1.001)
                  AND year = {year} AND {part_filter}
            ),
            train_data AS (
                SELECT t.*
                FROM train_val_data t
                JOIN {SPLIT} s ON t.site_id = s.site_id AND t.actual_day = s.actual_day
                WHERE s.day_type = 'train'
            ),
            model AS (
                SELECT site_id, tod_bin,
                       (1 - regr_slope(y, x)) AS a,   -- a = 1 - b  (line through (1,1))
                       regr_slope(y, x)       AS b,
                       count(*)               AS n
                FROM train_data
                GROUP BY site_id, tod_bin
            )
            SELECT * FROM model
        """, database=database)
        results.append(f"fitted year={year} part={part}/{n_parts}")
        print(results[-1])
    return results

# LEGACY-MODEL AND YEAR-POOLING NOTE
# ----------------------------------
# This function intentionally reproduces the inherited GHI-model algorithm used for the milestone 3 report:
#
#     x = GHI / GHI_cs
#     y = P_kw_norm / P_kw_norm_cs
#     b = regr_slope(y, x)
#     a = 1 - b
#
# The slope is therefore estimated using ordinary unconstrained regression and
# the intercept is subsequently set to 1-b so that a+b=1. This is retained for
# comparability with the inherited CICCADA results. It should not be described
# as the mathematically constrained least-squares fit through (1, 1).
#
# The selected years are pooled into one fit because the model key is only
# (site_id, tod_bin), with no year dimension. Do not call run_year() separately
# for multiple years and append them to the same target table, because that
# creates duplicate model keys and fans out the downstream application join.
#
# Any future constrained-regression implementation should be written to a new
# versioned model table so that legacy and revised results remain auditable.

def run(aq, database, years=(2024, 2025), n_parts=1, parts=None):
    """
    Fit the model ONCE across all years.

    The model key is (site_id, tod_bin) -- there is no year dimension. Calling
    run_year() once per year therefore inserts a SECOND row for every
    (site_id, tod_bin) that has data in both years, which fans out the join in
    build_all_uncurtailedpv and doubles the counterfactual table.
    Fit across all years in a single INSERT instead.
    """
    if parts is None:
        parts = list(range(n_parts))

    year_list = ", ".join(str(y) for y in years)
    results = []
    for part in parts:
        part_filter = f"site_id % {n_parts} = {part}"
        aq(f"""
            INSERT INTO {TARGET}
            WITH train_val_data AS (
                SELECT
                    site_id, actual_day, t_stamp,
                    CAST(CAST(date_trunc('minute', t_stamp + interval '10' hour)
                              - interval '1' minute * (minute(t_stamp + interval '10' hour) % {TIME_BIN_MIN})
                              AS TIME) AS VARCHAR) AS tod_bin,
                    GHI / GHI_cs AS x,
                    P_kw_norm / NULLIF(P_kw_norm_cs, 0.0) AS y
                FROM {SD}
                WHERE P_kw_norm_cs > 0.2 AND GHI > 50 AND P_kw_norm > 0.05
                  AND P_kw_norm <= P_kw_norm_cs
                  AND V <= 253 AND (P_kw_norm >= 1 OR S_norm < 1.001)
                  AND year IN ({year_list}) AND {part_filter}
            ),
            train_data AS (
                SELECT t.*
                FROM train_val_data t
                JOIN {SPLIT} s ON t.site_id = s.site_id AND t.actual_day = s.actual_day
                WHERE s.day_type = 'train'
            ),
            model AS (
                SELECT site_id, tod_bin,
                       (1 - regr_slope(y, x)) AS a,
                       regr_slope(y, x)       AS b,
                       count(*)               AS n
                FROM train_data
                GROUP BY site_id, tod_bin
            )
            SELECT * FROM model
        """, database=database)
        results.append(f"fitted years={year_list} part={part}/{n_parts}")
        print(results[-1])
    return results

def validate(aq, database):
    n = aq(f"SELECT count(DISTINCT site_id) AS n_sites, count(*) AS n_rows FROM {TARGET}",
           database=database)
    dupes = aq(f"""
        SELECT count(*) AS n FROM (
            SELECT site_id, tod_bin FROM {TARGET}
            GROUP BY site_id, tod_bin HAVING count(*) > 1
        )
    """, database=database)
    print(f"Model rows: {int(n['n_rows'].iloc[0]):,}  "
          f"across {int(n['n_sites'].iloc[0]):,} sites")
    print(f"Duplicate (site_id, tod_bin) keys (MUST be 0): {int(dupes['n'].iloc[0]):,}")
    # A few example fits
    sample = aq(f"""
        SELECT site_id, tod_bin, round(a,3) AS a, round(b,3) AS b, n
        FROM {TARGET}
        WHERE n > 20
        ORDER BY site_id, tod_bin
        LIMIT 8
    """, database=database)
    print("Sample fits (expect a+b near 1.0):")
    print(sample.to_string(index=False))
    return n, dupes, sample
