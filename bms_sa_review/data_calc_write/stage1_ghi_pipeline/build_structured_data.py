"""
Data calc-write pipeline: Stage 1. Step 1 of 4
=================================================

WHAT IT DOES
------------
Reads raw `ts` (telemetry) + `meta_up23c` (metadata) + `bom_nci.solar`(satellite irradiance),
Builds one clean row per site per 5-minute interval with: 
- normalised P/Q, 
- voltage, 
- GHI, and 
- an empirical clear-sky reference profile (P_kw_norm_cs, GHI_cs). 
This is the foundational feature table that the GHI model and everything downstream is built on.


DELIBERATELY NOT CHANGED HERE
-----------------------------
  * Storage partition columns `year`/`month` stay UTC-derived (year(t_stamp),
    month(t_stamp)) so this table stays ALIGNED with `ts` (which is partitioned
    on UTC year/month). 

SAFE-BY-DEFAULT
---------------
Writes to a table named `structured_data{TABLE_SUFFIX}` (default suffix
"_v2") so original `structured_data` is NOT overwritten.

USAGE (from the orchestrator notebook)
--------------------------------------
    from build_structured_data import create_table, run_slice, validate
    create_table(aq, database=SAI)                      # once
    run_slice(aq, database=SAI, year=2024, months=[1],  # small test slice
              n_parts=8, parts=[0])
    validate(aq, database=SAI)
"""

TABLE_SUFFIX = "_v2"      # change to "" only if you deliberately want to overwrite
TARGET = f"structured_data{TABLE_SUFFIX}"
WAREHOUSE = "Trino-Warehouse/solar_analytics"

# ---------------------------------------------------------------------------
# DDL create the empty target table (Iceberg, partitioned by year/month)
# ---------------------------------------------------------------------------
def create_table(aq, database):
    """Drop & recreate the empty target table. Run once before loading."""
    aq(f"DROP TABLE IF EXISTS {TARGET}", database=database)
    aq(f"""
        CREATE TABLE {TARGET} (
            site_id       BIGINT,
            t_stamp       TIMESTAMP,
            actual_day    DATE,
            actual_tod    STRING,
            V             DOUBLE,
            Q_kvar_norm   DOUBLE,
            P_kw_norm     DOUBLE,
            S_norm        DOUBLE,
            GHI           DOUBLE,
            cloud_type    INT,
            cs_day        DATE,
            cs_tod        STRING,
            P_kw_norm_cs  DOUBLE,
            GHI_cs        DOUBLE,
            cloud_type_cs INT,
            S_99          DOUBLE,
            year          INT,
            month         INT
        )
        PARTITIONED BY (year, month)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{TARGET}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {TARGET}"


# ---------------------------------------------------------------------------
# INSERT one (year, month-set, site-slice) at a time
# ---------------------------------------------------------------------------
def _insert_sql(year, month_filter, part_filter, meta_filter):
    """
    Build the INSERT ... SELECT statement for one slice.
    """
    return f"""
    INSERT INTO {TARGET}
    WITH
    -- circuit -> site aggregation ------------------------------------------
    data AS (
        SELECT
            site_id, t_stamp,
            sum(power * circuit_polarity) / 1000 / max(S_99)            AS P_kw_norm,
            sum(energy_reactive * circuit_polarity) / 1000 / max(S_99) * 12 AS Q_kvar_norm,
            max(voltage) AS V,          -- FIX R1: was avg(voltage)
            max(S_99)    AS S_99
        FROM ts
        JOIN (
            SELECT site_id, circuit_id, circuit_polarity, S_99
            FROM meta_up23c
            WHERE {meta_filter}
        ) AS m ON ts.circuit_id = m.circuit_id
        WHERE year = {year} AND {month_filter} AND {part_filter}
          AND ts.is_pv = True AND voltage > 0 AND voltage < 300
        GROUP BY site_id, t_stamp
    ),
    -- BOM satellite irradiance (10-min native) -----------------------------
    bom10min AS (
        SELECT DISTINCT time, b.latitude, b.longitude,
               surface_global_irradiance AS GHI, cloud_type
        FROM bom_nci.solar AS b
        JOIN (SELECT DISTINCT site_id, n_lat, n_long FROM meta_up23c WHERE {meta_filter}) AS m
          ON b.latitude = m.n_lat AND b.longitude = m.n_long
        WHERE year({{time_col}}) = {year}
    ),
    -- duplicate each 10-min reading into the +5-min slot -------------------
    bom5min AS (
        (SELECT time AS time_5min, latitude, longitude, GHI, cloud_type FROM bom10min)
        UNION ALL
        (SELECT date_add('minute', 5, time) AS time_5min, latitude, longitude, GHI, cloud_type FROM bom10min)
    ),
    daily_cloud AS (
        SELECT latitude, longitude,
               date_trunc('day',   time + interval '10' hour) AS day,
               date_trunc('month', time + interval '10' hour) AS month,
               sum(cloud_type) AS cloud_sum, max(GHI) AS max_GHI
        FROM bom10min
        GROUP BY 1, 2, 3, 4
    ),
    clear_sky AS (
        SELECT day, latitude, longitude
        FROM (
            SELECT day, latitude, longitude, cloud_sum, max_GHI,
                   row_number() OVER (PARTITION BY month, latitude, longitude
                                      ORDER BY cloud_sum ASC, day ASC) AS rn
            FROM daily_cloud
        )
        WHERE rn < 4 AND cloud_sum < 60 AND max_GHI > 200
    ),
    daily_site_days AS (
        SELECT n_lat, n_long, date_trunc('day', t_stamp + interval '10' hour) AS day
        FROM data d
        JOIN (SELECT DISTINCT site_id, n_lat, n_long FROM meta_up23c WHERE {meta_filter}) m
          ON d.site_id = m.site_id
        GROUP BY n_lat, n_long, date_trunc('day', t_stamp + interval '10' hour)
    ),
    nearest_clear_sky_day AS (
        SELECT dy.n_lat, dy.n_long, dy.day AS actual_day, c.day AS clear_sky_day,
               row_number() OVER (
                   PARTITION BY dy.n_lat, dy.n_long, dy.day
                   ORDER BY abs(date_diff('day', dy.day, c.day)),
                            date_diff('day', c.day, dy.day)
               ) AS rn
        FROM daily_site_days dy
        JOIN clear_sky c ON dy.n_lat = c.latitude AND dy.n_long = c.longitude
    ),
    nearest_clear_sky AS (
        SELECT n_lat, n_long, actual_day, clear_sky_day
        FROM nearest_clear_sky_day WHERE rn = 1
    ),
    nearest_cs_days AS (
        SELECT DISTINCT site_id, clear_sky_day AS cs_day
        FROM nearest_clear_sky n
        JOIN (SELECT DISTINCT site_id, n_lat, n_long FROM meta_up23c WHERE {meta_filter}) m
          ON n.n_lat = m.n_lat AND n.n_long = m.n_long
    ),
    base AS (
        SELECT d.*,
               lag(t_stamp) OVER (
                   PARTITION BY site_id, date_trunc('day', t_stamp + interval '10' hour)
                   ORDER BY t_stamp
               ) AS prev_ts
        FROM data d
    ),
    gaps AS (
        SELECT *,
               CASE WHEN prev_ts IS NULL THEN 0
                    WHEN t_stamp - prev_ts > interval '30' minute THEN 1
                    ELSE 0 END AS gap_start
        FROM base
    ),
    segments AS (
        SELECT *,
               sum(gap_start) OVER (
                   PARTITION BY site_id, date_trunc('day', t_stamp + interval '10' hour)
                   ORDER BY t_stamp ROWS UNBOUNDED PRECEDING
               ) AS segment_id
        FROM gaps
    ),
    nearest_cs_profiles AS (
        SELECT
            s.site_id,
            date_trunc('day', s.t_stamp + interval '10' hour) AS cs_day,
            CAST(date_trunc('minute', s.t_stamp + interval '10' hour)
                 - interval '1' minute * (minute(s.t_stamp + interval '10' hour) % 5)
                 AS TIME) AS cs_tod,
            approx_percentile(P_kw_norm, 0.6) OVER (
                PARTITION BY s.site_id, date_trunc('day', s.t_stamp + interval '10' hour), segment_id
                ORDER BY t_stamp ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
            ) AS P_kw_norm_cs,
            approx_percentile(GHI, 0.6) OVER (
                PARTITION BY s.site_id, date_trunc('day', s.t_stamp + interval '10' hour), segment_id
                ORDER BY t_stamp ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
            ) AS GHI_cs,
            cloud_type AS cloud_type_cs
        FROM segments s
        JOIN nearest_cs_days n
          ON s.site_id = n.site_id
         AND date_trunc('day', s.t_stamp + interval '10' hour) = n.cs_day
        JOIN (SELECT DISTINCT site_id, n_lat, n_long FROM meta_up23c WHERE {meta_filter}) m
          ON s.site_id = m.site_id
        JOIN bom5min b
          ON m.n_lat = b.latitude AND m.n_long = b.longitude AND b.time_5min = s.t_stamp
    ),
    assembled AS (
        SELECT
            d.site_id, d.t_stamp,
            CAST(date_trunc('day', d.t_stamp + interval '10' hour) AS DATE) AS actual_day,
            CAST(CAST(date_trunc('minute', d.t_stamp + interval '10' hour)
                 - interval '1' minute * (minute(d.t_stamp + interval '10' hour) % 5)
                 AS TIME) AS VARCHAR) AS actual_tod,
            d.V, d.Q_kvar_norm, d.P_kw_norm,
            sqrt(pow(d.Q_kvar_norm, 2) + pow(d.P_kw_norm, 2)) AS S_norm,
            GHI, cloud_type,
            CAST(ncs.cs_day AS DATE) AS cs_day,
            CAST(ncs.cs_tod AS VARCHAR) AS cs_tod,
            ncs.P_kw_norm_cs, ncs.GHI_cs, ncs.cloud_type_cs,
            d.S_99,
            CAST(year(d.t_stamp) AS INT)  AS year,
            CAST(month(d.t_stamp) AS INT) AS month
        FROM data d
        JOIN (SELECT DISTINCT site_id, n_lat, n_long FROM meta_up23c WHERE {meta_filter}) m
          ON d.site_id = m.site_id
        JOIN nearest_cs_profiles ncs
          ON d.site_id = ncs.site_id
         AND ncs.cs_tod = CAST(date_trunc('minute', d.t_stamp + interval '10' hour)
                               - interval '1' minute * (minute(d.t_stamp + interval '10' hour) % 5)
                               AS TIME)
        JOIN nearest_clear_sky n
          ON n.n_lat = m.n_lat AND n.n_long = m.n_long
         AND n.actual_day = date_trunc('day', d.t_stamp + interval '10' hour)
         AND n.clear_sky_day = ncs.cs_day
        JOIN bom5min b
          ON m.n_lat = b.latitude AND m.n_long = b.longitude AND b.time_5min = d.t_stamp
        WHERE abs(date_diff('day', ncs.cs_day, n.actual_day)) < 45
    )
    SELECT * FROM assembled
    """.replace("{time_col}", "time")


def run_slice(aq, database, year, months, n_parts=8, parts=None):
    """
    Load one slice at a time

    Parameters
    ----------
    year   : int          calendar year in `ts`
    months : list[int]    e.g. [1] for Jan only (test), or range(1,13) for full year
    n_parts: int          how many site-slices to split into (mod site_id)
    parts  : list[int]    which slices to run, e.g. [0] to test one slice;
                          None means all parts 0..n_parts-1
    """
    if parts is None:
        parts = list(range(n_parts))

    results = []
    for month in months:
        for part in parts:
            month_filter = f"month = {month}"
            part_filter  = f"site_id % {n_parts} = {part}"
            # FIX R2: exclude flex-export sites (matches the GHI-model cohort)
            meta_filter  = f"is_pv = True AND flex_export_detected = False AND {part_filter}"
            sql = _insert_sql(year, month_filter, part_filter, meta_filter)
            aq(sql, database=database)
            results.append(f"loaded year={year} month={month} part={part}/{n_parts}")
            print(results[-1])
    return results


def validate(aq, database):
    """sanity checks after loading."""
    n_sites = aq(f"SELECT count(DISTINCT site_id) AS n FROM {TARGET}", database=database)
    v_stats = aq(f"""
        SELECT round(min(V),1) AS v_min, round(avg(V),1) AS v_avg,
               round(max(V),1) AS v_max,
               round(avg(P_kw_norm),3) AS p_norm_avg
        FROM {TARGET}
    """, database=database)
    print(f"Distinct sites: {int(n_sites['n'].iloc[0]):,}")
    print("Voltage / P_norm sanity:")
    print(v_stats.to_string(index=False))
    return n_sites, v_stats
