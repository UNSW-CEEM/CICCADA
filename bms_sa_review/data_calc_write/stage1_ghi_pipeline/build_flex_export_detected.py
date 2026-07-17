"""Reproduce Original telemetry-based ``flex_export_detected`` heuristic.

This builder does not append a column to ``ts``. 
The flag is site metadata: one value is joined to many telemetry intervals when a cohort is selected.

Detection rule (legacy ``Flex_export_sites.ipynb`` cell 4)
--------------------------------------------------------------------------
A site-day is flagged when at least one ordered sequence of the current plus
four previous available readings has a power range below 0.004 kW, while:

* current P is at least 20% of ``ac_capacity_kw``;
* current P is no more than 75% of that site's daily maximum P;
* ``ac_capacity_kw`` is at least 5 kW; and
* all four previous readings exist.

At a complete five-minute cadence this is approximately a 25-minute plateau.
The legacy rule does not verify that the five rows are exactly five minutes
apart, so this implementation deliberately does not add a gap rule.

The heuristic is not ground truth. False positives can arise from fixed export
limits, ordinary inverter/battery/load control, telemetry quantisation, flat
weather or missing-row sequences. False negatives can arise from variable
limits, shorter events, noisy measurements or no suitable operating day.

Safe workflow
--------------------------------------------------------------------------
``create_table`` + ``run`` build an auditable side table.

This module never updates ``meta_up23c``. Candidate detections remain in the
separate target table and can be compared with the existing metadata flag.
"""

TARGET = "flex_export_detection_v2"
WAREHOUSE = "Trino-Warehouse/solar_analytics"
METHOD_VERSION = "original_plateau_v1"

MIN_CAPACITY_KW = 5.0
MIN_POWER_CAPACITY_FRAC = 0.20
MAX_POWER_DAILY_MAX_FRAC = 0.75
PLATEAU_SPREAD_KW = 0.004
N_READINGS = 5


def create_table(aq, database, target=TARGET):
    aq(f"DROP TABLE IF EXISTS {target}", database=database)
    aq(f"""
        CREATE TABLE {target} (
            site_id BIGINT,
            flex_export_detected BOOLEAN,
            detected_day_count BIGINT,
            first_detected_day DATE,
            last_detected_day DATE,
            method_version STRING,
            analysis_start_year INT,
            analysis_end_year INT
        )
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{target}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {target}"


def _year_predicate(years, alias="ts"):
    years = sorted({int(year) for year in years})
    if not years:
        raise ValueError("years must not be empty")
    return years, " OR ".join(f"{alias}.year = {year}" for year in years)


def build_sql(years=(2024, 2025), target=TARGET):
    years, year_predicate = _year_predicate(years)
    return f"""
    INSERT INTO {target}
    WITH metadata AS (
        SELECT circuit_id,
               max(site_id) AS site_id,
               max(circuit_polarity) AS circuit_polarity,
               max(ac_capacity_kw) AS ac_capacity_kw
        FROM meta_up23c
        WHERE is_pv = True
          AND ac_capacity_kw > 0
        GROUP BY circuit_id
    ),
    data AS (
        SELECT m.site_id,
               ts.t_stamp,
               sum(ts.power * m.circuit_polarity) / 1000 AS P_kw,
               max(m.ac_capacity_kw) AS ac_capacity_kw
        FROM ts
        JOIN metadata m ON ts.circuit_id = m.circuit_id
        WHERE ({year_predicate})
          AND ts.is_pv = True
        GROUP BY m.site_id, ts.t_stamp
    ),
    daily_max AS (
        SELECT site_id,
               CAST(date_trunc('day', t_stamp + interval '10' hour) AS DATE) AS day_aest,
               max(P_kw) AS max_P_kw
        FROM data
        GROUP BY site_id, date_trunc('day', t_stamp + interval '10' hour)
    ),
    lagged AS (
        SELECT d.*,
               lag(P_kw, 1) OVER (PARTITION BY site_id, date_trunc('day', t_stamp + interval '10' hour) ORDER BY t_stamp) AS p_1,
               lag(P_kw, 2) OVER (PARTITION BY site_id, date_trunc('day', t_stamp + interval '10' hour) ORDER BY t_stamp) AS p_2,
               lag(P_kw, 3) OVER (PARTITION BY site_id, date_trunc('day', t_stamp + interval '10' hour) ORDER BY t_stamp) AS p_3,
               lag(P_kw, 4) OVER (PARTITION BY site_id, date_trunc('day', t_stamp + interval '10' hour) ORDER BY t_stamp) AS p_4
        FROM data d
    ),
    candidate_rows AS (
        SELECT l.site_id,
               dm.day_aest,
               greatest(P_kw, p_1, p_2, p_3, p_4)
                   - least(P_kw, p_1, p_2, p_3, p_4) AS spread_kw
        FROM lagged l
        JOIN daily_max dm
          ON l.site_id = dm.site_id
         AND CAST(date_trunc('day', l.t_stamp + interval '10' hour) AS DATE) = dm.day_aest
        WHERE l.ac_capacity_kw >= {MIN_CAPACITY_KW}
          AND l.P_kw >= {MIN_POWER_CAPACITY_FRAC} * l.ac_capacity_kw
          AND l.P_kw <= {MAX_POWER_DAILY_MAX_FRAC} * dm.max_P_kw
          AND p_1 IS NOT NULL AND p_2 IS NOT NULL
          AND p_3 IS NOT NULL AND p_4 IS NOT NULL
    ),
    candidate_days AS (
        SELECT site_id, day_aest
        FROM candidate_rows
        GROUP BY site_id, day_aest
        HAVING min(spread_kw) < {PLATEAU_SPREAD_KW}
    ),
    detected AS (
        SELECT site_id,
               count(*) AS detected_day_count,
               min(day_aest) AS first_detected_day,
               max(day_aest) AS last_detected_day
        FROM candidate_days
        GROUP BY site_id
    ),
    all_sites AS (
        SELECT DISTINCT site_id
        FROM meta_up23c
        WHERE is_pv = True
    )
    SELECT s.site_id,
           d.site_id IS NOT NULL AS flex_export_detected,
           coalesce(d.detected_day_count, 0) AS detected_day_count,
           d.first_detected_day,
           d.last_detected_day,
           '{METHOD_VERSION}' AS method_version,
           {years[0]} AS analysis_start_year,
           {years[-1]} AS analysis_end_year
    FROM all_sites s
    LEFT JOIN detected d ON s.site_id = d.site_id
    """


def run(aq, database, years=(2024, 2025), target=TARGET):
    aq(build_sql(years=years, target=target), database=database)
    return validate(aq, database, target=target)


def validate(aq, database, target=TARGET):
    result = aq(f"""
        SELECT flex_export_detected,
               count(*) AS n_sites,
               sum(detected_day_count) AS detected_site_days
        FROM {target}
        GROUP BY flex_export_detected
        ORDER BY flex_export_detected
    """, database=database)
    print(result.to_string(index=False))
    return result


#### TO AVOID ANY ACCIDENTS, KEEP THE FOLLOWING COMMENTED OUT

'''
def write_back_meta(aq, database, target=TARGET, *, confirmation=None):
    """Replace ``meta_up23c.flex_export_detected`` from the audit table.

    This reproduces the effect of original cells 7-9 but only after the caller
    explicitly acknowledges that every metadata row will be updated.
    """
    required = "UPDATE_META_UP23C_FROM_AUDITED_FLEX_TABLE"
    if confirmation != required:
        raise ValueError(f"Pass confirmation={required!r} to permit metadata UPDATE")

    aq(f"""
        UPDATE meta_up23c
        SET flex_export_detected = CASE
            WHEN site_id IN (
                SELECT site_id FROM {target} WHERE flex_export_detected = True
            ) THEN True
            ELSE False
        END
    """, database=database)
    return "meta_up23c.flex_export_detected replaced from audited detection table"
'''