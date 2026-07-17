"""Build an auditable reconstruction of site ``S_99`` values.

The original job that populated ``meta_up23c.S_99`` is not present in this repository. 

This module therefore does not claim to reproduce its provenance,
and it never updates ``meta_up23c``. 

It estimates the 99th percentile of site-level apparent power using the same P/Q conversions as Stage 1, 
while recording the population and method beside every result.

Use this as a sensitivity input or evidence for investigating metadata, 
not as an unlabelled replacement for manufacturer ``S_rated``.

Source-unit assumptions: ``power`` is instantaneous W, ``energy_reactive`` is
five-minute kvarh, and ``circuit_polarity`` makes PV generation positive. 
The ``*12`` conversion must be changed if the provider field is already a reactive
power measurement. 

Those contracts are not available in this repository.
"""

TARGET = "s99_estimates_v1"
WAREHOUSE = "Trino-Warehouse/solar_analytics"
METHOD_VERSION = "observed_site_apparent_power_p99_v1"


def create_table(aq, database, target=TARGET):
    aq(f"DROP TABLE IF EXISTS {target}", database=database)
    aq(f"""
        CREATE TABLE {target} (
            site_id BIGINT,
            S_99_estimate DOUBLE,
            n_intervals BIGINT,
            first_t_stamp TIMESTAMP,
            last_t_stamp TIMESTAMP,
            analysis_start_year INT,
            analysis_end_year INT,
            min_active_power_fraction DOUBLE,
            method_version STRING
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


def build_sql(
    years=(2024, 2025),
    *,
    target=TARGET,
    min_active_power_fraction=0.0,
):
    """Return SQL for one reconstruction population.

    ``min_active_power_fraction`` is relative to provider ``ac_capacity_kw``.
    Keep 0.0 for the primary reconstruction. Values such as 0.05 or 0.20 are
    sensitivity variants that remove low-generation/night-time rows.
    """
    years, year_predicate = _year_predicate(years)
    fraction = float(min_active_power_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("min_active_power_fraction must be in [0, 1)")

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
    site_intervals AS (
        SELECT m.site_id,
               ts.t_stamp,
               sum(ts.power * m.circuit_polarity) / 1000 AS P_kw,
               sum(ts.energy_reactive * m.circuit_polarity) / 1000 * 12 AS Q_kvar,
               max(m.ac_capacity_kw) AS ac_capacity_kw
        FROM ts
        JOIN metadata m ON ts.circuit_id = m.circuit_id
        WHERE ({year_predicate})
          AND ts.is_pv = True
          AND ts.power IS NOT NULL
          AND ts.energy_reactive IS NOT NULL
        GROUP BY m.site_id, ts.t_stamp
    ),
    apparent_power AS (
        SELECT site_id,
               t_stamp,
               sqrt(pow(P_kw, 2) + pow(Q_kvar, 2)) AS S_kVA
        FROM site_intervals
        WHERE P_kw >= {fraction} * ac_capacity_kw
    )
    SELECT site_id,
           approx_percentile(S_kVA, 0.99) AS S_99_estimate,
           count(*) AS n_intervals,
           min(t_stamp) AS first_t_stamp,
           max(t_stamp) AS last_t_stamp,
           {years[0]} AS analysis_start_year,
           {years[-1]} AS analysis_end_year,
           {fraction} AS min_active_power_fraction,
           '{METHOD_VERSION}' AS method_version
    FROM apparent_power
    WHERE S_kVA > 0
    GROUP BY site_id
    """


def run(
    aq,
    database,
    years=(2024, 2025),
    *,
    target=TARGET,
    min_active_power_fraction=0.0,
):
    aq(build_sql(
        years,
        target=target,
        min_active_power_fraction=min_active_power_fraction,
    ), database=database)
    return validate(aq, database, target=target)


def validate(aq, database, target=TARGET):
    result = aq(f"""
        SELECT method_version,
               min_active_power_fraction,
               count(*) AS n_sites,
               min(S_99_estimate) AS min_S_99,
               approx_percentile(S_99_estimate, 0.5) AS median_S_99,
               max(S_99_estimate) AS max_S_99,
               min(n_intervals) AS min_intervals
        FROM {target}
        GROUP BY method_version, min_active_power_fraction
    """, database=database)
    print(result.to_string(index=False))
    return result
