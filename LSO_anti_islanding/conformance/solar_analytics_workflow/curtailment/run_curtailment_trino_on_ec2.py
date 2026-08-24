# This script is to be run on the EC2 instance for Solar Analytics curtailment.
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

CONFORMANCE_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_DIR = Path(__file__).resolve().parents[4]
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from solar_analytics_workflow.config import (
    DAY_END,
    DAY_EXTRACTION_START,
    LOCAL_TIMEZONE,
)
from solar_analytics_workflow.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertWToKw,
    deduplicateMeasurements,
)
from solar_analytics_workflow.preprocessing import STATE_TIMEZONES
from solar_analytics_workflow.site_preparation import (
    calculate_site_day_voltage_signals,
    extract_site_day,
    map_circuit_data_to_site,
    trim_site_day_analysis_window,
)
from solar_analytics_workflow.rated_capacity import add_s_rated_capacity
from solar_analytics_workflow.solar_paths import TRINO_OUTPUT_DIR
from solar_analytics_workflow.trino.trino_connection_on_ec2 import (
    engine,
    iceberg_exec,
)

iceberg_exec("DROP TABLE IF EXISTS v10_vinst_assessed_sites")
iceberg_exec("""CREATE TABLE IF NOT EXISTS v10_vinst_assessed_sites (
    site_id BIGINT,
    utc_tstamp TIMESTAMP(6) WITH TIME ZONE,
    local_tstamp TIMESTAMP(6),
    timezone VARCHAR,
    s_rated DOUBLE,
    v10m_avg DOUBLE,
    vinst_max DOUBLE
)
WITH (
    format = 'PARQUET',
    partitioning = ARRAY['month(utc_tstamp)'],
    sorted_by = ARRAY['site_id', 'utc_tstamp']
)
""")

iceberg_exec("DROP TABLE IF EXISTS curtailment_lso_anti_islanding")
iceberg_exec("""CREATE TABLE curtailment_lso_anti_islanding (
    year INTEGER,
    month INTEGER,
    day INTEGER,
    site_id BIGINT,
    curtailment_lso_anti_islanding_sum DOUBLE,
    curtailment_lso_anti_islanding_count BIGINT
)
WITH (
    format = 'PARQUET',
    partitioning = ARRAY['year', 'month'],
    sorted_by = ARRAY['site_id', 'day']
)
""")

TRINO_SITE_BATCH_SIZE = 10

conformance_summary = pl.read_csv(
    TRINO_OUTPUT_DIR / "solA_conformance_trino_summary_new.csv"
).filter(pl.col("assessment_status") != "unassessed")

acceptable_site_ids = (
    conformance_summary.get_column("site_id")
    .drop_nulls()
    .cast(pl.Int64)
    .unique(maintain_order=True)
)
if len(acceptable_site_ids) == 0:
    raise ValueError("No acceptable sites found in the conformance summary.")

acceptable_site_ids_sql = ", ".join(
    acceptable_site_ids.cast(pl.String).to_list()
)

SITE_QUERY = f"""
SELECT DISTINCT
    m.site_id,
    m.state,
    m.postcode,
    system.bucket(CAST(m.postcode AS INTEGER), 16) AS postcode_bucket,
    m.ac_capacity_kw,
    m.s_99
FROM iceberg.solar_analytics_iceberg.meta_up23c AS m
WHERE m.site_id IN ({acceptable_site_ids_sql})
"""

CIRCUIT_QUERY = f"""
SELECT
    c.site_id,
    c.circuit_id,
    c.circuit_polarity
FROM iceberg.solar_analytics_iceberg.circuits AS c
WHERE c.site_id IN ({acceptable_site_ids_sql})
    AND c.is_pv = TRUE
"""


def _iter_site_timeseries_batches(engine, acceptable_sites, circuit_data):
    postcode_buckets = acceptable_sites.get_column("postcode_bucket").unique(
        maintain_order=True
    )

    for postcode_bucket in postcode_buckets:
        bucket_sites = acceptable_sites.filter(
            pl.col("postcode_bucket") == postcode_bucket
        )
        batch_timezones = bucket_sites.get_column("timezone").unique(
            maintain_order=True
        )

        for batch_timezone in batch_timezones:
            timezone_sites = bucket_sites.filter(
                pl.col("timezone") == batch_timezone
            )

            for batch_start in range(
                0,
                timezone_sites.height,
                TRINO_SITE_BATCH_SIZE,
            ):
                batch_sites = timezone_sites.slice(
                    batch_start,
                    TRINO_SITE_BATCH_SIZE,
                )
                batch_site_ids = batch_sites.get_column("site_id")
                batch_circuit_data = circuit_data.filter(
                    pl.col("site_id").is_in(batch_site_ids.implode())
                )
                if batch_circuit_data.is_empty():
                    continue

                batch_circuit_ids = batch_circuit_data.get_column("c_id").unique(
                    maintain_order=True
                )
                batch_postcodes = batch_sites.get_column("postcode").unique(
                    maintain_order=True
                )
                circuit_ids = ", ".join(
                    batch_circuit_ids.cast(pl.Int64).cast(pl.String).to_list()
                )
                postcodes = ", ".join(
                    batch_postcodes.cast(pl.Int64).cast(pl.String).to_list()
                )

                print(
                    "Querying time series: "
                    f"postcode_bucket={postcode_bucket} "
                    f"timezone={batch_timezone} "
                    f"sites={batch_sites.height}",
                    flush=True,
                )

                # These predicates align with the Iceberg partitions and local window.
                batch_query = f"""
                    SELECT
                        circuit_id,
                        t_stamp,
                        power,
                        voltage
                    FROM iceberg.solar_analytics_iceberg.ts
                    WHERE system.bucket(postcode, 16) = {int(postcode_bucket)}
                        AND postcode IN ({postcodes})
                        AND circuit_id IN ({circuit_ids})
                        AND is_pv = TRUE
                        AND year IN (2024, 2025)
                        AND CAST(
                            at_timezone(
                                with_timezone(t_stamp, 'UTC'),
                                '{batch_timezone}'
                            ) AS time
                        ) BETWEEN TIME '{DAY_EXTRACTION_START.isoformat()}'
                            AND TIME '{DAY_END.isoformat()}'
                """

                batch_timeseries_data = pl.read_database(
                    query=batch_query,
                    connection=engine,
                ).rename(
                    {
                        "circuit_id": "c_id",
                        "t_stamp": "utc_tstamp",
                    }
                )
                yield batch_sites, batch_circuit_data, batch_timeseries_data


try:
    acceptable_sites = (
        pl.read_database(query=SITE_QUERY, connection=engine)
        .unique(subset=["site_id"], keep="first", maintain_order=True)
        .drop_nulls(["site_id", "postcode", "postcode_bucket"])
        .with_columns(
            pl.col("state")
            .replace_strict(STATE_TIMEZONES, default=LOCAL_TIMEZONE)
            .fill_null(LOCAL_TIMEZONE)
            .alias("timezone")
        )
    )

    acceptable_sites = add_s_rated_capacity(acceptable_sites)
    acceptable_sites = acceptable_sites.join(
        conformance_summary.select(
            ["site_id", "los_threshold_used", "ov1_threshold_used"]
        ),
        on="site_id",
        how="inner",
    )
    

    circuit_data = pl.read_database(
        query=CIRCUIT_QUERY,
        connection=engine,
    ).rename(
        {
            "circuit_id": "c_id",
            "circuit_polarity": "polarity",
        }
    )

    if circuit_data.is_empty():
        print("No PV circuits found for acceptable sites.", flush=True)
    else:
        pv_site_ids = circuit_data.get_column("site_id").unique()
        acceptable_sites = acceptable_sites.filter(
            pl.col("site_id").is_in(pv_site_ids.implode())
        )
        print(f"Acceptable sites to process: {acceptable_sites.height}", flush=True)

        site_idx = 0
        for (
            batch_sites,
            batch_circuit_data,
            batch_timeseries_data,
        ) in _iter_site_timeseries_batches(
            engine,
            acceptable_sites,
            circuit_data,
        ):
            batch_site_day_voltage_frames = []
            batch_timeseries_data = batch_timeseries_data.lazy().filter(
                pl.col("utc_tstamp").is_not_null()
            )
            batch_timeseries_data = convertWToKw(batch_timeseries_data)
            batch_timeseries_data = deduplicateMeasurements(batch_timeseries_data)
            batch_timeseries_data = batch_timeseries_data.with_columns(
                pl.lit(batch_sites.get_column("timezone")[0]).alias("timezone")
            )
            batch_timeseries_data = addLocalTStamp(batch_timeseries_data)
            batch_timeseries_data = addValidVoltage(batch_timeseries_data)
            batch_timeseries_data = addPolarityToPower(
                batch_timeseries_data,
                batch_circuit_data,
            )
            batch_timeseries_data = batch_timeseries_data.select(
                [
                    "c_id",
                    "timezone",
                    "utc_tstamp",
                    "local_tstamp",
                    "power",
                    "voltage",
                    "voltage_valid",
                ]
            ).collect(engine="streaming")
            if batch_timeseries_data.is_empty():
                continue

            for site in batch_sites.iter_rows(named=True):
                site_idx += 1
                site_circuit_ids = (
                    batch_circuit_data.filter(
                        pl.col("site_id") == site["site_id"]
                    )
                    .get_column("c_id")
                    .unique(maintain_order=True)
                )
                site_timeseries_data = batch_timeseries_data.filter(
                    pl.col("c_id").is_in(site_circuit_ids.implode())
                )

                print(
                    f"[{site_idx}/{acceptable_sites.height}] "
                    f"Processing site {site['site_id']}",
                    flush=True,
                )
                if site_timeseries_data.is_empty():
                    continue

                local_dates = (
                    site_timeseries_data.select(
                        pl.col("local_tstamp").dt.date().alias("local_date")
                    )
                    .drop_nulls()
                    .unique()
                    .sort("local_date")
                    .get_column("local_date")
                    .to_list()
                )

                for day in local_dates:
                    site_day_long = extract_site_day(
                        site_timeseries_data,
                        datetime.combine(day, DAY_EXTRACTION_START),
                        datetime.combine(day, DAY_END),
                    )
                    if site_day_long.is_empty():
                        continue

                    site_day_data = map_circuit_data_to_site(
                        site_day_long,
                        site["site_id"],
                    )
                    site_day_voltage_signals = calculate_site_day_voltage_signals(
                        site_day_data,
                        voltage_prefix="voltage_valid",
                    )
                    site_day_voltage_signals = trim_site_day_analysis_window(
                        site_day_voltage_signals
                    ).select(
                        [
                            "site_id",
                            "utc_tstamp",
                            "local_tstamp",
                            pl.lit(site["timezone"]).alias("timezone"),
                            pl.lit(site["s_rated"]).alias("s_rated"),
                            "v10m_avg",
                            "vinst_max",
                        ]
                    )
                    batch_site_day_voltage_frames.append(site_day_voltage_signals)

            if batch_site_day_voltage_frames:
                batch_signals = pl.concat(batch_site_day_voltage_frames, how="vertical").with_columns(
                    pl.col("utc_tstamp").cast(
                        pl.Datetime(time_unit="us", time_zone="UTC")
                    )
                )
                if batch_signals.is_empty():
                    continue
                batch_signals.write_database(
                    table_name="v10_vinst_assessed_sites",
                    connection=engine,
                    if_table_exists="append",
                    engine_options={"chunksize": 250, "method": "multi"},
                )

                batch_signal_site_ids = batch_signals.get_column("site_id").unique()
                batch_signal_site_ids_sql = ", ".join(
                    batch_signal_site_ids.cast(pl.Int64).cast(pl.String).to_list()
                )
                batch_uncurtailed_query = f"""
                    SELECT
                        site_id,
                        with_timezone(t_stamp, 'UTC') AS utc_tstamp,
                        P_kw AS "P_kw",
                        uncurtailed_P AS "uncurtailed_P"
                    FROM all_uncurtailedPV_LSO
                    WHERE site_id IN ({batch_signal_site_ids_sql})
                        AND year IN (2024, 2025)
                """
                batch_uncurtailed = pl.read_database(
                    query=batch_uncurtailed_query,
                    connection=engine,
                ).with_columns(
                    pl.col("utc_tstamp").cast(
                        pl.Datetime(time_unit="us", time_zone="UTC")
                    )
                )

                batch_curtailment = (
                    batch_signals.join(
                        batch_sites.select(
                            [
                                "site_id",
                                "los_threshold_used",
                                "ov1_threshold_used",
                            ]
                        ),
                        on="site_id",
                        how="inner",
                    )
                    .join(
                        batch_uncurtailed,
                        on=["site_id", "utc_tstamp"],
                        how="inner",
                    )
                    .with_columns(
                        [
                            (
                                (pl.col("v10m_avg") >= pl.col("los_threshold_used"))
                                | (
                                    pl.col("vinst_max")
                                    >= pl.col("ov1_threshold_used")
                                )
                            )
                            .fill_null(False)
                            .alias("voltage_triggered"),
                            (0.04 * pl.col("s_rated")).alias(
                                "disconnect_limit_kw"
                            ),
                        ]
                    )
                    .with_columns(
                        (pl.col("P_kw") <= pl.col("disconnect_limit_kw"))
                        .fill_null(False)
                        .alias("power_at_or_below_disconnect_limit")
                    )
                    .with_columns(
                        (
                            pl.col("voltage_triggered")
                            & pl.col("power_at_or_below_disconnect_limit")
                        ).alias("curtailment_triggered")
                    )
                    .with_columns(
                        pl.when(pl.col("curtailment_triggered"))
                        .then(
                            pl.max_horizontal(
                                pl.col("uncurtailed_P") - pl.col("P_kw"),
                                pl.lit(0.0),
                            )
                        )
                        .otherwise(0.0)
                        .alias("curtailed_power_kw")
                    )
                )

                batch_daily_curtailment = (
                    batch_curtailment.with_columns(
                        [
                            pl.col("local_tstamp").dt.year().cast(pl.Int32).alias("year"),
                            pl.col("local_tstamp").dt.month().cast(pl.Int32).alias("month"),
                            pl.col("local_tstamp").dt.day().cast(pl.Int32).alias("day"),
                        ]
                    )
                    .group_by(["year", "month", "day", "site_id"])
                    .agg(
                        [
                            pl.col("curtailed_power_kw")
                            .sum()
                            .alias("curtailment_lso_anti_islanding_sum"),
                            (pl.col("curtailed_power_kw") > 0)
                            .sum()
                            .cast(pl.Int64)
                            .alias("curtailment_lso_anti_islanding_count"),
                        ]
                    )
                    .select(
                        [
                            "year",
                            "month",
                            "day",
                            "site_id",
                            "curtailment_lso_anti_islanding_sum",
                            "curtailment_lso_anti_islanding_count",
                        ]
                    )
                )
                if not batch_daily_curtailment.is_empty():
                    batch_daily_curtailment.write_database(
                        table_name="curtailment_lso_anti_islanding",
                        connection=engine,
                        if_table_exists="append",
                        engine_options={"chunksize": 250, "method": "multi"},
                    )

                print(
                    "Calculated triggered curtailment for "
                    f"{batch_curtailment.height} matched rows.",
                    flush=True,
                )
finally:
    engine.dispose()
