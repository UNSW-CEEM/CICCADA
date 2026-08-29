# This script is to be run on the EC2 instance for daily LSO anti-islanding
# conformance results for the assessed Solar Analytics site cohort.
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

from core.phase_a import run_phase_a_for_site
from core.phase_b import run_phase_b_for_site
from core.site_day_signals import build_site_day_signals
from solar_analytics_workflow.config import (
    DAY_ANALYSIS_START,
    DAY_END,
    DAY_EXTRACTION_START,
    LOCAL_TIMEZONE,
    PRIMARY_PHASE_B_METHOD,
)
from solar_analytics_workflow.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertWToKw,
    deduplicateMeasurements,
)
from solar_analytics_workflow.preprocessing import STATE_TIMEZONES
from solar_analytics_workflow.rated_capacity import add_s_rated_capacity
from solar_analytics_workflow.site_day_filtering import (
    summarize_solar_analytics_day_eligibility,
)
from solar_analytics_workflow.site_preparation import (
    calculate_site_day_voltage_signals,
    extract_site_day,
    map_circuit_data_to_site,
    trim_site_day_analysis_window,
)
from solar_analytics_workflow.trino.trino_connection_on_ec2 import (
    engine,
    iceberg_exec,
)

DAILY_CONFORMANCE_SITE_BATCH_SIZE = 10

iceberg_exec("DROP TABLE IF EXISTS lso_anti_islanding_conformance_daily")
iceberg_exec("""
    CREATE TABLE lso_anti_islanding_conformance_daily (
        year INTEGER,
        month INTEGER,
        day INTEGER,
        site_id BIGINT,
        threshold_method VARCHAR,
        los_responsible_count BIGINT,
        los_compliant_count BIGINT,
        ov1_responsible_count BIGINT,
        ov1_compliant_count BIGINT
    )
    WITH (
        format = 'PARQUET',
        partitioning = ARRAY['year', 'month'],
        sorted_by = ARRAY['site_id', 'day']
    )
""")


def _iter_site_timeseries_batches(engine, assessed_sites, circuit_data):
    postcode_buckets = assessed_sites.get_column("postcode_bucket").unique(
        maintain_order=True
    )

    for postcode_bucket in postcode_buckets:
        bucket_sites = assessed_sites.filter(
            pl.col("postcode_bucket") == postcode_bucket
        )
        batch_timezones = bucket_sites.get_column("timezone").unique(
            maintain_order=True
        )

        for batch_timezone in batch_timezones:
            timezone_sites = bucket_sites.filter(pl.col("timezone") == batch_timezone)

            for batch_start in range(
                0,
                timezone_sites.height,
                DAILY_CONFORMANCE_SITE_BATCH_SIZE,
            ):
                batch_sites = timezone_sites.slice(
                    batch_start,
                    DAILY_CONFORMANCE_SITE_BATCH_SIZE,
                )
                batch_site_ids = batch_sites.get_column("site_id")

                print(
                    "Querying time series: "
                    f"postcode_bucket={postcode_bucket} "
                    f"timezone={batch_timezone} "
                    f"sites={batch_sites.height}",
                    flush=True,
                )

                batch_circuit_data = circuit_data.filter(
                    pl.col("site_id").is_in(batch_site_ids.implode())
                )
                batch_circuit_ids = batch_circuit_data.get_column("c_id").unique(
                    maintain_order=True
                )
                batch_postcodes = batch_sites.get_column("postcode").unique(
                    maintain_order=True
                )

                circuit_ids = ", ".join(batch_circuit_ids.cast(pl.String).to_list())
                postcodes = ", ".join(
                    batch_postcodes.cast(pl.Int64).cast(pl.String).to_list()
                )

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
    assessed_sites_query = f"""
    SELECT DISTINCT site_id
    FROM iceberg.solar_analytics_iceberg.lso_anti_islanding_conformance
    WHERE threshold_method = '{PRIMARY_PHASE_B_METHOD}'
        AND assessment_status IN ('conformant', 'non-conformant')
    """

    assessed_site_ids = (
        pl.read_database(query=assessed_sites_query, connection=engine)
        .get_column("site_id")
        .cast(pl.Int64)
        .unique(maintain_order=True)
    )
    assessed_site_ids_sql = ", ".join(assessed_site_ids.cast(pl.String).to_list())

    site_query = f"""
    SELECT DISTINCT
        m.site_id,
        m.state,
        m.postcode,
        system.bucket(CAST(m.postcode AS INTEGER), 16) AS postcode_bucket,
        m.ac_capacity_kw,
        m.s_99
    FROM iceberg.solar_analytics_iceberg.meta_up23c AS m
    WHERE m.site_id IN ({assessed_site_ids_sql})
        AND m.inverter_count = 1
    """

    circuit_query = f"""
    SELECT
        c.site_id,
        c.circuit_id,
        c.circuit_polarity
    FROM iceberg.solar_analytics_iceberg.circuits AS c
    WHERE c.site_id IN ({assessed_site_ids_sql})
        AND c.is_pv = TRUE
    """

    site_data = pl.read_database(query=site_query, connection=engine).unique(
        subset=["site_id"],
        keep="first",
        maintain_order=True,
    )
    site_data = add_s_rated_capacity(site_data)

    circuit_data = pl.read_database(
        query=circuit_query,
        connection=engine,
    ).rename(
        {
            "circuit_id": "c_id",
            "circuit_polarity": "polarity",
        }
    )

    pv_circuit_counts = circuit_data.group_by("site_id").agg(
        pl.col("c_id").n_unique().alias("pv_circuit_count")
    )
    eligible_site_ids = pv_circuit_counts.filter(
        pl.col("pv_circuit_count").is_between(1, 3)
    )["site_id"]

    assessed_sites = site_data.filter(
        pl.col("site_id").is_in(eligible_site_ids.implode())
    ).with_columns(
        pl.col("state")
        .replace_strict(STATE_TIMEZONES, default=LOCAL_TIMEZONE)
        .alias("timezone")
    )

    total_sites = assessed_sites.height
    print(f"Assessed sites to process: {total_sites}", flush=True)

    site_idx = 0
    for (
        batch_sites,
        batch_circuit_data,
        batch_timeseries_data,
    ) in _iter_site_timeseries_batches(
        engine,
        assessed_sites,
        circuit_data,
    ):
        batch_daily_conformance_frames = []

        for site in batch_sites.iter_rows(named=True):
            site_idx += 1
            site_circuit_ids = (
                batch_circuit_data.filter(pl.col("site_id") == site["site_id"])
                .get_column("c_id")
                .unique(maintain_order=True)
            )
            site_timeseries_data = batch_timeseries_data.filter(
                pl.col("c_id").is_in(site_circuit_ids.implode())
            )

            print(
                f"[{site_idx}/{total_sites}] Processing site {site['site_id']}",
                flush=True,
            )

            if site_timeseries_data.is_empty():
                continue

            site_timeseries_data = site_timeseries_data.lazy().filter(
                pl.col("utc_tstamp").is_not_null()
            )
            site_timeseries_data = convertWToKw(site_timeseries_data)
            site_timeseries_data = deduplicateMeasurements(site_timeseries_data)

            site_timezone = STATE_TIMEZONES.get(site["state"], LOCAL_TIMEZONE)
            site_timeseries_data = site_timeseries_data.with_columns(
                pl.lit(site_timezone).alias("timezone")
            )
            site_timeseries_data = addLocalTStamp(site_timeseries_data)
            site_timeseries_data = addValidVoltage(site_timeseries_data)
            site_timeseries_data = addPolarityToPower(
                site_timeseries_data,
                circuit_data,
            )
            site_timeseries_data = site_timeseries_data.select(
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

            eligible_analysis_days = []
            local_dates = (
                site_timeseries_data.select(
                    pl.col("local_tstamp").dt.date().alias("local_date")
                )
                .drop_nulls()
                .unique()
                .sort("local_date")["local_date"]
                .to_list()
            )

            for day in local_dates:
                start_day = datetime.combine(day, DAY_EXTRACTION_START)
                end_day = datetime.combine(day, DAY_END)

                site_day_long = extract_site_day(
                    site_timeseries_data,
                    start_day,
                    end_day,
                )
                if site_day_long.is_empty():
                    continue

                site_day_df = map_circuit_data_to_site(
                    site_day_long,
                    site["site_id"],
                )
                prepared_day_df = calculate_site_day_voltage_signals(
                    site_day_df,
                    voltage_prefix="voltage_valid",
                )
                analysis_day_df = trim_site_day_analysis_window(
                    prepared_day_df,
                    DAY_ANALYSIS_START,
                    DAY_END,
                )
                eligibility = summarize_solar_analytics_day_eligibility(analysis_day_df)

                if not eligibility["eligible"]:
                    continue

                eligible_analysis_days.append(
                    {
                        "analysis_date": day,
                        "analysis_frame": analysis_day_df,
                    }
                )

            if not eligible_analysis_days:
                continue

            capacity_row = site_data.filter(
                pl.col("site_id") == site["site_id"]
            ).select("s_rated")
            s_rated = None if capacity_row.is_empty() else capacity_row["s_rated"][0]
            if s_rated is None:
                print(
                    f"No S_rated for site {site['site_id']}; skipping.",
                    flush=True,
                )
                continue

            prepared_site_days = [
                {
                    "analysis_date": day_info["analysis_date"],
                    "signal_frame": build_site_day_signals(
                        day_info["analysis_frame"], s_rated
                    ),
                }
                for day_info in eligible_analysis_days
            ]

            phase_a_result = run_phase_a_for_site(
                site["site_id"],
                prepared_site_days,
                s_rated,
            )
            phase_b_result = run_phase_b_for_site(
                site["site_id"],
                prepared_site_days,
                site_thresholds=phase_a_result["site_thresholds"],
                threshold_method=PRIMARY_PHASE_B_METHOD,
            )

            site_compliance_timestamp_detail = phase_b_result[
                "site_compliance_timestamp_detail"
            ]
            if site_compliance_timestamp_detail.is_empty():
                continue

            daily_conformance = (
                site_compliance_timestamp_detail.group_by(["event_day", "site_id"])
                .agg(
                    [
                        pl.col("los_responsible")
                        .sum()
                        .cast(pl.Int64)
                        .alias("los_responsible_count"),
                        pl.col("los_compliant")
                        .sum()
                        .cast(pl.Int64)
                        .alias("los_compliant_count"),
                        pl.col("ov1_responsible")
                        .sum()
                        .cast(pl.Int64)
                        .alias("ov1_responsible_count"),
                        pl.col("ov1_compliant")
                        .sum()
                        .cast(pl.Int64)
                        .alias("ov1_compliant_count"),
                    ]
                )
                .with_columns(
                    [
                        pl.col("event_day").dt.year().cast(pl.Int32).alias("year"),
                        pl.col("event_day").dt.month().cast(pl.Int32).alias("month"),
                        pl.col("event_day").dt.day().cast(pl.Int32).alias("day"),
                        pl.lit(PRIMARY_PHASE_B_METHOD).alias("threshold_method"),
                    ]
                )
                .select(
                    [
                        "year",
                        "month",
                        "day",
                        "site_id",
                        "threshold_method",
                        "los_responsible_count",
                        "los_compliant_count",
                        "ov1_responsible_count",
                        "ov1_compliant_count",
                    ]
                )
                .sort(["site_id", "year", "month", "day"])
            )
            batch_daily_conformance_frames.append(daily_conformance)

        if batch_daily_conformance_frames:
            batch_daily_conformance = pl.concat(
                batch_daily_conformance_frames,
                how="vertical",
            )
            rows_written = batch_daily_conformance.write_database(
                table_name="lso_anti_islanding_conformance_daily",
                connection=engine,
                if_table_exists="append",
                engine_options={"chunksize": 250, "method": "multi"},
            )
            print(
                "Uploaded daily conformance rows: " f"{rows_written}",
                flush=True,
            )

finally:
    engine.dispose()
