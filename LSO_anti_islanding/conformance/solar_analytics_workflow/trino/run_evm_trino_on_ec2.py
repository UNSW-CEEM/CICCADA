# this script is to be run on the EC2 instance
#  for conformance analysis on solA data already on S3
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

from core.phase_a import SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA, run_phase_a_for_site
from core.phase_b import run_phase_b_for_site
from core.site_day_signals import build_site_day_signals
from solar_analytics_workflow.config import (
    DAY_ANALYSIS_START,
    DAY_END,
    DAY_EXTRACTION_START,
    LOCAL_TIMEZONE,
    PHASE_B_METHODS,
    SAVE_SITE_LEVEL_VARIOUS_VOLTAGES,
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
from solar_analytics_workflow.reporting import (
    SITE_COMPLIANCE_SCHEMA as REPORTING_SITE_COMPLIANCE_SCHEMA,
)
from solar_analytics_workflow.reporting import (
    SITE_COMPLIANCE_TIME_DISTRIBUTION_SCHEMA,
    write_method_compliance_final_table,
)
from solar_analytics_workflow.site_day_filtering import (
    summarize_solar_analytics_day_eligibility,
)
from solar_analytics_workflow.site_preparation import (
    calculate_site_day_voltage_signals,
    extract_site_day,
    map_circuit_data_to_site,
    trim_site_day_analysis_window,
)
from solar_analytics_workflow.solar_paths import TRINO_OUTPUT_DIR
from solar_analytics_workflow.trino.trino_connection_on_ec2 import (
    engine,
    iceberg_exec,
)

EVM_TRINO_SITE_BATCH_SIZE = 10  # num sites queried at once

# these are the columns for conformance results that will be pushed to trino
# and utilised for grafana plotting
SITE_COMPLIANCE_SCHEMA = {
    **REPORTING_SITE_COMPLIANCE_SCHEMA,
    "disconnect_supported_assessment_status": pl.Utf8,
}
site_compliance_rows = []
site_compliance_time_distribution_rows = []
site_level_various_voltage_rows = []

# Select distinct site-level metadata from the eligible inverter cohort.
SITE_QUERY = """
SELECT DISTINCT
    m.site_id,
    m.state,
    m.postcode,
    system.bucket(CAST(m.postcode AS INTEGER), 16) AS postcode_bucket,
    m.ac_capacity_kw,
    m.s_99
FROM iceberg.solar_analytics_iceberg.meta_up23c AS m
WHERE m.inverter_count = 1
-- LIMIT 10
"""


def _iter_site_timeseries_batches(engine, eligible_sites, circuit_data):
    # This generator receives the complete eligible-site cohort once. It groups
    # and downloads/queries sites in batches, then yields one complete batch to
    # the existing processing loop.

    # Preserve the order in which Iceberg postcode buckets first appear.
    postcode_buckets = eligible_sites.get_column("postcode_bucket").unique(
        maintain_order=True
    )

    for postcode_bucket in postcode_buckets:
        bucket_sites = eligible_sites.filter(
            pl.col("postcode_bucket") == postcode_bucket
        )

        # A postcode hash bucket can contain sites from multiple Australian
        # timezones. Keep each query timezone-homogeneous so its UTC-to-local
        # analysis-window predicate is correct for every site in the batch.
        batch_timezones = bucket_sites.get_column("timezone").unique(
            maintain_order=True
        )
        for batch_timezone in batch_timezones:
            timezone_sites = bucket_sites.filter(pl.col("timezone") == batch_timezone)

            # Keep each Trino result small enough for the EC2 instance to hold
            # one complete batch in memory while its sites are processed.
            site_batch_size = EVM_TRINO_SITE_BATCH_SIZE
            for batch_start in range(
                0,
                timezone_sites.height,
                site_batch_size,
            ):
                batch_sites = timezone_sites.slice(
                    batch_start,
                    site_batch_size,
                )
                batch_site_ids = batch_sites.get_column("site_id")

                print(
                    "Querying time series: "
                    f"postcode_bucket={postcode_bucket} "
                    f"timezone={batch_timezone} "
                    f"sites={batch_sites.height}",
                    flush=True,
                )

                # Use the selected site IDs to include every eligible PV circuit
                # belonging to this configured site batch.
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

                # Retrieve the 2024 and 2025 measurements needed by the
                # configured local conformance extraction window. year, is_pv,
                # and postcode bucket predicates align with the Iceberg
                # partition definition. Month is deliberately not filtered
                # because all months are required.
                # casting UTC as local time to further limit num rows filtered based on loacl time
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

                # Read this bounded batch directly from Trino. Trino handles
                # access to the underlying Hive/Iceberg storage.
                batch_timeseries_data = pl.read_database(
                    query=batch_query,
                    connection=engine,
                ).rename(
                    {
                        "circuit_id": "c_id",
                        "t_stamp": "utc_tstamp",
                    }
                )

                # Yield the complete batch and pause this generator with the
                # current batch_start retained. After the caller processes every
                # site in the batch, the generator resumes and advances.
                yield batch_sites, batch_circuit_data, batch_timeseries_data


try:
    # Select the site cohort before looking up its circuits.
    site_data = pl.read_database(query=SITE_QUERY, connection=engine)
    site_data = site_data.unique(
        subset=["site_id"],
        keep="first",
        maintain_order=True,
    )
    site_data = add_s_rated_capacity(site_data)

    # Match the selected site cohort to its PV circuits within Trino.

    # Retrieve only PV circuits linked to the selected site cohort instead of
    # loading every circuit type from the circuit metadata table.
    circuit_query = """
        SELECT
            c.site_id,
            c.circuit_id,
            c.circuit_polarity
        FROM iceberg.solar_analytics_iceberg.circuits AS c
        INNER JOIN (
            SELECT DISTINCT site_id
            FROM iceberg.solar_analytics_iceberg.meta_up23c
            WHERE inverter_count = 1
        ) AS selected_sites
            ON c.site_id = selected_sites.site_id
        WHERE c.is_pv = TRUE
    """
    circuit_data = pl.read_database(
        query=circuit_query,
        connection=engine,
    ).rename(
        {
            "circuit_id": "c_id",
            "circuit_polarity": "polarity",
        }
    )

    # Count distinct PV circuits per site. Sites with one to three circuits are
    # eligible; sites with no PV circuit or more than three are excluded.
    pv_circuit_counts = circuit_data.group_by("site_id").agg(
        pl.col("c_id").n_unique().alias("pv_circuit_count")
    )

    eligible_site_ids = pv_circuit_counts.filter(
        pl.col("pv_circuit_count").is_between(1, 3)
    )["site_id"]

    eligible_sites = site_data.filter(
        pl.col("site_id").is_in(eligible_site_ids.implode())
    ).with_columns(
        pl.col("state")
        .replace_strict(STATE_TIMEZONES, default=LOCAL_TIMEZONE)
        .alias("timezone")
    )

    total_sites = eligible_sites.height
    print(f"Eligible sites to process: {total_sites}", flush=True)

    # temporary csv files are stored here to push it on trino
    # and then deleted after
    conformance_output_dir = TRINO_OUTPUT_DIR
    conformance_output_dir.mkdir(parents=True, exist_ok=True)
    conformance_output_path = (
        conformance_output_dir / "solA_conformance_trino_summary.csv"
    )
    pl.DataFrame(schema=SITE_COMPLIANCE_SCHEMA).write_csv(conformance_output_path)
    time_distribution_output_path = (
        conformance_output_dir / "solA_conformance_trino_time_distribution.csv"
    )
    pl.DataFrame(schema=SITE_COMPLIANCE_TIME_DISTRIBUTION_SCHEMA).write_csv(
        time_distribution_output_path
    )
    site_level_various_voltages_path = (
        conformance_output_dir / "site_level_various_voltages.csv"
    )
    final_table_output_path = conformance_output_dir / "site_compliance_final_table.csv"
    final_table_output_path.unlink(missing_ok=True)
    if SAVE_SITE_LEVEL_VARIOUS_VOLTAGES:
        pl.DataFrame(schema=SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA).write_csv(
            site_level_various_voltages_path
        )

    site_idx = 0
    # Call the generator once for the complete cohort. Each iteration receives
    # one queried batch containing the configured number of sites from one
    # postcode bucket and timezone.
    for (
        batch_sites,
        batch_circuit_data,
        batch_timeseries_data,
    ) in _iter_site_timeseries_batches(
        engine,
        eligible_sites,
        circuit_data,
    ):
        # Split the downloaded batch back into individual sites using the
        # site-to-circuit mapping.
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

            # pre processing
            site_timeseries_data = convertWToKw(site_timeseries_data)
            site_timeseries_data = deduplicateMeasurements(site_timeseries_data)

            # Reuse the timezone assigned when the eligible-site cohort was built.
            site_timezone = site["timezone"]
            site_timeseries_data = site_timeseries_data.with_columns(
                pl.lit(site_timezone).alias("timezone")
            )

            # add lcoal timestamp and pre process
            site_timeseries_data = addLocalTStamp(site_timeseries_data)
            site_timeseries_data = addValidVoltage(site_timeseries_data)
            site_timeseries_data = addPolarityToPower(
                site_timeseries_data,
                batch_circuit_data,
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

                # Reuse the shared inclusive site-day extraction
                # Extract all circuit measurements for this site and day in long format.
                site_day_long = extract_site_day(
                    site_timeseries_data,
                    start_day,
                    end_day,
                )
                if site_day_long.is_empty():
                    continue

                # Reuse the shared long-to-wide circuit mapping, then calculate the
                # per-circuit rolling voltage and site-level V10m/instantaneous-max
                # signals required by the conformance checks.

                # Pivot the site's circuits into one row per timestamp.
                site_day_df = map_circuit_data_to_site(
                    site_day_long,
                    site["site_id"],
                )

                # Add per-circuit rolling voltages and site-level V10m and maximum voltage.
                prepared_day_df = calculate_site_day_voltage_signals(
                    site_day_df,
                    voltage_prefix="voltage_valid",
                )
                # Apply the same configured analysis window to the long and wide
                # forms before testing day eligibility.
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

            s_rated = site["s_rated"]
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
            if SAVE_SITE_LEVEL_VARIOUS_VOLTAGES:
                site_level_various_voltage_rows.append(
                    phase_a_result["site_level_various_voltages"]
                )

            # run the three cases of phase b
            for threshold_method in PHASE_B_METHODS:
                phase_b_calculated = run_phase_b_for_site(
                    site["site_id"],
                    prepared_site_days,
                    site_thresholds=phase_a_result["site_thresholds"],
                    threshold_method=threshold_method,
                    threshold_source="calculated",
                    disconnect_support=False,
                    tau=0.0,
                )
                phase_b_disconnect_supported = run_phase_b_for_site(
                    site["site_id"],
                    prepared_site_days,
                    site_thresholds=phase_a_result["site_thresholds"],
                    threshold_method=threshold_method,
                    threshold_source="calculated",
                    disconnect_support=True,
                    tau=0.0,
                )
                phase_b_lowest_disconnect = run_phase_b_for_site(
                    site["site_id"],
                    prepared_site_days,
                    site_thresholds=phase_a_result["site_thresholds"],
                    threshold_method=threshold_method,
                    threshold_source="lowest_disconnect",
                    disconnect_support=False,
                    tau=0.0,
                )

                calculated_compliance = (
                    phase_b_calculated["site_compliance"]
                    .select(
                        [
                            "site_id",
                            "threshold_method",
                            "los_threshold_used",
                            "ov1_threshold_used",
                            "los_lowest_disconnect_voltage",
                            "ov1_lowest_disconnect_voltage",
                            "los_responsible_count",
                            "los_compliant_count",
                            "los_compliance_pct",
                            "los_pass",
                            "ov1_responsible_count",
                            "ov1_compliant_count",
                            "ov1_compliance_pct",
                            "ov1_pass",
                            "overall_responsible_count",
                            "overall_compliant_count",
                            "overall_compliance_pct",
                            "overall_pass",
                        ]
                    )
                    .rename(
                        {
                            "los_threshold_used": "los_calculated_threshold_used",
                            "ov1_threshold_used": "ov1_calculated_threshold_used",
                            "los_responsible_count": "los_calculated_responsible_count",
                            "los_compliant_count": "los_calculated_compliant_count",
                            "los_compliance_pct": "los_calculated_compliance_pct",
                            "los_pass": "los_calculated_pass",
                            "ov1_responsible_count": "ov1_calculated_responsible_count",
                            "ov1_compliant_count": "ov1_calculated_compliant_count",
                            "ov1_compliance_pct": "ov1_calculated_compliance_pct",
                            "ov1_pass": "ov1_calculated_pass",
                            "overall_responsible_count": "overall_calculated_responsible_count",
                            "overall_compliant_count": "overall_calculated_compliant_count",
                            "overall_compliance_pct": "overall_calculated_compliance_pct",
                            "overall_pass": "overall_calculated_pass",
                        }
                    )
                )
                disconnect_supported_compliance = (
                    phase_b_disconnect_supported["site_compliance"]
                    .select(
                        [
                            "los_disconnect_support_added_count",
                            "ov1_disconnect_support_added_count",
                            "los_disconnect_supported_responsible_count",
                            "los_disconnect_supported_compliant_count",
                            "los_disconnect_supported_compliance_pct",
                            "los_disconnect_supported_pass",
                            "ov1_disconnect_supported_responsible_count",
                            "ov1_disconnect_supported_compliant_count",
                            "ov1_disconnect_supported_compliance_pct",
                            "ov1_disconnect_supported_pass",
                            "overall_disconnect_supported_responsible_count",
                            "overall_disconnect_supported_compliant_count",
                            "overall_disconnect_supported_compliance_pct",
                            "overall_disconnect_supported_pass",
                        ]
                    )
                )
                lowest_disconnect_compliance = (
                    phase_b_lowest_disconnect["site_compliance"]
                    .select(
                        [
                            "los_threshold_used",
                            "ov1_threshold_used",
                            "los_responsible_count",
                            "los_compliant_count",
                            "los_compliance_pct",
                            "los_pass",
                            "ov1_responsible_count",
                            "ov1_compliant_count",
                            "ov1_compliance_pct",
                            "ov1_pass",
                            "overall_responsible_count",
                            "overall_compliant_count",
                            "overall_compliance_pct",
                            "overall_pass",
                        ]
                    )
                    .rename(
                        {
                            "los_threshold_used": "los_lowest_disconnect_threshold_used",
                            "ov1_threshold_used": "ov1_lowest_disconnect_threshold_used",
                            "los_responsible_count": "los_lowest_disconnect_responsible_count",
                            "los_compliant_count": "los_lowest_disconnect_compliant_count",
                            "los_compliance_pct": "los_lowest_disconnect_compliance_pct",
                            "los_pass": "los_lowest_disconnect_pass",
                            "ov1_responsible_count": "ov1_lowest_disconnect_responsible_count",
                            "ov1_compliant_count": "ov1_lowest_disconnect_compliant_count",
                            "ov1_compliance_pct": "ov1_lowest_disconnect_compliance_pct",
                            "ov1_pass": "ov1_lowest_disconnect_pass",
                            "overall_responsible_count": "overall_lowest_disconnect_responsible_count",
                            "overall_compliant_count": "overall_lowest_disconnect_compliant_count",
                            "overall_compliance_pct": "overall_lowest_disconnect_compliance_pct",
                            "overall_pass": "overall_lowest_disconnect_pass",
                        }
                    )
                )
                site_compliance_frame = pl.concat(
                    [
                        calculated_compliance,
                        disconnect_supported_compliance,
                        lowest_disconnect_compliance,
                    ],
                    how="horizontal",
                )
                site_compliance = site_compliance_frame.to_dicts()[0]
                overall_pass = site_compliance["overall_disconnect_supported_pass"]
                if overall_pass is None:
                    assessment_status = "unassessed"
                elif overall_pass:
                    assessment_status = "conformant"
                else:
                    assessment_status = "non-conformant"
                site_compliance_rows.append(
                    {
                        **site_compliance,
                        "disconnect_supported_assessment_status": assessment_status,
                    }
                )

                calculated_distribution = phase_b_calculated["site_compliance"].select(
                    [
                        "site_id",
                        "threshold_method",
                        pl.lit("calculated").alias("case"),
                        pl.col("overall_responsible_count").alias(
                            "eligible_timestamp_count"
                        ),
                        pl.col("overall_compliant_count").alias(
                            "compliant_timestamp_count"
                        ),
                        (
                            pl.col("overall_responsible_count")
                            - pl.col("overall_compliant_count")
                        ).alias("non_compliant_timestamp_count"),
                        pl.col("overall_compliance_pct").alias("compliant_pct"),
                        (100.0 - pl.col("overall_compliance_pct")).alias(
                            "non_compliant_pct"
                        ),
                        "disconnected_below_threshold_count",
                        "disconnected_unknown_voltage_count",
                    ]
                )
                disconnect_supported_distribution = phase_b_disconnect_supported[
                    "site_compliance"
                ].select(
                    [
                        "site_id",
                        "threshold_method",
                        pl.lit("disconnect_supported").alias("case"),
                        pl.col("overall_disconnect_supported_responsible_count").alias(
                            "eligible_timestamp_count"
                        ),
                        pl.col("overall_disconnect_supported_compliant_count").alias(
                            "compliant_timestamp_count"
                        ),
                        (
                            pl.col("overall_disconnect_supported_responsible_count")
                            - pl.col("overall_disconnect_supported_compliant_count")
                        ).alias("non_compliant_timestamp_count"),
                        pl.col("overall_disconnect_supported_compliance_pct").alias(
                            "compliant_pct"
                        ),
                        (
                            100.0
                            - pl.col("overall_disconnect_supported_compliance_pct")
                        ).alias("non_compliant_pct"),
                        "disconnected_below_threshold_count",
                        "disconnected_unknown_voltage_count",
                    ]
                )
                lowest_disconnect_distribution = phase_b_lowest_disconnect[
                    "site_compliance"
                ].select(
                    [
                        "site_id",
                        "threshold_method",
                        pl.lit("lowest_disconnect").alias("case"),
                        pl.col("overall_responsible_count").alias(
                            "eligible_timestamp_count"
                        ),
                        pl.col("overall_compliant_count").alias(
                            "compliant_timestamp_count"
                        ),
                        (
                            pl.col("overall_responsible_count")
                            - pl.col("overall_compliant_count")
                        ).alias("non_compliant_timestamp_count"),
                        pl.col("overall_compliance_pct").alias("compliant_pct"),
                        (100.0 - pl.col("overall_compliance_pct")).alias(
                            "non_compliant_pct"
                        ),
                        "disconnected_below_threshold_count",
                        "disconnected_unknown_voltage_count",
                    ]
                )
                site_compliance_time_distribution_rows.extend(
                    [
                        calculated_distribution,
                        disconnect_supported_distribution,
                        lowest_disconnect_distribution,
                    ]
                )

            # investigate

            # save in table form back to trino

        if site_compliance_rows:
            print("appending data to csv")
            site_compliance = pl.DataFrame(
                site_compliance_rows,
                schema=SITE_COMPLIANCE_SCHEMA,
            )
            with conformance_output_path.open("ab") as output_file:
                site_compliance.write_csv(
                    output_file,
                    include_header=False,
                )
            site_compliance_rows.clear()
        if site_compliance_time_distribution_rows:
            time_distribution = pl.concat(
                site_compliance_time_distribution_rows,
                how="vertical",
            ).cast(SITE_COMPLIANCE_TIME_DISTRIBUTION_SCHEMA, strict=False)
            with time_distribution_output_path.open("ab") as output_file:
                time_distribution.write_csv(output_file, include_header=False)
            site_compliance_time_distribution_rows.clear()
        if SAVE_SITE_LEVEL_VARIOUS_VOLTAGES and site_level_various_voltage_rows:
            site_level_various_voltages = pl.concat(
                site_level_various_voltage_rows,
                how="vertical",
            )
            with site_level_various_voltages_path.open("ab") as output_file:
                site_level_various_voltages.write_csv(
                    output_file,
                    include_header=False,
                )
            site_level_various_voltage_rows.clear()

    site_compliance = pl.read_csv(
        conformance_output_path,
        schema_overrides=SITE_COMPLIANCE_SCHEMA,
    )
    write_method_compliance_final_table(
        site_compliance,
        final_table_output_path,
    )

    # push data to trino
    iceberg_exec("DROP TABLE IF EXISTS lso_anti_islanding_conformance")
    iceberg_exec("""
        CREATE TABLE lso_anti_islanding_conformance (
            site_id BIGINT,
            threshold_method VARCHAR,
            los_calculated_threshold_used DOUBLE,
            ov1_calculated_threshold_used DOUBLE,
            los_lowest_disconnect_voltage DOUBLE,
            ov1_lowest_disconnect_voltage DOUBLE,
            los_lowest_disconnect_threshold_used DOUBLE,
            ov1_lowest_disconnect_threshold_used DOUBLE,
            los_calculated_responsible_count BIGINT,
            los_calculated_compliant_count BIGINT,
            los_calculated_compliance_pct DOUBLE,
            los_calculated_pass BOOLEAN,
            ov1_calculated_responsible_count BIGINT,
            ov1_calculated_compliant_count BIGINT,
            ov1_calculated_compliance_pct DOUBLE,
            ov1_calculated_pass BOOLEAN,
            overall_calculated_responsible_count BIGINT,
            overall_calculated_compliant_count BIGINT,
            overall_calculated_compliance_pct DOUBLE,
            overall_calculated_pass BOOLEAN,
            los_disconnect_support_added_count BIGINT,
            ov1_disconnect_support_added_count BIGINT,
            los_disconnect_supported_responsible_count BIGINT,
            los_disconnect_supported_compliant_count BIGINT,
            los_disconnect_supported_compliance_pct DOUBLE,
            los_disconnect_supported_pass BOOLEAN,
            ov1_disconnect_supported_responsible_count BIGINT,
            ov1_disconnect_supported_compliant_count BIGINT,
            ov1_disconnect_supported_compliance_pct DOUBLE,
            ov1_disconnect_supported_pass BOOLEAN,
            overall_disconnect_supported_responsible_count BIGINT,
            overall_disconnect_supported_compliant_count BIGINT,
            overall_disconnect_supported_compliance_pct DOUBLE,
            overall_disconnect_supported_pass BOOLEAN,
            los_lowest_disconnect_responsible_count BIGINT,
            los_lowest_disconnect_compliant_count BIGINT,
            los_lowest_disconnect_compliance_pct DOUBLE,
            los_lowest_disconnect_pass BOOLEAN,
            ov1_lowest_disconnect_responsible_count BIGINT,
            ov1_lowest_disconnect_compliant_count BIGINT,
            ov1_lowest_disconnect_compliance_pct DOUBLE,
            ov1_lowest_disconnect_pass BOOLEAN,
            overall_lowest_disconnect_responsible_count BIGINT,
            overall_lowest_disconnect_compliant_count BIGINT,
            overall_lowest_disconnect_compliance_pct DOUBLE,
            overall_lowest_disconnect_pass BOOLEAN,
            disconnect_supported_assessment_status VARCHAR
        )
        WITH (format = 'PARQUET')
    """)

    rows_written = site_compliance.write_database(
        table_name="lso_anti_islanding_conformance",
        connection=engine,
        if_table_exists="append",
        engine_options={"chunksize": 250, "method": "multi"},
    )
    print(
        "Uploaded site compliance to lso_anti_islanding_conformance: "
        f"{rows_written} rows",
        flush=True,
    )

    time_distribution = pl.read_csv(
        time_distribution_output_path,
        schema_overrides=SITE_COMPLIANCE_TIME_DISTRIBUTION_SCHEMA,
    )
    iceberg_exec(
        "DROP TABLE IF EXISTS lso_anti_islanding_conformance_time_distribution"
    )
    iceberg_exec("""
        CREATE TABLE lso_anti_islanding_conformance_time_distribution (
            site_id BIGINT,
            threshold_method VARCHAR,
            "case" VARCHAR,
            eligible_timestamp_count BIGINT,
            compliant_timestamp_count BIGINT,
            non_compliant_timestamp_count BIGINT,
            compliant_pct DOUBLE,
            non_compliant_pct DOUBLE,
            disconnected_below_threshold_count BIGINT,
            disconnected_unknown_voltage_count BIGINT
        )
        WITH (format = 'PARQUET')
    """)
    rows_written = time_distribution.write_database(
        table_name="lso_anti_islanding_conformance_time_distribution",
        connection=engine,
        if_table_exists="append",
        engine_options={"chunksize": 250, "method": "multi"},
    )
    print(
        "Uploaded time distribution to "
        "lso_anti_islanding_conformance_time_distribution: "
        f"{rows_written} rows",
        flush=True,
    )

    final_table = pl.read_csv(final_table_output_path).rename(
        {
            "Method Used": "threshold_method",
            "Case": "case",
            "Eligible Sites After Filtering": "eligible_sites_after_filtering",
            "Sites Assessed": "sites_assessed",
            "Unassessed Sites": "unassessed_sites",
            "Conformant Sites": "conformant_sites",
            "Non-Conformant Sites": "non_conformant_sites",
            "Conformance Percentage (% of Assessed)": "conformance_percentage_pct",
        }
    )
    iceberg_exec("DROP TABLE IF EXISTS lso_anti_islanding_conformance_final_table")
    iceberg_exec("""
        CREATE TABLE lso_anti_islanding_conformance_final_table (
            threshold_method VARCHAR,
            "case" VARCHAR,
            eligible_sites_after_filtering BIGINT,
            sites_assessed BIGINT,
            unassessed_sites BIGINT,
            conformant_sites BIGINT,
            non_conformant_sites BIGINT,
            conformance_percentage_pct DOUBLE
        )
        WITH (format = 'PARQUET')
    """)
    rows_written = final_table.write_database(
        table_name="lso_anti_islanding_conformance_final_table",
        connection=engine,
        if_table_exists="append",
    )
    print(
        "Uploaded final table to "
        "lso_anti_islanding_conformance_final_table: "
        f"{rows_written} rows",
        flush=True,
    )

    if SAVE_SITE_LEVEL_VARIOUS_VOLTAGES:
        site_level_various_voltages = pl.read_csv(
            site_level_various_voltages_path,
            schema_overrides=SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA,
        )
        iceberg_exec("DROP TABLE IF EXISTS lso_ov_site_level_various_voltages")
        iceberg_exec("""
            CREATE TABLE lso_ov_site_level_various_voltages (
                site_id BIGINT,
                los_threshold DOUBLE,
                los_lowest_disconnect_voltage DOUBLE,
                los_median_all_disconnect_voltages DOUBLE,
                los_lowest_reconnect_voltage DOUBLE,
                los_median_all_reconnect_voltages DOUBLE,
                ov1_threshold DOUBLE,
                ov1_lowest_disconnect_voltage DOUBLE,
                ov1_median_all_disconnect_voltages DOUBLE,
                ov1_lowest_reconnect_voltage DOUBLE,
                ov1_median_all_reconnect_voltages DOUBLE
            )
            WITH (format = 'PARQUET')
        """)
        rows_written = site_level_various_voltages.write_database(
            table_name="lso_ov_site_level_various_voltages",
            connection=engine,
            if_table_exists="append",
            engine_options={"chunksize": 250, "method": "multi"},
        )
        print(
            "Uploaded site voltages to lso_ov_site_level_various_voltages: "
            f"{rows_written} rows",
            flush=True,
        )

    conformance_output_path.unlink(missing_ok=True)
    time_distribution_output_path.unlink(missing_ok=True)
    final_table_output_path.unlink(missing_ok=True)
    if SAVE_SITE_LEVEL_VARIOUS_VOLTAGES:
        site_level_various_voltages_path.unlink(missing_ok=True)
    print("Removed temporary conformance CSV files", flush=True)

finally:
    engine.dispose()
