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

from core.check_pv_behaviour import CheckPVBehaviour
from core.phase_a import run_phase_a_for_site
from core.phase_b import run_phase_b_for_site
from solar_analytics_workflow.config import (
    DAY_ANALYSIS_START,
    DAY_END,
    DAY_EXTRACTION_START,
    LOCAL_TIMEZONE,
    PHASE_B_METHODS,
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
from solar_analytics_workflow.solar_paths import TRINO_OUTPUT_DIR
from solar_analytics_workflow.trino.trino_connection_on_ec2 import (
    engine,
    iceberg_exec,
)

EVM_TRINO_SITE_BATCH_SIZE = 10 # num sites queried at once

# these are the columns for conformance results that will be pushed to trino
# and utilised for grafana plotting
CONFORMANCE_SUMMARY_SCHEMA = {
    "site_id": pl.Int64,
    "method_key": pl.Utf8,
    "assessment_status": pl.Utf8,
    "overall_pass": pl.Boolean,
    "los_pass": pl.Boolean,
    "los_compliance_pct": pl.Float64,
    "los_threshold_used": pl.Float64,
    "ov1_pass": pl.Boolean,
    "ov1_compliance_pct": pl.Float64,
    "ov1_threshold_used": pl.Float64,
    "pass_basis": pl.Utf8,
    "threshold_selection_basis": pl.Utf8,
    "threshold_confidence_tier": pl.Utf8,
}
conformance_summary_rows = []

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

    conformance_output_dir = TRINO_OUTPUT_DIR
    conformance_output_dir.mkdir(parents=True, exist_ok=True)
    conformance_output_path = (
        conformance_output_dir / "solA_conformance_trino_summary.csv"
    )
    pl.DataFrame(schema=CONFORMANCE_SUMMARY_SCHEMA).write_csv(conformance_output_path)

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

            # get local timestamp zone based on state name
            site_timezone = STATE_TIMEZONES.get(site["state"], LOCAL_TIMEZONE)
            site_timeseries_data = site_timeseries_data.with_columns(
                pl.lit(site_timezone).alias("timezone")
            )

            # add lcoal timestamp and pre process
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

            # Build one CheckPVBehaviour object for each eligible local day.
            day_behaviours = []
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

                day_behaviours.append(
                    {
                        "day": day,
                        "behaviour": CheckPVBehaviour(
                            analysis_day_df,
                            volCol="voltage_valid",
                        ),
                    }
                )

            if not day_behaviours:
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

            phase_a_result = run_phase_a_for_site(
                site["site_id"],
                day_behaviours,  # this has the CheckPVBehaviour obj
                s_rated,
            )
            for phase_b_method in PHASE_B_METHODS:
                phase_b_result = run_phase_b_for_site(
                    site["site_id"],
                    day_behaviours,  # this has the CheckPVBehaviour obj
                    s_rated,
                    raw_thresholds=phase_a_result["raw_thresholds"],
                    confidence_info=phase_a_result["confidence_info"],
                    phase_b_method=phase_b_method,
                )
                # get columns to save results in csv later to be pushed on trino
                phase_b_summary = phase_b_result["summary_row"].to_dicts()[0]
                phase_b_thresholds = phase_b_result["threshold_row"].to_dicts()[0]
                overall_pass = phase_b_summary["overall_pass"]
                if overall_pass is None:
                    assessment_status = "unassessed"
                elif overall_pass:
                    assessment_status = "conformant"
                else:
                    assessment_status = "non-conformant"
                conformance_summary_rows.append(
                    {
                        "site_id": phase_b_summary["site_id"],
                        "method_key": phase_b_method,
                        "assessment_status": assessment_status,
                        "overall_pass": overall_pass,
                        "los_pass": phase_b_summary["los_pass"],
                        "los_compliance_pct": phase_b_summary["los_compliance_pct"],
                        "los_threshold_used": phase_b_summary["los_threshold_used"],
                        "ov1_pass": phase_b_summary["ov1_pass"],
                        "ov1_compliance_pct": phase_b_summary["ov1_compliance_pct"],
                        "ov1_threshold_used": phase_b_thresholds["ov1_test_site"],
                        "pass_basis": phase_b_summary["pass_basis"],
                        "threshold_selection_basis": phase_b_thresholds[
                            "threshold_selection_basis"
                        ],
                        "threshold_confidence_tier": phase_b_thresholds[
                            "threshold_confidence_tier"
                        ],
                    }
                )
                # print("yo")

            # investigate

            # save in table form back to trino

        if conformance_summary_rows:
            print("appending data to csv")
            conformance_summary = pl.DataFrame(
                conformance_summary_rows,
                schema=CONFORMANCE_SUMMARY_SCHEMA,
            )
            with conformance_output_path.open("ab") as output_file:
                conformance_summary.write_csv(
                    output_file,
                    include_header=False,
                )
            conformance_summary_rows.clear()

    conformance_summary = pl.read_csv(
        conformance_output_path,
        schema_overrides=CONFORMANCE_SUMMARY_SCHEMA,
    )

    iceberg_exec("DROP TABLE IF EXISTS lso_anti_islanding_conformance")
    iceberg_exec("""
        CREATE TABLE lso_anti_islanding_conformance (
            site_id BIGINT,
            method_key VARCHAR,
            assessment_status VARCHAR,
            overall_pass BOOLEAN,
            los_pass BOOLEAN,
            los_compliance_pct DOUBLE,
            los_threshold_used DOUBLE,
            ov1_pass BOOLEAN,
            ov1_compliance_pct DOUBLE,
            ov1_threshold_used DOUBLE,
            pass_basis VARCHAR,
            threshold_selection_basis VARCHAR,
            threshold_confidence_tier VARCHAR
        )
        WITH (format = 'PARQUET')
    """)

    rows_written = conformance_summary.write_database(
        table_name="lso_anti_islanding_conformance",
        connection=engine,
        if_table_exists="append",
        engine_options={"chunksize": 250, "method": "multi"},
    )
    print(
        "Uploaded conformance summary to lso_anti_islanding_conformance: "
        f"{rows_written} rows",
        flush=True,
    )

finally:
    engine.dispose()
