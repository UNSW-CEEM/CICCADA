import sys
from datetime import datetime
from pathlib import Path

import polars as pl

CONFORMANCE_DIR = Path(__file__).resolve().parents[2]
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

from core.phase_a import SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA, run_phase_a_for_site
from core.phase_b import evaluate_compliance_for_day, run_phase_b_for_site
from core.site_day_signals import build_site_day_signals
from solar_analytics_workflow.config import (
    DAY_ANALYSIS_START,
    DAY_END,
    DAY_EXTRACTION_START,
    GENERATE_SITE_PLOTS,
    LOCAL_TIMEZONE,
    PLOT_NO_RESPONSIBLE_TIMESTAMP_DAYS,
    PRIMARY_PHASE_B_METHOD,
    SAVE_SITE_LEVEL_VARIOUS_VOLTAGES,
)
from solar_analytics_workflow.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertWToKw,
    deduplicateMeasurements,
)
from solar_analytics_workflow.plotting import (
    plot_site_compliance_day,
    plot_site_threshold_distribution,
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
from solar_analytics_workflow.solar_paths import (
    TRINO_LIMITED_OUTPUT_DIR,
    TRINO_OUTPUT_DIR,
)
from solar_analytics_workflow.trino.trino_connection_local_to_s3 import (
    local_trino_engine,
    read_query_via_parquet,
)

CLEANED_DATA_SCHEMA = {
    "c_id": pl.Int64,
    "site_id": pl.Int64,
    "state": pl.Utf8,
    "timezone": pl.Utf8,
    "utc_tstamp": pl.Datetime("us", "UTC"),
    "local_tstamp": pl.Datetime("us"),
    "power": pl.Float64,
    "voltage": pl.Float64,
    "voltage_valid": pl.Float64,
}

SITE_COMPLIANCE_SCHEMA = {
    "site_id": pl.Int64,
    "threshold_method": pl.Utf8,
    "assessment_status": pl.Utf8,
    "overall_pass": pl.Boolean,
    "los_responsible_count": pl.Int64,
    "los_compliant_count": pl.Int64,
    "los_pass": pl.Boolean,
    "los_compliance_pct": pl.Float64,
    "los_threshold_used": pl.Float64,
    "ov1_responsible_count": pl.Int64,
    "ov1_compliant_count": pl.Int64,
    "ov1_pass": pl.Boolean,
    "ov1_compliance_pct": pl.Float64,
    "ov1_threshold_used": pl.Float64,
}

LIMITED_OUTPUT_DIR = TRINO_LIMITED_OUTPUT_DIR
LIMITED_SITE_PLOT_DIR = LIMITED_OUTPUT_DIR / "overall_site_plots"
LIMITED_THRESHOLD_PLOT_DIR = LIMITED_OUTPUT_DIR / "threshold_distribution_plots"
LIMITED_SUMMARY_PATH = LIMITED_OUTPUT_DIR / "solA_conformance_trino_limited_summary.csv"
ASSESSMENT_SUMMARY_PATH = TRINO_OUTPUT_DIR / "solA_conformance_trino_summary.csv"
MAX_ASSESSED_SITES = 1000


def _site_compliance_report_row(site_result):
    """Build the limited tier-based compliance row for one completed site."""
    site_compliance = site_result["site_compliance"]
    if site_compliance.is_empty():
        return None
    if site_compliance.height != 1:
        raise ValueError("Expected exactly one primary Phase B result per site.")

    compliance = site_compliance.to_dicts()[0]
    overall_pass = compliance["overall_pass"]
    if overall_pass is None:
        assessment_status = "unassessed"
    elif overall_pass:
        assessment_status = "conformant"
    else:
        assessment_status = "non-conformant"

    return {
        "site_id": compliance["site_id"],
        "threshold_method": compliance["threshold_method"],
        "assessment_status": assessment_status,
        "overall_pass": overall_pass,
        "los_responsible_count": compliance["los_responsible_count"],
        "los_compliant_count": compliance["los_compliant_count"],
        "los_pass": compliance["los_pass"],
        "los_compliance_pct": compliance["los_compliance_pct"],
        "los_threshold_used": compliance["los_threshold_used"],
        "ov1_responsible_count": compliance["ov1_responsible_count"],
        "ov1_compliant_count": compliance["ov1_compliant_count"],
        "ov1_pass": compliance["ov1_pass"],
        "ov1_compliance_pct": compliance["ov1_compliance_pct"],
        "ov1_threshold_used": compliance["ov1_threshold_used"],
    }


def _threshold_stats(phase_a_records, mechanism, voltage_column):
    """Return per-site Phase A threshold statistics for one mechanism."""
    return (
        phase_a_records.filter(
            (pl.col("mechanism") == mechanism)
            & pl.col(voltage_column).is_not_null()
        )
        .group_by("site_id")
        .agg(
            [
                pl.col(voltage_column).min().alias("min_v"),
                pl.col(voltage_column).median().alias("median_v"),
                pl.col(voltage_column).max().alias("max_v"),
                pl.col(voltage_column).std().alias("std_v"),
                pl.len().alias("n_events"),
            ]
        )
    )


def _generate_threshold_distribution_plots(phase_a_record_frames):
    """Generate one cohort-level threshold distribution plot per mechanism."""
    if not phase_a_record_frames:
        print("No Phase A events; skipping threshold-distribution plots.")
        return

    phase_a_records = pl.concat(phase_a_record_frames, how="vertical")
    plot_specs = (
        (
            "LOS",
            "v10m_disc",
            "LOS Thresholds Across Limited Trino Sites — "
            "Min / Median / Max (Std on right)",
            "los_threshold_distribution.png",
        ),
        (
            "OV1",
            "vinst_disc",
            "OV1 Thresholds Across Limited Trino Sites — "
            "Min / Median / Max (Std on right)",
            "ov1_threshold_distribution.png",
        ),
    )
    for mechanism, voltage_column, title, filename in plot_specs:
        stats = _threshold_stats(
            phase_a_records,
            mechanism,
            voltage_column,
        )
        if stats.is_empty():
            print(
                f"No {mechanism} Phase A events; skipping {filename}.",
                flush=True,
            )
            continue
        plot_site_threshold_distribution(
            stats,
            title=title,
            save_path=LIMITED_THRESHOLD_PLOT_DIR / filename,
        )
        print(
            f"Saved {mechanism} threshold-distribution plot to "
            f"{LIMITED_THRESHOLD_PLOT_DIR / filename}",
            flush=True,
        )


def _iter_site_timeseries_batches(engine, selected_sites, circuit_data):
    """Yield one postcode-bucket/timezone batch at a time."""
    postcode_buckets = selected_sites.get_column("postcode_bucket").unique(
        maintain_order=True
    )

    for postcode_bucket in postcode_buckets:
        bucket_sites = selected_sites.filter(
            pl.col("postcode_bucket") == postcode_bucket
        )
        batch_timezones = bucket_sites.get_column("timezone").unique(
            maintain_order=True
        )

        for batch_timezone in batch_timezones:
            batch_sites = bucket_sites.filter(pl.col("timezone") == batch_timezone)
            batch_site_ids = batch_sites.get_column("site_id")
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

            print(
                "Querying time series: "
                f"postcode_bucket={postcode_bucket} "
                f"timezone={batch_timezone} "
                f"sites={batch_sites.height}",
                flush=True,
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

            batch_timeseries_data = read_query_via_parquet(
                trino_engine=engine,
                query=batch_query,
            ).rename(
                {
                    "circuit_id": "c_id",
                    "t_stamp": "utc_tstamp",
                }
            )

            yield batch_sites, batch_circuit_data, batch_timeseries_data


def _clean_site_timeseries_data(site_timeseries_data, site_circuit_data, site):
    """Return one site's measurements in the standard cleaned-data schema."""
    cleaned_data = (
        site_timeseries_data.lazy()
        .filter(pl.col("utc_tstamp").is_not_null())
        .with_columns(
            [
                pl.col("c_id").cast(pl.Int64),
                pl.col("power").cast(pl.Float64, strict=False),
                pl.col("voltage").cast(pl.Float64, strict=False),
            ]
        )
    )

    cleaned_data = convertWToKw(cleaned_data)
    cleaned_data = deduplicateMeasurements(cleaned_data)
    cleaned_data = cleaned_data.with_columns(
        [
            pl.lit(site["site_id"]).cast(pl.Int64).alias("site_id"),
            pl.lit(site["state"]).cast(pl.Utf8).alias("state"),
            pl.lit(site["timezone"]).cast(pl.Utf8).alias("timezone"),
        ]
    )

    cleaned_data = addLocalTStamp(cleaned_data)
    cleaned_data = addValidVoltage(cleaned_data)
    cleaned_data = addPolarityToPower(cleaned_data, site_circuit_data)

    return cleaned_data.select(
        [
            pl.col(column).cast(dtype).alias(column)
            for column, dtype in CLEANED_DATA_SCHEMA.items()
        ]
    ).collect(engine="streaming")


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
"""


# Keep one Trino connection open
LIMITED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
site_compliance_rows = []
site_level_various_voltage_rows = []
phase_a_record_frames = []
assessed_site_ids = (
    pl.read_csv(
        ASSESSMENT_SUMMARY_PATH,
        schema_overrides={
            "site_id": pl.Int64,
            "overall_pass": pl.Boolean,
        },
    )
    .filter(pl.col("overall_pass").is_not_null())
    .select("site_id")
    .drop_nulls()
    .unique(maintain_order=True)
    .get_column("site_id")
)
print(f"Assessed sites in summary: {assessed_site_ids.len()}", flush=True)
print(f"Configured assessed-site cap: {MAX_ASSESSED_SITES}", flush=True)

with local_trino_engine(
    catalog="iceberg",
    schema="solar_analytics_iceberg",
) as engine:
    site_data = pl.read_database(query=SITE_QUERY, connection=engine)
    site_data = site_data.unique(
        subset=["site_id"],
        keep="first",
        maintain_order=True,
    )
    site_data = site_data.with_columns(
        [
            pl.col("site_id").cast(pl.Int64),
            pl.col("state").cast(pl.Utf8),
            pl.col("state")
            .replace_strict(STATE_TIMEZONES, default=LOCAL_TIMEZONE)
            .cast(pl.Utf8)
            .alias("timezone"),
        ]
    )
    site_data = add_s_rated_capacity(site_data)

    # Retrieve PV circuits linked to the single-inverter site cohort.
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
        ) AS single_inverter_sites
            ON c.site_id = single_inverter_sites.site_id
        WHERE c.is_pv = TRUE
    """
    circuit_data = (
        pl.read_database(
            query=circuit_query,
            connection=engine,
        )
        .rename(
            {
                "circuit_id": "c_id",
                "circuit_polarity": "polarity",
            }
        )
        .with_columns(
            [
                pl.col("site_id").cast(pl.Int64),
                pl.col("c_id").cast(pl.Int64),
                pl.col("polarity").cast(pl.Float64, strict=False),
            ]
        )
    )

    pv_circuit_counts = circuit_data.group_by("site_id").agg(
        pl.col("c_id").n_unique().alias("pv_circuit_count")
    )
    eligible_site_ids = pv_circuit_counts.filter(
        pl.col("pv_circuit_count").is_between(1, 3)
    )["site_id"]
    selected_sites = site_data.filter(
        pl.col("site_id").is_in(eligible_site_ids.implode())
        & pl.col("site_id").is_in(assessed_site_ids.implode())
    ).sort("site_id")
    if MAX_ASSESSED_SITES is not None:
        selected_sites = selected_sites.head(MAX_ASSESSED_SITES)
    print(f"Selected assessed sites: {selected_sites.height}", flush=True)

    selected_site_ids = selected_sites.get_column("site_id")
    circuit_data = circuit_data.filter(
        pl.col("site_id").is_in(selected_site_ids.implode())
    )

    processed_sites = 0
    completed_phase_sites = 0
    for (
        batch_sites,
        batch_circuit_data,
        batch_timeseries_data,
    ) in _iter_site_timeseries_batches(
        engine,
        selected_sites,
        circuit_data,
    ):
        for site in batch_sites.iter_rows(named=True):
            site_circuit_data = batch_circuit_data.filter(
                pl.col("site_id") == site["site_id"]
            )
            site_circuit_ids = site_circuit_data.get_column("c_id").unique(
                maintain_order=True
            )
            site_timeseries_data = batch_timeseries_data.filter(
                pl.col("c_id").is_in(site_circuit_ids.implode())
            )

            if site_timeseries_data.is_empty():
                print(
                    f"No time-series data for site {site['site_id']}; skipping.",
                    flush=True,
                )
                continue

            site_timeseries_data = _clean_site_timeseries_data(
                site_timeseries_data,
                site_circuit_data,
                site,
            )
            if site_timeseries_data.is_empty():
                print(
                    f"No cleaned time-series data for site {site['site_id']}; "
                    "skipping.",
                    flush=True,
                )
                continue

            print(
                f"Cleaned site {site['site_id']}: "
                f"{site_timeseries_data.height} rows",
                flush=True,
            )
            processed_sites += 1

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
            for local_date in local_dates:
                site_day_long = extract_site_day(
                    site_timeseries_data,
                    datetime.combine(local_date, DAY_EXTRACTION_START),
                    datetime.combine(local_date, DAY_END),
                )
                if site_day_long.is_empty():
                    continue
                prepared_day = calculate_site_day_voltage_signals(
                    map_circuit_data_to_site(
                        site_day_long,
                        site["site_id"],
                    ),
                    voltage_prefix="voltage_valid",
                )
                analysis_day = trim_site_day_analysis_window(
                    prepared_day,
                    DAY_ANALYSIS_START,
                    DAY_END,
                )
                eligibility = summarize_solar_analytics_day_eligibility(analysis_day)
                if eligibility["eligible"]:
                    eligible_analysis_days.append(
                        {
                            "analysis_date": local_date,
                            "analysis_frame": analysis_day,
                        }
                    )

            if not eligible_analysis_days:
                print(
                    f"No eligible days for site {site['site_id']}; skipping.",
                    flush=True,
                )
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
            phase_b_result = run_phase_b_for_site(
                site["site_id"],
                prepared_site_days,
                site_thresholds=phase_a_result["site_thresholds"],
                threshold_method=PRIMARY_PHASE_B_METHOD,
            )
            site_result = {
                "site_compliance": phase_b_result["site_compliance"],
            }

            compliance = phase_b_result["site_compliance"].to_dicts()[0]
            if GENERATE_SITE_PLOTS and compliance["overall_pass"] is not None:
                plot_folder = (
                    "compliant"
                    if compliance["overall_pass"] is True
                    else "non_compliant"
                )
                for day_info in prepared_site_days:
                    evaluated_day = evaluate_compliance_for_day(
                        day_info["signal_frame"],
                        los_threshold=compliance["los_threshold_used"],
                        ov1_threshold=compliance["ov1_threshold_used"],
                    )
                    plot_site_compliance_day(
                        evaluated_day,
                        site["site_id"],
                        day_info["analysis_date"],
                        p_rated=s_rated,
                        lso_threshold=compliance["los_threshold_used"],
                        ov1_threshold=compliance["ov1_threshold_used"],
                        overall_pass=compliance["overall_pass"],
                        plot_no_responsible_timestamp_days=(
                            PLOT_NO_RESPONSIBLE_TIMESTAMP_DAYS
                        ),
                        save_path=(
                            LIMITED_SITE_PLOT_DIR
                            / plot_folder
                            / f"Site_{site['site_id']}_Day_"
                            f"{day_info['analysis_date']}_{plot_folder}.png"
                        ),
                    )

            compliance_row = _site_compliance_report_row(site_result)
            if compliance_row is not None:
                site_compliance_rows.append(compliance_row)

            phase_a_records = phase_a_result["records"]
            if not phase_a_records.is_empty():
                phase_a_record_frames.append(phase_a_records)

            completed_phase_sites += 1
            del site_result
            del site_timeseries_data

site_compliance = pl.DataFrame(
    site_compliance_rows,
    schema=SITE_COMPLIANCE_SCHEMA,
)
if not site_compliance.is_empty():
    site_compliance = site_compliance.sort("site_id")
site_compliance.write_csv(LIMITED_SUMMARY_PATH)
print(f"Saved limited site compliance to {LIMITED_SUMMARY_PATH}")
if SAVE_SITE_LEVEL_VARIOUS_VOLTAGES:
    site_level_various_voltages = (
        pl.concat(site_level_various_voltage_rows, how="vertical")
        if site_level_various_voltage_rows
        else pl.DataFrame(schema=SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA)
    )
    site_level_various_voltages.write_csv(
        LIMITED_OUTPUT_DIR / "site_level_various_voltages.csv"
    )

_generate_threshold_distribution_plots(phase_a_record_frames)

print(f"Selected sites: {selected_sites.height}")
print(f"Selected PV circuits: {circuit_data['c_id'].n_unique()}")
print(f"Sites processed through conformance: {processed_sites}")
print(f"Sites completing Phase A and Phase B: {completed_phase_sites}")
