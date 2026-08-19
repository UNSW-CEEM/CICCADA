import sys
from functools import partial
from pathlib import Path

import polars as pl

CONFORMANCE_DIR = Path(__file__).resolve().parents[1]
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

from config import (
    GENERATE_SITE_PLOTS_DEFAULT,
    LOCAL_TIMEZONE,
    PHASE_B_METHODS,
    PLOT_NO_ELIGIBLE_TIMESTAMP_DAYS,
    PRIMARY_PHASE_B_METHOD,
    SITE_DAY_END,
    SITE_DAY_EXTRACTION_START,
)
from core.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertWToKw,
    deduplicateMeasurements,
)
from core.pipeline import run_conformance
from core.workflow import build_workflow_inputs, prepare_site
from reporting.plotting import plot_site_threshold_distribution
from solar_analytics_workflow.adapter import SOLAR_ANALYTICS_CONFORMANCE_CONFIG
from solar_analytics_workflow.preprocessing import STATE_TIMEZONES
from trino_connection_local_to_s3 import local_trino_engine, read_query_via_parquet


CLEANED_DATA_SCHEMA = {
    "c_id": pl.Int64,
    "site_id": pl.Int64,
    "con_type": pl.Utf8,
    "state": pl.Utf8,
    "timezone": pl.Utf8,
    "utc_tstamp": pl.Datetime("us", "UTC"),
    "local_tstamp": pl.Datetime("us"),
    "power": pl.Float64,
    "voltage": pl.Float64,
    "voltage_valid": pl.Float64,
}

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

LIMITED_OUTPUT_DIR = (
    SOLAR_ANALYTICS_CONFORMANCE_CONFIG.output_dir / "trino_limited"
)
LIMITED_SITE_PLOT_DIR = LIMITED_OUTPUT_DIR / "overall_site_plots"
LIMITED_THRESHOLD_PLOT_DIR = LIMITED_OUTPUT_DIR / "threshold_distribution_plots"
LIMITED_SUMMARY_PATH = (
    LIMITED_OUTPUT_DIR / "solA_conformance_trino_limited_summary.csv"
)


def _conformance_summary_row(site_result):
    """Build the limited tier-based summary row for one completed site."""
    phase_b_summary = site_result["phase_b_site_summary"]
    phase_b_thresholds = site_result["site_thresholds"]
    if phase_b_summary.is_empty() or phase_b_thresholds.is_empty():
        return None
    if phase_b_summary.height != 1 or phase_b_thresholds.height != 1:
        raise ValueError("Expected exactly one primary Phase B result per site.")

    summary = phase_b_summary.to_dicts()[0]
    thresholds = phase_b_thresholds.to_dicts()[0]
    overall_pass = summary["overall_pass"]
    if overall_pass is None:
        assessment_status = "unassessed"
    elif overall_pass:
        assessment_status = "conformant"
    else:
        assessment_status = "non-conformant"

    return {
        "site_id": summary["site_id"],
        "method_key": PRIMARY_PHASE_B_METHOD,
        "assessment_status": assessment_status,
        "overall_pass": overall_pass,
        "los_pass": summary["los_pass"],
        "los_compliance_pct": summary["los_compliance_pct"],
        "los_threshold_used": summary["los_threshold_used"],
        "ov1_pass": summary["ov1_pass"],
        "ov1_compliance_pct": summary["ov1_compliance_pct"],
        "ov1_threshold_used": thresholds["ov1_test_site"],
        "pass_basis": summary["pass_basis"],
        "threshold_selection_basis": thresholds["threshold_selection_basis"],
        "threshold_confidence_tier": thresholds[
            "threshold_confidence_tier"
        ],
    }


def _threshold_stats(phase_a_records, mechanism, voltage_column):
    """Return per-site Phase A threshold statistics for one mechanism."""
    return (
        phase_a_records.filter(
            (pl.col("mech") == mechanism)
            & pl.col(voltage_column).is_not_null()
        )
        .group_by("site_id")
        .agg([
            pl.col(voltage_column).min().alias("min_v"),
            pl.col(voltage_column).median().alias("median_v"),
            pl.col(voltage_column).max().alias("max_v"),
            pl.col(voltage_column).std().alias("std_v"),
            pl.len().alias("n_events"),
        ])
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
            "v_los_recorded",
            "LOS Thresholds Across Limited Trino Sites — "
            "Min / Median / Max (Std on right)",
            "los_threshold_distribution.png",
        ),
        (
            "OV1",
            "v_ov1_recorded",
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
            batch_sites = bucket_sites.filter(
                pl.col("timezone") == batch_timezone
            )
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

            circuit_ids = ", ".join(
                batch_circuit_ids.cast(pl.String).to_list()
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
                    ) BETWEEN TIME '{SITE_DAY_EXTRACTION_START.isoformat()}'
                        AND TIME '{SITE_DAY_END.isoformat()}'
            """

            batch_timeseries_data = read_query_via_parquet(
                trino_engine=engine,
                query=batch_query,
            ).rename({
                "circuit_id": "c_id",
                "t_stamp": "utc_tstamp",
            })

            yield batch_sites, batch_circuit_data, batch_timeseries_data


def _clean_site_timeseries_data(site_timeseries_data, site_circuit_data, site):
    """Return one site's measurements in the standard cleaned-data schema."""
    cleaned_data = (
        site_timeseries_data.lazy()
        .filter(pl.col("utc_tstamp").is_not_null())
        .with_columns([
            pl.col("c_id").cast(pl.Int64),
            pl.col("power").cast(pl.Float64, strict=False),
            pl.col("voltage").cast(pl.Float64, strict=False),
        ])
    )

    cleaned_data = convertWToKw(cleaned_data)
    cleaned_data = deduplicateMeasurements(cleaned_data)
    cleaned_data = (
        cleaned_data.join(
            site_circuit_data.select(["c_id", "con_type"])
            .unique()
            .lazy(),
            on="c_id",
            how="inner",
        )
        .with_columns([
            pl.lit(site["site_id"]).cast(pl.Int64).alias("site_id"),
            pl.lit(site["state"]).cast(pl.Utf8).alias("state"),
            pl.lit(site["timezone"]).cast(pl.Utf8).alias("timezone"),
        ])
    )

    cleaned_data = addLocalTStamp(cleaned_data)
    cleaned_data = addValidVoltage(cleaned_data)
    cleaned_data = addPolarityToPower(cleaned_data, site_circuit_data)

    return (
        cleaned_data.select([
            pl.col(column).cast(dtype).alias(column)
            for column, dtype in CLEANED_DATA_SCHEMA.items()
        ])
        .collect(engine="streaming")
    )


# Select ten sites with one grouped metadata row and one to three PV circuits.
SITE_QUERY = """
WITH eligible_sites AS (
    SELECT site_id
    FROM hive.solar_analytics.circuits
    WHERE circuit_type = 'pv_site_net'
    GROUP BY site_id
    HAVING COUNT(DISTINCT circuit_id) BETWEEN 1 AND 3
)
SELECT
    s.site_id,
    MAX(s.state) AS state,
    CAST(MAX(s.postcode) AS INTEGER) AS postcode,
    system.bucket(CAST(MAX(s.postcode) AS INTEGER), 16) AS postcode_bucket,
    MAX(s.ac_capacity_kw) AS ac_capacity_kw
FROM hive.solar_analytics.sites AS s
INNER JOIN eligible_sites AS e
    ON s.site_id = e.site_id
GROUP BY s.site_id
ORDER BY s.site_id
LIMIT 2000
"""


# Keep one Trino connection open
LIMITED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
conformance_summary_rows = []
phase_a_record_frames = []

with local_trino_engine(
    catalog="iceberg",
    schema="solar_analytics_iceberg",
) as engine:
    site_data = pl.read_database(query=SITE_QUERY, connection=engine)
    site_data = site_data.with_columns([
        pl.col("site_id").cast(pl.Int64),
        pl.col("state").cast(pl.Utf8),
        pl.col("ac_capacity_kw")
        .cast(pl.Float64, strict=False)
        .alias("capacity_kw"),
        pl.col("state")
        .replace_strict(STATE_TIMEZONES, default=LOCAL_TIMEZONE)
        .cast(pl.Utf8)
        .alias("timezone"),
    ])

    # Retrieve PV circuits only for the selected ten sites.
    site_ids = ", ".join(site_data["site_id"].cast(pl.String).to_list())
    circuit_query = f"""
        SELECT DISTINCT
            site_id,
            circuit_id,
            circuit_polarity,
            circuit_type
        FROM hive.solar_analytics.circuits
        WHERE site_id IN ({site_ids})
            AND circuit_type = 'pv_site_net'
    """
    circuit_data = (
        pl.read_database(
            query=circuit_query,
            connection=engine,
        )
        .rename({
            "circuit_id": "c_id",
            "circuit_polarity": "polarity",
            "circuit_type": "con_type",
        })
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("c_id").cast(pl.Int64),
            pl.col("con_type").cast(pl.Utf8),
            pl.col("polarity").cast(pl.Float64, strict=False),
        ])
    )

    processed_sites = 0
    for batch_sites, batch_circuit_data, batch_timeseries_data in (
        _iter_site_timeseries_batches(
            engine,
            site_data,
            circuit_data,
        )
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

            site_workflow_inputs = build_workflow_inputs(
                site_data.filter(pl.col("site_id") == site["site_id"]),
                site_circuit_data,
                site_timeseries_data,
            )
            prepare_trino_site = partial(
                prepare_site,
                inputs=site_workflow_inputs,
                workflow_config=SOLAR_ANALYTICS_CONFORMANCE_CONFIG,
            )
            site_result = run_conformance(
                candidate_site_ids=site_workflow_inputs["candidate_site_ids"],
                prepare_site=prepare_trino_site,
                methods=PHASE_B_METHODS,
                primary_method=PRIMARY_PHASE_B_METHOD,
                generate_site_plots=GENERATE_SITE_PLOTS_DEFAULT,
                plot_no_eligible_timestamp_days=(
                    PLOT_NO_ELIGIBLE_TIMESTAMP_DAYS
                ),
                site_plot_dir=LIMITED_SITE_PLOT_DIR,
            )

            summary_row = _conformance_summary_row(site_result)
            if summary_row is not None:
                conformance_summary_rows.append(summary_row)

            phase_a_records = site_result["phase_a_trip_attribution"]
            if not phase_a_records.is_empty():
                phase_a_record_frames.append(phase_a_records)

            processed_sites += 1
            del site_result
            del prepare_trino_site
            del site_workflow_inputs
            del site_timeseries_data

conformance_summary = pl.DataFrame(
    conformance_summary_rows,
    schema=CONFORMANCE_SUMMARY_SCHEMA,
)
if not conformance_summary.is_empty():
    conformance_summary = conformance_summary.sort("site_id")
conformance_summary.write_csv(LIMITED_SUMMARY_PATH)
print(f"Saved limited conformance summary to {LIMITED_SUMMARY_PATH}")

_generate_threshold_distribution_plots(phase_a_record_frames)

print(f"Selected sites: {site_data.height}")
print(f"Selected PV circuits: {circuit_data['c_id'].n_unique()}")
print(f"Sites processed through conformance: {processed_sites}")
