"""Solar Analytics inputs and policies for the shared conformance workflow."""

from datetime import datetime
from pathlib import Path

import polars as pl

from config import (
    SITE_DAY_END,
    SITE_DAY_EXTRACTION_START,
    SOLAR_ANALYTICS_DAY_COVERAGE_THRESHOLD,
)
from core.workflow import DatasetDefinition, build_workflow_inputs
from solar_analytics_workflow.preprocessing import load_circuit_details
from solar_analytics_workflow.site_day_filtering import (
    summarize_solar_analytics_day_eligibility,
)
from solar_analytics_workflow.solar_paths import (
    CLEANED_DATA_PATH,
    CONFORMANCE_OUTPUT_DIR,
    SITE_METADATA_PATH,
)


def load_cleaned_site_data(cleaned_path=CLEANED_DATA_PATH):
    cleaned_path = Path(cleaned_path)
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned site data at {cleaned_path}. "
            "Run run_solar_analytics_preprocessing.py first."
        )
    all_data = pl.scan_parquet(cleaned_path)
    required_columns = {
        "c_id",
        "local_tstamp",
        "utc_tstamp",
        "power",
        "voltage_valid",
    }
    missing_columns = required_columns.difference(all_data.collect_schema())
    if missing_columns:
        raise ValueError(
            f"Solar cleaned data at {cleaned_path} uses an older schema and is "
            f"missing {sorted(missing_columns)}. Run "
            "run_solar_analytics_preprocessing.py again."
        )
    return all_data


def load_solar_analytics_inputs():
    site_details = pl.read_csv(SITE_METADATA_PATH).with_columns(
        pl.col("ac_capacity_kw")
        .cast(pl.Float64, strict=False)
        .alias("capacity_kw")
    )
    circuit_details = load_circuit_details()
    all_data = load_cleaned_site_data()
    return build_workflow_inputs(site_details, circuit_details, all_data)


def _solar_analytics_days(site_data):
    local_dates = (
        site_data.select(pl.col("local_tstamp").dt.date().alias("local_date"))
        .drop_nulls()
        .unique()
        .sort("local_date")["local_date"]
        .to_list()
    )
    return [
        (
            local_date,
            datetime.combine(local_date, SITE_DAY_EXTRACTION_START),
            datetime.combine(local_date, SITE_DAY_END),
        )
        for local_date in local_dates
    ]


def _summarize_solar_analytics_day(site_day_long, prepared_day_df):
    del site_day_long
    return summarize_solar_analytics_day_eligibility(
        prepared_day_df,
        coverage_threshold=SOLAR_ANALYTICS_DAY_COVERAGE_THRESHOLD,
    )


SOLAR_ANALYTICS_DEFINITION = DatasetDefinition(
    name="solar_analytics",
    load_inputs=load_solar_analytics_inputs,
    day_provider=_solar_analytics_days,
    eligibility_function=_summarize_solar_analytics_day,
    output_dir=CONFORMANCE_OUTPUT_DIR,
    coverage_threshold=SOLAR_ANALYTICS_DAY_COVERAGE_THRESHOLD,
    exclusion_fields=(
        "common_power_v10m_coverage_pct",
        "rows_common_power_v10m",
        "rows_with_power",
        "rows_with_v10m",
        "qualifying_timestamps",
        "expected_timestamps",
        "total_rows",
    ),
    excluded_day_schema={
        "site_id": pl.Int64,
        "day": pl.Date,
        "reason": pl.Utf8,
        "common_power_v10m_coverage_pct": pl.Float64,
        "rows_common_power_v10m": pl.Int64,
        "rows_with_power": pl.Int64,
        "rows_with_v10m": pl.Int64,
        "qualifying_timestamps": pl.Int64,
        "expected_timestamps": pl.Int64,
        "total_rows": pl.Int64,
        "coverage_threshold_pct": pl.Float64,
    },
)
