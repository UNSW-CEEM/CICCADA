"""SAPN2022 inputs and policies for the shared conformance workflow."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from config import (
    SAPN2022_DAY_COVERAGE_THRESHOLD,
    SAPN2022_DAY_END,
    SAPN2022_DAY_START,
    SAPN2022_EVENT_DAYS,
    SAPN2022_REQUIRED_SITE_METADATA_ROWS,
)
from core.workflow import DatasetConformanceConfig, build_workflow_inputs
from sapn2022_workflow.rated_capacity import generate_rated_capacity
from sapn2022_workflow.sapn_paths import (
    CAPACITY_DERIVED_PATH,
    CIRCUIT_DETAILS_PATH,
    CLEANED_SITE_DATA_PATH,
    CONFORMANCE_OUTPUT_DIR,
    SITE_DETAILS_PATH,
)
from sapn2022_workflow.site_day_filtering import (
    summarize_nov2022_day_eligibility,
)


def load_cleaned_site_data(cleaned_path=CLEANED_SITE_DATA_PATH):
    cleaned_path = Path(cleaned_path)
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned site data at {cleaned_path}. "
            "Run run_sapn2022_preprocessing.py first."
        )
    return pl.scan_parquet(cleaned_path)


def _capacity_w_to_kw(capacity_w):
    if capacity_w is None:
        return None
    try:
        return float(capacity_w) / 1_000.0
    except (TypeError, ValueError):
        return None


def _load_sapn2022_site_details(site_details_path=SITE_DETAILS_PATH):
    """Load SAPN sites having exactly one source metadata row."""
    return (
        pl.read_csv(site_details_path)
        .with_columns(pl.len().over("site_id").alias("_site_metadata_rows"))
        .filter(
            pl.col("_site_metadata_rows")
            == SAPN2022_REQUIRED_SITE_METADATA_ROWS
        )
        .drop("_site_metadata_rows")
    )


def load_sapn2022_inputs(update_ac_capacity=False):
    site_details = _load_sapn2022_site_details()
    site_details = site_details.with_columns(
        pl.Series(
            "capacity_kw",
            [_capacity_w_to_kw(capacity_w) for capacity_w in site_details["ac_cap_w"]],
            dtype=pl.Float64,
        )
    )
    circuit_details = pl.read_csv(CIRCUIT_DETAILS_PATH)
    all_data = load_cleaned_site_data()
    inputs = build_workflow_inputs(site_details, circuit_details, all_data)

    if update_ac_capacity or not CAPACITY_DERIVED_PATH.exists():
        generate_rated_capacity(
            site_details,
            inputs,
            SAPN2022_CONFORMANCE_CONFIG,
            CAPACITY_DERIVED_PATH,
        )

    capacity_derived = pl.read_csv(CAPACITY_DERIVED_PATH).select(
        "site_id",
        "metadata_ac_capacity_kw",
        "calculated_ac_capacity_kw",
        "chosen_ac_capacity_kw",
    )
    inputs["site_details"] = (
        site_details.join(capacity_derived, on="site_id", how="left")
        .with_columns(pl.col("chosen_ac_capacity_kw").alias("capacity_kw"))
    )
    return inputs


def _sapn2022_days(site_data):
    timestamp_dtype = site_data.schema["local_tstamp"]
    timezone_name = (
        timestamp_dtype.time_zone if isinstance(timestamp_dtype, pl.Datetime) else None
    )
    timezone = ZoneInfo(timezone_name) if timezone_name else None
    return [
        (
            day,
            datetime(
                2022,
                11,
                day,
                SAPN2022_DAY_START.hour,
                SAPN2022_DAY_START.minute,
                SAPN2022_DAY_START.second,
                tzinfo=timezone,
            ),
            datetime(
                2022,
                11,
                day,
                SAPN2022_DAY_END.hour,
                SAPN2022_DAY_END.minute,
                SAPN2022_DAY_END.second,
                tzinfo=timezone,
            ),
        )
        for day in SAPN2022_EVENT_DAYS
    ]


def _summarize_sapn2022_day(site_day_long, prepared_day_df):
    return summarize_nov2022_day_eligibility(
        site_day_long,
        prepared_day_df,
        coverage_threshold=SAPN2022_DAY_COVERAGE_THRESHOLD,
    )


SAPN2022_CONFORMANCE_CONFIG = DatasetConformanceConfig(
    name="sapn2022",
    load_inputs=load_sapn2022_inputs,
    day_provider=_sapn2022_days,
    eligibility_function=_summarize_sapn2022_day,
    capacity_estimator=None,
    output_dir=CONFORMANCE_OUTPUT_DIR,
    coverage_threshold=SAPN2022_DAY_COVERAGE_THRESHOLD,
    exclusion_fields=(
        "common_power_v10m_coverage_pct",
        "rows_common_power_v10m",
        "rows_with_power",
        "rows_with_v10m",
        "covered_seconds",
        "window_seconds",
        "total_rows",
    ),
    excluded_day_schema={
        "site_id": pl.Int64,
        "day": pl.Int64,
        "reason": pl.Utf8,
        "common_power_v10m_coverage_pct": pl.Float64,
        "rows_common_power_v10m": pl.Int64,
        "rows_with_power": pl.Int64,
        "rows_with_v10m": pl.Int64,
        "covered_seconds": pl.Float64,
        "window_seconds": pl.Float64,
        "total_rows": pl.Int64,
        "coverage_threshold_pct": pl.Float64,
    },
)
