"""Solar Analytics local input loading."""

from pathlib import Path

import polars as pl
from solar_analytics_workflow.preprocessing import load_circuit_details
from solar_analytics_workflow.rated_capacity import add_s_rated_capacity
from solar_analytics_workflow.solar_paths import (
    CIRCUIT_METADATA_PATH,
    CLEANED_DATA_PATH,
    SITE_METADATA_PATH,
)


def load_solar_analytics_site_details(site_metadata_path=SITE_METADATA_PATH):
    site_details = (
        pl.read_csv(site_metadata_path)
        .unique(subset=["site_id"], keep="first", maintain_order=True)
        .filter(pl.col("inverter_count") == 1)
    )
    return add_s_rated_capacity(site_details)


def load_solar_analytics_circuit_details(
    circuit_metadata_path=CIRCUIT_METADATA_PATH,
):
    return load_circuit_details(circuit_metadata_path)


def load_solar_analytics_cleaned_data(cleaned_path=CLEANED_DATA_PATH):
    all_data = pl.scan_parquet(Path(cleaned_path))
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
            f"Solar cleaned data at {cleaned_path} is missing "
            f"{sorted(missing_columns)}. Run "
            "run_solar_analytics_preprocessing.py again."
        )
    return all_data
