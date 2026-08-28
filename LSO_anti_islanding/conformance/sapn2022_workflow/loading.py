"""SAPN input loading."""

from pathlib import Path

import polars as pl
from sapn2022_workflow.config import REQUIRED_SITE_METADATA_ROWS
from sapn2022_workflow.sapn_paths import (
    CIRCUIT_DETAILS_PATH,
    CLEANED_SITE_DATA_PATH,
    SITE_DETAILS_PATH,
)


def load_sapn_site_details(site_details_path=SITE_DETAILS_PATH):
    """Load SAPN sites having exactly one source metadata row."""
    return (
        pl.read_csv(site_details_path)
        .with_columns(pl.len().over("site_id").alias("_site_metadata_rows"))
        .filter(pl.col("_site_metadata_rows") == REQUIRED_SITE_METADATA_ROWS)
        .drop("_site_metadata_rows")
        .with_columns(
            (pl.col("ac_cap_w").cast(pl.Float64, strict=False) / 1_000.0).alias(
                "capacity_kw"
            )
        )
    )


def load_sapn_circuit_details(circuit_details_path=CIRCUIT_DETAILS_PATH):
    return pl.read_csv(circuit_details_path)


def load_sapn_cleaned_data(cleaned_path=CLEANED_SITE_DATA_PATH):
    return pl.scan_parquet(Path(cleaned_path))
