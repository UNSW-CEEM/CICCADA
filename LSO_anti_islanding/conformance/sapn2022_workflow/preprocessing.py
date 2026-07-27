"""SAPN-specific orchestration of shared measurement cleaning operations."""

from pathlib import Path

import polars as pl

from config import LOCAL_TIMEZONE
from core.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertcWToKw,
    deduplicateMeasurements,
)
from sapn2022_workflow.sapn_paths import (
    CIRCUIT_DETAILS_PATH,
    CLEANED_SITE_DATA_PATH,
    RAW_SITE_DATA_PATH,
)


def build_cleaned_site_data(
    raw_path=RAW_SITE_DATA_PATH,
    circuit_details_path=CIRCUIT_DETAILS_PATH,
    cleaned_path=CLEANED_SITE_DATA_PATH,
    *,
    deduplicate=False,
):
    """Build cleaned SAPN metrology with low peak-memory behaviour.

    Whole-dataset deduplication is optional here because the SAPN raw parquet is
    large enough for the anti-join implementation to exhaust memory. The
    conformance workflow still applies the same duplicate policy later when it
    slices each site-day in ``extract_site_day``.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw processed site data at {raw_path}.")
    circuit_details_path = Path(circuit_details_path)
    if not circuit_details_path.exists():
        raise FileNotFoundError(f"Missing circuit details at {circuit_details_path}.")

    circuit_details = pl.read_csv(circuit_details_path)
    mapped_circuit_ids = (
        circuit_details.filter(pl.col("site_id").is_not_null())
        .select("c_id")
        .unique()
        .lazy()
    )
    all_data = (
        pl.scan_parquet(raw_path)
        .join(mapped_circuit_ids, on="c_id", how="inner")
        .with_columns(pl.lit(LOCAL_TIMEZONE).alias("timezone"))
    )
    if deduplicate:
        all_data = deduplicateMeasurements(all_data)
    all_data = convertcWToKw(all_data)
    all_data = addLocalTStamp(all_data)
    all_data = addValidVoltage(all_data, fallback_col="vmean")
    all_data = addPolarityToPower(all_data, circuit_details)

    cleaned_path = Path(cleaned_path)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    all_data.sink_parquet(cleaned_path, compression="zstd")
    return cleaned_path
