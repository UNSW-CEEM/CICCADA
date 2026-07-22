"""SAPN-specific orchestration of shared measurement cleaning operations."""

from pathlib import Path

import polars as pl

from core.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertPowerToKw,
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
):
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw processed site data at {raw_path}.")
    circuit_details_path = Path(circuit_details_path)
    if not circuit_details_path.exists():
        raise FileNotFoundError(f"Missing circuit details at {circuit_details_path}.")

    circuit_details = pl.read_csv(circuit_details_path)
    all_data = pl.scan_parquet(raw_path)
    all_data = convertPowerToKw(all_data, convert=True)
    all_data = addLocalTStamp(all_data, add=True)
    all_data = addValidVoltage(all_data)
    all_data = addPolarityToPower(all_data, circuit_details)

    cleaned_path = Path(cleaned_path)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    all_data.sink_parquet(cleaned_path, compression="zstd")
    return cleaned_path
