"""Repository-relative paths for the Solar Analytics workflow."""

from pathlib import Path

CONFORMANCE_DIR = Path(__file__).resolve().parents[1]
SOLAR_ANALYTICS_DATA_DIR = CONFORMANCE_DIR / "datasets" / "Solar Analytics"

RAW_DATA_FILE_EG = SOLAR_ANALYTICS_DATA_DIR / "20240102.parquet"

CIRCUIT_METADATA_PATH = SOLAR_ANALYTICS_DATA_DIR / "circuit_metadata.csv"

SITE_METADATA_PATH = SOLAR_ANALYTICS_DATA_DIR / "site_metadata.csv"

CLEANED_DATA_PATH = SOLAR_ANALYTICS_DATA_DIR / "data_cleaned.parquet"

CONFORMANCE_OUTPUT_DIR = CONFORMANCE_DIR / "updated results" / "solar_analytics"
