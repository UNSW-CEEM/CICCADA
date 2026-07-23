"""Solar Analytics orchestration of shared measurement cleaning operations."""

from pathlib import Path

import polars as pl

from config import LOCAL_TIMEZONE
from core.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertWToKw,
    deduplicateMeasurements,
)
from solar_analytics_workflow.solar_paths import (
    CIRCUIT_METADATA_PATH,
    CLEANED_DATA_PATH,
    SITE_METADATA_PATH,
    SOLAR_ANALYTICS_DATA_DIR,
)


STATE_TIMEZONES = {
    "ACT": "Australia/Sydney",
    "NSW": "Australia/Sydney",
    "NT": "Australia/Darwin",
    "QLD": "Australia/Brisbane",
    "SA": "Australia/Adelaide",
    "TAS": "Australia/Hobart",
    "VIC": "Australia/Melbourne",
    "WA": "Australia/Perth",
}


def _raw_parquet_paths(data_dir, cleaned_path):
    cleaned_path = cleaned_path.resolve()
    return sorted(
        path
        for path in data_dir.glob("*.parquet")
        if path.resolve() != cleaned_path
        and path.name != CLEANED_DATA_PATH.name
    )


def _load_circuit_details(path):
    circuit_details = pl.read_csv(path).rename({
        "circuit_id": "c_id",
        "circuit_polarity": "polarity",
        "circuit_type": "con_type",
    })
    if circuit_details["c_id"].n_unique() != circuit_details.height:
        raise ValueError(
            "Solar Analytics circuit_id must be unique when device_id is ignored."
        )
    return circuit_details


def build_cleaned_site_data(
    data_dir=SOLAR_ANALYTICS_DATA_DIR,
    circuit_metadata_path=CIRCUIT_METADATA_PATH,
    site_metadata_path=SITE_METADATA_PATH,
    cleaned_path=CLEANED_DATA_PATH,
):
    """Build standardised long-form Solar Analytics metrology."""
    data_dir = Path(data_dir)
    circuit_metadata_path = Path(circuit_metadata_path)
    site_metadata_path = Path(site_metadata_path)
    cleaned_path = Path(cleaned_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing Solar Analytics folder at {data_dir}.")
    if not circuit_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing circuit metadata at {circuit_metadata_path}."
        )
    if not site_metadata_path.exists():
        raise FileNotFoundError(f"Missing site metadata at {site_metadata_path}.")

    raw_parquet_paths = _raw_parquet_paths(data_dir, cleaned_path)
    if not raw_parquet_paths:
        raise FileNotFoundError(
            f"No raw parquet files found in {data_dir}. Expected one or more "
            f"source files alongside {circuit_metadata_path.name}."
        )

    circuit_details = _load_circuit_details(circuit_metadata_path)
    circuit_site_lookup = (
        circuit_details.filter(pl.col("site_id").is_not_null())
        .select(["c_id", "site_id", "con_type"])
        .lazy()
    )
    site_lookup = (
        pl.read_csv(site_metadata_path)
        .select(["site_id", "state"])
        .unique(subset=["site_id"], keep="first")
        .lazy()
    )

    all_data = (
        pl.scan_parquet([str(path) for path in raw_parquet_paths])
        .rename({
            "circuit_id": "c_id",
            "t_stamp": "utc_tstamp",
        })
        .filter(pl.col("utc_tstamp").is_not_null())
        .select([
            pl.col("c_id").cast(pl.Int64),
            pl.col("utc_tstamp"),
            pl.col("power").cast(pl.Float64, strict=False),
            pl.col("voltage").cast(pl.Float64, strict=False),
        ])
    )
    all_data = deduplicateMeasurements(all_data)
    all_data = (
        all_data.join(circuit_site_lookup, on="c_id", how="inner")
        .join(site_lookup, on="site_id", how="left")
        .with_columns(
            pl.col("state")
            .replace_strict(STATE_TIMEZONES, default=LOCAL_TIMEZONE)
            .alias("timezone")
        )
        .with_columns(pl.col("voltage").alias("vmean"))
    )
    all_data = convertWToKw(all_data)
    all_data = addLocalTStamp(all_data)
    all_data = addValidVoltage(all_data)
    all_data = addPolarityToPower(all_data, circuit_details)
    all_data = all_data.select([
        "c_id",
        "site_id",
        "con_type",
        "state",
        "timezone",
        "utc_tstamp",
        "local_tstamp",
        "power",
        "voltage",
        "voltage_valid",
    ])

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    all_data.sink_parquet(cleaned_path, compression="zstd")
    return cleaned_path
