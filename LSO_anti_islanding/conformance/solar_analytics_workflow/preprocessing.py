"""Solar Analytics orchestration of shared measurement cleaning operations."""

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from config import LOCAL_TIMEZONE
from core.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    clipNegativePower,
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


def load_circuit_details(path=CIRCUIT_METADATA_PATH):
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
    *,
    deduplicate=True,
    num_buckets=128,
):
    """Build Solar Analytics metrology one circuit bucket at a time."""
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

    circuit_details = load_circuit_details(circuit_metadata_path)
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

    raw_data = (
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

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_writer = None
    for bucket_number in range(num_buckets):
        print(
            f"Processing bucket {bucket_number + 1}/{num_buckets}...",
            flush=True,
        )
        all_data = raw_data.filter(
            (pl.col("c_id") % num_buckets) == bucket_number
        )
        all_data = (
            all_data.join(circuit_site_lookup, on="c_id", how="inner")
            .join(site_lookup, on="site_id", how="left")
            .with_columns(
                pl.col("state")
                .replace_strict(STATE_TIMEZONES, default=LOCAL_TIMEZONE)
                .alias("timezone")
            )
        )
        all_data = convertWToKw(all_data)
        if deduplicate:
            all_data = deduplicateMeasurements(all_data)
        all_data = addLocalTStamp(all_data)
        all_data = addValidVoltage(all_data)
        all_data = addPolarityToPower(all_data, circuit_details)
        # commenting this as it appliies to ac_load_net too which is incorrect
        # but clipping of pv below 0 still happens in checkpvbehaviour
        # and _robust_observed_peak_kw
        # after polarity is applied here
        # all_data = clipNegativePower(all_data)

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

        cleaned_bucket = all_data.collect(engine="streaming")
        if cleaned_bucket.is_empty():
            print("No output rows for this bucket.", flush=True)
            del cleaned_bucket
            continue

        table = cleaned_bucket.to_arrow()
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                cleaned_path,
                table.schema,
                compression="zstd",
            )

        parquet_writer.write_table(table)
        print(
            f"Wrote {table.num_rows:,} rows from bucket {bucket_number}.",
            flush=True,
        )
        del table
        del cleaned_bucket

    if parquet_writer is None:
        raise RuntimeError("Solar Analytics preprocessing produced no cleaned rows.")

    parquet_writer.close()
    print(f"Saved cleaned Solar Analytics data to {cleaned_path}.", flush=True)
    return cleaned_path
