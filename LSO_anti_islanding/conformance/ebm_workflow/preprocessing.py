"""SAPN measurement preprocessing."""

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from ebm_workflow.data_cleaning import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertWToKw,
    deduplicateMeasurements,
)
from ebm_workflow.loading import load_ebm_circuit_details


def write_cleaned_site_data(
    raw_path,
    raw_csv_glob,
    circuit_details_path,
    cleaned_path,
    local_timezone,
    *,
    deduplicate=True,
    num_buckets=128,
):
    """Write cleaned SAPN metrology, deduplicating one circuit-bucket at a time.

    Each bucket is selected directly from the raw parquet, passed through the
    established cleaning sequence, and appended to the same output parquet.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raw_csv_glob = Path(raw_csv_glob)
        csv_files = sorted(raw_csv_glob.parent.glob(raw_csv_glob.name))
        if not csv_files:
            raise FileNotFoundError(f"No raw CSV files match {raw_csv_glob}.")
        pl.scan_csv(csv_files).sink_parquet(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw processed site data at {raw_path}.")
    circuit_details_path = Path(circuit_details_path)
    if not circuit_details_path.exists():
        raise FileNotFoundError(f"Missing circuit details at {circuit_details_path}.")
    cleaned_path = Path(cleaned_path)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    circuit_details = load_ebm_circuit_details(circuit_details_path)
    mapped_circuit_ids = (
        circuit_details.filter(pl.col("site_id").is_not_null())
        .select("c_id", "site_id", "con_type")
        .unique()
        .lazy()
    )
    raw_data = pl.scan_parquet(raw_path)

    parquet_writer = None
    for bucket_number in range(num_buckets):
        print(
            f"Processing bucket {bucket_number + 1}/{num_buckets}...",
            flush=True,
        )
        all_data = raw_data.filter((pl.col("c_id") % num_buckets) == bucket_number)
        all_data = all_data.join(
            mapped_circuit_ids, on="c_id", how="inner"
        ).with_columns(pl.lit(local_timezone).alias("timezone"))
        all_data = convertWToKw(all_data)
        if deduplicate:
            all_data = deduplicateMeasurements(all_data)
        all_data = addLocalTStamp(all_data, local_timezone=local_timezone)
        all_data = addValidVoltage(all_data, fallback_col="vmean")
        all_data = addPolarityToPower(all_data, circuit_details)
        # Negative power is intentionally not clipped here because that would
        # also affect ac_load_net. PV clipping is handled later.

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
        raise RuntimeError("SAPN preprocessing produced no cleaned rows.")

    parquet_writer.close()
    print(f"Saved cleaned SAPN data to {cleaned_path}.", flush=True)
    return cleaned_path
