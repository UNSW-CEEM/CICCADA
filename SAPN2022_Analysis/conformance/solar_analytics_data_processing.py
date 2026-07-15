from pathlib import Path

import polars as pl

from funcs import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    convertPowerToKw,
)


DATA_DIR = Path(__file__).resolve().parent / "Solar Analytics"
CIRCUIT_METADATA_PATH = DATA_DIR / "circuit_metadata.csv"
OUTPUT_PATH = DATA_DIR / "data_cleaned.parquet"
TERMINAL_DURATION_SECONDS = 300


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing Solar Analytics folder at {DATA_DIR}")
    if not CIRCUIT_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing circuit metadata at {CIRCUIT_METADATA_PATH}")

    raw_parquet_paths = sorted(
        path
        for path in DATA_DIR.glob("*.parquet")
        if path.name != OUTPUT_PATH.name
    )
    if not raw_parquet_paths:
        raise FileNotFoundError(
            f"No raw parquet files found in {DATA_DIR}. "
            f"Expected one or more sample files alongside {CIRCUIT_METADATA_PATH.name}."
        )

    circuit_details = pl.read_csv(CIRCUIT_METADATA_PATH).rename(
        {
            "circuit_id": "c_id",
            "circuit_polarity": "polarity",
        }
    )

    all_data = (
        pl.scan_parquet([str(path) for path in raw_parquet_paths])
        .rename(
            {
                "circuit_id": "c_id",
                "t_stamp": "utc_tstamp",
            }
        )
        .filter(pl.col("utc_tstamp").is_not_null())
        .select(
            [
                pl.col("c_id").cast(pl.Int64),
                pl.col("utc_tstamp").cast(pl.Datetime),
                # the shared SAPN2022 cleaning path expects processed power in cW
                # before convertPowerToKw() divides by 100*1000
                (pl.col("power").cast(pl.Float64, strict=False) * 100.0).alias("power"), # convert to cW
                pl.col("voltage").cast(pl.Float64, strict=False).alias("voltage"),
            ]
        )
        # Duration must be derived on the full combined timeline, not per file.
        .sort(["c_id", "utc_tstamp"])
        .with_columns(
            [
                # Use the next timestamp within each circuit to recover the
                # measurement duration expected by the conformance code.
                pl.col("utc_tstamp").shift(-1).over("c_id").alias("_next_utc_tstamp"),
                # The shared voltage cleaning logic expects a vmean column.
                pl.col("voltage").alias("vmean"),
            ]
        )
        .with_columns(
            (
                pl.col("_next_utc_tstamp") - pl.col("utc_tstamp")
            ).dt.total_seconds().alias("duration")
        )
        .with_columns(
            [
                # Only the final row in each circuit has no next timestamp;
                # preserve real longer gaps (600, 900, ...) between observations.
                pl.col("duration")
                .fill_null(TERMINAL_DURATION_SECONDS)
                .cast(pl.Int64, strict=False),
                pl.col("utc_tstamp")
                .dt.strftime("%Y-%m-%d %H:%M:%S%.f")
                .alias("utc_tstamp"),
            ]
        )
        .drop("_next_utc_tstamp")
    )

    all_data = convertPowerToKw(all_data, convert=True)
    all_data = addLocalTStamp(all_data, add=True)
    all_data = addValidVoltage(all_data)
    all_data = addPolarityToPower(all_data, circuit_details)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_data.sink_parquet(OUTPUT_PATH, compression="zstd")
    print(f"Saved cleaned Solar Analytics data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
