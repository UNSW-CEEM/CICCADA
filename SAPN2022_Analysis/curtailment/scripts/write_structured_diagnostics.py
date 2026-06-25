"""Write diagnostics from an existing structured_data.parquet file.

This keeps diagnostics generation separate from the heavier structured-data
build so coverage checks can be regenerated even when the main build is slow
or resource-constrained.
"""

import argparse
from pathlib import Path

import polars as pl

import build_structured_high_resolution as bsl
from structured_data_shared_params import bom_daily_parquets, prepare_bom10min


DEFAULT_STRUCTURED_PARQUET = (
    bsl.PROJECT_ROOT / "outputs" / "all_structured_data_test" / "structured_data.parquet"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write diagnostics from an existing structured_data.parquet file."
    )
    parser.add_argument(
        "--structured-parquet",
        type=Path,
        default=DEFAULT_STRUCTURED_PARQUET,
        help="Path to an existing structured_data.parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where the diagnostics/ folder should be written. Defaults to the parquet folder.",
    )
    parser.add_argument(
        "--bom-root",
        type=Path,
        default=bsl.DEFAULT_BOM_ROOT,
        help="Root directory containing BOM parquet files.",
    )
    parser.add_argument(
        "--site-id",
        type=int,
        help="Optional site_id filter for one-site diagnostics regeneration.",
    )
    return parser.parse_args()


def main(
    structured_parquet=DEFAULT_STRUCTURED_PARQUET,
    output_dir=None,
    bom_root=bsl.DEFAULT_BOM_ROOT,
    site_id=None,
):
    structured_parquet = Path(structured_parquet)
    if not structured_parquet.exists():
        raise FileNotFoundError(f"Structured parquet not found: {structured_parquet}")

    output_dir = Path(output_dir) if output_dir else structured_parquet.parent

    print(f"Reading {structured_parquet}")
    structured = pl.read_parquet(structured_parquet)
    if site_id is not None:
        structured = structured.filter(pl.col("site_id") == site_id)
    if structured.is_empty():
        raise ValueError("No rows found for the requested parquet/site filter")
    structured_lf = structured.lazy()

    train_start_day, train_end_day = (
        structured
        .filter(pl.col("dataset_role") == "train")
        .select([
            pl.col("actual_day").min().alias("train_start_day"),
            pl.col("actual_day").max().alias("train_end_day"),
        ])
        .row(0)
    )
    if train_start_day is None or train_end_day is None:
        raise ValueError("Structured parquet has no train rows; cannot derive clear-sky candidate window")

    bom_mapping = (
        structured
        .select(["site_id", "n_lat", "n_long"])
        .unique()
        .sort("site_id")
    )

    print("Preparing BOM diagnostics context")
    bom_files = bom_daily_parquets(bom_root, train_start_day, train_end_day)
    bom10min = prepare_bom10min(bom_files, bom_mapping)
    clear_sky_day_diagnostics = bsl.build_clear_sky_day_diagnostics(
        structured_lf,
        structured_lf,
        bom10min,
    )

    print(f"Writing diagnostics to {output_dir / 'diagnostics'}")
    bsl.write_diagnostic_files(
        output_dir,
        bom_mapping,
        structured,
        clear_sky_day_diagnostics=clear_sky_day_diagnostics,
    )
    print("Done")


if __name__ == "__main__":
    main(**vars(parse_args()))
