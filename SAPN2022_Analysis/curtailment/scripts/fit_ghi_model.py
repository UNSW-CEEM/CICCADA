"""
Fit script 2 for the local SAPN2022 curtailment workflow.

This should work with either of the data resolution

This reads the structured parquet produced by build_structured_local.py and
fits the same GHI-normalised model as the original model_ghi_norm notebook.

Original model:
    x = GHI / GHI_cs
    y = P_kw_norm / P_kw_norm_cs
    y = a + b*x

The original Trino code stores:
    b = regr_slope(y, x)
    a = 1 - b

That forces the fitted line through the clear-sky point x=1, y=1.
"""

import argparse
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STRUCTURED_DATA = (
    PROJECT_ROOT / "outputs" / "all_structured_data_5m" / "structured_data_5m.parquet"
    # PROJECT_ROOT / "outputs" / "local_structured" / "structured_data.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model"
MODEL_OUTPUT_NAME = "pv_ghi_norm_model_5m.parquet"


MODEL_COLUMNS = [
    "site_id",
    "tod_bin",
    "a",
    "b",
    "n",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the local GHI-normalised PV model from structured data."
    )
    parser.add_argument("--structured-data", type=Path, default=DEFAULT_STRUCTURED_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=MODEL_OUTPUT_NAME)
    return parser.parse_args()


def training_rows(structured_data):
    """Apply the original model_ghi_norm training filters.

    The local structured data replaces original split_days with dataset_role.
    Only dataset_role == train is used to fit this model.
    """
    return (
        structured_data
        .filter(pl.col("dataset_role") == "train")
        .filter(pl.col("P_kw_norm_cs") > 0.2)
        .filter(pl.col("GHI") > 50)
        .filter(pl.col("P_kw_norm") > 0.05)
        .filter(pl.col("P_kw_norm") <= pl.col("P_kw_norm_cs"))
        .filter(pl.col("V") <= 253)
        # confirm S_norm works here as we dont have Q
        .filter((pl.col("P_kw_norm") >= 1) | (pl.col("S_norm") < 1.001))
        .with_columns([
            pl.col("actual_tod").alias("tod_bin"),
            (pl.col("GHI") / pl.col("GHI_cs")).alias("x"),
            (pl.col("P_kw_norm") / pl.col("P_kw_norm_cs")).alias("y"),
        ])
        .select(["site_id", "tod_bin", "x", "y"])
    )


def fit_model(train_data):
    """Fit one slope per site and 5-minute time-of-day bin."""
    valid_xy = pl.col("x").is_not_null() & pl.col("y").is_not_null()
    grouped = (
        train_data
        .group_by(["site_id", "tod_bin"])
        .agg([
            pl.len().alias("n"),
            pl.col("x").filter(valid_xy).count().alias("n_regr"),
            pl.col("x").filter(valid_xy).sum().alias("sum_x"),
            pl.col("y").filter(valid_xy).sum().alias("sum_y"),
            (pl.col("x") * pl.col("y")).filter(valid_xy).sum().alias("sum_xy"),
            (pl.col("x") * pl.col("x")).filter(valid_xy).sum().alias("sum_x2"),
        ])
        .with_columns(
            ((pl.col("n_regr") * pl.col("sum_xy")) - (pl.col("sum_x") * pl.col("sum_y"))).alias("slope_num")
        )
        .with_columns(
            ((pl.col("n_regr") * pl.col("sum_x2")) - (pl.col("sum_x") * pl.col("sum_x"))).alias("slope_den")
        )
        .with_columns(
            pl.when(pl.col("slope_den") != 0)
            .then(pl.col("slope_num") / pl.col("slope_den"))
            .otherwise(None)
            .alias("b")
        )
        .with_columns((1 - pl.col("b")).alias("a"))
        .select(MODEL_COLUMNS)
        .sort(["site_id", "tod_bin"])
    )
    return grouped


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.structured_data}")
    structured_data = pl.scan_parquet(args.structured_data)

    print("Preparing training rows")
    train_data = training_rows(structured_data)

    print("Fitting model")
    model = fit_model(train_data)

    output_path = args.output_dir / args.output_name
    print(f"Writing {output_path}")
    model.collect().write_parquet(output_path, compression="zstd")
    print("Done")


if __name__ == "__main__":
    main()
