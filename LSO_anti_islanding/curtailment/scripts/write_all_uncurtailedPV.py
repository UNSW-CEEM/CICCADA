"""
Write script 3 for the local SAPN2022 curtailment workflow.

This works with either of the data resolution

This is the local Polars equivalent of the original
Write_All_uncartailedPV notebook. It applies the fitted GHI-normalised model
to validation rows and writes an all_uncurtailedPV-style parquet table.

Original scoring model:
    x = GHI / GHI_cs
    P_kw_norm_est_raw = P_kw_norm_cs * (a + b*x)
    P_kw_norm_est = max(P_kw_norm_est_raw, P_kw_norm)

The original output column named GHI contains x, the normalised irradiance,
not the raw GHI value.
"""

import argparse
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADELAIDE_TZ = "Australia/Adelaide"
VALIDATION_P_KW_NORM_EPS = 0.05

# all data
DEFAULT_STRUCTURED_DATA = (
    PROJECT_ROOT / "outputs" / "all_structured_data_5m" / "structured_data_5m.parquet"
)

# trained modle parameters
DEFAULT_MODEL_DATA = PROJECT_ROOT / "outputs" / "model" / "pv_ghi_norm_model_5m.parquet"

# output estimate of this file
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prediction"
OUTPUT_NAME = "all_uncurtailedPV_5m.parquet"


ALL_UNCURTAILED_COLUMNS = [
    "site_id",
    "t_stamp",
    "local_tstamp",
    "tod_bin",
    "year",
    "month",
    "uncurtailed_P",
    "P_kw",
    "GHI",
    "n_train",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write local all_uncurtailedPV estimates from structured data and a GHI model."
    )
    parser.add_argument("--structured-data", type=Path, default=DEFAULT_STRUCTURED_DATA)
    parser.add_argument("--model-data", type=Path, default=DEFAULT_MODEL_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--site-id", type=int, default=None)
    return parser.parse_args()


def as_lazy(df):
    if isinstance(df, pl.LazyFrame):
        return df
    return df.lazy()


def eligible_validation_rows(structured_data, site_id=None):
    """Apply the original Write_All_uncartailedPV row filters."""
    rows = (
        structured_data.filter(pl.col("dataset_role") == "validation")
        .filter(pl.col("P_kw_norm_cs") > 0.2)
        .filter(pl.col("GHI") > 50)
        # .filter(pl.col("P_kw_norm") > 0.05)
        .filter(pl.col("P_kw_norm") >= 0)  # modified this as it can be 0 in our case
        # .filter(pl.col("P_kw_norm") <= pl.col("P_kw_norm_cs"))
        # Validation-only tolerance: does not affect model training,
        # but can affect plotting and downstream curtailment results.
        .filter(
            pl.col("P_kw_norm") <= (pl.col("P_kw_norm_cs") + VALIDATION_P_KW_NORM_EPS)
        )
        .with_columns(
            [
                pl.col("actual_tod").alias("tod_bin"),
                (pl.col("GHI") / pl.col("GHI_cs")).alias("x"),
            ]
        )
    )

    if site_id is not None:
        rows = rows.filter(pl.col("site_id") == site_id)

    return rows


def score_uncurtailed(eligible_data, model_data):
    """Join the model and produce the all_uncurtailedPV output shape."""
    model = as_lazy(model_data).select(
        [
            "site_id",
            "tod_bin",
            "a",
            "b",
            pl.col("n").alias("n_train"),
        ]
    )

    return (
        eligible_data
        # need to verify how it will handle multiple tod_bin, becasue there should be multiple!
        # perhaps it does not matter that much becasue the timestamps would still be unique
        # make sure this is in line with script 4
        .join(model, on=["site_id", "tod_bin"], how="inner")
        .with_columns(
            (pl.col("P_kw_norm_cs") * (pl.col("a") + pl.col("b") * pl.col("x"))).alias(
                "P_kw_norm_est_raw"
            )
        )
        .with_columns(
            pl.when(pl.col("P_kw_norm_est_raw") >= pl.col("P_kw_norm"))
            .then(pl.col("P_kw_norm_est_raw"))
            .otherwise(pl.col("P_kw_norm"))
            .alias("P_kw_norm_est")
        )
        .filter(pl.col("P_kw_norm_est").is_not_null())
        .with_columns(
            [
                # t_stamp is preserved as the original UTC/source validation
                # timestamp. local_tstamp and tod_bin are added only for clarity.
                pl.col("t_stamp")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone(ADELAIDE_TZ)
                .alias("local_tstamp"),
                pl.col("t_stamp").dt.year().alias("year"),
                pl.col("t_stamp").dt.month().alias("month"),
                (pl.col("P_kw_norm_est") * pl.col("S_99")).alias("uncurtailed_P"),
                (pl.col("P_kw_norm") * pl.col("S_99")).alias("P_kw"),
                pl.col("x").alias("GHI"),
                pl.col("n_train").cast(pl.Int64),
            ]
        )
        .select(ALL_UNCURTAILED_COLUMNS)
        .sort(["site_id", "t_stamp"])
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading structured data: {args.structured_data}")
    structured_data = pl.scan_parquet(args.structured_data)

    print(f"Reading model data: {args.model_data}")
    model_data = pl.scan_parquet(args.model_data)

    print("Preparing eligible validation rows")
    eligible_data = eligible_validation_rows(structured_data, site_id=args.site_id)

    print("Scoring uncurtailed PV")
    scored = score_uncurtailed(eligible_data, model_data)

    output_path = args.output_dir / args.output_name
    print(f"Writing {output_path}")
    scored.sink_parquet(output_path, compression="zstd")
    print("Done")


if __name__ == "__main__":
    main()
