"""Run SAPN2022 5-minute curtailment metrics from eligible buckets and uncurtailed PV."""

import argparse
from datetime import time
from pathlib import Path

import polars as pl
from path_config import require_local_path

import sapn2022_metrics_5m_data_checks as data_checks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADELAIDE_TZ = "Australia/Adelaide"
BUCKET_WINDOW_START = time(6, 0, 0)
BUCKET_WINDOW_END = time(18, 0, 0)

DEFAULT_ALL_UNCURTAILED = (
    PROJECT_ROOT / "outputs" / "prediction" / "all_uncurtailedPV_5m.parquet"
)
# The external tier-bucket export lives outside this repo, so the local SAPN
# root is defined in the ignored `local_paths.py` file instead of here.
SAPN_ROOT = require_local_path(
    "SAPN_ROOT",
    "root folder containing `updated results/phase b info for curtailment/tier based/`.",
)
DEFAULT_ELIGIBLE_BUCKETS5M = (
    SAPN_ROOT
    / "updated results"
    / "phase b info for curtailment"
    / "tier based"
    / "tier_based_5min_buckets.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "curtailed_estimates_5m"
OUTPUT_NAME = "curtailment_sapn2022_5m.parquet"

OUTPUT_COLUMNS = [
    "year",
    "month",
    "day",
    "site_id",
    "curtailment_sapn2022_sum",
    "curtailment_sapn2022_count",
    "null_uncurtailed_P_count",
    "total_count",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write SAPN2022 5-minute curtailed PV summaries from eligible buckets."
    )
    parser.add_argument("--eligible-buckets5m", type=Path, default=DEFAULT_ELIGIBLE_BUCKETS5M)
    parser.add_argument("--all-uncurtailed", type=Path, default=DEFAULT_ALL_UNCURTAILED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--site-id", type=int, default=None)
    return parser.parse_args()


def as_lazy(df):
    if isinstance(df, pl.LazyFrame):
        return df
    return df.lazy()


def eligible_bucket_timestamp_expr(column_name):
    """Parse bucket_5min_local directly as Adelaide local time."""
    return (
        pl.col(column_name)
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime(time_zone=ADELAIDE_TZ), "%Y-%m-%d %H:%M:%S%z", strict=False)
    )


def bucket_rows(eligible_buckets5m, site_id=None):
    """Prepare raw 5-minute bucket rows before LOS/OV1 eligibility filtering."""
    rows = (
        as_lazy(eligible_buckets5m)
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("los_or_ov1_flag").cast(pl.Int64),
            eligible_bucket_timestamp_expr("bucket_5min_local").alias("local_tstamp"),
        ])
        .filter(pl.col("local_tstamp").is_not_null())
    )

    if site_id is not None:
        rows = rows.filter(pl.col("site_id") == site_id)

    return rows.select(["site_id", "local_tstamp", "los_or_ov1_flag"])


def uncurtailed_rows(all_uncurtailed, site_id=None):
    """Prepare 5-minute uncurtailed PV rows at site/local timestamp grain."""
    rows = (
        as_lazy(all_uncurtailed)
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("local_tstamp").cast(pl.Datetime(time_zone=ADELAIDE_TZ)),
            pl.col("P_kw").cast(pl.Float64),
            pl.col("uncurtailed_P").cast(pl.Float64),
        ])
        .select([
            "site_id",
            "local_tstamp",
            "t_stamp",
            "P_kw",
            "uncurtailed_P",
            "GHI",
            "n_train",
        ])
    )

    if site_id is not None:
        rows = rows.filter(pl.col("site_id") == site_id)

    return rows


def bucket_window_uncurtailed_rows(all_uncurtailed):
    """Restrict all_uncurtailed rows to the raw tier-bucket day window for coverage checks."""
    return (
        as_lazy(all_uncurtailed)
        .filter(
            (pl.col("local_tstamp").dt.time() >= BUCKET_WINDOW_START)
            & (pl.col("local_tstamp").dt.time() <= BUCKET_WINDOW_END)
        )
    )


def score_curtailment(eligible_rows, all_uncurtailed):
    """Join eligible buckets to uncurtailed PV and calculate curtailed power."""
    return (
        eligible_rows
        .join(all_uncurtailed, on=["site_id", "local_tstamp"], how="inner")
        .with_columns([
            pl.col("local_tstamp").dt.year().alias("year"),
            pl.col("local_tstamp").dt.month().alias("month"),
            pl.col("local_tstamp").dt.day().alias("day"),
        ])
        .with_columns(
            pl.when(
                pl.col("uncurtailed_P").is_not_null()
                & pl.col("P_kw").is_not_null()
            )
            .then(pl.max_horizontal(pl.col("uncurtailed_P") - pl.col("P_kw"), pl.lit(0.0)))
            .otherwise(0.0)
            .alias("curtailed_P")
        )
    )


def daily_site_summary(scored_bins):
    """Aggregate to the original daily/site summary shape."""
    return (
        scored_bins
        .group_by(["year", "month", "day", "site_id"])
        .agg([
            pl.col("curtailed_P").sum().alias("curtailment_sapn2022_sum"),
            (pl.col("curtailed_P") > 0).sum().cast(pl.Int64).alias("curtailment_sapn2022_count"),
            pl.col("local_tstamp").count().cast(pl.Int64).alias("total_count"),
        ])
        .with_columns(pl.lit(0).cast(pl.Int64).alias("null_uncurtailed_P_count"))
        .select(OUTPUT_COLUMNS)
        .sort(["year", "month", "day", "site_id"])
    )


def diagnostics(scored_bins, eligible_bucket_count):
    """Summarise successful 5-minute join/scoring counts for stdout."""
    return (
        scored_bins
        .select([
            pl.lit(int(eligible_bucket_count)).alias("eligible_bucket_count"),
            pl.len().cast(pl.Int64).alias("matched_bucket_count"),
            (pl.col("curtailed_P") > 0).sum().cast(pl.Int64).alias("positive_curtailment_buckets"),
            pl.col("curtailed_P").sum().alias("curtailment_sapn2022_sum"),
        ])
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading eligible buckets: {args.eligible_buckets5m}")
    eligible_buckets5m = pl.scan_csv(args.eligible_buckets5m)

    print(f"Reading all_uncurtailedPV_5m: {args.all_uncurtailed}")
    all_uncurtailed = pl.scan_parquet(args.all_uncurtailed)

    print("Preparing raw 5-minute bucket timestamps")
    raw_bucket_rows = bucket_rows(eligible_buckets5m, site_id=args.site_id)

    print("Preparing 5-minute uncurtailed PV rows")
    uncurtailed = uncurtailed_rows(all_uncurtailed, site_id=args.site_id)
    uncurtailed_in_bucket_window = bucket_window_uncurtailed_rows(uncurtailed)

    print("Running data checks")
    data_checks.assert_unique_site_timestamp_keys(raw_bucket_rows, "tier_based_5min_buckets")
    data_checks.assert_unique_site_timestamp_keys(uncurtailed, "all_uncurtailedPV_5m")
    data_checks.assert_all_rows_have_match(
        uncurtailed_in_bucket_window,
        raw_bucket_rows,
        "all_uncurtailedPV_5m bucket-window",
        "tier_based_5min_buckets",
    )

    print("Matching shared 5-minute timestamps before LOS/OV1 filtering")
    matched_bucket_rows = (
        raw_bucket_rows
        .join(
            uncurtailed_in_bucket_window.select(["site_id", "local_tstamp"]),
            on=["site_id", "local_tstamp"],
            how="inner",
        )
    )

    print("Preparing eligible 5-minute buckets")
    eligible_rows = matched_bucket_rows.filter(pl.col("los_or_ov1_flag") == 1).select([
        "site_id",
        "local_tstamp",
    ])

    eligible_bucket_count = (
        eligible_rows
        .select(pl.len().alias("eligible_bucket_count"))
        .collect()
        .item(0, 0)
    )

    print("Scoring curtailed PV")
    scored_bins = score_curtailment(eligible_rows, uncurtailed)

    diag = diagnostics(scored_bins, eligible_bucket_count).collect().to_dicts()[0]
    print(
        "Diagnostics: "
        f"eligible_bucket_count={diag['eligible_bucket_count']}, "
        f"matched_bucket_count={diag['matched_bucket_count']}, "
        f"positive_curtailment_buckets={diag['positive_curtailment_buckets']}, "
        f"curtailment_sapn2022_sum={diag['curtailment_sapn2022_sum']}"
    )

    output_path = args.output_dir / args.output_name
    print(f"Writing {output_path}")
    daily_site_summary(scored_bins).collect().write_parquet(output_path, compression="zstd")
    print("Done")


if __name__ == "__main__":
    main()
