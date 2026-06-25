"""
Run script 4 for the local SAPN2022 curtailment workflow.

This is the SAPN2022-specific curtailment summary layer. It combines:

- SAPN Phase B timestamp responsibility flags; and
- all_uncurtailedPV estimates from write_all_uncurtailedPV.py.

The final calculation is done at the validation timestamp grain. Phase B detail
is already a subset of SAPN validation timestamps: it only contains the local
event-day windows exported by the compliance workflow, not every validation
timestamp. all_uncurtailedPV is broader and covers all model-eligible validation
rows. The curtailment join therefore intentionally starts from Phase B rows and
matches them to all_uncurtailedPV on the exact Adelaide local timestamp after
normalising both sides to the same naive local datetime representation.
"""

import argparse
from pathlib import Path

import polars as pl
from path_config import require_local_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADELAIDE_TZ = "Australia/Adelaide"
DISCONNECT_POWER_FRACTION = 0.04

DEFAULT_ALL_UNCURTAILED = (
    PROJECT_ROOT / "outputs" / "local_scored" / "all_uncurtailedPV.parquet"
)
DEFAULT_STRUCTURED_DATA = (
    PROJECT_ROOT / "outputs" / "all_structured_data_test" / "structured_data.parquet"
)
# The Phase B detail CSV lives outside this repo, so its machine-specific path
# is defined in the ignored `local_paths.py` file instead of being committed.
DEFAULT_PHASE_B_DETAIL = require_local_path(
    "PHASE_B_TIMESTAMP_DETAIL_PATH",
    "Phase B timestamp detail CSV used by the legacy exact-timestamp curtailment summary.",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "local_metrics"
OUTPUT_NAME = "curtailment_sapn2022.parquet"


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
        description="Write SAPN2022 curtailed PV summaries from Phase B flags and all_uncurtailedPV."
    )
    parser.add_argument("--all-uncurtailed", type=Path, default=DEFAULT_ALL_UNCURTAILED)
    parser.add_argument("--structured-data", type=Path, default=DEFAULT_STRUCTURED_DATA)
    parser.add_argument("--phase-b-detail", type=Path, default=DEFAULT_PHASE_B_DETAIL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--site-id", type=int, default=None)
    return parser.parse_args()


def as_lazy(df):
    if isinstance(df, pl.LazyFrame):
        return df
    return df.lazy()


def utc_to_adelaide_local_timestamp_expr(column_name):
    """Convert a UTC/source timestamp to naive Adelaide local wall time."""
    return (
        pl.col(column_name)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(ADELAIDE_TZ)
        .dt.replace_time_zone(None)
    )


def phase_b_local_timestamp_expr(column_name):
    """Parse Phase B local timestamp strings as naive Adelaide local wall time."""
    return (
        pl.col(column_name)
        .cast(pl.Utf8)
        .str.slice(0, 26)
        .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
    )


def phase_b_responsible_rows(phase_b_detail, site_id=None):
    """Build one SAPN-responsibility flag per exact site/local timestamp."""
    rows = (
        as_lazy(phase_b_detail)
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            phase_b_local_timestamp_expr("local_tstamp").alias("local_tstamp_naive"),
            pl.col("los_responsible").cast(pl.Boolean, strict=False).fill_null(False),
            pl.col("ov1_responsible").cast(pl.Boolean, strict=False).fill_null(False),
        ])
        .with_columns(
            (pl.col("los_responsible") | pl.col("ov1_responsible")).alias("is_responsible")
        )
    )

    if site_id is not None:
        rows = rows.filter(pl.col("site_id") == site_id)

    return (
        rows
        .filter(pl.col("local_tstamp_naive").is_not_null())
        .group_by(["site_id", "local_tstamp_naive"])
        .agg([
            pl.col("is_responsible").any().alias("timestamp_is_responsible"),
            pl.col("is_responsible").sum().cast(pl.Int64).alias("responsible_count"),
            pl.len().alias("phase_b_row_count"),
        ])
        .filter(pl.col("timestamp_is_responsible"))
    )


def uncurtailed_rows(all_uncurtailed, site_id=None):
    """Prepare model-scored validation rows at exact Adelaide local timestamp grain."""
    rows = (
        as_lazy(all_uncurtailed)
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            utc_to_adelaide_local_timestamp_expr("t_stamp").alias("local_tstamp_naive"),
            pl.col("P_kw").cast(pl.Float64),
            pl.col("uncurtailed_P").cast(pl.Float64),
        ])
        .select([
            "site_id",
            "local_tstamp_naive",
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


def capacity_by_site(structured_data, site_id=None):
    """Read site capacity needed for the original-style SAPN max-power gate."""
    rows = (
        as_lazy(structured_data)
        .filter(pl.col("dataset_role") == "validation")
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("ac_capacity_kw").cast(pl.Float64, strict=False),
        ])
        .group_by("site_id")
        .agg(pl.col("ac_capacity_kw").max().alias("ac_capacity_kw"))
    )

    if site_id is not None:
        rows = rows.filter(pl.col("site_id") == site_id)

    return rows


def score_curtailment(phase_bins, all_uncurtailed, capacities):
    """Join responsible timestamps to uncurtailed PV and calculate curtailed PV."""
    return (
        phase_bins
        .join(all_uncurtailed, on=["site_id", "local_tstamp_naive"], how="left")
        .join(capacities, on="site_id", how="left")
        .with_columns([
            (pl.col("ac_capacity_kw") * DISCONNECT_POWER_FRACTION).alias("max_P_sapn2022"),
            pl.col("local_tstamp_naive").dt.year().alias("year"),
            pl.col("local_tstamp_naive").dt.month().alias("month"),
            pl.col("local_tstamp_naive").dt.day().alias("day"),
        ])
        .with_columns(
            pl.when(
                pl.col("timestamp_is_responsible")
                & pl.col("uncurtailed_P").is_not_null()
                & pl.col("P_kw").is_not_null()
                & pl.col("max_P_sapn2022").is_not_null()
                & (pl.col("uncurtailed_P") > pl.col("max_P_sapn2022"))
                & (pl.col("P_kw") < pl.col("max_P_sapn2022"))
            )
            .then(pl.col("uncurtailed_P") - pl.col("P_kw"))
            .otherwise(0.0)
            .alias("curtailed_P")
        )
    )


def daily_site_summary(scored_bins):
    """Aggregate to the original daily/site summary shape, curtailment only."""
    return (
        scored_bins
        .filter(
            (pl.col("uncurtailed_P") > pl.col("max_P_sapn2022"))
            | pl.col("uncurtailed_P").is_null()
        )
        .group_by(["year", "month", "day", "site_id"])
        .agg([
            pl.col("curtailed_P").sum().alias("curtailment_sapn2022_sum"),
            (pl.col("curtailed_P") > 0).sum().cast(pl.Int64).alias("curtailment_sapn2022_count"),
            pl.col("uncurtailed_P").is_null().sum().cast(pl.Int64).alias("null_uncurtailed_P_count"),
            pl.len().alias("total_count"),
        ])
        .select(OUTPUT_COLUMNS)
        .sort(["year", "month", "day", "site_id"])
    )


def diagnostics(scored_bins):
    """Summarise join coverage and positive curtailment counts for stdout."""
    return (
        scored_bins
        .select([
            pl.len().alias("responsible_timestamps"),
            pl.col("uncurtailed_P").is_not_null().sum().alias("matched_timestamps"),
            pl.col("uncurtailed_P").is_null().sum().alias("unmatched_responsible_timestamps"),
            pl.col("ac_capacity_kw").is_null().sum().alias("missing_capacity_timestamps"),
            (
                (pl.col("uncurtailed_P") > pl.col("max_P_sapn2022"))
                | pl.col("uncurtailed_P").is_null()
            ).sum().alias("original_style_metric_timestamps"),
            (pl.col("curtailed_P") > 0).sum().alias("positive_curtailment_timestamps"),
            pl.col("curtailed_P").sum().alias("curtailment_sapn2022_sum"),
        ])
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Phase B detail: {args.phase_b_detail}")
    phase_b_detail = pl.scan_csv(args.phase_b_detail)

    print(f"Reading all_uncurtailedPV: {args.all_uncurtailed}")
    all_uncurtailed = pl.scan_parquet(args.all_uncurtailed)

    print(f"Reading structured data for capacities: {args.structured_data}")
    structured_data = pl.scan_parquet(args.structured_data)

    print("Building SAPN-responsible timestamps")
    phase_bins = phase_b_responsible_rows(phase_b_detail, site_id=args.site_id)

    print("Preparing uncurtailed PV rows")
    uncurtailed = uncurtailed_rows(all_uncurtailed, site_id=args.site_id)

    print("Preparing site capacities")
    capacities = capacity_by_site(structured_data, site_id=args.site_id)

    print("Scoring curtailed PV")
    scored_bins = score_curtailment(phase_bins, uncurtailed, capacities)

    diag = diagnostics(scored_bins).collect().to_dicts()[0]
    print(
        "Diagnostics: "
        f"responsible_timestamps={diag['responsible_timestamps']}, "
        f"matched_timestamps={diag['matched_timestamps']}, "
        f"unmatched_responsible_timestamps={diag['unmatched_responsible_timestamps']}, "
        f"missing_capacity_timestamps={diag['missing_capacity_timestamps']}, "
        f"original_style_metric_timestamps={diag['original_style_metric_timestamps']}, "
        f"positive_curtailment_timestamps={diag['positive_curtailment_timestamps']}, "
        f"curtailment_sapn2022_sum={diag['curtailment_sapn2022_sum']}"
    )

    output_path = args.output_dir / args.output_name
    print(f"Writing {output_path}")
    daily_site_summary(scored_bins).collect().write_parquet(output_path, compression="zstd")
    print("Done")


if __name__ == "__main__":
    main()
