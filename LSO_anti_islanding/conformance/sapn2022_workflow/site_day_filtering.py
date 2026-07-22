"""SAPN duration-based site-day eligibility policy."""

import polars as pl

from config import SAPN2022_DAY_COVERAGE_THRESHOLD


def summarize_nov2022_day_eligibility(
    site_day_long,
    prepared_day_df,
    coverage_threshold=SAPN2022_DAY_COVERAGE_THRESHOLD,
    window_seconds=12 * 60 * 60,
):
    required_columns = {"local_tstamp", "utc_tstamp", "duration", "power"}
    if site_day_long.is_empty() or not required_columns.issubset(site_day_long.columns):
        return {
            "total_rows": 0,
            "rows_with_power": 0,
            "rows_with_v10m": 0,
            "rows_common_power_v10m": 0,
            "covered_seconds": 0.0,
            "window_seconds": float(window_seconds),
            "common_power_v10m_coverage_pct": 0.0,
            "eligible": False,
            "reason": "insufficient_raw_columns",
        }

    v10m_lookup = prepared_day_df.select(["local_tstamp", "v10m_avg"]).unique(
        subset=["local_tstamp"], keep="first"
    )
    df = (
        site_day_long.group_by(["local_tstamp", "utc_tstamp", "duration"])
        .agg(pl.col("power").is_not_null().any().alias("_has_power"))
        .join(v10m_lookup, on="local_tstamp", how="left")
        .with_columns([
            pl.col("v10m_avg").is_not_null().alias("_has_v10m"),
            pl.col("duration")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("_duration_s"),
        ])
        .with_columns(
            (pl.col("_has_power") & pl.col("_has_v10m")).alias(
                "_has_common_power_v10m"
            )
        )
    )

    total_rows = int(df.height)
    rows_with_power = int(df.filter(pl.col("_has_power")).height)
    rows_with_v10m = int(df.filter(pl.col("_has_v10m")).height)
    rows_common = int(df.filter(pl.col("_has_common_power_v10m")).height)
    covered_seconds = float(
        df.select(
            pl.when(pl.col("_has_common_power_v10m"))
            .then(pl.col("_duration_s"))
            .otherwise(0)
            .sum()
        ).item()
    )
    coverage_pct = (
        0.0 if window_seconds == 0 else covered_seconds / window_seconds * 100.0
    )
    eligible = coverage_pct >= coverage_threshold * 100.0
    return {
        "total_rows": total_rows,
        "rows_with_power": rows_with_power,
        "rows_with_v10m": rows_with_v10m,
        "rows_common_power_v10m": rows_common,
        "covered_seconds": covered_seconds,
        "window_seconds": float(window_seconds),
        "common_power_v10m_coverage_pct": coverage_pct,
        "eligible": eligible,
        "reason": None if eligible else "common_power_v10m_coverage_below_threshold",
    }
