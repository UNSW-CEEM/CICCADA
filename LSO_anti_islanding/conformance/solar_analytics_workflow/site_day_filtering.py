"""Solar Analytics count-based site-day eligibility policy."""

import polars as pl
from solar_analytics_workflow.config import (
    DAY_COVERAGE_THRESHOLD,
    EXPECTED_TIMESTAMPS,
)


def summarize_solar_analytics_day_eligibility(
    prepared_day_df,
    coverage_threshold=DAY_COVERAGE_THRESHOLD,
    expected_timestamps=EXPECTED_TIMESTAMPS,
):
    """Summarize complete PV-power and V10m timestamps in a prepared day."""
    required_columns = {"local_tstamp", "v10m_avg"}
    power_columns = [
        column
        for column in prepared_day_df.columns
        if column.startswith("power")
        and not column.endswith("_next")
        and not column.endswith("_logic")
    ]
    if (
        prepared_day_df.is_empty()
        or not required_columns.issubset(prepared_day_df.columns)
        or not power_columns
        or expected_timestamps <= 0
    ):
        return {
            "total_rows": 0,
            "rows_with_power": 0,
            "rows_with_v10m": 0,
            "rows_common_power_v10m": 0,
            "qualifying_timestamps": 0,
            "expected_timestamps": int(expected_timestamps),
            "common_power_v10m_coverage_pct": 0.0,
            "eligible": False,
            "reason": "insufficient_prepared_columns",
        }

    timestamp_status = (
        prepared_day_df.select(
            [
                "local_tstamp",
                pl.all_horizontal(
                    [pl.col(column).is_not_null() for column in power_columns]
                ).alias("_has_all_power"),
                pl.col("v10m_avg").is_not_null().alias("_has_v10m"),
            ]
        )
        .unique(subset=["local_tstamp"], keep="first")
        .with_columns(
            (pl.col("_has_all_power") & pl.col("_has_v10m")).alias(
                "_has_common_power_v10m"
            )
        )
    )

    total_rows = int(timestamp_status.height)
    rows_with_power = int(timestamp_status.filter(pl.col("_has_all_power")).height)
    rows_with_v10m = int(timestamp_status.filter(pl.col("_has_v10m")).height)
    qualifying_timestamps = int(
        timestamp_status.filter(pl.col("_has_common_power_v10m")).height
    )
    coverage_pct = qualifying_timestamps / expected_timestamps * 100.0
    eligible = coverage_pct >= coverage_threshold * 100.0

    return {
        "total_rows": total_rows,
        "rows_with_power": rows_with_power,
        "rows_with_v10m": rows_with_v10m,
        "rows_common_power_v10m": qualifying_timestamps,
        "qualifying_timestamps": qualifying_timestamps,
        "expected_timestamps": int(expected_timestamps),
        "common_power_v10m_coverage_pct": coverage_pct,
        "eligible": eligible,
        "reason": None if eligible else "common_power_v10m_coverage_below_threshold",
    }
