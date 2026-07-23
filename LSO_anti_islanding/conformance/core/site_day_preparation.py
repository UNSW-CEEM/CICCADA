"""Shared circuit mapping and voltage preparation for one site-day."""

from datetime import time

import polars as pl

from config import (
    SITE_DAY_ANALYSIS_START,
    SITE_DAY_END,
    VOLTAGE_ROLLING_WINDOW,
)


def map_circuit_data_to_site(site_day_long, site_number):
    index_columns = ["local_tstamp", "utc_tstamp"]
    select_columns = [
        "c_id",
        "local_tstamp",
        "utc_tstamp",
        "power",
        "voltage_valid",
    ]
    if "duration" in site_day_long.columns:
        index_columns.append("duration")
        select_columns.insert(3, "duration")

    analysis_long = site_day_long.select(select_columns)
    if analysis_long.is_empty():
        return (
            analysis_long.select([c for c in index_columns if c != "duration"])
            .with_row_index("row_id")
            .with_columns(pl.lit(site_number).alias("site_id"))
        )

    return (
        analysis_long.pivot(
            values=["power", "voltage_valid"],
            index=index_columns,
            on="c_id",
        )
        .sort("local_tstamp")
        .drop("duration", strict=False)
        .with_row_index("row_id")
        .with_columns(pl.lit(site_number).alias("site_id"))
    )


def calculate_site_day_voltage_signals(
    site_day_df,
    voltage_prefix="voltage_valid",
    rolling_window=VOLTAGE_ROLLING_WINDOW,
):
    """Add circuit rolling voltages plus site-level V10m and instantaneous max."""
    voltage_cols = [c for c in site_day_df.columns if c.startswith(voltage_prefix)]
    df = site_day_df.clone()

    if voltage_cols:
        for column in voltage_cols:
            rolled_name = (
                f"vmean_rolling_10m{column.replace(voltage_prefix, '', 1)}"
            )
            valid_voltage = df.filter(pl.col(column).is_not_null())
            if valid_voltage.is_empty():
                df = df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias(rolled_name)
                )
                continue
            first_voltage_timestamp = valid_voltage["local_tstamp"].min()
            rolled = (
                valid_voltage
                .with_columns(
                    pl.col(column).rolling_mean_by(
                        by="local_tstamp",
                        window_size=rolling_window,
                        closed="both",
                    ).alias(rolled_name),
                )
                .with_columns(
                    pl.when(
                        pl.col("local_tstamp")
                        >= pl.lit(first_voltage_timestamp).dt.offset_by(rolling_window)
                    )
                    .then(pl.col(rolled_name))
                    .otherwise(None)
                    .alias(rolled_name)
                )
                .select(["local_tstamp", rolled_name])
            )
            df = df.join(rolled, on="local_tstamp", how="left")

        rolling_cols = [
            c for c in df.columns if c.startswith("vmean_rolling_10m")
        ]
        return df.with_columns([
            pl.mean_horizontal([pl.col(c) for c in rolling_cols]).alias("v10m_avg"),
            pl.max_horizontal([pl.col(c) for c in voltage_cols]).alias("vinst_max"),
        ])

    return df.with_columns([
        pl.lit(None).cast(pl.Float64).alias("v10m_avg"),
        pl.lit(None).cast(pl.Float64).alias("vinst_max"),
    ])


def trim_site_day_analysis_window(
    site_day_df,
    start_time: time = SITE_DAY_ANALYSIS_START,
    end_time: time = SITE_DAY_END,
):
    """Keep the inclusive local-time window used for eligibility and analysis."""
    return site_day_df.filter(
        pl.col("local_tstamp")
        .dt.time()
        .is_between(start_time, end_time, closed="both")
    )
