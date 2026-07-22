"""Shared circuit mapping and voltage preparation for one site-day."""

import polars as pl

from config import VOLTAGE_ROLLING_WINDOW


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
            rolled = (
                df.filter(pl.col(column).is_not_null())
                .with_columns(
                    pl.col(column)
                    .rolling_mean_by(
                        by="local_tstamp",
                        window_size=rolling_window,
                    )
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
