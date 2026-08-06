"""Shared circuit mapping and voltage preparation for one site-day."""

from datetime import time

import polars as pl

from config import (
    SITE_DAY_ANALYSIS_START,
    SITE_DAY_END,
    VOLTAGE_ROLLING_WINDOW,
)
from core.data_cleaning import deduplicateMeasurements


def select_site_pv_data(all_data, circuit_details, site_number):
    """Collect one site's ``pv_site_net`` measurements using circuit metadata."""
    circuit_details_df = (
        circuit_details.collect()
        if isinstance(circuit_details, pl.LazyFrame)
        else circuit_details
    )
    pv_circuit_ids = (
        circuit_details_df.filter(
            (pl.col("site_id") == site_number)
            & (pl.col("con_type") == "pv_site_net")
        )
        .select("c_id")
        .unique()
        .sort("c_id")["c_id"]
        .to_list()
    )
    all_data_ldf = all_data if isinstance(all_data, pl.LazyFrame) else all_data.lazy()
    if not pv_circuit_ids:
        return all_data_ldf.limit(0).collect()
    return (
        all_data_ldf.filter(pl.col("c_id").is_in(pv_circuit_ids))
        .collect()
    )


def extract_site_day(site_data, start_day, end_day):
    """Extract one inclusive local-time site-day from selected PV measurements."""
    site_data_ldf = (
        site_data if isinstance(site_data, pl.LazyFrame) else site_data.lazy()
    )
    schema = site_data_ldf.collect_schema()
    select_columns = [
        "c_id",
        "local_tstamp",
        "utc_tstamp",
        "power",
        "voltage_valid",
    ]
    has_duration = "duration" in schema
    if has_duration:
        select_columns.insert(3, "duration")

    site_day = (
        site_data_ldf.select(select_columns)
        .filter(
            pl.col("local_tstamp").is_between(
                start_day,
                end_day,
                closed="both",
            )
        )
        .collect()
    )
    # SAPN uses duration as part of the later pivot index. Preserve its source
    # order within equal timestamps so the established edge sequence is stable.
    # Datasets without duration can use an explicit circuit tie-breaker.
    
    sort_columns = ["local_tstamp"] if has_duration else ["local_tstamp", "c_id"]

    # This is unnecessary when deduplication was completed in preprocessing,
    # but is retained as a safeguard when preparing site-days from data that
    # was not deduplicated during preprocessing.
    return deduplicateMeasurements(site_day).sort(sort_columns)


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
            rolled = (
                valid_voltage
                .with_columns(
                    pl.col(column).rolling_mean_by(
                        by="local_tstamp",
                        window_size=rolling_window,
                        closed="right",
                    ).alias(rolled_name),
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
