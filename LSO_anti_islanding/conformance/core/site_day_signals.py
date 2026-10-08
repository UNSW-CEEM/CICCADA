"""Shared timestamp-level signal preparation for one site-day."""

import polars as pl


def build_site_day_signals(
    circuit_data,
    PRated,
    *,
    power_measurement_error=0.04,
):
    """Return the prepared one-day signal frame shared by Phase A and Phase B."""
    df = circuit_data.with_columns(
        (
            pl.col("local_tstamp").cast(pl.Datetime).shift(-1)
            - pl.col("local_tstamp").cast(pl.Datetime)
        )
        .dt.total_seconds()
        .fill_null(0)
        .alias("dt_next_s")
    )
    power_cols = [
        column
        for column in df.columns
        if column.startswith("power")
        and not column.endswith("_next")
    ]
    if not power_cols:
        return pl.DataFrame()

    p_disconnect = power_measurement_error * PRated
    df = df.with_columns(
        pl.when(
            pl.all_horizontal(
                [pl.col(column).is_not_null() for column in power_cols]
            )
        )
        .then(pl.sum_horizontal([pl.col(column) for column in power_cols]))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("raw_site_net_power")
    )

    large_negative_at_timestamp = pl.col("raw_site_net_power").is_not_null() & (
        pl.any_horizontal(
            [(pl.col(column) < -p_disconnect).fill_null(False) for column in power_cols]
        )
        | (pl.col("raw_site_net_power") < -p_disconnect)
    )
    df = df.with_columns(
        large_negative_at_timestamp.fill_null(False).alias(
            "large_negative_power_timestamp"
        )
    )
    df = df.with_columns(
        pl.when(
            pl.col("raw_site_net_power").is_null()
            | pl.col("large_negative_power_timestamp")
        )
        .then(pl.lit(None, dtype=pl.Float64))
        .when(pl.col("raw_site_net_power") < 0)
        .then(pl.lit(0.0))
        .otherwise(pl.col("raw_site_net_power"))
        .alias("site_power_calculated")
    )

    any_negative_current = pl.any_horizontal(
        [(pl.col(column) < 0).fill_null(False) for column in power_cols]
    )
    df = df.with_columns(
        [
            (
                pl.col("site_power_calculated").is_not_null()
                & (pl.col("raw_site_net_power") <= 0)
                & any_negative_current
            )
            .fill_null(False)
            .alias("within_tolerance_negative_power_timestamp"),
            (
                pl.col("site_power_calculated").is_not_null()
                & (pl.col("raw_site_net_power") > 0)
            )
            .fill_null(False)
            .alias("positive_site_power_timestamp"),
            (
                pl.col("site_power_calculated").is_not_null()
                & (pl.col("raw_site_net_power") == 0)
                & ~any_negative_current
            )
            .fill_null(False)
            .alias("zero_site_power_timestamp"),
        ]
    )
    df = df.with_columns(
        pl.when(pl.col("site_power_calculated").is_not_null())
        .then(
            pl.all_horizontal(
                [pl.col(column) <= p_disconnect for column in power_cols]
            )
            & (pl.col("site_power_calculated") <= p_disconnect)
        )
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("is_disc")
    )
    df = df.with_columns(pl.col("is_disc").shift(-1).alias("is_disc_next"))
    df = df.with_columns(
        [
            (
                pl.col("is_disc").is_not_null()
                & (
                    pl.col("is_disc").fill_null(False)
                    | pl.col("is_disc_next").is_not_null()
                )
            ).alias("_power_assessable"),
            (
                pl.col("site_power_calculated").shift(1)
                - pl.col("site_power_calculated")
            ).alias("site_power_drop"),
            (
                pl.col("site_power_calculated")
                - pl.col("site_power_calculated").shift(1)
            ).alias("site_power_rise"),
        ]
    )
    return df.with_columns(
        [
            (pl.col("v10m_avg").is_not_null() & pl.col("_power_assessable")).alias(
                "los_signals_available"
            ),
            (pl.col("vinst_max").is_not_null() & pl.col("_power_assessable")).alias(
                "ov1_signals_available"
            ),
        ]
    )
