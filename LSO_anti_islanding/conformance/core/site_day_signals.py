"""Shared timestamp-level signal preparation for one site-day."""

import polars as pl


def build_site_day_signals(
    circuit_data,
    PRated,
    *,
    power_measurement_error=0.04,
):
    """Return the prepared one-day signal frame shared by Phase A and Phase B."""
    df = circuit_data.clone()
    df = df.with_columns(
        (
            pl.col("local_tstamp").cast(pl.Datetime).shift(-1)
            - pl.col("local_tstamp").cast(pl.Datetime)
        )
        .dt.total_seconds()
        .fill_null(0)
        .alias("dt_next_s")
    )
    df = df.with_columns(pl.col("^power(_.*)?$").shift(-1).name.suffix("_next"))
    df = df.with_columns(pl.col("local_tstamp").shift(-1).alias("ts_next"))

    power_cols = [
        column
        for column in df.columns
        if column.startswith("power")
        and not column.endswith("_next")
        and not column.endswith("_logic")
    ]
    power_cols_next = [
        column
        for column in df.columns
        if column.startswith("power")
        and column.endswith("_next")
        and not column.endswith("_logic_next")
    ]
    if not power_cols:
        return pl.DataFrame()

    p_disconnect = power_measurement_error * PRated
    logic_current = []
    logic_next = []
    for column in power_cols:
        logic_name = f"{column}_logic"
        logic_next_name = f"{column}_logic_next"
        df = df.with_columns(
            pl.when(pl.col(column) < 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col(column))
            .alias(logic_name)
        )
        logic_current.append(logic_name)

        next_name = f"{column}_next"
        if next_name in power_cols_next:
            df = df.with_columns(
                pl.when(pl.col(next_name) < 0)
                .then(pl.lit(0.0))
                .otherwise(pl.col(next_name))
                .alias(logic_next_name)
            )
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(logic_next_name))
        logic_next.append(logic_next_name)

    df = df.with_columns(
        [
            pl.when(
                pl.all_horizontal(
                    [pl.col(column).is_not_null() for column in logic_current]
                )
            )
            .then(pl.sum_horizontal([pl.col(column) for column in logic_current]))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("site_power"),
            pl.when(
                pl.all_horizontal(
                    [pl.col(column).is_not_null() for column in logic_next]
                )
            )
            .then(pl.sum_horizontal([pl.col(column) for column in logic_next]))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("site_power_next"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(pl.col("site_power").is_not_null())
            .then(
                pl.all_horizontal(
                    [pl.col(column) <= p_disconnect for column in logic_current]
                )
                & (pl.col("site_power") <= p_disconnect)
            )
            .otherwise(pl.lit(None, dtype=pl.Boolean))
            .alias("is_disc"),
            pl.when(pl.col("site_power_next").is_not_null())
            .then(
                pl.all_horizontal(
                    [pl.col(column) <= p_disconnect for column in logic_next]
                )
                & (pl.col("site_power_next") <= p_disconnect)
            )
            .otherwise(pl.lit(None, dtype=pl.Boolean))
            .alias("is_disc_next"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("is_disc").is_not_null()
                & (
                    pl.col("is_disc").fill_null(False)
                    | pl.col("is_disc_next").is_not_null()
                )
            ).alias("_power_assessable"),
            (pl.col("site_power").shift(1) - pl.col("site_power")).alias(
                "site_power_drop"
            ),
            (pl.col("site_power") - pl.col("site_power").shift(1)).alias(
                "site_power_rise"
            ),
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
