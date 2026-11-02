import polars as pl


def estimate_rated_capacity(
    site_metadata_capacity,
    site_telemetry,
    tol=0.04,
):
    apparent_power = (
        site_telemetry.select(
            "inverter_ac_real_power",
            "inverter_ac_reactive_power",
        )
        .drop_nulls()
        .with_columns(
            (
                (
                    pl.col("inverter_ac_real_power") ** 2
                    + pl.col("inverter_ac_reactive_power") ** 2
                ).sqrt()
                / 1000
            ).alias("apparent_power_kva")
        )
    )

    s99_capacity = apparent_power["apparent_power_kva"].quantile(
        0.99, interpolation="linear"
    )

    calc_capacity = s99_capacity
    if s99_capacity <= site_metadata_capacity * (1 + tol):
        calc_capacity = site_metadata_capacity

    return site_metadata_capacity, s99_capacity, calc_capacity
