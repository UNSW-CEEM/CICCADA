"""AS/NZS 4777 conformance calculations for Tesla site telemetry."""

import polars as pl


def volt_watt_conformance(
    site_telemetry: pl.DataFrame,
    system_power_kw_ac: float,
    min_eligible_timestamps: int = 1,
) -> pl.DataFrame:
    """Return site-level Volt-Watt conformance from interval telemetry.
       At the moment voltages V>253 are assessed including V>=260
       if combining with other standards, needs modification or 
       some sort of superseding standards
    """
    scored_telemetry = (
        site_telemetry.with_columns(
            pl.when(pl.col("grid_voltage") <= 253.0)
            .then(1.0)
            .when(pl.col("grid_voltage") < 260.0)
            .then(1.0 - (pl.col("grid_voltage") - 253.0) * (0.8 / 7.0))
            .otherwise(0.2)
            .alias("volt_watt_power_fraction")
        )
        .with_columns(
            (
                system_power_kw_ac * pl.col("volt_watt_power_fraction")
                + 0.04 * system_power_kw_ac
            ).alias("volt_watt_ceiling_kw"),
            (pl.col("inverter_ac_real_power") / 1000.0).alias(
                "inverter_ac_real_power_kw"
            ),
        )
        .filter(
            (pl.col("grid_voltage") > 253.0)
            & pl.col("grid_voltage").is_not_null()
            & pl.col("inverter_ac_real_power").is_not_null()
        )
        .with_columns(
            (
                pl.col("inverter_ac_real_power_kw")
                <= pl.col("volt_watt_ceiling_kw")
            ).alias("conformant")
        )
    )

    num_eligible_timestamps = scored_telemetry.height
    num_conformant_timestamps = scored_telemetry["conformant"].sum()

    if num_eligible_timestamps < min_eligible_timestamps:
        conformant_percentage = None
        conformant = None
    else:
        conformant_percentage = (
            100.0 * num_conformant_timestamps / num_eligible_timestamps
        )
        conformant = conformant_percentage >= 90.0

    return pl.DataFrame(
        {
            "num_eligible_timestamps": [num_eligible_timestamps],
            "num_conformant_timestamps": [num_conformant_timestamps],
            "conformant_percentage": [conformant_percentage],
            "conformant": [conformant],
        },
        schema={
            "num_eligible_timestamps": pl.Int64,
            "num_conformant_timestamps": pl.Int64,
            "conformant_percentage": pl.Float64,
            "conformant": pl.Boolean,
        },
    )
