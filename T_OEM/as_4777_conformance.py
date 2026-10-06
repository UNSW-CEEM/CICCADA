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


def passive_anti_islanding(
    site_telemetry: pl.DataFrame,
    system_power_kw_ac: float,
    min_eligible_timestamps: int = 1,
) -> pl.DataFrame:
    """Return site-level passive anti-islanding conformance."""
    scored_telemetry = (
        site_telemetry.sort("timestamp")
        .with_columns(
            (pl.col("inverter_ac_real_power") / 1000.0).alias(
                "inverter_ac_real_power_kw"
            ),
            (pl.col("inverter_ac_real_power").shift(-1) / 1000.0).alias(
                "next_inverter_ac_real_power_kw"
            ),
            (pl.col("timestamp").shift(-1) - pl.col("timestamp"))
            .dt.total_seconds()
            .alias("seconds_to_next_timestamp"),
        )
        .filter(
            (pl.col("grid_voltage") >= 265.0)
            & pl.col("grid_voltage").is_not_null()
            & pl.col("inverter_ac_real_power").is_not_null()
        )
        .with_columns(
            (
                (pl.col("inverter_ac_real_power_kw") <= 0.04 * system_power_kw_ac)
                | (
                    pl.col("next_inverter_ac_real_power_kw").is_not_null()
                    & pl.col("seconds_to_next_timestamp").is_not_null()
                    & (pl.col("seconds_to_next_timestamp") <= 60)
                    & (
                        pl.col("next_inverter_ac_real_power_kw")
                        <= 0.04 * system_power_kw_ac
                    )
                )
            )
            .fill_null(False)
            .alias("conformant")
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


def limit_for_sustained_operation(
    site_telemetry: pl.DataFrame,
    system_power_kw_ac: float,
    min_eligible_timestamps: int = 1,
    v_nom_max: float = 258.0,
) -> pl.DataFrame:
    """Return site-level sustained-operation conformance.

    The 60-second telemetry cannot verify the three-second disconnection
    requirement directly. A response is therefore treated as conformant when
    inverter real power is no greater than the 4% measurement-error allowance
    at the current timestamp or at the next timestamp, provided the latter is
    no more than 60 seconds later.
    """
    scored_telemetry = (
        site_telemetry.sort("timestamp")
        .with_columns(
            pl.col("grid_voltage")
            .rolling_mean_by(
                "timestamp",
                window_size="10m",
                min_samples=8,
            )
            .alias("grid_voltage_10m_avg"),
            (pl.col("inverter_ac_real_power") / 1000.0).alias(
                "inverter_ac_real_power_kw"
            ),
            (pl.col("inverter_ac_real_power").shift(-1) / 1000.0).alias(
                "next_inverter_ac_real_power_kw"
            ),
            (pl.col("timestamp").shift(-1) - pl.col("timestamp"))
            .dt.total_seconds()
            .alias("seconds_to_next_timestamp"),
        )
        .filter(
            (pl.col("grid_voltage_10m_avg") >= v_nom_max)
            & pl.col("grid_voltage").is_not_null()
            & pl.col("inverter_ac_real_power").is_not_null()
        )
        .with_columns(
            (
                (
                    pl.col("inverter_ac_real_power_kw")
                    <= 0.04 * system_power_kw_ac
                )
                | (
                    pl.col("next_inverter_ac_real_power_kw").is_not_null()
                    & pl.col("seconds_to_next_timestamp").is_not_null()
                    & (pl.col("seconds_to_next_timestamp") <= 60)
                    & (
                        pl.col("next_inverter_ac_real_power_kw")
                        <= 0.04 * system_power_kw_ac
                    )
                )
            )
            .fill_null(False)
            .alias("conformant")
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
