"""AS/NZS 4777 conformance calculations for Tesla site telemetry."""

import polars as pl


def volt_var_interval_conformance(
    site_telemetry: pl.DataFrame,
    system_apparent_power_kva: float,
) -> pl.DataFrame:
    """Return timestamp-level Volt-VAr conformance.

    ``S`` is the supplied system capacity in VA. The required reactive power
    ``Q_req`` follows this piecewise curve:

    ==================  =========================  =============================
    Grid voltage (V)    ``Q_req / S``              Required inverter response
    ==================  =========================  =============================
    ``V <= 207``        ``+0.44``                  Supply reactive power
    ``207 < V < 220``   ``+0.44 * (220-V) / 13``   Ramp supply down to zero
    ``220 <= V <= 240`` ``0``                      No reactive power
    ``240 < V < 258``   ``-0.60 * (V-240) / 18``   Ramp absorption up
    ``V >= 258``        ``-0.60``                  Absorb reactive power
    ==================  =========================  =============================

    Positive reactive power is supplied; negative reactive power is absorbed.
    A timestamp conforms when measured reactive power is within
    ``Q_req +/- 4% of S``. It is not assessable (null) when active-power
    magnitude is below 20% of ``S`` or any required telemetry is missing.
    """
    capacity_va = system_apparent_power_kva * 1000.0
    tolerance_var = 0.04 * capacity_va
    input_columns = site_telemetry.columns

    return (
        site_telemetry.with_columns(
            pl.when(pl.col("grid_voltage").is_null())
            .then(None)
            .when(pl.col("grid_voltage") <= 207.0)
            .then(0.44 * capacity_va)
            .when(pl.col("grid_voltage") < 220.0)
            .then(
                0.44 * capacity_va * (220.0 - pl.col("grid_voltage")) / (220.0 - 207.0)
            )
            .when(pl.col("grid_voltage") <= 240.0)
            .then(0.0)
            .when(pl.col("grid_voltage") < 258.0)
            .then(
                -0.60 * capacity_va * (pl.col("grid_voltage") - 240.0) / (258.0 - 240.0)
            )
            .otherwise(-0.60 * capacity_va)
            .alias("volt_var_required_reactive_power_var"),
        )
        .with_columns(
            (pl.col("volt_var_required_reactive_power_var") - tolerance_var).alias(
                "volt_var_minimum_reactive_power_var"
            ),
            (pl.col("volt_var_required_reactive_power_var") + tolerance_var).alias(
                "volt_var_maximum_reactive_power_var"
            ),
        )
        .with_columns(
            pl.when(
                pl.col("grid_voltage").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
                & pl.col("inverter_ac_reactive_power").is_not_null()
                & (pl.col("inverter_ac_real_power").abs() >= 0.20 * capacity_va)
            )
            .then(
                (
                    pl.col("inverter_ac_reactive_power")
                    >= pl.col("volt_var_minimum_reactive_power_var")
                )
                & (
                    pl.col("inverter_ac_reactive_power")
                    <= pl.col("volt_var_maximum_reactive_power_var")
                )
            )
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("volt_var_conformant")
        )
        .select(
            input_columns
            + [
                "volt_var_required_reactive_power_var",
                "volt_var_minimum_reactive_power_var",
                "volt_var_maximum_reactive_power_var",
                "volt_var_conformant",
            ]
        )
    )


def volt_watt_interval_conformance(
    site_telemetry: pl.DataFrame,
    system_power_kw_ac: float,
) -> pl.DataFrame:
    """Return timestamp-level Volt-Watt conformance at measured voltage.

    The curve permits 100% of rated power up to 253 V, ramps linearly to
    20% at 260 V, and remains at 20% above 260 V. The conformance ceiling
    includes a 4% rated-power allowance. A timestamp is assessable only when
    voltage is above 253 V and the required telemetry is available.
    """
    input_columns = site_telemetry.columns
    result_columns = [
        "volt_watt_curve_limit_kw",
        "volt_watt_ceiling_kw",
        "volt_watt_conformant",
    ]

    return (
        site_telemetry.with_columns(
            pl.when(pl.col("grid_voltage").is_null())
            .then(None)
            .when(pl.col("grid_voltage") <= 253.0)
            .then(system_power_kw_ac)
            .when(pl.col("grid_voltage") < 260.0)
            .then(
                system_power_kw_ac
                * (1.0 - (pl.col("grid_voltage") - 253.0) * (0.8 / 7.0))
            )
            .otherwise(0.2 * system_power_kw_ac)
            .alias("volt_watt_curve_limit_kw")
        )
        .with_columns(
            (pl.col("volt_watt_curve_limit_kw") + 0.04 * system_power_kw_ac).alias(
                "volt_watt_ceiling_kw"
            )
        )
        .with_columns(
            pl.when(
                (pl.col("grid_voltage") > 253.0)
                & pl.col("grid_voltage").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
            )
            .then(
                pl.col("inverter_ac_real_power") / 1000.0
                <= pl.col("volt_watt_ceiling_kw")
            )
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("volt_watt_conformant")
        )
        .select(
            input_columns
            + [column for column in result_columns if column not in input_columns]
        )
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


def combined_conformance(
    site_telemetry: pl.DataFrame,
    system_apparent_power_kva: float,
    solar_coupling: str,
    v_nom_max: float = 258.0,
    csip_export_limit_usable: bool = True,
) -> pl.DataFrame:
    """Return timestamp-level results for coordinated conformance checks.

    The input is expected to contain telemetry that has already passed the
    coupling-specific inverter, battery, and solar consistency checks. Passive
    anti-islanding supersedes the sustained-operation check at 265 V or above.
    Otherwise, the sustained-operation check applies when the 10-minute
    average voltage reaches ``v_nom_max``.
    Volt-Watt, Volt-VAr, and CSIP (for DC-only systems)
    are assessed independently of those two checks.
    Set ``csip_export_limit_usable`` false when the raw CSIP signal has no
    finite, positive export limit; CSIP results will then remain null.

    A nullable Boolean is returned for each check:
    true or false when the check applies, and null when it does not.
    ``as_4777_conformant`` contains the conjunction of all
    applicable AS 4777 results and is null when no AS 4777 check applies.
    CSIP remains a separate result.
    """

    input_columns = site_telemetry.columns
    tolerance_w = system_apparent_power_kva * 1000.0 * 0.04

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
            pl.col("inverter_ac_real_power")
            .shift(-1)
            .alias("_next_inverter_ac_real_power"),
            (pl.col("timestamp").shift(-1) - pl.col("timestamp"))
            .dt.total_seconds()
            .alias("_seconds_to_next_timestamp"),
        )
        .with_columns(
            (
                (pl.col("grid_voltage") >= 265.0)
                & pl.col("grid_voltage").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
            )
            .fill_null(False)
            .alias("_passive_anti_islanding_applicable")
        )
        .with_columns(
            (
                ~pl.col("_passive_anti_islanding_applicable")
                & (pl.col("grid_voltage_10m_avg") >= v_nom_max)
                & pl.col("grid_voltage").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
            )
            .fill_null(False)
            .alias("_sustained_operation_applicable")
        )
        .with_columns(
            pl.when(pl.col("_passive_anti_islanding_applicable"))
            .then(
                (pl.col("inverter_ac_real_power") <= tolerance_w)
                | (
                    pl.col("_next_inverter_ac_real_power").is_not_null()
                    & pl.col("_seconds_to_next_timestamp").is_not_null()
                    & (pl.col("_seconds_to_next_timestamp") <= 60)
                    & (pl.col("_next_inverter_ac_real_power") <= tolerance_w)
                )
            )
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("passive_anti_islanding_conformant"),
            pl.when(pl.col("_sustained_operation_applicable"))
            .then(
                (pl.col("inverter_ac_real_power") <= tolerance_w)
                | (
                    pl.col("_next_inverter_ac_real_power").is_not_null()
                    & pl.col("_seconds_to_next_timestamp").is_not_null()
                    & (pl.col("_seconds_to_next_timestamp") <= 60)
                    & (pl.col("_next_inverter_ac_real_power") <= tolerance_w)
                )
            )
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("sustained_operation_conformant"),
        )
    )

    scored_telemetry = volt_var_interval_conformance(
        scored_telemetry,
        system_apparent_power_kva,
    )
    scored_telemetry = volt_watt_interval_conformance(
        scored_telemetry,
        system_apparent_power_kva,
    ).with_columns(
        pl.col("volt_watt_ceiling_kw").alias("volt_watt_assessment_ceiling_kw")
    )

    if solar_coupling == "dc_only" and csip_export_limit_usable:
        scored_telemetry = scored_telemetry.with_columns(
            pl.when(
                pl.col("site_instant_power").is_not_null()
                & pl.col("csip_export_limit_w_ffill").is_not_null()
            )
            .then(
                (-pl.col("site_instant_power")).clip(lower_bound=0)
                <= pl.col("csip_export_limit_w_ffill") + tolerance_w
            )
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("csip_conformant")
        )
    else:
        scored_telemetry = scored_telemetry.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("csip_conformant")
        )

    as_4777_result_columns = [
        "passive_anti_islanding_conformant",
        "sustained_operation_conformant",
        "volt_watt_conformant",
        "volt_var_conformant",
    ]
    individual_result_columns = [
        *as_4777_result_columns,
        "csip_conformant",
    ]
    scored_telemetry = scored_telemetry.with_columns(
        pl.when(
            pl.any_horizontal(
                [pl.col(column).is_not_null() for column in as_4777_result_columns]
            )
        )
        .then(
            pl.all_horizontal(
                [pl.col(column).fill_null(True) for column in as_4777_result_columns]
            )
        )
        .otherwise(None)
        .cast(pl.Boolean)
        .alias("as_4777_conformant")
    )

    return scored_telemetry.select(
        input_columns
        + [
            "grid_voltage_10m_avg",
            "volt_watt_curve_limit_kw",
            "volt_watt_ceiling_kw",
            "volt_watt_assessment_ceiling_kw",
            "volt_var_required_reactive_power_var",
            "volt_var_minimum_reactive_power_var",
            "volt_var_maximum_reactive_power_var",
            *individual_result_columns,
            "as_4777_conformant",
        ]
    )


def combined_conformance_with_voltage_tolerance(
    site_telemetry: pl.DataFrame,
    system_apparent_power_kva: float,
    solar_coupling: str,
    v_nom_max: float = 258.0,
    csip_export_limit_usable: bool = True,
    voltage_tolerance_v: float = 2.3,
    vw_require_measured_above_threshold: bool = False,
) -> pl.DataFrame:
    """Return conservative timestamp-level conformance results.

    Voltage measurement uncertainty is represented by the interval from
    measured voltage minus ``voltage_tolerance_v`` to measured voltage plus
    ``voltage_tolerance_v``.
    1. A result is true when a requirement could apply and the observed response is
    conformant for at least one voltage in that interval.
    2. A result is false only when the requirement definitely applies
    and the response is non-conformant for every voltage in the interval.
    3. Otherwise, the result is null.

    When ``vw_require_measured_above_threshold`` is true, Volt-Watt is
    assessed only when the measured voltage is above 253 V.

    Passive anti-islanding is assessed first. Limit for sustained operation
    is assessed only when passive anti-islanding is inconclusive or does not
    apply, so no timestamp has results for both cessation mechanisms.
    Volt-Watt, Volt-VAr, and CSIP are assessed independently.
    """

    input_columns = site_telemetry.columns
    tolerance_w = system_apparent_power_kva * 1000.0 * 0.04

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
            pl.col("inverter_ac_real_power")
            .shift(-1)
            .alias("_next_inverter_ac_real_power"),
            (pl.col("timestamp").shift(-1) - pl.col("timestamp"))
            .dt.total_seconds()
            .alias("_seconds_to_next_timestamp"),
        )
        .with_columns(
            (pl.col("grid_voltage") - voltage_tolerance_v).alias("_grid_voltage_lower"),
            (pl.col("grid_voltage") + voltage_tolerance_v).alias("_grid_voltage_upper"),
            (pl.col("grid_voltage_10m_avg") - voltage_tolerance_v).alias(
                "_grid_voltage_10m_avg_lower"
            ),
            (pl.col("grid_voltage_10m_avg") + voltage_tolerance_v).alias(
                "_grid_voltage_10m_avg_upper"
            ),
            (
                (pl.col("inverter_ac_real_power") <= tolerance_w)
                | (
                    pl.col("_next_inverter_ac_real_power").is_not_null()
                    & pl.col("_seconds_to_next_timestamp").is_not_null()
                    & (pl.col("_seconds_to_next_timestamp") <= 60)
                    & (pl.col("_next_inverter_ac_real_power") <= tolerance_w)
                )
            )
            .fill_null(False)
            .alias("_disconnected"),
        )
        .with_columns(
            pl.when(
                pl.col("grid_voltage").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
                & (pl.col("_grid_voltage_upper") >= 265.0)
                & pl.col("_disconnected")
            )
            .then(True)
            .when(
                pl.col("grid_voltage").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
                & (pl.col("_grid_voltage_lower") >= 265.0)
                & ~pl.col("_disconnected")
            )
            .then(False)
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("passive_anti_islanding_conformant")
        )
        .with_columns(
            pl.when(
                pl.col("passive_anti_islanding_conformant").is_null()
                & pl.col("grid_voltage").is_not_null()
                & pl.col("grid_voltage_10m_avg").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
                & (pl.col("_grid_voltage_10m_avg_upper") >= v_nom_max)
                & pl.col("_disconnected")
            )
            .then(True)
            .when(
                pl.col("passive_anti_islanding_conformant").is_null()
                & pl.col("grid_voltage").is_not_null()
                & pl.col("grid_voltage_10m_avg").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
                & (pl.col("_grid_voltage_10m_avg_lower") >= v_nom_max)
                & ~pl.col("_disconnected")
            )
            .then(False)
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("sustained_operation_conformant"),
            (
                pl.max_horizontal(
                    pl.col("_grid_voltage_lower"),
                    pl.lit(253.0),
                )
                if vw_require_measured_above_threshold
                else pl.col("_grid_voltage_lower")
            ).alias("_volt_watt_lower_applicable_voltage"),
        )
    )

    volt_var_lower_voltage_results = volt_var_interval_conformance(
        scored_telemetry.with_columns(
            pl.col("_grid_voltage_lower").alias("grid_voltage")
        ),
        system_apparent_power_kva,
    )
    volt_var_upper_voltage_results = volt_var_interval_conformance(
        scored_telemetry.with_columns(
            pl.col("_grid_voltage_upper").alias("grid_voltage")
        ),
        system_apparent_power_kva,
    )
    scored_telemetry = volt_var_interval_conformance(
        scored_telemetry,
        system_apparent_power_kva,
    ).with_columns(
        volt_var_upper_voltage_results[
            "volt_var_minimum_reactive_power_var"
        ].alias("_volt_var_assessment_minimum_reactive_power_var"),
        volt_var_lower_voltage_results[
            "volt_var_maximum_reactive_power_var"
        ].alias("_volt_var_assessment_maximum_reactive_power_var"),
    )
    scored_telemetry = scored_telemetry.with_columns(
        pl.when(pl.col("volt_var_conformant").is_not_null())
        .then(
            (
                pl.col("inverter_ac_reactive_power")
                >= pl.col("_volt_var_assessment_minimum_reactive_power_var")
            )
            & (
                pl.col("inverter_ac_reactive_power")
                <= pl.col("_volt_var_assessment_maximum_reactive_power_var")
            )
        )
        .otherwise(None)
        .cast(pl.Boolean)
        .alias("volt_var_conformant")
    )

    scored_telemetry = volt_watt_interval_conformance(
        scored_telemetry,
        system_apparent_power_kva,
    )
    assessment_results = volt_watt_interval_conformance(
        scored_telemetry.with_columns(
            pl.when(pl.col("grid_voltage").is_not_null())
            .then(pl.col("_volt_watt_lower_applicable_voltage"))
            .otherwise(None)
            .alias("grid_voltage")
        ),
        system_apparent_power_kva,
    )
    scored_telemetry = (
        scored_telemetry.with_columns(
            assessment_results["volt_watt_ceiling_kw"].alias(
                "volt_watt_assessment_ceiling_kw"
            )
        )
        .with_columns(
            (
                pl.col("inverter_ac_real_power") / 1000.0
                <= pl.col("volt_watt_assessment_ceiling_kw")
            )
            .fill_null(False)
            .alias("_volt_watt_power_conformant")
        )
        .with_columns(
            pl.when(
                pl.col("grid_voltage").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
                & (pl.col("_grid_voltage_upper") > 253.0)
                & (
                    (pl.col("grid_voltage") > 253.0)
                    if vw_require_measured_above_threshold
                    else pl.lit(True)
                )
                & pl.col("_volt_watt_power_conformant")
            )
            .then(True)
            .when(
                pl.col("grid_voltage").is_not_null()
                & pl.col("inverter_ac_real_power").is_not_null()
                & (pl.col("_grid_voltage_lower") > 253.0)
                & ~pl.col("_volt_watt_power_conformant")
            )
            .then(False)
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("volt_watt_conformant")
        )
    )

    if solar_coupling == "dc_only" and csip_export_limit_usable:
        scored_telemetry = scored_telemetry.with_columns(
            pl.when(
                pl.col("site_instant_power").is_not_null()
                & pl.col("csip_export_limit_w_ffill").is_not_null()
            )
            .then(
                (-pl.col("site_instant_power")).clip(lower_bound=0)
                <= pl.col("csip_export_limit_w_ffill") + tolerance_w
            )
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("csip_conformant")
        )
    else:
        scored_telemetry = scored_telemetry.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("csip_conformant")
        )

    as_4777_result_columns = [
        "passive_anti_islanding_conformant",
        "sustained_operation_conformant",
        "volt_watt_conformant",
        "volt_var_conformant",
    ]
    individual_result_columns = [
        *as_4777_result_columns,
        "csip_conformant",
    ]
    scored_telemetry = scored_telemetry.with_columns(
        pl.when(
            pl.any_horizontal(
                [pl.col(column).is_not_null() for column in as_4777_result_columns]
            )
        )
        .then(
            pl.all_horizontal(
                [pl.col(column).fill_null(True) for column in as_4777_result_columns]
            )
        )
        .otherwise(None)
        .cast(pl.Boolean)
        .alias("as_4777_conformant")
    )

    return scored_telemetry.select(
        input_columns
        + [
            "grid_voltage_10m_avg",
            "volt_watt_curve_limit_kw",
            "volt_watt_ceiling_kw",
            "volt_watt_assessment_ceiling_kw",
            "volt_var_required_reactive_power_var",
            "volt_var_minimum_reactive_power_var",
            "volt_var_maximum_reactive_power_var",
            *individual_result_columns,
            "as_4777_conformant",
        ]
    )
