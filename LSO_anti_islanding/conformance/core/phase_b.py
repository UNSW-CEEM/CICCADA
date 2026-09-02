"""Site-level Phase B threshold selection and compliance scoring."""

import polars as pl


def select_thresholds_for_compliance(
    site_thresholds,
    *,
    threshold_method="tier_based",
    threshold_source="calculated",
    tau=0.0,
):
    """Select the LOS and OV1 thresholds used by one compliance case."""
    if threshold_source == "lowest_disconnect":
        return site_thresholds.select(
            [
                "site_id",
                pl.lit(threshold_method, dtype=pl.Utf8).alias("threshold_method"),
                pl.min_horizontal(
                    pl.col("los_lowest_disconnect_voltage").fill_null(258.0),
                    pl.lit(258.0, dtype=pl.Float64),
                ).alias("los_threshold_used"),
                pl.min_horizontal(
                    pl.col("ov1_lowest_disconnect_voltage").fill_null(265.0),
                    pl.lit(265.0, dtype=pl.Float64),
                ).alias("ov1_threshold_used"),
                "los_lowest_disconnect_voltage",
                "ov1_lowest_disconnect_voltage",
            ]
        )

    # automaticalaly goes to properly calcualted threshodls if not lowest_disconnect
    if threshold_method == "tier_based":
        return site_thresholds.select(
            [
                "site_id",
                pl.lit(threshold_method, dtype=pl.Utf8).alias("threshold_method"),
                pl.min_horizontal(
                    pl.col("los_threshold"),
                    pl.lit(258.0, dtype=pl.Float64),
                ).alias("los_threshold_used"),
                pl.min_horizontal(
                    pl.col("ov1_threshold") - tau,
                    pl.lit(265.0, dtype=pl.Float64),
                ).alias("ov1_threshold_used"),
                "los_lowest_disconnect_voltage",
                "ov1_lowest_disconnect_voltage",
            ]
        )
    if threshold_method == "default":
        return site_thresholds.select(
            [
                "site_id",
                pl.lit(threshold_method, dtype=pl.Utf8).alias("threshold_method"),
                pl.lit(258.0, dtype=pl.Float64).alias("los_threshold_used"),
                pl.lit(265.0 - tau, dtype=pl.Float64).alias(
                    "ov1_threshold_used"
                ),
                "los_lowest_disconnect_voltage",
                "ov1_lowest_disconnect_voltage",
            ]
        )
    raise KeyError(
        f"Unknown active Phase B method: {threshold_method}. "
        "Supported methods are 'tier_based' and 'default'."
    )


def evaluate_compliance_for_day(
    signal_frame,
    *,
    los_threshold,
    ov1_threshold,
    disconnect_support=False,
    los_lowest_disconnect_voltage=None,
    ov1_lowest_disconnect_voltage=None,
):
    """Assign responsibility and compliance for one prepared site-day."""
    if signal_frame.is_empty():
        return signal_frame

    frame = signal_frame.with_columns(
        (
            pl.col("ov1_signals_available") & (pl.col("vinst_max") >= ov1_threshold)
        ).alias("ov1_responsible")
    )
    frame = frame.with_columns(
        (
            pl.col("los_signals_available")
            & (~pl.col("ov1_responsible"))
            & (pl.col("v10m_avg") >= los_threshold)
        ).alias("los_responsible")
    )

    if disconnect_support:
        frame = frame.with_columns(
            (
                pl.col("is_disc").fill_null(False)
                & ~(pl.col("ov1_responsible") | pl.col("los_responsible"))
            ).alias("eligible_for_disconnect_support")
        )
        frame = frame.with_columns(
            (
                pl.col("eligible_for_disconnect_support")
                & pl.col("ov1_signals_available")
                & (
                    pl.col("vinst_max")
                    >= pl.lit(ov1_lowest_disconnect_voltage, dtype=pl.Float64)
                )
            )
            .fill_null(False)
            .alias("ov1_disconnect_support_added")
        )
        frame = frame.with_columns(
            (
                pl.col("eligible_for_disconnect_support")
                & (~pl.col("ov1_disconnect_support_added"))
                & pl.col("los_signals_available")
                & (
                    pl.col("v10m_avg")
                    >= pl.lit(los_lowest_disconnect_voltage, dtype=pl.Float64)
                )
            )
            .fill_null(False)
            .alias("los_disconnect_support_added")
        )
    else:
        frame = frame.with_columns(
            [
                pl.lit(False).alias("eligible_for_disconnect_support"),
                pl.lit(False).alias("ov1_disconnect_support_added"),
                pl.lit(False).alias("los_disconnect_support_added"),
            ]
        )

    is_disc_current_or_next = pl.col("is_disc").fill_null(False) | pl.col(
        "is_disc_next"
    ).fill_null(False)
    frame = frame.with_columns(
        [
            (pl.col("los_responsible") & is_disc_current_or_next).alias(
                "los_compliant"
            ),
            (pl.col("ov1_responsible") & is_disc_current_or_next).alias(
                "ov1_compliant"
            ),
            (
                pl.col("is_disc").fill_null(False)
                & ~pl.col("los_signals_available").fill_null(False)
                & ~pl.col("ov1_signals_available").fill_null(False)
            ).alias("disconnected_unknown_voltage"),
        ]
    )
    return frame.with_columns(
        (
            pl.col("is_disc").fill_null(False)
            & ~(
                pl.col("los_responsible")
                | pl.col("ov1_responsible")
                | pl.col("los_disconnect_support_added")
                | pl.col("ov1_disconnect_support_added")
            )
            & ~pl.col("disconnected_unknown_voltage")
        ).alias("disconnected_below_threshold")
    )


def aggregate_all_daily_compliance_for_site(site_id, evaluated_site_days):
    """Combine evaluated days into timestamp detail and one site counts row."""
    detail_columns = [
        "site_id",
        "event_day",
        "local_tstamp",
        "utc_tstamp",
        "v10m_avg",
        "vinst_max",
        "los_signals_available",
        "ov1_signals_available",
        "is_disc",
        "is_disc_next",
        "los_responsible",
        "ov1_responsible",
        "eligible_for_disconnect_support",
        "los_disconnect_support_added",
        "ov1_disconnect_support_added",
        "los_compliant",
        "ov1_compliant",
        "disconnected_below_threshold",
        "disconnected_unknown_voltage",
    ]
    daily_frames = [
        evaluated_day["frame"]
        .with_columns(pl.lit(evaluated_day["event_day"]).alias("event_day"))
        .select(detail_columns)
        for evaluated_day in evaluated_site_days
        if not evaluated_day["frame"].is_empty()
    ]
    site_compliance_timestamp_detail = pl.concat(
        daily_frames,
        how="vertical",
    )
    compliance_counts = site_compliance_timestamp_detail.select(
        [
            pl.lit(site_id, dtype=pl.Int64).alias("site_id"),
            pl.col("los_responsible")
            .sum()
            .cast(pl.Int64)
            .alias("los_responsible_count"),
            pl.col("los_compliant").sum().cast(pl.Int64).alias("los_compliant_count"),
            pl.col("ov1_responsible")
            .sum()
            .cast(pl.Int64)
            .alias("ov1_responsible_count"),
            pl.col("ov1_compliant")
            .sum()
            .cast(pl.Int64)
            .alias("ov1_compliant_count"),
            pl.col("los_disconnect_support_added")
            .sum()
            .cast(pl.Int64)
            .alias("los_disconnect_support_added_count"),
            pl.col("ov1_disconnect_support_added")
            .sum()
            .cast(pl.Int64)
            .alias("ov1_disconnect_support_added_count"),
            pl.col("disconnected_below_threshold")
            .sum()
            .cast(pl.Int64)
            .alias("disconnected_below_threshold_count"),
            pl.col("disconnected_unknown_voltage")
            .sum()
            .cast(pl.Int64)
            .alias("disconnected_unknown_voltage_count"),
        ]
    )
    return site_compliance_timestamp_detail, compliance_counts


def score_site_compliance(
    compliance_counts,
    selected_thresholds,
    *,
    compliance_threshold_pct=90.0,
):
    """Calculate base and disconnect-supported compliance results."""
    site_compliance = compliance_counts.join(
        selected_thresholds,
        on="site_id",
        how="inner",
    ).with_columns(
        [
            (
                pl.col("los_responsible_count")
                + pl.col("ov1_responsible_count")
            )
            .cast(pl.Int64)
            .alias("overall_responsible_count"),
            (
                pl.col("los_compliant_count")
                + pl.col("ov1_compliant_count")
            )
            .cast(pl.Int64)
            .alias("overall_compliant_count"),
            (
                pl.col("los_responsible_count")
                + pl.col("los_disconnect_support_added_count")
            )
            .cast(pl.Int64)
            .alias("los_disconnect_supported_responsible_count"),
            (
                pl.col("los_compliant_count")
                + pl.col("los_disconnect_support_added_count")
            )
            .cast(pl.Int64)
            .alias("los_disconnect_supported_compliant_count"),
            (
                pl.col("ov1_responsible_count")
                + pl.col("ov1_disconnect_support_added_count")
            )
            .cast(pl.Int64)
            .alias("ov1_disconnect_supported_responsible_count"),
            (
                pl.col("ov1_compliant_count")
                + pl.col("ov1_disconnect_support_added_count")
            )
            .cast(pl.Int64)
            .alias("ov1_disconnect_supported_compliant_count"),
        ]
    )
    site_compliance = site_compliance.with_columns(
        [
            (
                pl.col("los_disconnect_supported_responsible_count")
                + pl.col("ov1_disconnect_supported_responsible_count")
            )
            .cast(pl.Int64)
            .alias("overall_disconnect_supported_responsible_count"),
            (
                pl.col("los_disconnect_supported_compliant_count")
                + pl.col("ov1_disconnect_supported_compliant_count")
            )
            .cast(pl.Int64)
            .alias("overall_disconnect_supported_compliant_count"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("los_responsible_count") == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("los_compliant_count") / pl.col("los_responsible_count") * 100.0
            )
            .alias("los_compliance_pct"),
            pl.when(pl.col("ov1_responsible_count") == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("ov1_compliant_count") / pl.col("ov1_responsible_count") * 100.0
            )
            .alias("ov1_compliance_pct"),
            pl.when(pl.col("overall_responsible_count") == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("overall_compliant_count")
                / pl.col("overall_responsible_count")
                * 100.0
            )
            .alias("overall_compliance_pct"),
            pl.when(pl.col("los_disconnect_supported_responsible_count") == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("los_disconnect_supported_compliant_count")
                / pl.col("los_disconnect_supported_responsible_count")
                * 100.0
            )
            .alias("los_disconnect_supported_compliance_pct"),
            pl.when(pl.col("ov1_disconnect_supported_responsible_count") == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("ov1_disconnect_supported_compliant_count")
                / pl.col("ov1_disconnect_supported_responsible_count")
                * 100.0
            )
            .alias("ov1_disconnect_supported_compliance_pct"),
            pl.when(pl.col("overall_disconnect_supported_responsible_count") == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("overall_disconnect_supported_compliant_count")
                / pl.col("overall_disconnect_supported_responsible_count")
                * 100.0
            )
            .alias("overall_disconnect_supported_compliance_pct"),
        ]
    )
    site_compliance = site_compliance.with_columns(
        [
            pl.when(pl.col("los_compliance_pct").is_null())
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(pl.col("los_compliance_pct") >= compliance_threshold_pct)
            .alias("los_pass"),
            pl.when(pl.col("ov1_compliance_pct").is_null())
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(pl.col("ov1_compliance_pct") >= compliance_threshold_pct)
            .alias("ov1_pass"),
            pl.when(pl.col("overall_compliance_pct").is_null())
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(pl.col("overall_compliance_pct") >= compliance_threshold_pct)
            .alias("overall_pass"),
            pl.when(pl.col("los_disconnect_supported_compliance_pct").is_null())
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(
                pl.col("los_disconnect_supported_compliance_pct")
                >= compliance_threshold_pct
            )
            .alias("los_disconnect_supported_pass"),
            pl.when(pl.col("ov1_disconnect_supported_compliance_pct").is_null())
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(
                pl.col("ov1_disconnect_supported_compliance_pct")
                >= compliance_threshold_pct
            )
            .alias("ov1_disconnect_supported_pass"),
            pl.when(pl.col("overall_disconnect_supported_compliance_pct").is_null())
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(
                pl.col("overall_disconnect_supported_compliance_pct")
                >= compliance_threshold_pct
            )
            .alias("overall_disconnect_supported_pass"),
        ]
    )
    return site_compliance.select(
        [
            "site_id",
            "threshold_method",
            "los_threshold_used",
            "ov1_threshold_used",
            "los_lowest_disconnect_voltage",
            "ov1_lowest_disconnect_voltage",
            "los_responsible_count",
            "los_compliant_count",
            "los_compliance_pct",
            "los_pass",
            "ov1_responsible_count",
            "ov1_compliant_count",
            "ov1_compliance_pct",
            "ov1_pass",
            "overall_responsible_count",
            "overall_compliant_count",
            "overall_compliance_pct",
            "overall_pass",
            "los_disconnect_support_added_count",
            "ov1_disconnect_support_added_count",
            "los_disconnect_supported_responsible_count",
            "los_disconnect_supported_compliant_count",
            "los_disconnect_supported_compliance_pct",
            "los_disconnect_supported_pass",
            "ov1_disconnect_supported_responsible_count",
            "ov1_disconnect_supported_compliant_count",
            "ov1_disconnect_supported_compliance_pct",
            "ov1_disconnect_supported_pass",
            "overall_disconnect_supported_responsible_count",
            "overall_disconnect_supported_compliant_count",
            "overall_disconnect_supported_compliance_pct",
            "overall_disconnect_supported_pass",
            "disconnected_below_threshold_count",
            "disconnected_unknown_voltage_count",
        ]
    )


def run_phase_b_for_site(
    site_id,
    prepared_site_days,
    *,
    site_thresholds,
    threshold_method="tier_based",
    threshold_source="calculated",
    disconnect_support=False,
    tau=0.0,
    compliance_threshold_pct=90.0,
):
    """Run Phase B compliance for one site."""
    selected_thresholds = select_thresholds_for_compliance(
        site_thresholds,
        threshold_method=threshold_method,
        threshold_source=threshold_source,
        tau=tau,
    )
    los_threshold_used = selected_thresholds.get_column(
        "los_threshold_used"
    ).item()
    ov1_threshold_used = selected_thresholds.get_column(
        "ov1_threshold_used"
    ).item()
    los_lowest_disconnect_voltage = selected_thresholds.get_column(
        "los_lowest_disconnect_voltage"
    ).item()
    ov1_lowest_disconnect_voltage = selected_thresholds.get_column(
        "ov1_lowest_disconnect_voltage"
    ).item()

    evaluated_site_days = [
        {
            "event_day": prepared_day["analysis_date"],
            "frame": evaluate_compliance_for_day(
                prepared_day["signal_frame"],
                los_threshold=los_threshold_used,
                ov1_threshold=ov1_threshold_used,
                disconnect_support=disconnect_support,
                los_lowest_disconnect_voltage=los_lowest_disconnect_voltage,
                ov1_lowest_disconnect_voltage=ov1_lowest_disconnect_voltage,
            ),
        }
        for prepared_day in prepared_site_days
    ]
    site_compliance_timestamp_detail, compliance_counts = (
        aggregate_all_daily_compliance_for_site(site_id, evaluated_site_days)
    )
    site_compliance = score_site_compliance(
        compliance_counts,
        selected_thresholds,
        compliance_threshold_pct=compliance_threshold_pct,
    )
    return {
        "site_compliance_timestamp_detail": site_compliance_timestamp_detail,
        "site_compliance": site_compliance,
    }
