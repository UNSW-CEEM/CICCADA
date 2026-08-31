"""Site-level Phase B threshold selection and compliance scoring."""

import polars as pl


def select_thresholds_for_compliance(
    site_thresholds,
    *,
    threshold_method="tier_based",
    tau=0.0,
):
    """Select the LOS and effective OV1 thresholds used for compliance."""
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
            ]
        )
    if threshold_method == "default":
        return site_thresholds.select(
            [
                "site_id",
                pl.lit(threshold_method, dtype=pl.Utf8).alias("threshold_method"),
                pl.lit(258.0, dtype=pl.Float64).alias("los_threshold_used"),
                pl.lit(265.0 - tau, dtype=pl.Float64).alias("ov1_threshold_used"),
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
    is_disc_current_or_next = pl.col("is_disc").fill_null(False) | pl.col(
        "is_disc_next"
    ).fill_null(False)
    return frame.with_columns(
        [
            (pl.col("los_responsible") & is_disc_current_or_next).alias(
                "los_compliant"
            ),
            (pl.col("ov1_responsible") & is_disc_current_or_next).alias(
                "ov1_compliant"
            ),
        ]
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
        "los_compliant",
        "ov1_compliant",
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
            pl.col("ov1_compliant").sum().cast(pl.Int64).alias("ov1_compliant_count"),
        ]
    )
    return site_compliance_timestamp_detail, compliance_counts


def score_site_compliance(
    compliance_counts,
    selected_thresholds,
    *,
    compliance_threshold_pct=90.0,
):
    """Calculate site compliance percentages and pass states."""
    site_compliance = compliance_counts.join(
        selected_thresholds,
        on="site_id",
        how="inner",
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
        ]
    )
    return site_compliance.with_columns(
        pl.when(pl.col("los_pass").is_null() & pl.col("ov1_pass").is_null())
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(
            pl.col("los_pass").fill_null(True) & pl.col("ov1_pass").fill_null(True)
        )
        .alias("overall_pass")
    ).select(
        [
            "site_id",
            "threshold_method",
            "los_responsible_count",
            "los_compliant_count",
            "los_compliance_pct",
            "los_pass",
            "los_threshold_used",
            "ov1_responsible_count",
            "ov1_compliant_count",
            "ov1_compliance_pct",
            "ov1_pass",
            "ov1_threshold_used",
            "overall_pass",
        ]
    )


def run_phase_b_for_site(
    site_id,
    prepared_site_days,
    *,
    site_thresholds,
    threshold_method="tier_based",
    tau=0.3,
    compliance_threshold_pct=90.0,
):
    """Run Phase B compliance for one site."""
    selected_thresholds = select_thresholds_for_compliance(
        site_thresholds,
        threshold_method=threshold_method,
        tau=tau,
    )
    los_threshold_used = selected_thresholds.get_column("los_threshold_used").item()
    ov1_threshold_used = selected_thresholds.get_column("ov1_threshold_used").item()
    evaluated_site_days = [
        {
            "event_day": prepared_day["analysis_date"],
            "frame": evaluate_compliance_for_day(
                prepared_day["signal_frame"],
                los_threshold=los_threshold_used,
                ov1_threshold=ov1_threshold_used,
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
