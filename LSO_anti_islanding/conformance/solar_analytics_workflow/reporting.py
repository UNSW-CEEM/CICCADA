"""Solar Analytics result-table construction and threshold plots."""

from pathlib import Path

import polars as pl
from solar_analytics_workflow.config import PRIMARY_PHASE_B_METHOD
from solar_analytics_workflow.plotting import (
    plot_site_threshold_distribution,
    plot_site_threshold_distribution_extremes,
)

SITE_CONFORMANCE_SUMMARY_NAME = "site_conformance_summary.csv"
CONFORMANCE_EXCLUSIONS_NAME = "conformance_exclusions.csv"

SITE_CONFORMANCE_SUMMARY_SCHEMA = {
    "site_id": pl.Int64,
    "method_key": pl.Utf8,
    "assessment_status": pl.Utf8,
    "overall_pass": pl.Boolean,
    "los_eligible": pl.Int64,
    "los_compliant": pl.Int64,
    "los_compliance_pct": pl.Float64,
    "los_pass": pl.Boolean,
    "los_threshold_used": pl.Float64,
    "los_threshold_basis": pl.Utf8,
    "ov1_eligible": pl.Int64,
    "ov1_compliant": pl.Int64,
    "ov1_compliance_pct": pl.Float64,
    "ov1_pass": pl.Boolean,
    "ov1_threshold_used": pl.Float64,
    "ov1_threshold_basis": pl.Utf8,
}

CONFORMANCE_EXCLUSIONS_SCHEMA = {
    "site_id": pl.Int64,
    "exclusion_scope": pl.Utf8,
    "day": pl.Date,
    "reason": pl.Utf8,
    "common_power_v10m_coverage_pct": pl.Float64,
    "rows_common_power_v10m": pl.Int64,
    "rows_with_power": pl.Int64,
    "rows_with_v10m": pl.Int64,
    "qualifying_timestamps": pl.Int64,
    "expected_timestamps": pl.Int64,
    "total_rows": pl.Int64,
    "coverage_threshold_pct": pl.Float64,
}


def build_sola_site_conformance_summary(results):
    phase_b_summary = results["phase_b_site_summary"]
    site_thresholds = results["site_thresholds"]
    if phase_b_summary.is_empty() and site_thresholds.is_empty():
        return pl.DataFrame(schema=SITE_CONFORMANCE_SUMMARY_SCHEMA)
    if phase_b_summary.is_empty() or site_thresholds.is_empty():
        raise ValueError(
            "SolA Phase B summary and threshold tables must contain the same sites."
        )

    combined = phase_b_summary.select(
        [
            "site_id",
            "overall_pass",
            "los_eligible",
            "los_compliant",
            "los_compliance_pct",
            "los_pass",
            "los_threshold_used",
            "ov1_eligible",
            "ov1_compliant",
            "ov1_compliance_pct",
            "ov1_pass",
        ]
    ).join(
        site_thresholds.select(
            [
                "site_id",
                "los_threshold_basis",
                "ov1_test_site",
                "ov1_threshold_basis",
            ]
        ),
        on="site_id",
        how="inner",
        validate="1:1",
    )
    if (
        combined.height != phase_b_summary.height
        or combined.height != site_thresholds.height
    ):
        raise ValueError(
            "SolA Phase B summary and threshold tables have different site IDs."
        )

    return (
        combined.with_columns(
            [
                pl.lit(PRIMARY_PHASE_B_METHOD).alias("method_key"),
                pl.when(pl.col("overall_pass").is_null())
                .then(pl.lit("unassessed"))
                .when(pl.col("overall_pass"))
                .then(pl.lit("conformant"))
                .otherwise(pl.lit("non-conformant"))
                .alias("assessment_status"),
                pl.col("ov1_test_site").alias("ov1_threshold_used"),
            ]
        )
        .select(list(SITE_CONFORMANCE_SUMMARY_SCHEMA))
        .cast(SITE_CONFORMANCE_SUMMARY_SCHEMA, strict=False)
        .sort("site_id")
    )


def build_sola_conformance_exclusions(results):
    rows = []
    for reason, site_ids in results["skipped_sites"].items():
        rows.extend(
            {
                "site_id": site_id,
                "exclusion_scope": "site",
                "day": None,
                "reason": reason,
            }
            for site_id in site_ids
        )
    rows.extend(
        {**row, "exclusion_scope": "site_day"} for row in results["excluded_day_rows"]
    )
    if not rows:
        return pl.DataFrame(schema=CONFORMANCE_EXCLUSIONS_SCHEMA)
    return pl.DataFrame(
        [
            {column: row.get(column) for column in CONFORMANCE_EXCLUSIONS_SCHEMA}
            for row in rows
        ],
        schema=CONFORMANCE_EXCLUSIONS_SCHEMA,
        strict=False,
    ).sort(["site_id", "exclusion_scope", "day"], nulls_last=True)


def write_sola_threshold_distribution_plots(phase_a, output_dir):
    if phase_a.is_empty():
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mechanism, voltage_column, prefix in (
        ("LOS", "v_los_recorded", "los"),
        ("OV1", "v_ov1_recorded", "ov1"),
    ):
        stats = (
            phase_a.filter(pl.col("mech") == mechanism)
            .group_by("site_id")
            .agg(
                [
                    pl.col(voltage_column).min().alias("min_v"),
                    pl.col(voltage_column).median().alias("median_v"),
                    pl.col(voltage_column).max().alias("max_v"),
                    pl.col(voltage_column).std().alias("std_v"),
                    pl.len().alias("n_events"),
                ]
            )
        )
        plot_site_threshold_distribution(
            stats,
            title=f"{mechanism} Thresholds Across Assessed Sites — Min / Median / Max (Std on right)",
            save_path=output_dir / f"{prefix}_threshold_distribution.png",
        )
        plot_site_threshold_distribution_extremes(
            stats,
            title=f"{mechanism} Thresholds — Lowest 20 Std Sites (n_events >= 3)",
            save_path=output_dir / f"{prefix}_threshold_lowest20_std.png",
            highest_std=False,
            min_events=3,
            n_sites=20,
        )
        plot_site_threshold_distribution_extremes(
            stats,
            title=f"{mechanism} Thresholds — Highest 20 Std Sites (n_events >= 3)",
            save_path=output_dir / f"{prefix}_threshold_highest20_std.png",
            highest_std=True,
            min_events=3,
            n_sites=20,
        )
