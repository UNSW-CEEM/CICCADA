"""SAPN result-table construction and threshold plots."""

from pathlib import Path

import polars as pl
from sapn2022_workflow.plotting import (
    plot_site_threshold_distribution,
    plot_site_threshold_distribution_extremes,
)

SITE_COMPLIANCE_NAME = "site_compliance.csv"
SITE_COMPLIANCE_FINAL_TABLE_NAME = "site_compliance_final_table.csv"
SITE_COMPLIANCE_TIME_DISTRIBUTION_NAME = "site_compliance_time_distribution.csv"
SITE_COMPLIANCE_TOD_DISTRIBUTION_NAME = "site_compliance_tod_distribution.csv"
SITE_LEVEL_VARIOUS_VOLTAGES_NAME = "site_level_various_voltages.csv"
CONFORMANCE_EXCLUSIONS_NAME = "conformance_exclusions.csv"

SITE_COMPLIANCE_SCHEMA = {
    "site_id": pl.Int64,
    "threshold_method": pl.Utf8,
    "los_calculated_threshold_used": pl.Float64,
    "ov1_calculated_threshold_used": pl.Float64,
    "los_lowest_disconnect_voltage": pl.Float64,
    "ov1_lowest_disconnect_voltage": pl.Float64,
    "los_lowest_disconnect_threshold_used": pl.Float64,
    "ov1_lowest_disconnect_threshold_used": pl.Float64,
    "los_calculated_responsible_count": pl.Int64,
    "los_calculated_compliant_count": pl.Int64,
    "los_calculated_compliance_pct": pl.Float64,
    "los_calculated_pass": pl.Boolean,
    "ov1_calculated_responsible_count": pl.Int64,
    "ov1_calculated_compliant_count": pl.Int64,
    "ov1_calculated_compliance_pct": pl.Float64,
    "ov1_calculated_pass": pl.Boolean,
    "overall_calculated_responsible_count": pl.Int64,
    "overall_calculated_compliant_count": pl.Int64,
    "overall_calculated_compliance_pct": pl.Float64,
    "overall_calculated_pass": pl.Boolean,
    "los_disconnect_support_added_count": pl.Int64,
    "ov1_disconnect_support_added_count": pl.Int64,
    "los_disconnect_supported_responsible_count": pl.Int64,
    "los_disconnect_supported_compliant_count": pl.Int64,
    "los_disconnect_supported_compliance_pct": pl.Float64,
    "los_disconnect_supported_pass": pl.Boolean,
    "ov1_disconnect_supported_responsible_count": pl.Int64,
    "ov1_disconnect_supported_compliant_count": pl.Int64,
    "ov1_disconnect_supported_compliance_pct": pl.Float64,
    "ov1_disconnect_supported_pass": pl.Boolean,
    "overall_disconnect_supported_responsible_count": pl.Int64,
    "overall_disconnect_supported_compliant_count": pl.Int64,
    "overall_disconnect_supported_compliance_pct": pl.Float64,
    "overall_disconnect_supported_pass": pl.Boolean,
    "los_lowest_disconnect_responsible_count": pl.Int64,
    "los_lowest_disconnect_compliant_count": pl.Int64,
    "los_lowest_disconnect_compliance_pct": pl.Float64,
    "los_lowest_disconnect_pass": pl.Boolean,
    "ov1_lowest_disconnect_responsible_count": pl.Int64,
    "ov1_lowest_disconnect_compliant_count": pl.Int64,
    "ov1_lowest_disconnect_compliance_pct": pl.Float64,
    "ov1_lowest_disconnect_pass": pl.Boolean,
    "overall_lowest_disconnect_responsible_count": pl.Int64,
    "overall_lowest_disconnect_compliant_count": pl.Int64,
    "overall_lowest_disconnect_compliance_pct": pl.Float64,
    "overall_lowest_disconnect_pass": pl.Boolean,
}

SITE_COMPLIANCE_TIME_DISTRIBUTION_SCHEMA = {
    "site_id": pl.Int64,
    "threshold_method": pl.Utf8,
    "case": pl.Utf8,
    "eligible_timestamp_count": pl.Int64,
    "compliant_timestamp_count": pl.Int64,
    "disconnect_support_timestamp_count": pl.Int64,
    "non_compliant_timestamp_count": pl.Int64,
    "compliant_pct": pl.Float64,
    "non_compliant_pct": pl.Float64,
    "disconnected_below_threshold_count": pl.Int64,
    "disconnected_unknown_voltage_count": pl.Int64,
}

CONFORMANCE_EXCLUSIONS_SCHEMA = {
    "site_id": pl.Int64,
    "exclusion_scope": pl.Utf8,
    "day": pl.Int64,
    "reason": pl.Utf8,
    "common_power_v10m_coverage_pct": pl.Float64,
    "rows_common_power_v10m": pl.Int64,
    "rows_with_power": pl.Int64,
    "rows_with_v10m": pl.Int64,
    "covered_seconds": pl.Float64,
    "window_seconds": pl.Float64,
    "total_rows": pl.Int64,
    "coverage_threshold_pct": pl.Float64,
}


def build_sapn_site_compliance(results):
    site_compliance = results["site_compliance"]
    site_thresholds = results["site_thresholds"]
    if site_compliance.is_empty() and site_thresholds.is_empty():
        return pl.DataFrame(schema=SITE_COMPLIANCE_SCHEMA)
    if site_compliance.is_empty() or site_thresholds.is_empty():
        raise ValueError(
            "SAPN compliance and threshold tables must contain the same sites."
        )

    combined_site_ids = site_compliance.select("site_id").join(
        site_thresholds.select("site_id"),
        on="site_id",
        how="inner",
        validate="1:1",
    )
    if (
        combined_site_ids.height != site_compliance.height
        or combined_site_ids.height != site_thresholds.height
    ):
        raise ValueError(
            "SAPN compliance and threshold tables have different site IDs."
        )

    return (
        site_compliance.select(list(SITE_COMPLIANCE_SCHEMA))
        .cast(SITE_COMPLIANCE_SCHEMA, strict=False)
        .sort("site_id")
    )


def build_site_compliance_tod_distribution(timestamp_detail):
    """Build per-site timestamp counts in right-closed five-minute TOD bins."""
    return (
        timestamp_detail.with_columns(
            [
                (
                    (
                        pl.col("local_tstamp") - pl.duration(microseconds=1)
                    ).dt.truncate("5m")
                    + pl.duration(minutes=5)
                )
                .dt.strftime("%H:%M")
                .alias("time_of_day_bin"),
                (
                    pl.col("los_responsible") | pl.col("ov1_responsible")
                ).alias("_eligible_threshold"),
                (
                    pl.col("los_disconnect_support_added")
                    | pl.col("ov1_disconnect_support_added")
                ).alias("_disconnect_support"),
                (pl.col("los_compliant") | pl.col("ov1_compliant")).alias(
                    "_base_compliant"
                ),
            ]
        )
        .group_by(["site_id", "time_of_day_bin"])
        .agg(
            [
                (
                    pl.col("_eligible_threshold").sum()
                    + pl.col("_disconnect_support").sum()
                )
                .cast(pl.Int64)
                .alias("eligible_timestamp_count"),
                pl.col("_eligible_threshold")
                .sum()
                .cast(pl.Int64)
                .alias("eligible_threshold_timestamp_count"),
                pl.col("_disconnect_support")
                .sum()
                .cast(pl.Int64)
                .alias("disconnect_support_timestamp_count"),
                (
                    pl.col("_base_compliant").sum()
                    + pl.col("_disconnect_support").sum()
                )
                .cast(pl.Int64)
                .alias("compliant_timestamp_count"),
                (
                    pl.col("_eligible_threshold").sum()
                    - pl.col("_base_compliant").sum()
                )
                .cast(pl.Int64)
                .alias("non_compliant_timestamp_count"),
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
        .select(
            [
                "site_id",
                "time_of_day_bin",
                "eligible_timestamp_count",
                "eligible_threshold_timestamp_count",
                "disconnect_support_timestamp_count",
                "compliant_timestamp_count",
                "non_compliant_timestamp_count",
                "disconnected_below_threshold_count",
                "disconnected_unknown_voltage_count",
            ]
        )
        .sort(["site_id", "time_of_day_bin"])
    )


def build_method_compliance_final_table(site_compliance):
    calculated = (
        site_compliance.group_by("threshold_method", maintain_order=True)
        .agg(
            [
                pl.col("site_id").n_unique().alias("Eligible Sites After Filtering"),
                pl.col("overall_calculated_pass")
                .is_not_null()
                .sum()
                .alias("Sites Assessed"),
                pl.col("overall_calculated_pass")
                .is_null()
                .sum()
                .alias("Unassessed Sites"),
                pl.col("overall_calculated_pass")
                .eq(True)
                .fill_null(False)
                .sum()
                .alias("Conformant Sites"),
                pl.col("overall_calculated_pass")
                .eq(False)
                .fill_null(False)
                .sum()
                .alias("Non-Conformant Sites"),
            ]
        )
        .with_columns(pl.lit("calculated").alias("Case"))
    )
    disconnect_supported = (
        site_compliance.group_by("threshold_method", maintain_order=True)
        .agg(
            [
                pl.col("site_id").n_unique().alias("Eligible Sites After Filtering"),
                pl.col("overall_disconnect_supported_pass")
                .is_not_null()
                .sum()
                .alias("Sites Assessed"),
                pl.col("overall_disconnect_supported_pass")
                .is_null()
                .sum()
                .alias("Unassessed Sites"),
                pl.col("overall_disconnect_supported_pass")
                .eq(True)
                .fill_null(False)
                .sum()
                .alias("Conformant Sites"),
                pl.col("overall_disconnect_supported_pass")
                .eq(False)
                .fill_null(False)
                .sum()
                .alias("Non-Conformant Sites"),
            ]
        )
        .with_columns(pl.lit("disconnect_supported").alias("Case"))
    )
    lowest_disconnect = (
        site_compliance.group_by("threshold_method", maintain_order=True)
        .agg(
            [
                pl.col("site_id").n_unique().alias("Eligible Sites After Filtering"),
                pl.col("overall_lowest_disconnect_pass")
                .is_not_null()
                .sum()
                .alias("Sites Assessed"),
                pl.col("overall_lowest_disconnect_pass")
                .is_null()
                .sum()
                .alias("Unassessed Sites"),
                pl.col("overall_lowest_disconnect_pass")
                .eq(True)
                .fill_null(False)
                .sum()
                .alias("Conformant Sites"),
                pl.col("overall_lowest_disconnect_pass")
                .eq(False)
                .fill_null(False)
                .sum()
                .alias("Non-Conformant Sites"),
            ]
        )
        .with_columns(pl.lit("lowest_disconnect").alias("Case"))
    )
    final_table = (
        pl.concat([calculated, disconnect_supported, lowest_disconnect])
        .with_columns(
            (pl.col("Conformant Sites") / pl.col("Sites Assessed") * 100.0)
            .round(2)
            .alias("Conformance Percentage (% of Assessed)")
        )
        .rename({"threshold_method": "Method Used"})
        .select(
            [
                "Method Used",
                "Case",
                "Eligible Sites After Filtering",
                "Sites Assessed",
                "Unassessed Sites",
                "Conformant Sites",
                "Non-Conformant Sites",
                "Conformance Percentage (% of Assessed)",
            ]
        )
    )
    return final_table


def build_sapn_conformance_exclusions(results):
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


def write_sapn_threshold_distribution_plots(phase_a, output_dir):
    if phase_a.is_empty():
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mechanism, voltage_column, prefix in (
        ("LOS", "v10m_disc", "los"),
        ("OV1", "vinst_disc", "ov1"),
    ):
        stats = (
            phase_a.filter(pl.col("mechanism") == mechanism)
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
