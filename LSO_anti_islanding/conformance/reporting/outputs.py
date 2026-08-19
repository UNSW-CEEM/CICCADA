"""Build and persist conformance output tables without assessment logic."""

from pathlib import Path

import polars as pl
from config import PRIMARY_PHASE_B_METHOD
from reporting.plotting import (
    plot_site_threshold_distribution,
    plot_site_threshold_distribution_extremes,
)

SITE_CONFORMANCE_SUMMARY_NAME = "site_conformance_summary.csv"
CONFORMANCE_EXCLUSIONS_NAME = "conformance_exclusions.csv"
PRIMARY_PHASE_B_DETAIL_NAME = (
    f"phase_b_timestamp_detail_{PRIMARY_PHASE_B_METHOD}.csv"
)

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

DEFAULT_EXCLUDED_DAY_SCHEMA = {
    "site_id": pl.Int64,
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


def _require_columns(frame, required_columns, table_name):
    missing_columns = set(required_columns).difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: {sorted(missing_columns)}"
        )


def _require_unique_site_ids(frame, table_name):
    if frame.is_empty():
        return
    duplicate_sites = (
        frame.group_by("site_id")
        .len()
        .filter(pl.col("len") > 1)
        .select("site_id")
    )
    if not duplicate_sites.is_empty():
        sample = duplicate_sites.head(10)["site_id"].to_list()
        raise ValueError(
            f"{table_name} must contain one row per site_id; duplicates include "
            f"{sample}"
        )


def _site_conformance_summary(results):
    phase_b_summary = results["phase_b_site_summary"]
    site_thresholds = results["site_thresholds"]
    if phase_b_summary.is_empty() and site_thresholds.is_empty():
        return pl.DataFrame(schema=SITE_CONFORMANCE_SUMMARY_SCHEMA)
    if phase_b_summary.is_empty() or site_thresholds.is_empty():
        raise ValueError(
            "Primary Phase B summary and threshold tables must contain the same sites."
        )

    summary_columns = [
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
    threshold_columns = [
        "site_id",
        "los_threshold_basis",
        "ov1_test_site",
        "ov1_threshold_basis",
    ]
    _require_columns(phase_b_summary, summary_columns, "phase_b_site_summary")
    _require_columns(site_thresholds, threshold_columns, "site_thresholds")
    _require_unique_site_ids(phase_b_summary, "phase_b_site_summary")
    _require_unique_site_ids(site_thresholds, "site_thresholds")

    combined = phase_b_summary.select(summary_columns).join(
        site_thresholds.select(threshold_columns),
        on="site_id",
        how="inner",
        validate="1:1",
    )
    if (
        combined.height != phase_b_summary.height
        or combined.height != site_thresholds.height
    ):
        raise ValueError(
            "Primary Phase B summary and threshold tables do not contain identical "
            "site_id values."
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


def _conformance_exclusions(results, excluded_day_schema):
    day_schema = dict(excluded_day_schema or DEFAULT_EXCLUDED_DAY_SCHEMA)
    ordered_columns = ["site_id", "exclusion_scope", "day", "reason"]
    ordered_columns.extend(
        column
        for column in day_schema
        if column not in {"site_id", "day", "reason"}
    )
    exclusion_schema = {
        column: (
            pl.Utf8
            if column == "exclusion_scope"
            else day_schema.get(column, pl.Utf8)
        )
        for column in ordered_columns
    }

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
        {**row, "exclusion_scope": "site_day"}
        for row in results["excluded_day_rows"]
    )

    normalised_rows = [
        {column: row.get(column) for column in ordered_columns} for row in rows
    ]
    if not normalised_rows:
        return pl.DataFrame(schema=exclusion_schema)
    return pl.DataFrame(
        normalised_rows,
        schema=exclusion_schema,
        strict=False,
    ).sort(["site_id", "exclusion_scope", "day"], nulls_last=True)


def _site_threshold_stats(phase_a, mechanism, voltage_column):
    return (
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


def build_output_tables(
    results,
    excluded_day_schema=None,
    include_by_method_outputs=False,
):
    """Build the consolidated persisted tables from pipeline results."""
    del include_by_method_outputs  # Retained for caller compatibility.
    return {
        "site_conformance_summary": _site_conformance_summary(results),
        "conformance_exclusions": _conformance_exclusions(
            results,
            excluded_day_schema,
        ),
        "phase_b_timestamp_detail": results["phase_b_timestamp_detail"],
    }


def write_outputs(
    results,
    output_dir,
    excluded_day_schema=None,
    include_by_method_outputs=False,
    include_timestamp_detail_output=False,
):
    """Write consolidated CSV outputs and the existing threshold plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_plot_dir = output_dir / "threshold_distribution_plots"
    threshold_plot_dir.mkdir(parents=True, exist_ok=True)
    tables = build_output_tables(
        results,
        excluded_day_schema=excluded_day_schema,
        include_by_method_outputs=include_by_method_outputs,
    )

    tables["site_conformance_summary"].write_csv(
        output_dir / SITE_CONFORMANCE_SUMMARY_NAME
    )
    tables["conformance_exclusions"].write_csv(
        output_dir / CONFORMANCE_EXCLUSIONS_NAME
    )
    if include_timestamp_detail_output:
        tables["phase_b_timestamp_detail"].write_csv(
            output_dir / PRIMARY_PHASE_B_DETAIL_NAME
        )

    phase_a = results["phase_a_trip_attribution"]
    if not phase_a.is_empty():
        los_stats = _site_threshold_stats(phase_a, "LOS", "v_los_recorded")
        ov1_stats = _site_threshold_stats(phase_a, "OV1", "v_ov1_recorded")
        plot_site_threshold_distribution(
            los_stats,
            title="LSO Thresholds Across Assessed Sites — Min / Median / Max (Std on right)",
            save_path=threshold_plot_dir / "los_threshold_distribution.png",
        )
        plot_site_threshold_distribution_extremes(
            los_stats,
            title="LSO Thresholds — Highest 20 Std Sites (n_events >= 3)",
            highest_std=True,
            min_events=3,
            n_sites=20,
            save_path=threshold_plot_dir / "los_threshold_highest20_std.png",
        )
        plot_site_threshold_distribution_extremes(
            los_stats,
            title="LSO Thresholds — Lowest 20 Std Sites (n_events >= 3)",
            highest_std=False,
            min_events=3,
            n_sites=20,
            save_path=threshold_plot_dir / "los_threshold_lowest20_std.png",
        )
        plot_site_threshold_distribution(
            ov1_stats,
            title="OV1 Thresholds Across Assessed Sites — Min / Median / Max (Std on right)",
            save_path=threshold_plot_dir / "ov1_threshold_distribution.png",
        )
        plot_site_threshold_distribution_extremes(
            ov1_stats,
            title="OV1 Thresholds — Highest 20 Std Sites (n_events >= 3)",
            highest_std=True,
            min_events=3,
            n_sites=20,
            save_path=threshold_plot_dir / "ov1_threshold_highest20_std.png",
        )
        plot_site_threshold_distribution_extremes(
            ov1_stats,
            title="OV1 Thresholds — Lowest 20 Std Sites (n_events >= 3)",
            highest_std=False,
            min_events=3,
            n_sites=20,
            save_path=threshold_plot_dir / "ov1_threshold_lowest20_std.png",
        )
    return tables
