"""Build and persist conformance output tables without assessment logic."""

from pathlib import Path

import polars as pl
from config import PRIMARY_PHASE_B_METHOD
from reporting.plotting import (
    plot_site_threshold_distribution,
    plot_site_threshold_distribution_extremes,
)

PRIMARY_SITE_THRESHOLDS_NAME = f"site_thresholds_{PRIMARY_PHASE_B_METHOD}.csv"
PRIMARY_PHASE_B_SUMMARY_NAME = f"phase_b_site_summary_{PRIMARY_PHASE_B_METHOD}.csv"
PRIMARY_PHASE_B_DETAIL_NAME = f"phase_b_timestamp_detail_{PRIMARY_PHASE_B_METHOD}.csv"

ACTIVE_THRESHOLD_COLUMNS = [
    "site_id",
    "delta_los_site",
    "delta_ov1_site",
    "los_anchor_site",
    "ov1_anchor_site",
    "ov1_work_site",
    "ov1_floor_site",
    "ov1_test_site",
    "delta_gap_v",
    "los_threshold_basis",
    "los_winning_window_count",
    "los_winning_window_median_v",
    "los_overall_range_v",
    "ov1_threshold_basis",
    "ov1_winning_window_count",
    "ov1_winning_window_median_v",
    "ov1_overall_range_v",
]


def _active_threshold_output(frame):
    columns = [column for column in ACTIVE_THRESHOLD_COLUMNS if column in frame.columns]
    if "method_key" in frame.columns:
        columns.append("method_key")
    return frame.select(columns)


def _active_phase_b_summary_output(frame):
    obsolete = [
        column
        for column in ("threshold_sensitive", "pass_basis")
        if column in frame.columns
    ]
    return frame.drop(obsolete)


def build_output_tables(results, excluded_day_schema=None):
    tables = {
        "site_thresholds": _active_threshold_output(results["site_thresholds"]),
        "site_thresholds_by_method": _active_threshold_output(
            results["site_thresholds_by_method"]
        ),
        "phase_a_trip_attribution": results["phase_a_trip_attribution"],
        "phase_a_brackets": results["phase_a_brackets"],
        "phase_b_site_summary": _active_phase_b_summary_output(
            results["phase_b_site_summary"]
        ),
        "phase_b_timestamp_detail": results["phase_b_timestamp_detail"],
        "phase_b_site_summary_by_method": _active_phase_b_summary_output(
            results["phase_b_site_summary_by_method"]
        ),
        "phase_b_timestamp_detail_by_method": results[
            "phase_b_timestamp_detail_by_method"
        ],
    }
    excluded_rows = results["excluded_day_rows"]
    if excluded_day_schema is None:
        excluded_day_schema = {
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
    tables["excluded_site_days"] = (
        pl.DataFrame(excluded_rows)
        if excluded_rows
        else pl.DataFrame(schema=excluded_day_schema)
    )

    phase_b_summary = tables["phase_b_site_summary"]
    tables["assessed_sites_overall"] = (
        phase_b_summary.filter(pl.col("overall_pass").is_not_null())
        if not phase_b_summary.is_empty()
        else pl.DataFrame()
    )

    phase_a = tables["phase_a_trip_attribution"]
    if phase_a.is_empty():
        tables["los_site_threshold_stats"] = pl.DataFrame()
        tables["ov1_site_threshold_stats"] = pl.DataFrame()
    else:
        tables["los_site_threshold_stats"] = (
            phase_a.filter(pl.col("mech") == "LOS")
            .group_by("site_id")
            .agg(
                [
                    pl.col("v_los_recorded").min().alias("min_v"),
                    pl.col("v_los_recorded").median().alias("median_v"),
                    pl.col("v_los_recorded").max().alias("max_v"),
                    pl.col("v_los_recorded").std().alias("std_v"),
                    pl.len().alias("n_events"),
                ]
            )
        )
        tables["ov1_site_threshold_stats"] = (
            phase_a.filter(pl.col("mech") == "OV1")
            .group_by("site_id")
            .agg(
                [
                    pl.col("v_ov1_recorded").min().alias("min_v"),
                    pl.col("v_ov1_recorded").median().alias("median_v"),
                    pl.col("v_ov1_recorded").max().alias("max_v"),
                    pl.col("v_ov1_recorded").std().alias("std_v"),
                    pl.len().alias("n_events"),
                ]
            )
        )

    return tables


def write_outputs(results, output_dir, excluded_day_schema=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_plot_dir = output_dir / "threshold_distribution_plots"
    threshold_plot_dir.mkdir(parents=True, exist_ok=True)
    tables = build_output_tables(
        results,
        excluded_day_schema=excluded_day_schema,
    )

    filenames = {
        "site_thresholds": PRIMARY_SITE_THRESHOLDS_NAME,
        "site_thresholds_by_method": "site_thresholds_by_method.csv",
        "phase_a_trip_attribution": "phase_a_trip_attribution.csv",
        "phase_a_brackets": "phase_a_brackets.csv",
        "phase_b_site_summary": PRIMARY_PHASE_B_SUMMARY_NAME,
        "phase_b_timestamp_detail": PRIMARY_PHASE_B_DETAIL_NAME,
        "phase_b_site_summary_by_method": "phase_b_site_summary_by_method.csv",
        "phase_b_timestamp_detail_by_method": "phase_b_timestamp_detail_by_method.csv",
        "excluded_site_days": "excluded_site_days.csv",
    }
    for table_name, filename in filenames.items():
        tables[table_name].write_csv(output_dir / filename)

    conditional_files = {
        "assessed_sites_overall": "assessed_sites_overall.csv",
        "los_site_threshold_stats": "los_site_threshold_stats.csv",
        "ov1_site_threshold_stats": "ov1_site_threshold_stats.csv",
    }
    for table_name, filename in conditional_files.items():
        if not tables[table_name].is_empty():
            tables[table_name].write_csv(output_dir / filename)

    los_stats = tables["los_site_threshold_stats"]
    ov1_stats = tables["ov1_site_threshold_stats"]
    if not tables["phase_a_trip_attribution"].is_empty():
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
