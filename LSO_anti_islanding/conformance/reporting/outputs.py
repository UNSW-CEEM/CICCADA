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


def summarize_multi_method_site_outputs(phase_b_summary_by_method_df):
    if phase_b_summary_by_method_df is None or phase_b_summary_by_method_df.is_empty():
        return {"site_comparison": pl.DataFrame()}

    comparison = (
        phase_b_summary_by_method_df.select("site_id").unique().sort("site_id")
    )
    pivot_specs = [
        "overall_pass",
        "los_pass",
        "ov1_pass",
        "los_eligible",
        "ov1_eligible",
        "los_compliance_pct",
        "ov1_compliance_pct",
        "los_threshold_used",
        "pass_basis",
    ]
    for value_col in pivot_specs:
        pivot = phase_b_summary_by_method_df.select([
            "site_id",
            "method_key",
            value_col,
        ]).pivot(
            values=value_col,
            index="site_id",
            on="method_key",
            aggregate_function="first",
        )
        comparison = comparison.join(
            pivot.rename({
                col: f"{value_col}__{col}"
                for col in pivot.columns
                if col != "site_id"
            }),
            on="site_id",
            how="left",
        )

    overall_cols = sorted(
        c for c in comparison.columns if c.startswith("overall_pass__")
    )
    comparison = comparison.with_columns([
        pl.struct(overall_cols)
        .map_elements(
            lambda row: len(set(row.values())) > 1,
            return_dtype=pl.Boolean,
        )
        .alias("any_disagreement"),
        pl.struct(overall_cols)
        .map_elements(
            lambda row: len({v for v in row.values() if v is not None}) > 1,
            return_dtype=pl.Boolean,
        )
        .alias("assessed_outcome_disagreement"),
        pl.struct(overall_cols)
        .map_elements(
            lambda row: any(v is not None for v in row.values()),
            return_dtype=pl.Boolean,
        )
        .alias("any_method_assessed"),
        pl.struct(overall_cols)
        .map_elements(
            lambda row: all(v is None for v in row.values()),
            return_dtype=pl.Boolean,
        )
        .alias("all_methods_unassessed"),
    ])
    return {"site_comparison": comparison}


def build_output_tables(results, excluded_day_schema=None):
    tables = {
        "site_thresholds": results["site_thresholds"],
        "site_thresholds_by_method": results["site_thresholds_by_method"],
        "phase_a_trip_attribution": results["phase_a_trip_attribution"],
        "phase_a_brackets": results["phase_a_brackets"],
        "phase_b_site_summary": results["phase_b_site_summary"],
        "phase_b_timestamp_detail": results["phase_b_timestamp_detail"],
        "phase_b_site_summary_by_method": results[
            "phase_b_site_summary_by_method"
        ],
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
            .agg([
                pl.col("v_los_recorded").min().alias("min_v"),
                pl.col("v_los_recorded").median().alias("median_v"),
                pl.col("v_los_recorded").max().alias("max_v"),
                pl.col("v_los_recorded").std().alias("std_v"),
                pl.len().alias("n_events"),
            ])
        )
        tables["ov1_site_threshold_stats"] = (
            phase_a.filter(pl.col("mech") == "OV1")
            .group_by("site_id")
            .agg([
                pl.col("v_ov1_recorded").min().alias("min_v"),
                pl.col("v_ov1_recorded").median().alias("median_v"),
                pl.col("v_ov1_recorded").max().alias("max_v"),
                pl.col("v_ov1_recorded").std().alias("std_v"),
                pl.len().alias("n_events"),
            ])
        )

    tables["five_method_site_comparison"] = summarize_multi_method_site_outputs(
        tables["phase_b_site_summary_by_method"]
    )["site_comparison"]
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
        "five_method_site_comparison": "five_method_site_comparison.csv",
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
