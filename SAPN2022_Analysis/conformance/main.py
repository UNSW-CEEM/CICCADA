import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from checkPVBehaviour import (
    run_phase_a_for_site,
    run_phase_b_for_site,
)
from funcs import (
    loadCleanedSiteData,
    ratedCapacityOfPV,
)
from plots.plots import (
    plot_site_compliance_day,
    plot_site_threshold_distribution,
    plot_site_threshold_distribution_extremes,
)
from nov2022_site_days import collect_site_days
from summaryStats import (
    summarize_multi_method_site_outputs,
)


DATA_DIR = REPO_ROOT / "Nov2022"
OUTPUT_DIR = REPO_ROOT / "updated results" / "site_compliance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "overall_site_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD_PLOT_DIR = OUTPUT_DIR / "threshold_distribution_plots"
THRESHOLD_PLOT_DIR.mkdir(parents=True, exist_ok=True)
GENERATE_SITE_PLOTS = os.getenv("SITE_COMPLIANCE_SKIP_PLOTS", "0") != "1"

# Available Phase B methods: default, original, tier_based, old_sweep, blended.
# Keep the primary method in the run list below.
PRIMARY_PHASE_B_METHOD = "tier_based"
PHASE_B_METHODS_TO_RUN = ["default", "original", "tier_based", "old_sweep", "blended"]
PRIMARY_SITE_THRESHOLDS_NAME = f"site_thresholds_{PRIMARY_PHASE_B_METHOD}.csv"
PRIMARY_PHASE_B_SUMMARY_NAME = f"phase_b_site_summary_{PRIMARY_PHASE_B_METHOD}.csv"
PRIMARY_PHASE_B_DETAIL_NAME = f"phase_b_timestamp_detail_{PRIMARY_PHASE_B_METHOD}.csv"


def build_site_selection_tables(site_details, circuit_details):
    site_metadata_rows = {
        row["site_id"]: int(row["site_metadata_rows"])
        for row in (
            site_details
            .group_by("site_id")
            .len()
            .rename({"len": "site_metadata_rows"})
            .to_dicts()
        )
    }

    pv_site_net_counts = {
        row["site_id"]: int(row["pv_site_net_count"])
        for row in (
            circuit_details
            .filter(pl.col("con_type") == "pv_site_net")
            .group_by("site_id")
            .len()
            .rename({"len": "pv_site_net_count"})
            .to_dicts()
        )
    }

    return site_metadata_rows, pv_site_net_counts


def main():
    # Load runtime inputs for the conformance run.
    site_details = pl.read_csv(DATA_DIR / "ebm_1_20221112_20221119_site_details.csv")
    circuit_details = pl.read_csv(DATA_DIR / "ebm_1_20221112_20221119_circuit_details.csv")
    all_data = loadCleanedSiteData()

    # Build site-level lookup tables used by the initial filters.
    site_metadata_rows, pv_site_net_counts = build_site_selection_tables(
        site_details, circuit_details
    )
    days_to_check = [13, 14, 15, 16, 17, 19]

    # Keep only sites that appear in both cleaned timeseries data and circuit metadata.
    sites_with_circuit_data = (
        all_data.select("c_id")
        .unique()
        .join(
            circuit_details.select(["c_id", "site_id"]).unique().lazy(),
            on="c_id",
            how="inner",
        )
        .select("site_id")
        .unique()
        .collect()["site_id"]
        .to_list()
    )

    threshold_rows = []
    threshold_rows_by_method = []
    phase_a_records = []
    bracket_rows = []
    phase_b_summary_rows = []
    phase_b_detail_rows = []
    phase_b_summary_rows_by_method = []
    phase_b_detail_rows_by_method = []

    skipped_not_single_inverter = []
    skipped_more_than_3 = []
    skipped_no_pv_site_net = []
    skipped_no_day_data = []
    skipped_no_eligible_days = []
    excluded_day_rows = []

    for idx, site_number in enumerate(sites_with_circuit_data, start=1):
        # Apply the site-level conformance filters before any day-level work.
        metadata_row_count = site_metadata_rows.get(site_number, 0)
        pv_site_net_count = pv_site_net_counts.get(site_number, 0)

        if metadata_row_count != 1:
            skipped_not_single_inverter.append(site_number)
            continue

        if pv_site_net_count == 0:
            skipped_no_pv_site_net.append(site_number)
            continue

        if pv_site_net_count > 3:
            skipped_more_than_3.append(site_number)
            continue

        # Build the per-day behaviour objects used by Phase A and Phase B.
        site_day_result = collect_site_days(
            site_number,
            circuit_details,
            all_data,
            days_to_check,
        )
        day_behaviours = site_day_result["eligible_day_behaviours"]
        excluded_day_rows.extend(site_day_result["excluded_day_rows"])

        if site_day_result["mapped_day_count"] == 0:
            skipped_no_day_data.append(site_number)
            continue

        if not day_behaviours:
            skipped_no_eligible_days.append(site_number)
            continue

        p_rated = ratedCapacityOfPV(
            site_details,
            site_number,
            day_behaviours=day_behaviours,
        )

        # Run Phase A once per site to build the candidate threshold inputs.
        phase_a = run_phase_a_for_site(site_number, day_behaviours, p_rated)

        if not phase_a["records"].is_empty():
            phase_a_records.append(phase_a["records"])
        if not phase_a["brackets"].is_empty():
            bracket_rows.append(phase_a["brackets"])

        # Run each selected Phase B method once and keep the configured primary result.
        primary_phase_b = None
        for method_key in PHASE_B_METHODS_TO_RUN:
            method_phase_b = run_phase_b_for_site(
                site_number,
                day_behaviours,
                p_rated,
                raw_thresholds=phase_a["raw_thresholds"],
                confidence_info=phase_a["confidence_info"],
                phase_b_method=method_key,
            )
            threshold_rows_by_method.append(
                method_phase_b["threshold_row"].with_columns([
                    pl.lit(method_key).alias("method_key"),
                ])
            )
            phase_b_summary_rows_by_method.append(
                method_phase_b["summary_row"].with_columns([
                    pl.lit(method_key).alias("method_key"),
                ])
            )
            if not method_phase_b["detail"].is_empty():
                phase_b_detail_rows_by_method.append(
                    method_phase_b["detail"].with_columns([
                        pl.lit(method_key).alias("method_key"),
                    ])
                )
            if method_key == PRIMARY_PHASE_B_METHOD:
                primary_phase_b = method_phase_b

        # Use the configured primary method for the main outputs and per-site plots.
        primary_thresholds = primary_phase_b["threshold_row"].to_dicts()[0]
        threshold_rows.append(primary_phase_b["threshold_row"])
        phase_b_summary_rows.append(primary_phase_b["summary_row"])
        if not primary_phase_b["detail"].is_empty():
            phase_b_detail_rows.append(primary_phase_b["detail"])

        summary = primary_phase_b["summary_row"].to_dicts()[0]
        if GENERATE_SITE_PLOTS and summary["overall_pass"] is not None:
            plot_folder = "compliant" if summary["overall_pass"] is True else "non_compliant"
            for day_info in day_behaviours:
                day_plot = day_info["behaviour"].phase_b_day(
                    p_rated,
                    los_threshold=summary["los_threshold_used"],
                    ov1_work_threshold=primary_thresholds["ov1_work_site"],
                )
                plot_path = (
                    PLOT_DIR
                    / plot_folder
                    / f"Site_{site_number}_Day_{day_info['day']}_{plot_folder}.png"
                )
                plot_site_compliance_day(
                    day_plot["frame"],
                    site_number,
                    day_info["day"],
                    p_rated=p_rated,
                    lso_threshold=summary["los_threshold_used"],
                    ov1_threshold=primary_thresholds["ov1_test_site"],
                    overall_pass=summary["overall_pass"],
                    day_summary=day_plot["summary"],
                    save_path=plot_path,
                )

        print(
            f"[{idx}/{len(sites_with_circuit_data)}] site {site_number} "
            f"LOS={summary['los_compliance_pct']} OV1={summary['ov1_compliance_pct']} "
            f"PASS={summary['overall_pass']}"
        )

    # Write the primary and by-method outputs collected across all assessed sites.
    thresholds_df = pl.concat(threshold_rows, how="vertical") if threshold_rows else pl.DataFrame()
    thresholds_by_method_df = (
        pl.concat(threshold_rows_by_method, how="vertical")
        if threshold_rows_by_method else pl.DataFrame()
    )
    phase_a_df = pl.concat(phase_a_records, how="vertical") if phase_a_records else pl.DataFrame()
    brackets_df = pl.concat(bracket_rows, how="vertical") if bracket_rows else pl.DataFrame()
    phase_b_summary_df = (
        pl.concat(phase_b_summary_rows, how="vertical") if phase_b_summary_rows else pl.DataFrame()
    )
    phase_b_detail_df = (
        pl.concat(phase_b_detail_rows, how="vertical") if phase_b_detail_rows else pl.DataFrame()
    )
    phase_b_summary_by_method_df = (
        pl.concat(phase_b_summary_rows_by_method, how="vertical")
        if phase_b_summary_rows_by_method else pl.DataFrame()
    )
    phase_b_detail_by_method_df = (
        pl.concat(phase_b_detail_rows_by_method, how="vertical")
        if phase_b_detail_rows_by_method else pl.DataFrame()
    )

    thresholds_df.write_csv(OUTPUT_DIR / PRIMARY_SITE_THRESHOLDS_NAME)
    thresholds_by_method_df.write_csv(OUTPUT_DIR / "site_thresholds_by_method.csv")
    phase_a_df.write_csv(OUTPUT_DIR / "phase_a_trip_attribution.csv")
    brackets_df.write_csv(OUTPUT_DIR / "phase_a_brackets.csv")
    phase_b_summary_df.write_csv(OUTPUT_DIR / PRIMARY_PHASE_B_SUMMARY_NAME)
    phase_b_detail_df.write_csv(OUTPUT_DIR / PRIMARY_PHASE_B_DETAIL_NAME)
    phase_b_summary_by_method_df.write_csv(OUTPUT_DIR / "phase_b_site_summary_by_method.csv")
    phase_b_detail_by_method_df.write_csv(OUTPUT_DIR / "phase_b_timestamp_detail_by_method.csv")
    excluded_site_days_df = (
        pl.DataFrame(excluded_day_rows)
        if excluded_day_rows
        else pl.DataFrame(
            schema={
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
        )
    )
    excluded_site_days_df.write_csv(OUTPUT_DIR / "excluded_site_days.csv")

    # Export the summary tables used by downstream inspection and plotting scripts.
    if not phase_b_summary_df.is_empty():
        assessed_overall = phase_b_summary_df.filter(pl.col("overall_pass").is_not_null())
        assessed_overall.write_csv(OUTPUT_DIR / "assessed_sites_overall.csv")

    if not phase_a_df.is_empty():
        los_threshold_stats = (
            phase_a_df
            .filter(pl.col("mech") == "LOS")
            .group_by("site_id")
            .agg([
                pl.col("v_los_recorded").min().alias("min_v"),
                pl.col("v_los_recorded").median().alias("median_v"),
                pl.col("v_los_recorded").max().alias("max_v"),
                pl.col("v_los_recorded").std().alias("std_v"),
                pl.len().alias("n_events"),
            ])
        )
        ov1_threshold_stats = (
            phase_a_df
            .filter(pl.col("mech") == "OV1")
            .group_by("site_id")
            .agg([
                pl.col("v_ov1_recorded").min().alias("min_v"),
                pl.col("v_ov1_recorded").median().alias("median_v"),
                pl.col("v_ov1_recorded").max().alias("max_v"),
                pl.col("v_ov1_recorded").std().alias("std_v"),
                pl.len().alias("n_events"),
            ])
        )

        los_threshold_stats.write_csv(OUTPUT_DIR / "los_site_threshold_stats.csv")
        ov1_threshold_stats.write_csv(OUTPUT_DIR / "ov1_site_threshold_stats.csv")

        plot_site_threshold_distribution(
            los_threshold_stats,
            title="LSO Thresholds Across Assessed Sites — Min / Median / Max (Std on right)",
            save_path=THRESHOLD_PLOT_DIR / "los_threshold_distribution.png",
        )
        plot_site_threshold_distribution_extremes(
            los_threshold_stats,
            title="LSO Thresholds — Highest 20 Std Sites (n_events >= 3)",
            highest_std=True,
            min_events=3,
            n_sites=20,
            save_path=THRESHOLD_PLOT_DIR / "los_threshold_highest20_std.png",
        )
        plot_site_threshold_distribution_extremes(
            los_threshold_stats,
            title="LSO Thresholds — Lowest 20 Std Sites (n_events >= 3)",
            highest_std=False,
            min_events=3,
            n_sites=20,
            save_path=THRESHOLD_PLOT_DIR / "los_threshold_lowest20_std.png",
        )
        plot_site_threshold_distribution(
            ov1_threshold_stats,
            title="OV1 Thresholds Across Assessed Sites — Min / Median / Max (Std on right)",
            save_path=THRESHOLD_PLOT_DIR / "ov1_threshold_distribution.png",
        )
        plot_site_threshold_distribution_extremes(
            ov1_threshold_stats,
            title="OV1 Thresholds — Highest 20 Std Sites (n_events >= 3)",
            highest_std=True,
            min_events=3,
            n_sites=20,
            save_path=THRESHOLD_PLOT_DIR / "ov1_threshold_highest20_std.png",
        )
        plot_site_threshold_distribution_extremes(
            ov1_threshold_stats,
            title="OV1 Thresholds — Lowest 20 Std Sites (n_events >= 3)",
            highest_std=False,
            min_events=3,
            n_sites=20,
            save_path=THRESHOLD_PLOT_DIR / "ov1_threshold_lowest20_std.png",
        )

    multi_method_outputs = summarize_multi_method_site_outputs(phase_b_summary_by_method_df)
    if not multi_method_outputs["site_comparison"].is_empty():
        multi_method_outputs["site_comparison"].write_csv(
            OUTPUT_DIR / "five_method_site_comparison.csv"
        )

    print("Saved outputs to", OUTPUT_DIR)
    print("Skipped (site metadata rows != 1):", len(skipped_not_single_inverter))
    print("Skipped (>3 PV circuits):", len(skipped_more_than_3))
    print("Skipped (no pv_site_net circuits):", len(skipped_no_pv_site_net))
    print("Skipped (no day data):", len(skipped_no_day_data))
    print("Skipped (no eligible days):", len(skipped_no_eligible_days))


if __name__ == "__main__":
    main()
