import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from checkPVBehaviour import (
    PHASE_B_METHOD_SPECS,
    CheckPVBehaviour,
    run_phase_a_for_site,
    run_phase_b_for_site,
)
from funcs import (
    loadCleanedSiteData,
    mapCircuitDataToSite,
    ratedCapacityOfPV,
)
from plots.plots import (
    plot_site_compliance_day,
    plot_site_threshold_distribution,
    plot_site_threshold_distribution_extremes,
)
from summaryStats import (
    summarize_multi_method_site_outputs,
    summarize_site_compliance_outputs,
)


OUTPUT_DIR = Path("updated results/site_compliance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "overall_site_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD_PLOT_DIR = OUTPUT_DIR / "threshold_distribution_plots"
THRESHOLD_PLOT_DIR.mkdir(parents=True, exist_ok=True)
OV1_PLOT_DIR = OUTPUT_DIR / "ov1_assessed_site_plots"
OV1_PLOT_DIR.mkdir(parents=True, exist_ok=True)
FIVE_METHOD_GROUP_DIR = OUTPUT_DIR / "five_method_site_groups"
FIVE_METHOD_GROUP_DIR.mkdir(parents=True, exist_ok=True)
FIVE_METHOD_ALL_DIR = FIVE_METHOD_GROUP_DIR / "all_sites"
FIVE_METHOD_ALL_DIR.mkdir(parents=True, exist_ok=True)
FIVE_METHOD_SAME_DIR = FIVE_METHOD_GROUP_DIR / "same_behavior"
FIVE_METHOD_SAME_DIR.mkdir(parents=True, exist_ok=True)
FIVE_METHOD_DIFF_DIR = FIVE_METHOD_GROUP_DIR / "different_behavior"
FIVE_METHOD_DIFF_DIR.mkdir(parents=True, exist_ok=True)
DAY_COVERAGE_THRESHOLD = 0.80
GENERATE_SITE_PLOTS = os.getenv("SITE_COMPLIANCE_SKIP_PLOTS", "0") != "1"


def prepare_inputs():
    site_details = pl.read_csv("Nov2022/ebm_1_20221112_20221119_site_details.csv")
    circuit_details = pl.read_csv("Nov2022/ebm_1_20221112_20221119_circuit_details.csv")

    all_sites = site_details["site_id"].unique().sort()
    all_sites_2 = circuit_details["site_id"].unique().sort()
    if not (all_sites == all_sites_2).all():
        raise ValueError("Num Sites not Consistent")

    all_data = loadCleanedSiteData()

    return site_details, circuit_details, all_data


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


def collect_site_days(site_number, circuit_details, all_data, days_to_check):
    day_behaviours = []
    pv_circuits = []

    for day in days_to_check:
        start_day = pl.datetime(2022, 11, day, 6, 0, 0, time_zone="Australia/Adelaide")
        end_day = pl.datetime(2022, 11, day, 18, 0, 0, time_zone="Australia/Adelaide")

        has_data, wide, pv_circuit_nos = mapCircuitDataToSite(
            all_data, circuit_details, site_number, start_day, end_day
        )
        if not has_data:
            continue

        pv_circuits = pv_circuit_nos
        behaviour = CheckPVBehaviour(wide, volCol="voltage_valid")
        day_behaviours.append(
            {
                "day": day,
                "behaviour": behaviour,
                "eligibility": behaviour.day_eligibility_summary(
                    coverage_threshold=DAY_COVERAGE_THRESHOLD
                ),
            }
        )

    return day_behaviours, pv_circuits


def main():
    site_details, circuit_details, all_data = prepare_inputs()
    site_metadata_rows, pv_site_net_counts = build_site_selection_tables(
        site_details, circuit_details
    )
    days_to_check = [13, 14, 15, 16, 17, 19]

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

        day_behaviours, pv_circuits = collect_site_days(
            site_number, circuit_details, all_data, days_to_check
        )

        if not day_behaviours:
            skipped_no_day_data.append(site_number)
            continue

        eligible_day_behaviours = []
        for day_info in day_behaviours:
            day_eligibility = day_info["eligibility"]
            if day_eligibility["eligible"]:
                eligible_day_behaviours.append(day_info)
            else:
                excluded_day_rows.append({
                    "site_id": site_number,
                    "day": day_info["day"],
                    "reason": day_eligibility["reason"],
                    "common_power_v10m_coverage_pct": day_eligibility["common_power_v10m_coverage_pct"],
                    "rows_common_power_v10m": day_eligibility["rows_common_power_v10m"],
                    "rows_with_power": day_eligibility["rows_with_power"],
                    "rows_with_v10m": day_eligibility["rows_with_v10m"],
                    "covered_seconds": day_eligibility["covered_seconds"],
                    "window_seconds": day_eligibility["window_seconds"],
                    "total_rows": day_eligibility["total_rows"],
                    "coverage_threshold_pct": DAY_COVERAGE_THRESHOLD * 100.0,
                })

        if not eligible_day_behaviours:
            skipped_no_eligible_days.append(site_number)
            continue

        p_rated = ratedCapacityOfPV(
            site_details,
            site_number,
            day_behaviours=eligible_day_behaviours,
        )

        phase_a = run_phase_a_for_site(site_number, eligible_day_behaviours, p_rated)
        thresholds = phase_a["thresholds"]
        phase_b = run_phase_b_for_site(
            site_number,
            eligible_day_behaviours,
            p_rated,
            los_threshold=thresholds["los_anchor_site"],
            los_threshold_p25=thresholds["los_anchor_p25_site"],
            los_threshold_p10=thresholds["los_anchor_p10_site"],
            los_threshold_min=thresholds["los_anchor_min_site"],
            ov1_work_threshold=thresholds["ov1_work_site"],
        )
        method_thresholds = phase_a["method_thresholds"]

        threshold_rows.append(phase_a["thresholds_row"])
        threshold_rows_by_method.extend(phase_a["method_threshold_rows"].values())
        if not phase_a["records"].is_empty():
            phase_a_records.append(phase_a["records"])
        if not phase_a["brackets"].is_empty():
            bracket_rows.append(phase_a["brackets"])
        phase_b_summary_rows.append(phase_b["summary_row"])
        if not phase_b["detail"].is_empty():
            phase_b_detail_rows.append(phase_b["detail"])

        for method_key, method_label in PHASE_B_METHOD_SPECS:
            if method_key == "tier_based":
                method_phase_b = phase_b
            else:
                method_profile = method_thresholds[method_key]
                method_phase_b = run_phase_b_for_site(
                    site_number,
                    eligible_day_behaviours,
                    p_rated,
                    los_threshold=method_profile["los_anchor_site"],
                    los_threshold_p25=method_profile["los_anchor_p25_site"],
                    los_threshold_p10=method_profile["los_anchor_p10_site"],
                    los_threshold_min=method_profile["los_anchor_min_site"],
                    ov1_work_threshold=method_profile["ov1_work_site"],
                )

            phase_b_summary_rows_by_method.append(
                method_phase_b["summary_row"].with_columns([
                    pl.lit(method_key).alias("method_key"),
                    pl.lit(method_label).alias("method_label"),
                ])
            )
            if not method_phase_b["detail"].is_empty():
                phase_b_detail_rows_by_method.append(
                    method_phase_b["detail"].with_columns([
                        pl.lit(method_key).alias("method_key"),
                        pl.lit(method_label).alias("method_label"),
                    ])
                )

        summary = phase_b["summary_row"].to_dicts()[0]
        if GENERATE_SITE_PLOTS and summary["overall_pass"] is not None:
            plot_folder = "compliant" if summary["overall_pass"] is True else "non_compliant"
            for day_info in eligible_day_behaviours:
                day_plot = day_info["behaviour"].phase_b_day(
                    p_rated,
                    los_threshold=summary["los_threshold_used"],
                    ov1_work_threshold=thresholds["ov1_work_site"],
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
                    los_threshold=summary["los_threshold_used"],
                    los_threshold_p25=thresholds["los_anchor_p25_site"],
                    los_threshold_p10=thresholds["los_anchor_p10_site"],
                    los_threshold_min=thresholds["los_anchor_min_site"],
                    ov1_threshold=thresholds["ov1_test_site"],
                    delta_los_site=thresholds["delta_los_site"],
                    delta_los_p25_site=thresholds["delta_los_p25_site"],
                    delta_los_p10_site=thresholds["delta_los_p10_site"],
                    delta_los_min_site=thresholds["delta_los_min_site"],
                    delta_ov1_site=thresholds["delta_ov1_site"],
                    ov1_basis=thresholds["ov1_basis"],
                    overall_pass=summary["overall_pass"],
                    pass_basis=summary["pass_basis"],
                    day_summary=day_plot["summary"],
                    save_path=plot_path,
                )
                if summary["ov1_eligible"] > 0:
                    ov1_plot_path = (
                        OV1_PLOT_DIR
                        / plot_folder
                        / f"Site_{site_number}_Day_{day_info['day']}_{plot_folder}_ov1_focus.png"
                    )
                    plot_site_compliance_day(
                        day_plot["frame"],
                        site_number,
                        day_info["day"],
                        p_rated=p_rated,
                        los_threshold=summary["los_threshold_used"],
                        los_threshold_p25=thresholds["los_anchor_p25_site"],
                        los_threshold_p10=thresholds["los_anchor_p10_site"],
                        los_threshold_min=thresholds["los_anchor_min_site"],
                        ov1_threshold=thresholds["ov1_test_site"],
                        delta_los_site=thresholds["delta_los_site"],
                        delta_los_p25_site=thresholds["delta_los_p25_site"],
                        delta_los_p10_site=thresholds["delta_los_p10_site"],
                        delta_los_min_site=thresholds["delta_los_min_site"],
                        delta_ov1_site=thresholds["delta_ov1_site"],
                        ov1_basis=thresholds["ov1_basis"],
                        overall_pass=summary["overall_pass"],
                        pass_basis=summary["pass_basis"],
                        day_summary=day_plot["summary"],
                        force_draw_los_threshold=True,
                        force_draw_ov1_threshold=True,
                        save_path=ov1_plot_path,
                    )

        print(
            f"[{idx}/{len(sites_with_circuit_data)}] site {site_number} "
            f"LOS={summary['los_compliance_pct']} OV1={summary['ov1_compliance_pct']} "
            f"PASS={summary['overall_pass']}"
        )

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

    thresholds_df.write_csv(OUTPUT_DIR / "site_thresholds.csv")
    thresholds_by_method_df.write_csv(OUTPUT_DIR / "site_thresholds_by_method.csv")
    phase_a_df.write_csv(OUTPUT_DIR / "phase_a_trip_attribution.csv")
    brackets_df.write_csv(OUTPUT_DIR / "phase_a_brackets.csv")
    phase_b_summary_df.write_csv(OUTPUT_DIR / "phase_b_site_summary.csv")
    phase_b_detail_df.write_csv(OUTPUT_DIR / "phase_b_timestamp_detail.csv")
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

    if not phase_b_summary_df.is_empty():
        assessed_overall = phase_b_summary_df.filter(pl.col("overall_pass").is_not_null())
        assessed_los = phase_b_summary_df.filter(pl.col("los_pass").is_not_null())
        assessed_ov1 = phase_b_summary_df.filter(pl.col("ov1_pass").is_not_null())
        unassessed = phase_b_summary_df.filter(pl.col("overall_pass").is_null())

        assessed_overall.write_csv(OUTPUT_DIR / "assessed_sites_overall.csv")
        assessed_los.write_csv(OUTPUT_DIR / "assessed_sites_los.csv")
        assessed_ov1.write_csv(OUTPUT_DIR / "assessed_sites_ov1.csv")
        unassessed.write_csv(OUTPUT_DIR / "unassessed_sites.csv")

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
            title="LOS Thresholds Across Assessed Sites — Min / Median / Max (Std on right)",
            save_path=THRESHOLD_PLOT_DIR / "los_threshold_distribution.png",
        )
        plot_site_threshold_distribution_extremes(
            los_threshold_stats,
            title="LOS Thresholds — Highest 20 Std Sites (n_events >= 3)",
            highest_std=True,
            min_events=3,
            n_sites=20,
            save_path=THRESHOLD_PLOT_DIR / "los_threshold_highest20_std.png",
        )
        plot_site_threshold_distribution_extremes(
            los_threshold_stats,
            title="LOS Thresholds — Lowest 20 Std Sites (n_events >= 3)",
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

    summary_outputs = summarize_site_compliance_outputs(phase_b_summary_df, thresholds_df)
    if not summary_outputs["overall_summary"].is_empty():
        summary_outputs["overall_summary"].write_csv(OUTPUT_DIR / "phase_b_overall_summary.csv")
    if not summary_outputs["mechanism_summary"].is_empty():
        summary_outputs["mechanism_summary"].write_csv(OUTPUT_DIR / "phase_b_mechanism_summary.csv")
    if not summary_outputs["threshold_summary"].is_empty():
        summary_outputs["threshold_summary"].write_csv(OUTPUT_DIR / "threshold_summary.csv")

    multi_method_outputs = summarize_multi_method_site_outputs(
        phase_b_summary_by_method_df,
        thresholds_by_method_df,
    )
    if not multi_method_outputs["method_summary"].is_empty():
        multi_method_outputs["method_summary"].write_csv(OUTPUT_DIR / "five_method_summary.csv")
    if not multi_method_outputs["site_comparison"].is_empty():
        multi_method_outputs["site_comparison"].write_csv(
            OUTPUT_DIR / "five_method_site_comparison.csv"
        )
    if not multi_method_outputs["disagreement_sites"].is_empty():
        multi_method_outputs["disagreement_sites"].write_csv(
            OUTPUT_DIR / "five_method_disagreement_sites.csv"
        )
    if not multi_method_outputs["threshold_summary"].is_empty():
        multi_method_outputs["threshold_summary"].write_csv(
            OUTPUT_DIR / "five_method_threshold_summary.csv"
        )
    if not multi_method_outputs["site_comparison"].is_empty():
        site_comparison_all = multi_method_outputs["site_comparison"]
        site_comparison_all.write_csv(
            FIVE_METHOD_ALL_DIR / "all_sites_five_method_comparison.csv"
        )
        site_comparison_all.filter(
            pl.col("any_disagreement") == False
        ).write_csv(
            FIVE_METHOD_SAME_DIR / "sites_same_behavior_all_methods.csv"
        )
        site_comparison_all.filter(
            pl.col("all_methods_unassessed") == True
        ).write_csv(
            FIVE_METHOD_SAME_DIR / "sites_unassessed_under_all_methods.csv"
        )
        site_comparison_all.filter(
            pl.col("any_disagreement") == True
        ).write_csv(
            FIVE_METHOD_DIFF_DIR / "sites_different_behavior_any_method.csv"
        )
        site_comparison_all.filter(
            pl.col("assessed_outcome_disagreement") == True
        ).write_csv(
            FIVE_METHOD_DIFF_DIR / "sites_different_assessed_outcomes.csv"
        )

    print("Saved outputs to", OUTPUT_DIR)
    print("Skipped (site metadata rows != 1):", len(skipped_not_single_inverter))
    print("Skipped (>3 PV circuits):", len(skipped_more_than_3))
    print("Skipped (no pv_site_net circuits):", len(skipped_no_pv_site_net))
    print("Skipped (no day data):", len(skipped_no_day_data))
    print("Skipped (no eligible days):", len(skipped_no_eligible_days))


if __name__ == "__main__":
    main()
