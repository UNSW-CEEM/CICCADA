"""Run the complete SAPN November 2022 conformance workflow."""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

CONFORMANCE_DIR = Path(__file__).resolve().parents[1]
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

from core.phase_a import run_phase_a_for_site
from core.phase_b import run_phase_b_for_day, run_phase_b_for_site
from core.site_day_signals import build_site_day_signals
from sapn2022_workflow.config import (
    DAY_ANALYSIS_START,
    DAY_COVERAGE_THRESHOLD,
    DAY_END,
    DAY_EXTRACTION_START,
    EVENT_DAYS,
    GENERATE_SITE_PLOTS,
    MAX_PV_SITE_NET_CIRCUITS,
    PLOT_NO_ELIGIBLE_TIMESTAMP_DAYS,
    PRIMARY_PHASE_B_METHOD,
)
from sapn2022_workflow.loading import (
    load_sapn_circuit_details,
    load_sapn_cleaned_data,
    load_sapn_site_details,
)
from sapn2022_workflow.plotting import plot_site_compliance_day
from sapn2022_workflow.reporting import (
    CONFORMANCE_EXCLUSIONS_NAME,
    SITE_CONFORMANCE_SUMMARY_NAME,
    build_sapn_conformance_exclusions,
    build_sapn_site_conformance_summary,
    write_method_conformance_final_table,
    write_sapn_threshold_distribution_plots,
)
from sapn2022_workflow.sapn_paths import (
    CAPACITY_DERIVED_PATH,
    CLEANED_SITE_DATA_PATH,
    CONFORMANCE_OUTPUT_DIR,
)
from sapn2022_workflow.site_day_filtering import summarize_nov2022_day_eligibility
from sapn2022_workflow.site_preparation import (
    calculate_site_day_voltage_signals,
    extract_site_day,
    map_circuit_data_to_site,
    select_site_pv_data,
    trim_site_day_analysis_window,
)

if not CLEANED_SITE_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned SAPN data not found: {CLEANED_SITE_DATA_PATH}\n"
        "Run sapn2022_workflow/run_sapn2022_preprocessing.py first."
    )
if not CAPACITY_DERIVED_PATH.exists():
    raise FileNotFoundError(
        f"SAPN capacity CSV not found: {CAPACITY_DERIVED_PATH}\n"
        "Run sapn2022_workflow/run_sapn2022_preprocessing.py first."
    )

site_details = load_sapn_site_details()
circuit_details = load_sapn_circuit_details()
all_data = load_sapn_cleaned_data()

eligible_site_ids = set(
    site_details.get_column("site_id").drop_nulls().unique().to_list()
)
pv_site_net_counts = {
    row["site_id"]: int(row["pv_site_net_count"])
    for row in (
        circuit_details.filter(pl.col("con_type") == "pv_site_net")
        .group_by("site_id")
        .len()
        .rename({"len": "pv_site_net_count"})
        .to_dicts()
    )
}
candidate_site_ids = (
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

print(f"Loading SAPN capacity CSV from {CAPACITY_DERIVED_PATH}.", flush=True)
capacity_derived = pl.read_csv(CAPACITY_DERIVED_PATH).select(
    [
        "site_id",
        "metadata_ac_capacity_kw",
        "calculated_ac_capacity_kw",
        "chosen_ac_capacity_kw",
    ]
)
site_details = site_details.join(capacity_derived, on="site_id", how="left")

threshold_rows = []
phase_a_records = []
bracket_rows = []
phase_b_summary_rows = []
phase_b_detail_rows = []
excluded_day_rows = []
skipped_sites = {
    "not_single_inverter": [],
    "more_than_3_pv_circuits": [],
    "no_pv_site_net": [],
    "no_day_data": [],
    "no_eligible_days": [],
    "missing_rated_capacity": [],
}

for site_index, site_id in enumerate(candidate_site_ids, start=1):
    pv_site_net_count = pv_site_net_counts.get(site_id, 0)
    if site_id not in eligible_site_ids:
        skipped_sites["not_single_inverter"].append(site_id)
        continue
    if pv_site_net_count == 0:
        skipped_sites["no_pv_site_net"].append(site_id)
        continue
    if pv_site_net_count > MAX_PV_SITE_NET_CIRCUITS:
        skipped_sites["more_than_3_pv_circuits"].append(site_id)
        continue

    site_data = select_site_pv_data(all_data, circuit_details, site_id)
    eligible_analysis_days = []
    mapped_day_count = 0
    if not site_data.is_empty():
        timestamp_dtype = site_data.schema["local_tstamp"]
        timezone_name = (
            timestamp_dtype.time_zone
            if isinstance(timestamp_dtype, pl.Datetime)
            else None
        )
        timezone = ZoneInfo(timezone_name) if timezone_name else None
        for day in EVENT_DAYS:
            site_day_long = extract_site_day(
                site_data,
                datetime(
                    2022,
                    11,
                    day,
                    DAY_EXTRACTION_START.hour,
                    DAY_EXTRACTION_START.minute,
                    DAY_EXTRACTION_START.second,
                    tzinfo=timezone,
                ),
                datetime(
                    2022,
                    11,
                    day,
                    DAY_END.hour,
                    DAY_END.minute,
                    DAY_END.second,
                    tzinfo=timezone,
                ),
            )
            if site_day_long.is_empty():
                continue

            mapped_day_count += 1
            prepared_day = calculate_site_day_voltage_signals(
                map_circuit_data_to_site(site_day_long, site_id),
                voltage_prefix="voltage_valid",
            )
            # careful that this fucntion is implemeted twice but for different purposes
            # do not delete it
            analysis_day_long = trim_site_day_analysis_window(
                site_day_long,
                DAY_ANALYSIS_START,
                DAY_END,
            )
            analysis_day = trim_site_day_analysis_window(
                prepared_day,
                DAY_ANALYSIS_START,
                DAY_END,
            )
            eligibility = summarize_nov2022_day_eligibility(
                analysis_day_long,
                analysis_day,
                coverage_threshold=DAY_COVERAGE_THRESHOLD,
            )
            if eligibility["eligible"]:
                eligible_analysis_days.append(
                    {
                        "analysis_date": day,
                        "analysis_frame": analysis_day,
                    }
                )
            else:
                excluded_day_rows.append(
                    {
                        "site_id": site_id,
                        "day": day,
                        "reason": eligibility["reason"],
                        "common_power_v10m_coverage_pct": eligibility[
                            "common_power_v10m_coverage_pct"
                        ],
                        "rows_common_power_v10m": eligibility["rows_common_power_v10m"],
                        "rows_with_power": eligibility["rows_with_power"],
                        "rows_with_v10m": eligibility["rows_with_v10m"],
                        "covered_seconds": eligibility["covered_seconds"],
                        "window_seconds": eligibility["window_seconds"],
                        "total_rows": eligibility["total_rows"],
                        "coverage_threshold_pct": DAY_COVERAGE_THRESHOLD * 100.0,
                    }
                )

    if mapped_day_count == 0:
        skipped_sites["no_day_data"].append(site_id)
        continue
    if not eligible_analysis_days:
        skipped_sites["no_eligible_days"].append(site_id)
        continue

    capacity_row = site_details.filter(pl.col("site_id") == site_id).select(
        "chosen_ac_capacity_kw"
    )
    rated_capacity = (
        None if capacity_row.is_empty() else capacity_row["chosen_ac_capacity_kw"][0]
    )
    if rated_capacity is None:
        skipped_sites["missing_rated_capacity"].append(site_id)
        continue

    prepared_site_days = [
        {
            "analysis_date": day_info["analysis_date"],
            "signal_frame": build_site_day_signals(
                day_info["analysis_frame"], rated_capacity
            ),
        }
        for day_info in eligible_analysis_days
    ]

    phase_a = run_phase_a_for_site(site_id, prepared_site_days, rated_capacity)
    if not phase_a["records"].is_empty():
        phase_a_records.append(phase_a["records"])
    if not phase_a["brackets"].is_empty():
        bracket_rows.append(phase_a["brackets"])

    phase_b = run_phase_b_for_site(
        site_id,
        prepared_site_days,
        raw_thresholds=phase_a["raw_thresholds"],
        confidence_info=phase_a["confidence_info"],
        phase_b_method=PRIMARY_PHASE_B_METHOD,
    )
    threshold_rows.append(phase_b["threshold_row"])
    phase_b_summary_rows.append(phase_b["summary_row"])
    if not phase_b["detail"].is_empty():
        phase_b_detail_rows.append(phase_b["detail"])

    summary = phase_b["summary_row"].to_dicts()[0]
    thresholds = phase_b["threshold_row"].to_dicts()[0]
    if GENERATE_SITE_PLOTS and summary["overall_pass"] is not None:
        plot_folder = (
            "compliant" if summary["overall_pass"] is True else "non_compliant"
        )
        for day_info in prepared_site_days:
            day_plot = run_phase_b_for_day(
                day_info["signal_frame"],
                los_threshold=summary["los_threshold_used"],
                ov1_work_threshold=thresholds["ov1_work_site"],
            )
            plot_site_compliance_day(
                day_plot["frame"],
                site_id,
                day_info["analysis_date"],
                p_rated=rated_capacity,
                lso_threshold=summary["los_threshold_used"],
                ov1_threshold=thresholds["ov1_test_site"],
                overall_pass=summary["overall_pass"],
                day_summary=day_plot["summary"],
                plot_no_eligible_timestamp_days=PLOT_NO_ELIGIBLE_TIMESTAMP_DAYS,
                save_path=(
                    CONFORMANCE_OUTPUT_DIR
                    / "overall_site_plots"
                    / plot_folder
                    / f"Site_{site_id}_Day_"
                    f"{day_info['analysis_date']}_{plot_folder}.png"
                ),
            )

    print(
        f"[{site_index}/{len(candidate_site_ids)}] site {site_id} "
        f"LOS={summary['los_compliance_pct']} "
        f"OV1={summary['ov1_compliance_pct']} PASS={summary['overall_pass']}"
    )

results = {
    "site_thresholds": (
        pl.concat(threshold_rows, how="vertical") if threshold_rows else pl.DataFrame()
    ),
    "phase_a_trip_attribution": (
        pl.concat(phase_a_records, how="vertical")
        if phase_a_records
        else pl.DataFrame()
    ),
    "phase_a_brackets": (
        pl.concat(bracket_rows, how="vertical") if bracket_rows else pl.DataFrame()
    ),
    "phase_b_site_summary": (
        pl.concat(phase_b_summary_rows, how="vertical")
        if phase_b_summary_rows
        else pl.DataFrame()
    ),
    "phase_b_timestamp_detail": (
        pl.concat(phase_b_detail_rows, how="vertical")
        if phase_b_detail_rows
        else pl.DataFrame()
    ),
    "excluded_day_rows": excluded_day_rows,
    "skipped_sites": skipped_sites,
}

CONFORMANCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
site_conformance_summary = build_sapn_site_conformance_summary(results)
conformance_exclusions = build_sapn_conformance_exclusions(results)
site_conformance_summary.write_csv(
    CONFORMANCE_OUTPUT_DIR / SITE_CONFORMANCE_SUMMARY_NAME
)
write_method_conformance_final_table(site_conformance_summary)
conformance_exclusions.write_csv(CONFORMANCE_OUTPUT_DIR / CONFORMANCE_EXCLUSIONS_NAME)
write_sapn_threshold_distribution_plots(
    results["phase_a_trip_attribution"],
    CONFORMANCE_OUTPUT_DIR / "threshold_distribution_plots",
)

print(f"Saved SAPN outputs to {CONFORMANCE_OUTPUT_DIR}")
for reason, site_ids in skipped_sites.items():
    print(f"Skipped ({reason}): {len(site_ids)}")
