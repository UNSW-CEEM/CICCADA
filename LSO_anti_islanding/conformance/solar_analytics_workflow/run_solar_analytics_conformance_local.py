"""Run the complete local Solar Analytics conformance workflow."""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

CONFORMANCE_DIR = Path(__file__).resolve().parents[1]
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

from core.phase_a import SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA, run_phase_a_for_site
from core.phase_b import evaluate_compliance_for_day, run_phase_b_for_site
from core.site_day_signals import build_site_day_signals
from solar_analytics_workflow.config import (
    CONSIDER_LOWEST_THRESHOLD_AT_DISCONNECT,
    DAY_ANALYSIS_START,
    DAY_COVERAGE_THRESHOLD,
    DAY_END,
    DAY_EXTRACTION_START,
    GENERATE_SITE_PLOTS,
    MAX_PV_SITE_NET_CIRCUITS,
    PLOT_NO_RESPONSIBLE_TIMESTAMP_DAYS,
    PRIMARY_PHASE_B_METHOD,
    SAVE_SITE_LEVEL_VARIOUS_VOLTAGES,
)
from solar_analytics_workflow.loading import (
    load_solar_analytics_circuit_details,
    load_solar_analytics_cleaned_data,
    load_solar_analytics_site_details,
)
from solar_analytics_workflow.plotting import plot_site_compliance_day
from solar_analytics_workflow.reporting import (
    CONFORMANCE_EXCLUSIONS_NAME,
    SITE_COMPLIANCE_NAME,
    build_sola_conformance_exclusions,
    build_sola_site_compliance,
    write_sola_threshold_distribution_plots,
)
from solar_analytics_workflow.site_day_filtering import (
    summarize_solar_analytics_day_eligibility,
)
from solar_analytics_workflow.site_preparation import (
    calculate_site_day_voltage_signals,
    extract_site_day,
    map_circuit_data_to_site,
    select_site_pv_data,
    trim_site_day_analysis_window,
)
from solar_analytics_workflow.solar_paths import (
    CLEANED_DATA_PATH,
    CONFORMANCE_OUTPUT_DIR,
)

if not CLEANED_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned Solar Analytics data not found: {CLEANED_DATA_PATH}\n"
        "Run solar_analytics_workflow/"
        "run_solar_analytics_preprocessing.py first."
    )

site_details = load_solar_analytics_site_details()
circuit_details = load_solar_analytics_circuit_details()
all_data = load_solar_analytics_cleaned_data()

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

# capacity is a little bit different here than SAPN2022
# mainlly becasue Hossein has already calcualted that
# you should reconsider how you want to do/handle it future

# SolA assumes capacity has already been managed upstream
# if missing, it excludes the site

site_threshold_rows = []
site_level_various_voltage_rows = []
phase_a_records = []
site_compliance_rows = []
site_compliance_timestamp_detail_rows = []
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
        local_dates = (
            site_data.select(pl.col("local_tstamp").dt.date().alias("local_date"))
            .drop_nulls()
            .unique()
            .sort("local_date")["local_date"]
            .to_list()
        )
        for local_date in local_dates:
            site_day_long = extract_site_day(
                site_data,
                datetime.combine(local_date, DAY_EXTRACTION_START),
                datetime.combine(local_date, DAY_END),
            )
            if site_day_long.is_empty():
                continue

            mapped_day_count += 1
            prepared_day = calculate_site_day_voltage_signals(
                map_circuit_data_to_site(site_day_long, site_id),
                voltage_prefix="voltage_valid",
            )
            analysis_day = trim_site_day_analysis_window(
                prepared_day,
                DAY_ANALYSIS_START,
                DAY_END,
            )
            eligibility = summarize_solar_analytics_day_eligibility(
                analysis_day,
                coverage_threshold=DAY_COVERAGE_THRESHOLD,
            )
            if eligibility["eligible"]:
                eligible_analysis_days.append(
                    {
                        "analysis_date": local_date,
                        "analysis_frame": analysis_day,
                    }
                )
            else:
                excluded_day_rows.append(
                    {
                        "site_id": site_id,
                        "day": local_date,
                        "reason": eligibility["reason"],
                        "common_power_v10m_coverage_pct": eligibility[
                            "common_power_v10m_coverage_pct"
                        ],
                        "rows_common_power_v10m": eligibility["rows_common_power_v10m"],
                        "rows_with_power": eligibility["rows_with_power"],
                        "rows_with_v10m": eligibility["rows_with_v10m"],
                        "qualifying_timestamps": eligibility["qualifying_timestamps"],
                        "expected_timestamps": eligibility["expected_timestamps"],
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

    capacity_row = site_details.filter(pl.col("site_id") == site_id).select("s_rated")
    s_rated = None if capacity_row.is_empty() else capacity_row["s_rated"][0]
    if s_rated is None:
        skipped_sites["missing_rated_capacity"].append(site_id)
        continue

    prepared_site_days = [
        {
            "analysis_date": day_info["analysis_date"],
            "signal_frame": build_site_day_signals(day_info["analysis_frame"], s_rated),
        }
        for day_info in eligible_analysis_days
    ]

    phase_a = run_phase_a_for_site(site_id, prepared_site_days, s_rated)
    site_threshold_rows.append(phase_a["site_thresholds"])
    site_level_various_voltage_rows.append(phase_a["site_level_various_voltages"])
    if not phase_a["records"].is_empty():
        phase_a_records.append(phase_a["records"])

    phase_b = run_phase_b_for_site(
        site_id,
        prepared_site_days,
        site_thresholds=phase_a["site_thresholds"],
        threshold_method=PRIMARY_PHASE_B_METHOD,
        consider_lowest_threshold_at_disconnect=
            CONSIDER_LOWEST_THRESHOLD_AT_DISCONNECT,
    )
    site_compliance_rows.append(phase_b["site_compliance"])
    if not phase_b["site_compliance_timestamp_detail"].is_empty():
        site_compliance_timestamp_detail_rows.append(
            phase_b["site_compliance_timestamp_detail"]
        )

    compliance = phase_b["site_compliance"].to_dicts()[0]
    if GENERATE_SITE_PLOTS and compliance["overall_total_pass"] is not None:
        plot_folder = (
            "compliant"
            if compliance["overall_total_pass"] is True
            else "non_compliant"
        )
        for day_info in prepared_site_days:
            evaluated_day = evaluate_compliance_for_day(
                day_info["signal_frame"],
                los_threshold=compliance["los_threshold_used"],
                ov1_threshold=compliance["ov1_threshold_used"],
                los_lowest_disconnect_voltage=compliance[
                    "los_lowest_disconnect_voltage"
                ],
                ov1_lowest_disconnect_voltage=compliance[
                    "ov1_lowest_disconnect_voltage"
                ],
                consider_lowest_threshold_at_disconnect=
                    CONSIDER_LOWEST_THRESHOLD_AT_DISCONNECT,
            )
            plot_site_compliance_day(
                evaluated_day,
                site_id,
                day_info["analysis_date"],
                p_rated=s_rated,
                lso_threshold=compliance["los_threshold_used"],
                ov1_threshold=compliance["ov1_threshold_used"],
                los_lowest_disconnect_voltage=compliance[
                    "los_lowest_disconnect_voltage"
                ],
                ov1_lowest_disconnect_voltage=compliance[
                    "ov1_lowest_disconnect_voltage"
                ],
                consider_lowest_threshold_at_disconnect=
                    CONSIDER_LOWEST_THRESHOLD_AT_DISCONNECT,
                overall_pass=compliance["overall_total_pass"],
                plot_no_responsible_timestamp_days=(
                    PLOT_NO_RESPONSIBLE_TIMESTAMP_DAYS
                ),
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
        f"LOS={compliance['los_total_compliance_pct']} "
        f"OV1={compliance['ov1_total_compliance_pct']} "
        f"PASS={compliance['overall_total_pass']}"
    )

results = {
    "site_thresholds": (
        pl.concat(site_threshold_rows, how="vertical")
        if site_threshold_rows
        else pl.DataFrame()
    ),
    "site_level_various_voltages": (
        pl.concat(site_level_various_voltage_rows, how="vertical")
        if site_level_various_voltage_rows
        else pl.DataFrame(schema=SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA)
    ),
    "phase_a_trip_attribution": (
        pl.concat(phase_a_records, how="vertical")
        if phase_a_records
        else pl.DataFrame()
    ),
    "site_compliance": (
        pl.concat(site_compliance_rows, how="vertical")
        if site_compliance_rows
        else pl.DataFrame()
    ),
    "site_compliance_timestamp_detail": (
        pl.concat(site_compliance_timestamp_detail_rows, how="vertical")
        if site_compliance_timestamp_detail_rows
        else pl.DataFrame()
    ),
    "excluded_day_rows": excluded_day_rows,
    "skipped_sites": skipped_sites,
}

CONFORMANCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
site_compliance = build_sola_site_compliance(results)
conformance_exclusions = build_sola_conformance_exclusions(results)
site_compliance.write_csv(CONFORMANCE_OUTPUT_DIR / SITE_COMPLIANCE_NAME)
if SAVE_SITE_LEVEL_VARIOUS_VOLTAGES:
    results["site_level_various_voltages"].write_csv(
        CONFORMANCE_OUTPUT_DIR / "site_level_various_voltages.csv"
    )
conformance_exclusions.write_csv(CONFORMANCE_OUTPUT_DIR / CONFORMANCE_EXCLUSIONS_NAME)
write_sola_threshold_distribution_plots(
    results["phase_a_trip_attribution"],
    CONFORMANCE_OUTPUT_DIR / "threshold_distribution_plots",
)

print(f"Saved Solar Analytics outputs to {CONFORMANCE_OUTPUT_DIR}")
for reason, site_ids in skipped_sites.items():
    print(f"Skipped ({reason}): {len(site_ids)}")
