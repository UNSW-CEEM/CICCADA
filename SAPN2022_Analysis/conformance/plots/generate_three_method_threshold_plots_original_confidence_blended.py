import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from checkPVBehaviour import CheckPVBehaviour
from funcs import (
    loadCleanedSiteData,
    mapCircuitDataToSite,
    ratedCapacityOfPV,
)
from plots.plots import plot_three_method_threshold_overlay_day


OUTPUT_DIR = Path("All Results/site_compliance")
THREE_METHOD_DIR = OUTPUT_DIR / "three_method_original_raw_confidence_tier_blended"
SAME_DIR = THREE_METHOD_DIR / "same"
DIFF_DIR = THREE_METHOD_DIR / "different"
DAYS_TO_CHECK = [13, 14, 15, 16, 17, 19]
DAY_COVERAGE_THRESHOLD = 0.80

METHOD_SPECS = (
    ("original_raw", "Original raw", {"original_raw", "original"}),
    ("confidence_tier", "Confidence tier", {"confidence_tier", "tier_based"}),
    ("blended", "Blended", {"high_blended", "blended"}),
)

METHOD_COLOR_MAP = {
    "original_raw": "#f58518",
    "confidence_tier": "#00a6a6",
    "blended": "#6f4ef2",
}


def _bool_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def prepare_inputs():
    site_details = pl.read_csv("Nov2022/ebm_1_20221112_20221119_site_details.csv")
    circuit_details = pl.read_csv("Nov2022/ebm_1_20221112_20221119_circuit_details.csv")

    all_data = loadCleanedSiteData()

    return site_details, circuit_details, all_data


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


def load_three_method_rows():
    summary_df = pl.read_csv(
        OUTPUT_DIR / "phase_b_site_summary_by_method.csv",
        try_parse_dates=True,
    )
    thresholds_df = pl.read_csv(OUTPUT_DIR / "site_thresholds_by_method.csv")
    joined = summary_df.join(
        thresholds_df,
        on=["site_id", "method_key", "method_label"],
        how="left",
        suffix="_threshold",
    )

    key_lookup = {}
    for target_key, display_label, raw_keys in METHOD_SPECS:
        for raw_key in raw_keys:
            key_lookup[raw_key] = (target_key, display_label)

    site_rows: dict[int, dict[str, dict]] = {}
    for row in joined.iter_rows(named=True):
        mapped = key_lookup.get(row["method_key"])
        if mapped is None:
            continue
        target_key, display_label = mapped
        overall_pass = _bool_or_none(row["overall_pass"])
        row = {
            **row,
            "target_key": target_key,
            "display_label": display_label,
            "overall_pass": overall_pass,
        }
        site_rows.setdefault(int(row["site_id"]), {})[target_key] = row
    return site_rows


def classify_site_group(method_rows: dict[str, dict]) -> str | None:
    assessed = []
    for target_key, _, _ in METHOD_SPECS:
        row = method_rows.get(target_key)
        if not row:
            continue
        if row["overall_pass"] is not None:
            assessed.append(bool(row["overall_pass"]))

    if len(assessed) < 2:
        return None
    if len(set(assessed)) == 1:
        return "same"
    return "different"


def build_site_group_tables(site_rows: dict[int, dict[str, dict]]):
    same_rows = []
    diff_rows = []
    skipped_rows = []

    for site_id in sorted(site_rows):
        method_rows = site_rows[site_id]
        group = classify_site_group(method_rows)

        flat = {"site_id": site_id}
        assessed_count = 0
        for target_key, _, _ in METHOD_SPECS:
            row = method_rows.get(target_key)
            result = None if row is None else row["overall_pass"]
            if result is True:
                flat[f"{target_key}_overall_pass"] = "true"
                assessed_count += 1
            elif result is False:
                flat[f"{target_key}_overall_pass"] = "false"
                assessed_count += 1
            else:
                flat[f"{target_key}_overall_pass"] = None

            if row is None:
                flat[f"{target_key}_los_threshold_used"] = None
            else:
                flat[f"{target_key}_los_threshold_used"] = row["los_threshold_used"]

        flat["assessed_method_count"] = assessed_count

        if group == "same":
            same_rows.append(flat)
        elif group == "different":
            diff_rows.append(flat)
        else:
            skipped_rows.append(flat)

    return (
        pl.DataFrame(same_rows) if same_rows else pl.DataFrame(),
        pl.DataFrame(diff_rows) if diff_rows else pl.DataFrame(),
        pl.DataFrame(skipped_rows) if skipped_rows else pl.DataFrame(),
    )


def export_three_method_plots():
    SAME_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)

    site_rows = load_three_method_rows()
    same_df, diff_df, skipped_df = build_site_group_tables(site_rows)
    same_df.write_csv(THREE_METHOD_DIR / "sites_same.csv")
    diff_df.write_csv(THREE_METHOD_DIR / "sites_different.csv")
    skipped_df.write_csv(THREE_METHOD_DIR / "sites_less_than_two_assessed.csv")

    site_details, circuit_details, all_data = prepare_inputs()

    plot_counts = {"same": 0, "different": 0}
    site_counts = {"same": 0, "different": 0}

    for site_id in sorted(site_rows):
        method_rows = site_rows[site_id]
        group = classify_site_group(method_rows)
        if group is None:
            continue

        day_behaviours, _ = collect_site_days(site_id, circuit_details, all_data, DAYS_TO_CHECK)
        eligible_day_behaviours = [d for d in day_behaviours if d["eligibility"]["eligible"]]
        if not eligible_day_behaviours:
            continue

        p_rated = ratedCapacityOfPV(
            site_details,
            site_id,
            day_behaviours=eligible_day_behaviours,
        )

        root = SAME_DIR if group == "same" else DIFF_DIR
        site_counts[group] += 1

        for day_info in eligible_day_behaviours:
            day = day_info["day"]
            base_frame = None
            method_thresholds = []
            day_has_any_eligible = False

            for target_key, display_label, _ in METHOD_SPECS:
                row = method_rows.get(target_key)
                if row is None or row["overall_pass"] is None:
                    continue

                los_threshold_used = float(row["los_threshold_used"])
                ov1_work_site = float(row["ov1_work_site"])
                day_plot = day_info["behaviour"].phase_b_day(
                    p_rated,
                    los_threshold=los_threshold_used,
                    ov1_work_threshold=ov1_work_site,
                )
                if base_frame is None:
                    base_frame = day_plot["frame"]

                total_eligible = (
                    int(day_plot["summary"].get("los_eligible", 0) or 0)
                    + int(day_plot["summary"].get("ov1_eligible", 0) or 0)
                )
                if total_eligible > 0:
                    day_has_any_eligible = True

                if row["overall_pass"] is True:
                    status = "compliant"
                elif row["overall_pass"] is False:
                    status = "non-compliant"
                else:
                    status = "unassessed"

                method_thresholds.append({
                    "label": display_label,
                    "los_threshold": los_threshold_used,
                    "status": status,
                    "color": METHOD_COLOR_MAP[target_key],
                })

            if not day_has_any_eligible or base_frame is None:
                continue

            filename = f"Site_{site_id}_Day_{day}.png"
            plot_three_method_threshold_overlay_day(
                base_frame,
                site_id,
                day,
                p_rated=p_rated,
                method_thresholds=method_thresholds,
                save_path=root / filename,
            )
            plot_counts[group] += 1

    print(
        "Finished three-method threshold plots:",
        {
            "same_sites": site_counts["same"],
            "different_sites": site_counts["different"],
            "same_plots": plot_counts["same"],
            "different_plots": plot_counts["different"],
        },
    )


if __name__ == "__main__":
    export_three_method_plots()
