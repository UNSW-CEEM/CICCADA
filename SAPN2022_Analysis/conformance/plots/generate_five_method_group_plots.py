import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from funcs import ratedCapacityOfPV
from main import collect_site_days, prepare_inputs
from plots.plots import plot_site_compliance_day


OUTPUT_DIR = Path("updated results/site_compliance")
GROUP_DIR = OUTPUT_DIR / "five_method_site_groups"
SAME_PLOTS_DIR = GROUP_DIR / "same_behavior" / "method_plots"
DIFF_PLOTS_DIR = GROUP_DIR / "different_behavior" / "method_plots"
DAYS_TO_CHECK = [13, 14, 15, 16, 17, 19]

METHOD_KEY_MAP = {
    "default": "default",
    "original_raw": "original",
    "original": "original",
    "confidence_tier": "tier_based",
    "tier_based": "tier_based",
    "old_sweep": "old_sweep",
    "high_blended": "blended",
    "blended": "blended",
}

METHOD_LABEL_MAP = {
    "Default thresholds": "Default",
    "Original Phase A raw": "Original",
    "Current confidence-tier": "Tier based",
    "Old sweep method": "Old sweep",
    "High -> blended": "Blended",
    "default": "Default",
    "original_raw": "Original",
    "original": "Original",
    "confidence_tier": "Tier based",
    "tier_based": "Tier based",
    "old_sweep": "Old sweep",
    "high_blended": "Blended",
    "blended": "Blended",
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


def _load_site_groups():
    comparison = pl.read_csv(OUTPUT_DIR / "five_method_site_comparison.csv")
    return {
        int(row["site_id"]): (
            "different_behavior" if row["any_disagreement"] else "same_behavior"
        )
        for row in comparison.select(["site_id", "any_disagreement"]).iter_rows(named=True)
    }


def _load_method_runs():
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

    site_runs: dict[int, list[dict]] = {}
    for row in joined.iter_rows(named=True):
        overall_pass = _bool_or_none(row["overall_pass"])
        if overall_pass is None:
            continue
        site_runs.setdefault(int(row["site_id"]), []).append({
            **row,
            "method_key": METHOD_KEY_MAP.get(row["method_key"], row["method_key"]),
            "method_label": METHOD_LABEL_MAP.get(row["method_label"], row["method_label"]),
            "overall_pass": overall_pass,
        })
    return site_runs


def _plot_root_for_group(group_name: str) -> Path:
    if group_name == "different_behavior":
        return DIFF_PLOTS_DIR
    return SAME_PLOTS_DIR


def export_group_plots():
    for folder in [SAME_PLOTS_DIR, DIFF_PLOTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    site_groups = _load_site_groups()
    site_runs = _load_method_runs()
    site_details, circuit_details, all_data = prepare_inputs()

    total_sites = len(site_runs)
    plotted = 0
    for idx, site_id in enumerate(sorted(site_runs), start=1):
        day_behaviours, _ = collect_site_days(site_id, circuit_details, all_data, DAYS_TO_CHECK)
        eligible_day_behaviours = [d for d in day_behaviours if d["eligibility"]["eligible"]]
        if not eligible_day_behaviours:
            continue

        p_rated = ratedCapacityOfPV(
            site_details,
            site_id,
            day_behaviours=eligible_day_behaviours,
        )
        group_name = site_groups.get(site_id, "same_behavior")
        group_root = _plot_root_for_group(group_name) / f"Site_{site_id}"

        for method_row in site_runs[site_id]:
            method_key = method_row["method_key"]
            method_label = method_row["method_label"]
            plot_folder = "compliant" if method_row["overall_pass"] else "non_compliant"
            method_group_root = group_root / method_key / plot_folder

            los_threshold_used = float(method_row["los_threshold_used"])
            ov1_work_site = float(method_row["ov1_work_site"])

            for day_info in eligible_day_behaviours:
                day_plot = day_info["behaviour"].phase_b_day(
                    p_rated,
                    los_threshold=los_threshold_used,
                    ov1_work_threshold=ov1_work_site,
                )
                total_eligible = (
                    int(day_plot["summary"].get("los_eligible", 0) or 0)
                    + int(day_plot["summary"].get("ov1_eligible", 0) or 0)
                )
                if total_eligible == 0:
                    continue
                day = day_info["day"]
                filename = f"Site_{site_id}_Day_{day}_{method_key}_{plot_folder}.png"
                plot_site_compliance_day(
                    day_plot["frame"],
                    site_id,
                    day,
                    p_rated=p_rated,
                    los_threshold=los_threshold_used,
                    los_threshold_p25=float(method_row["los_anchor_p25_site"]),
                    los_threshold_p10=float(method_row["los_anchor_p10_site"]),
                    los_threshold_min=float(method_row["los_anchor_min_site"]),
                    ov1_threshold=float(method_row["ov1_test_site"]),
                    delta_los_site=method_row["delta_los_site"],
                    delta_los_p25_site=method_row["delta_los_p25_site"],
                    delta_los_p10_site=method_row["delta_los_p10_site"],
                    delta_los_min_site=method_row["delta_los_min_site"],
                    delta_ov1_site=method_row["delta_ov1_site"],
                    ov1_basis=method_row["ov1_basis"],
                    overall_pass=method_row["overall_pass"],
                    pass_basis=method_row["pass_basis"],
                    day_summary=day_plot["summary"],
                    method_label=method_label,
                    save_path=method_group_root / filename,
                )

            plotted += 1
        if idx % 25 == 0:
            print(f"[{idx}/{total_sites}] plotted site {site_id}")

    print("Finished grouped five-method plots")


if __name__ == "__main__":
    export_group_plots()
