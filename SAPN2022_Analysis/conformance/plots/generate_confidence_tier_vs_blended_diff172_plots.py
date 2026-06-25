import csv
import shutil
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from funcs import ratedCapacityOfPV
from main import collect_site_days, prepare_inputs
from plots.plots import plot_three_method_threshold_overlay_day


OUTPUT_DIR = Path("All Results/site_compliance")
ROOT_DIR = (
    OUTPUT_DIR
    / "three_method_original_raw_confidence_tier_blended"
    / "confidence_tier_vs_blended_threshold_diff_172_sites"
)

BUCKETS = {
    "both_compliant": ROOT_DIR / "both_compliant",
    "both_non_compliant": ROOT_DIR / "both_non_compliant",
    "ct_non_compliant_blended_compliant": ROOT_DIR / "confidence_tier_non_compliant_blended_compliant",
    "ct_assessed_blended_unassessed": ROOT_DIR / "confidence_tier_assessed_blended_unassessed",
}

METHOD_SPECS = (
    ("confidence_tier", "Confidence tier", {"confidence_tier", "tier_based"}, "#00a6a6"),
    ("blended", "Blended", {"blended", "high_blended"}, "#6f4ef2"),
)

DAYS_TO_CHECK = [13, 14, 15, 16, 17, 19]


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


def _load_site_rows():
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
    for target_key, display_label, raw_keys, color in METHOD_SPECS:
        for raw_key in raw_keys:
            key_lookup[raw_key] = (target_key, display_label, color)

    site_rows: dict[int, dict[str, dict]] = {}
    for row in joined.iter_rows(named=True):
        mapped = key_lookup.get(row["method_key"])
        if mapped is None:
            continue
        target_key, display_label, color = mapped
        site_rows.setdefault(int(row["site_id"]), {})[target_key] = {
            **row,
            "target_key": target_key,
            "display_label": display_label,
            "color": color,
            "overall_pass": _bool_or_none(row["overall_pass"]),
            "eligible_timestamps": int(row.get("los_eligible") or 0) + int(row.get("ov1_eligible") or 0),
            "compliant_timestamps": int(row.get("los_compliant") or 0) + int(row.get("ov1_compliant") or 0),
        }
    return site_rows


def _different_threshold_sites(site_rows: dict[int, dict[str, dict]]) -> dict[int, dict[str, dict]]:
    selected = {}
    for site_id, method_rows in site_rows.items():
        ct = method_rows.get("confidence_tier")
        blended = method_rows.get("blended")
        if not ct or not blended:
            continue
        try:
            ct_los = float(ct["los_anchor_site"])
            blended_los = float(blended["los_anchor_site"])
        except Exception:
            continue
        if abs(ct_los - blended_los) > 1e-9:
            selected[site_id] = method_rows
    return selected


def _bucket_name(method_rows: dict[str, dict]) -> str | None:
    ct = method_rows["confidence_tier"]["overall_pass"]
    blended = method_rows["blended"]["overall_pass"]
    if ct is True and blended is True:
        return "both_compliant"
    if ct is False and blended is False:
        return "both_non_compliant"
    if ct is False and blended is True:
        return "ct_non_compliant_blended_compliant"
    if ct is not None and blended is None:
        return "ct_assessed_blended_unassessed"
    return None


def _write_bucket_csv(path: Path, site_ids: list[int]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["site_id"])
        for site_id in sorted(site_ids):
            writer.writerow([site_id])


def export_confidence_vs_blended_diff172():
    if ROOT_DIR.exists():
        shutil.rmtree(ROOT_DIR)
    for folder in BUCKETS.values():
        folder.mkdir(parents=True, exist_ok=True)

    site_rows = _different_threshold_sites(_load_site_rows())
    bucket_site_ids = {key: [] for key in BUCKETS}

    site_details, circuit_details, all_data = prepare_inputs()
    plot_counts = {key: 0 for key in BUCKETS}

    for site_id in sorted(site_rows):
        method_rows = site_rows[site_id]
        bucket = _bucket_name(method_rows)
        if bucket is None:
            continue
        bucket_site_ids[bucket].append(site_id)

        day_behaviours, _ = collect_site_days(site_id, circuit_details, all_data, DAYS_TO_CHECK)
        eligible_day_behaviours = [d for d in day_behaviours if d["eligibility"]["eligible"]]
        if not eligible_day_behaviours:
            continue

        p_rated = ratedCapacityOfPV(
            site_details,
            site_id,
            day_behaviours=eligible_day_behaviours,
        )

        for day_info in eligible_day_behaviours:
            day = day_info["day"]
            base_frame = None
            method_thresholds = []
            day_has_any_eligible = False

            for target_key, display_label, _, color in METHOD_SPECS:
                row = method_rows[target_key]
                los_threshold_used = float(row["los_threshold_used"])
                ov1_work_site = float(row["ov1_work_site"])
                day_plot = day_info["behaviour"].phase_b_day(
                    p_rated,
                    los_threshold=los_threshold_used,
                    ov1_work_threshold=ov1_work_site,
                )
                if base_frame is None:
                    base_frame = day_plot["frame"]

                total_day_eligible = (
                    int(day_plot["summary"].get("los_eligible", 0) or 0)
                    + int(day_plot["summary"].get("ov1_eligible", 0) or 0)
                )
                if total_day_eligible > 0:
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
                    "color": color,
                    "eligible_timestamps": row["eligible_timestamps"],
                    "compliant_timestamps": row["compliant_timestamps"],
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
                save_path=BUCKETS[bucket] / filename,
            )
            plot_counts[bucket] += 1

    all_rows = []
    for bucket, site_ids in bucket_site_ids.items():
        _write_bucket_csv(ROOT_DIR / f"{bucket}.csv", site_ids)
        for site_id in site_ids:
            method_rows = site_rows[site_id]
            all_rows.append({
                "site_id": site_id,
                "bucket": bucket,
                "confidence_tier_status": method_rows["confidence_tier"]["overall_pass"],
                "blended_status": method_rows["blended"]["overall_pass"],
                "confidence_tier_los": method_rows["confidence_tier"]["los_anchor_site"],
                "blended_los": method_rows["blended"]["los_anchor_site"],
                "confidence_tier_compliant_ts": method_rows["confidence_tier"]["compliant_timestamps"],
                "confidence_tier_eligible_ts": method_rows["confidence_tier"]["eligible_timestamps"],
                "blended_compliant_ts": method_rows["blended"]["compliant_timestamps"],
                "blended_eligible_ts": method_rows["blended"]["eligible_timestamps"],
            })

    if all_rows:
        with (ROOT_DIR / "site_bucket_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(all_rows, key=lambda row: int(row["site_id"])))

    print({
        "total_sites": len(site_rows),
        "bucket_counts": {bucket: len(ids) for bucket, ids in bucket_site_ids.items()},
        "plot_counts": plot_counts,
    })


if __name__ == "__main__":
    export_confidence_vs_blended_diff172()
