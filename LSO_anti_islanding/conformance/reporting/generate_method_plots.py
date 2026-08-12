"""Generate the optional method-comparison plots."""

from pathlib import Path

import polars as pl
from config import METHOD_COMPARISON_METHODS
from reporting.plotting import plot_method_threshold_overlay_day

METHOD_COLOR_MAP = {
    "default": "#2ca02c",
    "original": "#f58518",
    "tier_based": "#00a6a6",
    "old_sweep": "#c62828",
    "blended": "#6f4ef2",
}

METHOD_EVENT_SHADE_MAP = {
    "tier_based": "#7c3aed",
    "blended": "#2563eb",
}

METHOD_EVENT_ALPHA_MAP = {
    "tier_based": 0.22,
    "blended": 0.18,
}

METHOD_OVERLAP_STYLE_MAP = {
    "tier_based_only": {"color": "#7c3aed", "alpha": 0.22},
    "both_methods": {"color": "#ff7a00", "alpha": 0.30},
    "blended_only": {"color": "#2563eb", "alpha": 0.22},
}

COMPARISON_BUCKETS = (
    "different_result",
    "assessed_vs_unassessed",
    "same_result",
)


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


def _status_label(value):
    if value is True:
        return "conformant"
    if value is False:
        return "non-conformant"
    return "unassessed"


def _comparison_methods():
    methods = tuple(METHOD_COMPARISON_METHODS)
    if len(methods) < 2:
        raise ValueError("METHOD_COMPARISON_METHODS must contain at least two methods.")
    if len(set(methods)) != len(methods):
        raise ValueError("METHOD_COMPARISON_METHODS must not contain duplicates.")
    return methods


def _load_method_rows(output_dir, methods):
    summary_path = output_dir / "phase_b_site_summary_by_method.csv"
    threshold_path = output_dir / "site_thresholds_by_method.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run main_run_conformance.main() first."
        )
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Missing {threshold_path}. Run main_run_conformance.main() first."
        )

    summary_df = pl.read_csv(summary_path, try_parse_dates=True).with_columns(
        pl.col("overall_pass").map_elements(_bool_or_none, return_dtype=pl.Boolean)
    )
    threshold_df = pl.read_csv(threshold_path)
    joined = summary_df.join(
        threshold_df,
        on=["site_id", "method_key"],
        how="left",
        suffix="_threshold",
    )

    available_methods = sorted(joined["method_key"].unique().to_list())
    missing_methods = [
        method_key for method_key in methods if method_key not in available_methods
    ]
    if missing_methods:
        raise ValueError(
            "Missing by-method outputs for "
            f"{missing_methods}. Enable them in config.py and rerun conformance."
        )

    site_rows: dict[int, dict[str, dict]] = {}
    for row in joined.iter_rows(named=True):
        method_key = row["method_key"]
        if method_key not in methods:
            continue
        row["method_label"] = method_key.replace("_", " ").title()
        site_rows.setdefault(int(row["site_id"]), {})[method_key] = row
    return site_rows


def _comparison_bucket(method_rows, methods):
    # These buckets are mutually exclusive so the same site does not get plotted
    # repeatedly in multiple folders.
    results = []
    for method_key in methods:
        row = method_rows.get(method_key)
        results.append(None if row is None else row["overall_pass"])

    assessed_results = [bool(value) for value in results if value is not None]
    if not assessed_results:
        return None
    if len(set(assessed_results)) > 1:
        return "different_result"
    if any(value is None for value in results):
        return "assessed_vs_unassessed"
    return "same_result"


def _prepared_site_for_plots(site_id, prepare_site):
    prepared_site = prepare_site(site_id)
    if prepared_site.get("skip_reason") is not None:
        return None
    return prepared_site


def _build_method_threshold_plot(day_info, p_rated, method_row):
    los_threshold_used = method_row["los_threshold_used"]
    ov1_work_site = method_row["ov1_work_site"]
    if los_threshold_used is None or ov1_work_site is None:
        return None
    return day_info["behaviour"].phase_b_day(
        p_rated,
        los_threshold=float(los_threshold_used),
        ov1_work_threshold=float(ov1_work_site),
    )


def _event_lookup_for_frame(frame: pl.DataFrame):
    if not {"los_responsible", "ov1_responsible"}.issubset(set(frame.columns)):
        return {}
    event_flags = (
        frame["los_responsible"].fill_null(False).cast(pl.Boolean)
        | frame["ov1_responsible"].fill_null(False).cast(pl.Boolean)
    ).to_list()
    return {
        ts: bool(active)
        for ts, active in zip(frame["local_tstamp"].to_list(), event_flags)
    }


def _build_method_event_overlays(base_timestamps, method_event_lookups):
    if len(method_event_lookups) != 2:
        overlays = []
        for overlay_info in method_event_lookups:
            overlays.append(
                {
                    "key": overlay_info["key"],
                    "label": overlay_info["label"],
                    "color": overlay_info["color"],
                    "alpha": overlay_info["alpha"],
                    "event_mask": [
                        bool(overlay_info["event_lookup"].get(ts, False))
                        for ts in base_timestamps
                    ],
                }
            )
        return overlays

    overlay_by_key = {overlay["key"]: overlay for overlay in method_event_lookups}
    if set(overlay_by_key) != {"tier_based", "blended"}:
        overlays = []
        for overlay_info in method_event_lookups:
            overlays.append(
                {
                    "key": overlay_info["key"],
                    "label": overlay_info["label"],
                    "color": overlay_info["color"],
                    "alpha": overlay_info["alpha"],
                    "event_mask": [
                        bool(overlay_info["event_lookup"].get(ts, False))
                        for ts in base_timestamps
                    ],
                }
            )
        return overlays

    tier_info = overlay_by_key["tier_based"]
    blended_info = overlay_by_key["blended"]
    tier_mask = [
        bool(tier_info["event_lookup"].get(ts, False)) for ts in base_timestamps
    ]
    blended_mask = [
        bool(blended_info["event_lookup"].get(ts, False)) for ts in base_timestamps
    ]

    tier_only_mask = [
        tier and not blended for tier, blended in zip(tier_mask, blended_mask)
    ]
    overlap_mask = [tier and blended for tier, blended in zip(tier_mask, blended_mask)]
    blended_only_mask = [
        blended and not tier for tier, blended in zip(tier_mask, blended_mask)
    ]

    overlays = []
    if any(tier_only_mask):
        overlays.append(
            {
                "key": "tier_based_only",
                "label": "Tier based only",
                "color": METHOD_OVERLAP_STYLE_MAP["tier_based_only"]["color"],
                "alpha": METHOD_OVERLAP_STYLE_MAP["tier_based_only"]["alpha"],
                "event_mask": tier_only_mask,
            }
        )
    if any(overlap_mask):
        overlays.append(
            {
                "key": "both_methods",
                "label": "Both methods",
                "color": METHOD_OVERLAP_STYLE_MAP["both_methods"]["color"],
                "alpha": METHOD_OVERLAP_STYLE_MAP["both_methods"]["alpha"],
                "event_mask": overlap_mask,
            }
        )
    if any(blended_only_mask):
        overlays.append(
            {
                "key": "blended_only",
                "label": "Blended only",
                "color": METHOD_OVERLAP_STYLE_MAP["blended_only"]["color"],
                "alpha": METHOD_OVERLAP_STYLE_MAP["blended_only"]["alpha"],
                "event_mask": blended_only_mask,
            }
        )
    return overlays


def _build_comparison_day_payload(day_info, p_rated, method_rows, methods):
    base_frame = None
    method_thresholds = []
    day_has_any_eligible = False
    event_lookup = {}
    method_event_lookups = []

    for method_key in methods:
        method_row = method_rows.get(method_key)
        if method_row is None:
            continue

        day_plot = _build_method_threshold_plot(day_info, p_rated, method_row)
        if day_plot is None:
            continue
        if base_frame is None:
            base_frame = day_plot["frame"]

        method_event_lookup = _event_lookup_for_frame(day_plot["frame"])
        for ts, active in method_event_lookup.items():
            event_lookup[ts] = event_lookup.get(ts, False) or active

        total_day_eligible = int(day_plot["summary"].get("los_eligible", 0) or 0) + int(
            day_plot["summary"].get("ov1_eligible", 0) or 0
        )
        if total_day_eligible > 0:
            day_has_any_eligible = True

        method_thresholds.append(
            {
                "label": method_row["method_label"],
                "lso_threshold": float(method_row["los_threshold_used"]),
                "status": _status_label(method_row["overall_pass"]),
                "color": METHOD_COLOR_MAP.get(method_key, "#7f7f7f"),
                "day_eligible_timestamps": total_day_eligible,
                "day_compliant_timestamps": (
                    int(day_plot["summary"].get("los_compliant", 0) or 0)
                    + int(day_plot["summary"].get("ov1_compliant", 0) or 0)
                ),
            }
        )
        method_event_lookups.append(
            {
                "key": method_key,
                "label": method_row["method_label"],
                "color": METHOD_EVENT_SHADE_MAP.get(
                    method_key,
                    METHOD_COLOR_MAP.get(method_key, "#7f7f7f"),
                ),
                "alpha": METHOD_EVENT_ALPHA_MAP.get(method_key, 0.12),
                "event_lookup": method_event_lookup,
            }
        )

    if not day_has_any_eligible or base_frame is None or not method_thresholds:
        return None

    base_timestamps = base_frame["local_tstamp"].to_list()
    comparison_event_mask = [
        bool(event_lookup.get(ts, False)) for ts in base_timestamps
    ]
    method_event_overlays = _build_method_event_overlays(
        base_timestamps, method_event_lookups
    )

    return {
        "base_frame": base_frame,
        "method_thresholds": method_thresholds,
        "method_event_overlays": method_event_overlays,
        "comparison_event_mask": comparison_event_mask,
    }


def _plot_method_comparison(
    site_rows,
    prepare_site,
    output_dir,
    methods,
    coverage_threshold,
):
    method_set_name = "__".join(methods)
    output_root = output_dir / "method_plots" / "comparison" / method_set_name
    bucket_counts = {bucket: 0 for bucket in COMPARISON_BUCKETS}
    site_counts = {bucket: 0 for bucket in COMPARISON_BUCKETS}

    for site_id in sorted(site_rows):
        method_rows = site_rows[site_id]

        # The comparison buckets are the main browsing paths for multi-method
        # reviews: different outcomes, assessed/unassessed splits, or same
        # outcomes across the selected methods.
        bucket = _comparison_bucket(method_rows, methods)
        if bucket is None:
            continue

        prepared_site = _prepared_site_for_plots(site_id, prepare_site)
        if prepared_site is None:
            continue
        eligible_day_behaviours = prepared_site["day_behaviours"]
        p_rated = prepared_site["p_rated"]
        site_counts[bucket] += 1

        for day_info in eligible_day_behaviours:
            day = day_info["day"]
            day_payload = _build_comparison_day_payload(
                day_info,
                p_rated,
                method_rows,
                methods,
            )
            if day_payload is None:
                continue

            filename = f"Site_{site_id}_Day_{day}.png"
            plot_method_threshold_overlay_day(
                day_payload["base_frame"],
                site_id,
                day,
                p_rated=p_rated,
                method_thresholds=day_payload["method_thresholds"],
                method_event_overlays=day_payload["method_event_overlays"],
                comparison_event_mask=day_payload["comparison_event_mask"],
                save_path=output_root / bucket / filename,
            )
            bucket_counts[bucket] += 1

    print(
        "Finished comparison plots:",
        {
            "methods": methods,
            "shared_day_coverage_threshold_pct": coverage_threshold * 100.0,
            "site_counts": site_counts,
            "plot_counts": bucket_counts,
            "output_root": str(output_root),
        },
    )


def generate_method_plots(
    output_dir,
    prepare_site,
    coverage_threshold,
):
    """Generate the method comparison configured in config.py."""
    output_dir = Path(output_dir)
    methods = _comparison_methods()

    site_rows = _load_method_rows(output_dir, methods)
    _plot_method_comparison(
        site_rows,
        prepare_site,
        output_dir,
        methods,
        coverage_threshold,
    )
