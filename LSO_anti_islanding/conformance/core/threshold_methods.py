"""Existing Phase B threshold-profile policies."""

import polars as pl


def _default_threshold_profile(*, tau=0.3, ov1_floor_offset=0.5):
    return _build_threshold_profile(
        los_anchor=258.0,
        los_anchor_p25=258.0,
        los_anchor_p10=258.0,
        los_anchor_min=258.0,
        ov1_anchor=265.0,
        ov1_basis="default",
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )


def _build_threshold_profile(
    *,
    los_anchor,
    los_anchor_p25,
    los_anchor_p10,
    los_anchor_min,
    ov1_anchor,
    ov1_basis,
    tau=0.3,
    ov1_floor_offset=0.5,
):
    delta_los = None if los_anchor == 258.0 else los_anchor - 258.0
    delta_los_p25 = None if los_anchor_p25 == 258.0 else los_anchor_p25 - 258.0
    delta_los_p10 = None if los_anchor_p10 == 258.0 else los_anchor_p10 - 258.0
    delta_los_min = None if los_anchor_min == 258.0 else los_anchor_min - 258.0
    delta_ov1 = None if ov1_anchor == 265.0 else ov1_anchor - 265.0
    return {
        "delta_los_site": delta_los,
        "delta_los_p25_site": delta_los_p25,
        "delta_los_p10_site": delta_los_p10,
        "delta_los_min_site": delta_los_min,
        "delta_ov1_site": delta_ov1,
        "los_anchor_site": float(los_anchor),
        "los_anchor_p25_site": float(los_anchor_p25),
        "los_anchor_p10_site": float(los_anchor_p10),
        "los_anchor_min_site": float(los_anchor_min),
        "ov1_anchor_site": float(ov1_anchor),
        "ov1_work_site": float(ov1_anchor),
        "ov1_floor_site": float(ov1_anchor) - ov1_floor_offset,
        "ov1_test_site": float(ov1_anchor) - tau,
        "ov1_basis": ov1_basis,
        "delta_gap_v": None
        if (delta_los is None or delta_ov1 is None)
        else abs(delta_ov1 - delta_los),
    }


def _raw_threshold_profile(raw_thresholds, *, tau=0.3, ov1_floor_offset=0.5):
    return _build_threshold_profile(
        los_anchor=raw_thresholds["los_anchor_site"],
        los_anchor_p25=raw_thresholds["los_anchor_p25_site"],
        los_anchor_p10=raw_thresholds["los_anchor_p10_site"],
        los_anchor_min=raw_thresholds["los_anchor_min_site"],
        ov1_anchor=raw_thresholds["ov1_anchor_site"],
        ov1_basis=raw_thresholds["ov1_basis"],
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )


def _profile_with_selection_metadata(profile, basis, score=None):
    return {
        **profile,
        "threshold_selection_basis": basis,
        "selection_score_rank": None if score is None else score[0],
        "selection_score_min_pct": None if score is None else score[1],
        "selection_score_mean_pct": None if score is None else score[2],
    }


def _thresholds_row_from_threshold_dict(
    site_id,
    thresholds,
    raw_thresholds,
    confidence_info,
    *,
    method_key=None,
):
    row = {
        "site_id": site_id,
        "delta_los_site": thresholds["delta_los_site"],
        "delta_los_p25_site": thresholds["delta_los_p25_site"],
        "delta_los_p10_site": thresholds["delta_los_p10_site"],
        "delta_los_min_site": thresholds["delta_los_min_site"],
        "delta_ov1_site": thresholds["delta_ov1_site"],
        "los_anchor_site": thresholds["los_anchor_site"],
        "los_anchor_p25_site": thresholds["los_anchor_p25_site"],
        "los_anchor_p10_site": thresholds["los_anchor_p10_site"],
        "los_anchor_min_site": thresholds["los_anchor_min_site"],
        "ov1_anchor_site": thresholds["ov1_anchor_site"],
        "ov1_work_site": thresholds["ov1_work_site"],
        "ov1_floor_site": thresholds["ov1_floor_site"],
        "ov1_test_site": thresholds["ov1_test_site"],
        "delta_gap_v": thresholds["delta_gap_v"],
        "delta_gap_flag": None
        if thresholds["delta_gap_v"] is None
        else thresholds["delta_gap_v"] > 2.0,
        "ov1_basis": thresholds["ov1_basis"],
        "ov1_event_count": raw_thresholds["ov1_event_count"],
        "ov1_reclassified_count": raw_thresholds["ov1_reclassified_count"],
        "los_removed_by_ov1_count": raw_thresholds["los_removed_by_ov1_count"],
        "raw_delta_los_site": raw_thresholds["delta_los_site"],
        "raw_delta_los_p25_site": raw_thresholds["delta_los_p25_site"],
        "raw_delta_los_p10_site": raw_thresholds["delta_los_p10_site"],
        "raw_delta_los_min_site": raw_thresholds["delta_los_min_site"],
        "raw_delta_ov1_site": raw_thresholds["delta_ov1_site"],
        "raw_los_anchor_site": raw_thresholds["los_anchor_site"],
        "raw_los_anchor_p25_site": raw_thresholds["los_anchor_p25_site"],
        "raw_los_anchor_p10_site": raw_thresholds["los_anchor_p10_site"],
        "raw_los_anchor_min_site": raw_thresholds["los_anchor_min_site"],
        "raw_ov1_anchor_site": raw_thresholds["ov1_anchor_site"],
        "raw_ov1_basis": raw_thresholds["ov1_basis"],
        "raw_delta_gap_v": raw_thresholds["delta_gap_v"],
        "threshold_confidence_tier": confidence_info["threshold_confidence_tier"],
        "confidence_primary_mech": confidence_info["confidence_primary_mech"],
        "confidence_event_count": confidence_info["confidence_event_count"],
        "confidence_drop20_count": confidence_info["confidence_drop20_count"],
        "confidence_drop10_count": confidence_info["confidence_drop10_count"],
        "confidence_spread_v": confidence_info["confidence_spread_v"],
        "threshold_selection_basis": thresholds["threshold_selection_basis"],
        "selection_score_rank": thresholds["selection_score_rank"],
        "selection_score_min_pct": thresholds["selection_score_min_pct"],
        "selection_score_mean_pct": thresholds["selection_score_mean_pct"],
    }
    if method_key is not None:
        row["method_key"] = method_key

    return pl.DataFrame([row]).with_columns(
        [
            pl.col("site_id").cast(pl.Int64),
            pl.col("delta_los_site").cast(pl.Float64),
            pl.col("delta_los_p25_site").cast(pl.Float64),
            pl.col("delta_los_p10_site").cast(pl.Float64),
            pl.col("delta_los_min_site").cast(pl.Float64),
            pl.col("delta_ov1_site").cast(pl.Float64),
            pl.col("los_anchor_site").cast(pl.Float64),
            pl.col("los_anchor_p25_site").cast(pl.Float64),
            pl.col("los_anchor_p10_site").cast(pl.Float64),
            pl.col("los_anchor_min_site").cast(pl.Float64),
            pl.col("ov1_anchor_site").cast(pl.Float64),
            pl.col("ov1_work_site").cast(pl.Float64),
            pl.col("ov1_floor_site").cast(pl.Float64),
            pl.col("ov1_test_site").cast(pl.Float64),
            pl.col("delta_gap_v").cast(pl.Float64),
            pl.col("delta_gap_flag").cast(pl.Boolean),
            pl.col("ov1_basis").cast(pl.Utf8),
            pl.col("ov1_event_count").cast(pl.Int64),
            pl.col("ov1_reclassified_count").cast(pl.Int64),
            pl.col("los_removed_by_ov1_count").cast(pl.Int64),
            pl.col("raw_delta_los_site").cast(pl.Float64),
            pl.col("raw_delta_los_p25_site").cast(pl.Float64),
            pl.col("raw_delta_los_p10_site").cast(pl.Float64),
            pl.col("raw_delta_los_min_site").cast(pl.Float64),
            pl.col("raw_delta_ov1_site").cast(pl.Float64),
            pl.col("raw_los_anchor_site").cast(pl.Float64),
            pl.col("raw_los_anchor_p25_site").cast(pl.Float64),
            pl.col("raw_los_anchor_p10_site").cast(pl.Float64),
            pl.col("raw_los_anchor_min_site").cast(pl.Float64),
            pl.col("raw_ov1_anchor_site").cast(pl.Float64),
            pl.col("raw_ov1_basis").cast(pl.Utf8),
            pl.col("raw_delta_gap_v").cast(pl.Float64),
            pl.col("threshold_confidence_tier").cast(pl.Utf8),
            pl.col("confidence_primary_mech").cast(pl.Utf8),
            pl.col("confidence_event_count").cast(pl.Int64),
            pl.col("confidence_drop20_count").cast(pl.Int64),
            pl.col("confidence_drop10_count").cast(pl.Int64),
            pl.col("confidence_spread_v").cast(pl.Float64),
            pl.col("threshold_selection_basis").cast(pl.Utf8),
            pl.col("selection_score_rank").cast(pl.Int64),
            pl.col("selection_score_min_pct").cast(pl.Float64),
            pl.col("selection_score_mean_pct").cast(pl.Float64),
            *([pl.col("method_key").cast(pl.Utf8)] if method_key is not None else []),
        ]
    )
