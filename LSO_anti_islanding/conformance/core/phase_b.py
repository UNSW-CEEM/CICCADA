"""Site-level Phase B profile selection and compliance aggregation."""

import polars as pl
from core.threshold_methods import (
    _default_threshold_profile,
    _profile_with_selection_metadata,
    _raw_threshold_profile,
    _thresholds_row_from_threshold_dict,
)


def _select_phase_b_threshold_profile_for_method(
    site_id,
    day_behaviours,
    PRated,
    raw_thresholds,
    confidence_info,
    *,
    phase_b_method="tier_based",
    tau=0.3,
    ov1_floor_offset=0.5,
):
    if phase_b_method == "default":
        return _profile_with_selection_metadata(
            _default_threshold_profile(tau=tau, ov1_floor_offset=ov1_floor_offset),
            "default",
        )
    if phase_b_method == "tier_based":
        return _profile_with_selection_metadata(
            _raw_threshold_profile(
                raw_thresholds, tau=tau, ov1_floor_offset=ov1_floor_offset
            ),
            "tier_based",
        )
    raise KeyError(
        f"Unknown active Phase B method: {phase_b_method}. "
        "Supported methods are 'tier_based' and 'default'."
    )


def _run_phase_b_with_thresholds(
    site_id,
    day_behaviours,
    PRated,
    *,
    los_threshold,
    los_threshold_p25=None,
    los_threshold_p10=None,
    los_threshold_min=None,
    ov1_work_threshold,
    tau=0.3,
    compliance_threshold_pct=90.0,
):
    """
    Aggregate Phase B time-compliance across all available days for one site.
    """

    def aggregate_for_los_threshold(los_threshold_used):
        detail_frames = []
        los_eligible = 0
        los_compliant = 0
        ov1_eligible = 0
        ov1_compliant = 0

        for day_info in day_behaviours:
            outcome = day_info["behaviour"].phase_b_day(
                PRated,
                los_threshold=los_threshold_used,
                ov1_work_threshold=ov1_work_threshold,
                tau=tau,
            )
            if not outcome["detail"].is_empty():
                detail_frames.append(
                    outcome["detail"].with_columns(
                        pl.lit(day_info["day"]).alias("event_day")
                    )
                )
            summary = outcome["summary"]
            los_eligible += summary["los_eligible"]
            los_compliant += summary["los_compliant"]
            ov1_eligible += summary["ov1_eligible"]
            ov1_compliant += summary["ov1_compliant"]

        detail_all = (
            pl.concat(detail_frames, how="vertical")
            if detail_frames
            else pl.DataFrame()
        )
        los_pct = None if los_eligible == 0 else (los_compliant / los_eligible) * 100.0
        ov1_pct = None if ov1_eligible == 0 else (ov1_compliant / ov1_eligible) * 100.0
        los_pass = None if los_pct is None else los_pct >= compliance_threshold_pct
        ov1_pass = None if ov1_pct is None else ov1_pct >= compliance_threshold_pct
        assessed_passes = [v for v in [los_pass, ov1_pass] if v is not None]
        overall_pass = None if not assessed_passes else all(assessed_passes)
        return {
            "detail": detail_all,
            "los_eligible": los_eligible,
            "los_compliant": los_compliant,
            "los_pct": los_pct,
            "ov1_eligible": ov1_eligible,
            "ov1_compliant": ov1_compliant,
            "ov1_pct": ov1_pct,
            "los_pass": los_pass,
            "ov1_pass": ov1_pass,
            "overall_pass": overall_pass,
            "los_threshold_used": los_threshold_used,
        }

    median_result = aggregate_for_los_threshold(los_threshold)
    chosen_result = median_result
    threshold_sensitive = False
    pass_basis = (
        "unassessed"
        if chosen_result["overall_pass"] is None
        else "median"
    )

    summary_row = pl.DataFrame(
        [
            {
                "site_id": site_id,
                "los_eligible": chosen_result["los_eligible"],
                "los_compliant": chosen_result["los_compliant"],
                "los_compliance_pct": chosen_result["los_pct"],
                "ov1_eligible": chosen_result["ov1_eligible"],
                "ov1_compliant": chosen_result["ov1_compliant"],
                "ov1_compliance_pct": chosen_result["ov1_pct"],
                "los_pass": chosen_result["los_pass"],
                "ov1_pass": chosen_result["ov1_pass"],
                "overall_pass": chosen_result["overall_pass"],
                "los_threshold_used": chosen_result["los_threshold_used"],
                "threshold_sensitive": threshold_sensitive,
                "pass_basis": pass_basis,
            }
        ]
    ).with_columns(
        [
            pl.col("site_id").cast(pl.Int64),
            pl.col("los_eligible").cast(pl.Int64),
            pl.col("los_compliant").cast(pl.Int64),
            pl.col("los_compliance_pct").cast(pl.Float64),
            pl.col("ov1_eligible").cast(pl.Int64),
            pl.col("ov1_compliant").cast(pl.Int64),
            pl.col("ov1_compliance_pct").cast(pl.Float64),
            pl.col("los_pass").cast(pl.Boolean),
            pl.col("ov1_pass").cast(pl.Boolean),
            pl.col("overall_pass").cast(pl.Boolean),
            pl.col("los_threshold_used").cast(pl.Float64),
            pl.col("threshold_sensitive").cast(pl.Boolean),
            pl.col("pass_basis").cast(pl.Utf8),
        ]
    )

    return {"detail": chosen_result["detail"], "summary_row": summary_row}


def run_phase_b_for_site(
    site_id,
    day_behaviours,
    PRated,
    *,
    raw_thresholds,
    confidence_info,
    phase_b_method="tier_based",
    tau=0.3,
    compliance_threshold_pct=90.0,
    ov1_floor_offset=None,
):
    """
    Run Phase B for one selected method using the thresholds learned in Phase A.
    """
    if ov1_floor_offset is None:
        ov1_floor_offset = float(raw_thresholds["ov1_anchor_site"]) - float(
            raw_thresholds["ov1_floor_site"]
        )

    selected_thresholds = _select_phase_b_threshold_profile_for_method(
        site_id,
        day_behaviours,
        PRated,
        raw_thresholds,
        confidence_info,
        phase_b_method=phase_b_method,
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )
    threshold_row = _thresholds_row_from_threshold_dict(
        site_id,
        selected_thresholds,
        raw_thresholds,
        confidence_info,
    )
    use_evidence_basis = phase_b_method == "tier_based"
    threshold_row = threshold_row.with_columns(
        [
            pl.lit(
                raw_thresholds["los_threshold_basis"]
                if use_evidence_basis
                else "default"
            ).alias("los_threshold_basis"),
            pl.lit(
                raw_thresholds["los_winning_window_count"],
                dtype=pl.Int64,
            ).alias("los_winning_window_count"),
            pl.lit(
                raw_thresholds["los_winning_window_median_v"],
                dtype=pl.Float64,
            ).alias("los_winning_window_median_v"),
            pl.lit(
                raw_thresholds["los_overall_range_v"],
                dtype=pl.Float64,
            ).alias("los_overall_range_v"),
            pl.lit(
                raw_thresholds["ov1_threshold_basis"]
                if use_evidence_basis
                else "default"
            ).alias("ov1_threshold_basis"),
            pl.lit(
                raw_thresholds["ov1_winning_window_count"],
                dtype=pl.Int64,
            ).alias("ov1_winning_window_count"),
            pl.lit(
                raw_thresholds["ov1_winning_window_median_v"],
                dtype=pl.Float64,
            ).alias("ov1_winning_window_median_v"),
            pl.lit(
                raw_thresholds["ov1_overall_range_v"],
                dtype=pl.Float64,
            ).alias("ov1_overall_range_v"),
        ]
    )
    phase_b = _run_phase_b_with_thresholds(
        site_id,
        day_behaviours,
        PRated,
        los_threshold=selected_thresholds["los_anchor_site"],
        los_threshold_p25=selected_thresholds["los_anchor_p25_site"],
        los_threshold_p10=selected_thresholds["los_anchor_p10_site"],
        los_threshold_min=selected_thresholds["los_anchor_min_site"],
        ov1_work_threshold=selected_thresholds["ov1_work_site"],
        tau=tau,
        compliance_threshold_pct=compliance_threshold_pct,
    )
    return {
        "detail": phase_b["detail"],
        "summary_row": phase_b["summary_row"],
        "threshold_row": threshold_row,
    }
