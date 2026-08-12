"""Site-level Phase B profile selection and compliance aggregation."""

from typing import Any

import polars as pl
from core.threshold_methods import (
    _blended_threshold_profile,
    _build_threshold_profile,
    _default_threshold_profile,
    _profile_with_selection_metadata,
    _raw_threshold_profile,
    _thresholds_row_from_threshold_dict,
)


def _phase_b_selection_score(summary_row: dict[str, Any]):
    overall_pass = summary_row["overall_pass"]
    overall_rank = 2 if overall_pass is True else 1 if overall_pass is None else 0
    assessed_pcts = [
        float(v)
        for v in [summary_row["los_compliance_pct"], summary_row["ov1_compliance_pct"]]
        if v is not None
    ]
    min_pct = min(assessed_pcts) if assessed_pcts else 0.0
    mean_pct = sum(assessed_pcts) / len(assessed_pcts) if assessed_pcts else 0.0
    return overall_rank, min_pct, mean_pct


def _evaluate_phase_b_profile_for_selection(
    site_id,
    day_behaviours,
    PRated,
    profile,
    *,
    tau=0.3,
):
    phase_b = _run_phase_b_with_thresholds(
        site_id,
        day_behaviours,
        PRated,
        los_threshold=profile["los_anchor_site"],
        los_threshold_p25=profile["los_anchor_p25_site"],
        los_threshold_p10=profile["los_anchor_p10_site"],
        los_threshold_min=profile["los_anchor_min_site"],
        ov1_work_threshold=profile["ov1_work_site"],
        tau=tau,
    )
    summary = phase_b["summary_row"].to_dicts()[0]
    score = _phase_b_selection_score(summary)
    return phase_b, summary, score


def _select_confidence_threshold_profile_for_phase_b(
    site_id,
    day_behaviours,
    PRated,
    raw_thresholds,
    confidence_info,
    *,
    high_profile_name="learned",
    tau=0.3,
    ov1_floor_offset=0.5,
):
    default_profile = _default_threshold_profile(
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )
    learned_profile = _raw_threshold_profile(
        raw_thresholds,
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )
    blended_profile = _blended_threshold_profile(
        raw_thresholds,
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )

    tier = confidence_info["threshold_confidence_tier"]
    if tier == "high":
        high_profiles = {
            "learned": learned_profile,
            "blended": blended_profile,
        }
        return _profile_with_selection_metadata(
            high_profiles[high_profile_name],
            f"high_{high_profile_name}",
        )

    if tier == "low":
        return _profile_with_selection_metadata(default_profile, "low_default")

    candidates = [
        ("default", default_profile),
        ("blended", blended_profile),
        ("learned", learned_profile),
    ]
    best_name = None
    best_profile = None
    best_score = None

    for name, profile in candidates:
        _, _, score = _evaluate_phase_b_profile_for_selection(
            site_id,
            day_behaviours,
            PRated,
            profile,
            tau=tau,
        )
        if best_score is None or score > best_score:
            best_name = name
            best_profile = profile
            best_score = score

    return _profile_with_selection_metadata(
        best_profile,
        f"medium_{best_name}",
        best_score,
    )


def _select_legacy_sweep_threshold_profile_for_phase_b(
    site_id,
    day_behaviours,
    PRated,
    *,
    tau=0.3,
    ov1_floor_offset=0.5,
):
    default_profile = _default_threshold_profile(
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )
    sweep_thresholds = [
        257.0,
        256.0,
        255.7,
        254.7,
        253.7,
        253.4,
        251.8,
        251.1,
        259.0,
        260.0,
        260.3,
    ]

    _, summary, best_score = _evaluate_phase_b_profile_for_selection(
        site_id,
        day_behaviours,
        PRated,
        default_profile,
        tau=tau,
    )
    if summary["overall_pass"] is True:
        return _profile_with_selection_metadata(
            default_profile, "sweep_default", best_score
        )

    best_profile = default_profile
    best_basis = "sweep_default"
    for los_threshold in sweep_thresholds:
        candidate_profile = _build_threshold_profile(
            los_anchor=los_threshold,
            los_anchor_p25=los_threshold,
            los_anchor_p10=los_threshold,
            los_anchor_min=los_threshold,
            ov1_anchor=265.0,
            ov1_basis="default",
            tau=tau,
            ov1_floor_offset=ov1_floor_offset,
        )
        _, summary, score = _evaluate_phase_b_profile_for_selection(
            site_id,
            day_behaviours,
            PRated,
            candidate_profile,
            tau=tau,
        )
        if summary["overall_pass"] is True:
            return _profile_with_selection_metadata(
                candidate_profile,
                f"sweep_{str(los_threshold).replace('.', 'p')}V",
                score,
            )
        if score > best_score:
            best_score = score
            best_profile = candidate_profile
            best_basis = f"sweep_{str(los_threshold).replace('.', 'p')}V"

    return _profile_with_selection_metadata(best_profile, best_basis, best_score)


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
    if phase_b_method == "original":
        return _profile_with_selection_metadata(
            _raw_threshold_profile(
                raw_thresholds, tau=tau, ov1_floor_offset=ov1_floor_offset
            ),
            "original",
        )
    if phase_b_method == "tier_based":
        return _select_confidence_threshold_profile_for_phase_b(
            site_id,
            day_behaviours,
            PRated,
            raw_thresholds,
            confidence_info,
            high_profile_name="learned",
            tau=tau,
            ov1_floor_offset=ov1_floor_offset,
        )
    if phase_b_method == "old_sweep":
        return _select_legacy_sweep_threshold_profile_for_phase_b(
            site_id,
            day_behaviours,
            PRated,
            tau=tau,
            ov1_floor_offset=ov1_floor_offset,
        )
    if phase_b_method == "blended":
        return _select_confidence_threshold_profile_for_phase_b(
            site_id,
            day_behaviours,
            PRated,
            raw_thresholds,
            confidence_info,
            high_profile_name="blended",
            tau=tau,
            ov1_floor_offset=ov1_floor_offset,
        )
    raise KeyError(f"Unknown Phase B method: {phase_b_method}")


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
    p25_result = None
    p10_result = None
    min_result = None
    if (
        median_result["los_pass"] is False
        and los_threshold_p25 is not None
        and los_threshold_p25 != los_threshold
    ):
        p25_result = aggregate_for_los_threshold(los_threshold_p25)

    if (
        median_result["los_pass"] is False
        and (p25_result is None or p25_result["los_pass"] is False)
        and los_threshold_p10 is not None
        and los_threshold_p10 not in [los_threshold, los_threshold_p25]
    ):
        p10_result = aggregate_for_los_threshold(los_threshold_p10)

    if (
        median_result["los_pass"] is False
        and (p25_result is None or p25_result["los_pass"] is False)
        and (p10_result is None or p10_result["los_pass"] is False)
        and los_threshold_min is not None
        and los_threshold_min
        not in [los_threshold, los_threshold_p25, los_threshold_p10]
    ):
        min_result = aggregate_for_los_threshold(los_threshold_min)

    use_p25_override = (
        p25_result is not None
        and median_result["los_pass"] is False
        and p25_result["los_pass"] is True
    )
    use_p10_override = (
        p10_result is not None
        and median_result["los_pass"] is False
        and (p25_result is None or p25_result["los_pass"] is not True)
        and p10_result["los_pass"] is True
    )
    use_min_override = (
        min_result is not None
        and median_result["los_pass"] is False
        and (p25_result is None or p25_result["los_pass"] is not True)
        and (p10_result is None or p10_result["los_pass"] is not True)
        and min_result["los_pass"] is True
    )

    if use_min_override:
        chosen_result = min_result
    elif use_p10_override:
        chosen_result = p10_result
    elif use_p25_override:
        chosen_result = p25_result
    else:
        chosen_result = median_result
    threshold_sensitive = use_p25_override or use_p10_override or use_min_override
    pass_basis = (
        "unassessed"
        if chosen_result["overall_pass"] is None
        else "min_override"
        if use_min_override
        else "p10_override"
        if use_p10_override
        else "p25_override"
        if use_p25_override
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
