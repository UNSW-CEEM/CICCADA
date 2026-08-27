"""Site-level Phase A aggregation and threshold learning."""

import polars as pl


def _median_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _range_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return max(values) - min(values)


def _dominant_voltage_window(values, width_v=0.5):
    """Return the densest inclusive window, preferring the lower start on ties."""
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return {"count": 0, "median_v": None, "values": []}

    best_values = []
    right = 0
    for left, start_v in enumerate(values):
        right = max(right, left)
        while right < len(values) and values[right] <= start_v + width_v:
            right += 1
        candidate = values[left:right]
        if len(candidate) > len(best_values):
            best_values = candidate

    return {
        "count": len(best_values),
        "median_v": _median_or_none(best_values),
        "values": best_values,
    }


def _mechanism_threshold_evidence(
    records, mechanism, voltage_column, default, min_events=3
):
    if records.is_empty() or voltage_column not in records.columns:
        values = []
    else:
        values = (
            records.filter(
                (pl.col("mech") == mechanism) & pl.col(voltage_column).is_not_null()
            )
            .get_column(voltage_column)
            .to_list()
        )

    values = [float(v) for v in values if v is not None]
    voltage_range = _range_or_none(values)
    window = _dominant_voltage_window(values)
    learned = (
        window["count"] >= min_events
        and voltage_range is not None
        and voltage_range <= 2.0
    )
    return {
        "threshold": window["median_v"] if learned else default,
        "basis": "learned" if learned else "default",
        "winning_window_count": window["count"],
        "winning_window_median_v": window["median_v"],
        "overall_range_v": voltage_range,
    }


def _threshold_confidence_from_records(records: pl.DataFrame):
    """Keep the existing return shape; confidence tiers are no longer used."""
    return {
        "threshold_confidence_tier": "not_used",
        "confidence_primary_mech": None,
        "confidence_event_count": records.height,
        "confidence_drop20_count": 0,
        "confidence_drop10_count": 0,
        "confidence_spread_v": None,
    }


def _site_thresholds_from_records(
    records: pl.DataFrame,
    *,
    tau: float = 0.3,
    ov1_floor_offset: float = 0.5,
):
    los = _mechanism_threshold_evidence(
        records,
        "LOS",
        "v_los_recorded",
        258.0, min_events=3,
    )
    ov1 = _mechanism_threshold_evidence(
        records,
        "OV1",
        "v_ov1_recorded",
        265.0, min_events=3,
    )

    los_anchor = float(los["threshold"])
    ov1_anchor = float(ov1["threshold"])
    delta_los = None if los["basis"] == "default" else los_anchor - 258.0
    delta_ov1 = None if ov1["basis"] == "default" else ov1_anchor - 265.0

    return {
        "delta_los_site": delta_los,
        "delta_los_p25_site": delta_los,
        "delta_los_p10_site": delta_los,
        "delta_los_min_site": delta_los,
        "delta_ov1_site": delta_ov1,
        "los_anchor_site": los_anchor,
        "los_anchor_p25_site": los_anchor,
        "los_anchor_p10_site": los_anchor,
        "los_anchor_min_site": los_anchor,
        "ov1_anchor_site": ov1_anchor,
        "ov1_work_site": ov1_anchor,
        "ov1_floor_site": ov1_anchor - ov1_floor_offset,
        "ov1_test_site": ov1_anchor - tau,
        "ov1_basis": ov1["basis"],
        "ov1_event_count": ov1["winning_window_count"],
        "ov1_reclassified_count": 0,
        "los_removed_by_ov1_count": 0,
        "delta_gap_v": None
        if (delta_los is None or delta_ov1 is None)
        else abs(delta_ov1 - delta_los),
        "los_threshold_basis": los["basis"],
        "los_winning_window_count": los["winning_window_count"],
        "los_winning_window_median_v": los["winning_window_median_v"],
        "los_overall_range_v": los["overall_range_v"],
        "ov1_threshold_basis": ov1["basis"],
        "ov1_winning_window_count": ov1["winning_window_count"],
        "ov1_winning_window_median_v": ov1["winning_window_median_v"],
        "ov1_overall_range_v": ov1["overall_range_v"],
    }


def run_phase_a_for_site(
    site_id,
    day_behaviours,
    PRated,
    *,
    tau=0.3,
    eps=0.02,
    delta_lower_daily_cap=0.5,
):
    """
    Run Phase A across all available days for one site and learn site thresholds
    from the disconnect-edge records.
    """
    last_records = pl.DataFrame()
    last_brackets = pl.DataFrame()
    phase_a_days = []
    records_all = []
    brackets_all = []
    phase_a_days = []
    for day_info in day_behaviours:
        outcome = day_info["behaviour"].phase_a_day(
            PRated,
            eps=eps,
        )
        phase_a_days.append({"day": day_info["day"], **outcome})
        if not outcome["records"].is_empty():
            records_all.append(
                outcome["records"].with_columns(
                    pl.lit(day_info["day"]).alias("event_day")
                )
            )
        if not outcome["brackets"].is_empty():
            brackets_all.append(
                outcome["brackets"].with_columns(
                    pl.lit(day_info["day"]).alias("event_day")
                )
            )

    last_records = (
        pl.concat(records_all, how="vertical") if records_all else pl.DataFrame()
    )
    last_brackets = (
        pl.concat(brackets_all, how="vertical") if brackets_all else pl.DataFrame()
    )

    raw_thresholds = _site_thresholds_from_records(
        last_records,
        tau=tau,
        ov1_floor_offset=delta_lower_daily_cap,
    )
    confidence_info = _threshold_confidence_from_records(last_records)

    return {
        "raw_thresholds": raw_thresholds,
        "confidence_info": confidence_info,
        "records": last_records,
        "brackets": last_brackets,
        "day_outputs": phase_a_days,
    }
