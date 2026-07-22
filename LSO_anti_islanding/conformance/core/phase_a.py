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


def _quantile_or_none(values, q: float):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _min_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return min(values)


def _range_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return max(values) - min(values)

def _threshold_confidence_from_records(records: pl.DataFrame):
    if records.is_empty():
        return {
            "threshold_confidence_tier": "low",
            "confidence_primary_mech": None,
            "confidence_event_count": 0,
            "confidence_drop20_count": 0,
            "confidence_drop10_count": 0,
            "confidence_spread_v": None,
        }

    los_rows = records.filter(pl.col("mech") == "LOS")
    ov1_rows = records.filter(pl.col("mech") == "OV1")

    if not los_rows.is_empty():
        primary_rows = los_rows
        primary_mech = "LOS"
        voltage_col = "v_los_recorded"
    elif not ov1_rows.is_empty():
        primary_rows = ov1_rows
        primary_mech = "OV1"
        voltage_col = "v_ov1_recorded"
    else:
        return {
            "threshold_confidence_tier": "low",
            "confidence_primary_mech": None,
            "confidence_event_count": 0,
            "confidence_drop20_count": 0,
            "confidence_drop10_count": 0,
            "confidence_spread_v": None,
        }

    values = [float(v) for v in primary_rows[voltage_col].to_list() if v is not None]
    drop_pcts = [
        float(v)
        for v in primary_rows["site_power_drop_pct_rated"].to_list()
        if v is not None
    ]

    event_count = len(values)
    drop20_count = sum(v >= 20.0 for v in drop_pcts)
    drop10_count = sum(v >= 10.0 for v in drop_pcts)
    spread_v = _range_or_none(values)

    if event_count >= 2 and drop20_count >= 2 and spread_v is not None and spread_v <= 2.0:
        tier = "high"
    elif event_count >= 2 and drop10_count >= 2 and spread_v is not None and spread_v <= 3.0:
        tier = "medium"
    else:
        tier = "low"

    return {
        "threshold_confidence_tier": tier,
        "confidence_primary_mech": primary_mech,
        "confidence_event_count": event_count,
        "confidence_drop20_count": drop20_count,
        "confidence_drop10_count": drop10_count,
        "confidence_spread_v": spread_v,
    }

def _site_thresholds_from_records(
    records: pl.DataFrame,
    *,
    tau: float = 0.3,
    ov1_floor_offset: float = 0.5,
):
    los_rows = records.filter(pl.col("mech") == "LOS") if not records.is_empty() else pl.DataFrame()
    ov1_rows = records.filter(pl.col("mech") == "OV1") if not records.is_empty() else pl.DataFrame()
    los_vals = los_rows["v_los_recorded"].to_list() if not los_rows.is_empty() else []
    ov1_vals_direct = ov1_rows["v_ov1_recorded"].to_list() if not ov1_rows.is_empty() else []

    delta_los = None
    delta_los_p25 = None
    delta_los_p10 = None
    delta_los_min = None
    delta_ov1 = None
    ov1_reclassified_count = 0
    los_removed_by_ov1_count = 0
    has_direct_ov1 = len(ov1_vals_direct) >= 1
    if has_direct_ov1:
        direct_ov1_anchor = _median_or_none(ov1_vals_direct)
        provisional_ov1_anchor = direct_ov1_anchor
        retained_los_vals: list[float] = []
        reclassified_ov1_vals: list[float] = []
        for row in los_rows.iter_rows(named=True):
            vlos = row["v_los_recorded"]
            if vlos is None:
                continue
            vlos = float(vlos)
            if vlos > provisional_ov1_anchor:
                los_removed_by_ov1_count += 1
                v10m = row["v10m_disc"]
                vinst = row["vinst_disc"]
                is_clear_ov1 = vinst is not None and float(vinst) > 260.3
                is_grey_ov1 = (
                    v10m is not None and
                    259.0 < float(v10m) <= 260.3 and
                    vinst is not None and
                    259.0 <= float(vinst) <= 260.3
                )
                if is_clear_ov1 or is_grey_ov1:
                    reclassified_ov1_vals.append(float(vinst))
            else:
                retained_los_vals.append(vlos)
        los_vals = retained_los_vals
        ov1_vals = [float(v) for v in ov1_vals_direct if v is not None] + reclassified_ov1_vals
        ov1_reclassified_count = len(reclassified_ov1_vals)
        delta_ov1 = _median_or_none(ov1_vals) - 265.0
    else:
        ov1_vals = []

    if los_vals:
        delta_los = _median_or_none(los_vals) - 258.0
        delta_los_p25 = _quantile_or_none(los_vals, 0.25) - 258.0
        delta_los_p10 = _quantile_or_none(los_vals, 0.10) - 258.0
        delta_los_min = _min_or_none(los_vals) - 258.0
    elif delta_ov1 is not None:
        # If OV1 is observed directly and no LOS samples survive, infer LOS from the same site delta.
        delta_los = delta_ov1
        delta_los_p25 = delta_ov1
        delta_los_p10 = delta_ov1
        delta_los_min = delta_ov1

    los_anchor = 258.0 if delta_los is None else 258.0 + delta_los
    los_anchor_p25 = 258.0 if delta_los_p25 is None else 258.0 + delta_los_p25
    los_anchor_p10 = 258.0 if delta_los_p10 is None else 258.0 + delta_los_p10
    los_anchor_min = 258.0 if delta_los_min is None else 258.0 + delta_los_min
    if delta_ov1 is not None:
        ov1_anchor = 265.0 + delta_ov1
        ov1_basis = "ov1_records"
    elif delta_los is not None:
        ov1_anchor = 265.0 + delta_los
        ov1_basis = "los_fallback"
    else:
        ov1_anchor = 265.0
        ov1_basis = "default"

    return {
        "delta_los_site": delta_los,
        "delta_los_p25_site": delta_los_p25,
        "delta_los_p10_site": delta_los_p10,
        "delta_los_min_site": delta_los_min,
        "delta_ov1_site": delta_ov1,
        "los_anchor_site": los_anchor,
        "los_anchor_p25_site": los_anchor_p25,
        "los_anchor_p10_site": los_anchor_p10,
        "los_anchor_min_site": los_anchor_min,
        "ov1_anchor_site": ov1_anchor,
        "ov1_work_site": ov1_anchor,
        "ov1_floor_site": ov1_anchor - ov1_floor_offset,
        "ov1_test_site": ov1_anchor - tau,
        "ov1_basis": ov1_basis,
        "ov1_event_count": len(ov1_vals),
        "ov1_reclassified_count": ov1_reclassified_count,
        "los_removed_by_ov1_count": los_removed_by_ov1_count,
        "delta_gap_v": None if (delta_los is None or delta_ov1 is None) else abs(delta_ov1 - delta_los),
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
            records_all.append(outcome["records"].with_columns(pl.lit(day_info["day"]).alias("event_day")))
        if not outcome["brackets"].is_empty():
            brackets_all.append(outcome["brackets"].with_columns(pl.lit(day_info["day"]).alias("event_day")))

    last_records = pl.concat(records_all, how="vertical") if records_all else pl.DataFrame()
    last_brackets = pl.concat(brackets_all, how="vertical") if brackets_all else pl.DataFrame()

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
