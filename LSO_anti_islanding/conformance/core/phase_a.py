"""Linear disconnect-edge detection, attribution, and threshold learning."""

from typing import Any

import polars as pl


MAX_DISCONNECT_EDGE_GAP_SECONDS = 300


def detect_edges(signal_frame: pl.DataFrame, PRated):
    """Detect strict disconnect and reconnect edges in one site-day."""
    if signal_frame.is_empty():
        return {
            "frame": signal_frame,
            "disconnect_edges": pl.DataFrame(),
            "reconnect_edges": pl.DataFrame(),
        }

    p_step_strict = 0.10 * PRated
    frame = signal_frame.with_columns(
        [
            (
                (~pl.col("is_disc").shift(1))
                & pl.col("is_disc")
                & (pl.col("site_power_drop") >= p_step_strict)
                & (pl.col("dt_next_s").shift(1) > 0)
                & (
                    pl.col("dt_next_s").shift(1)
                    <= MAX_DISCONNECT_EDGE_GAP_SECONDS
                )
            )
            .fill_null(False)
            .alias("disconnect_edge"),
            (
                pl.col("is_disc").shift(1)
                & (~pl.col("is_disc"))
                & (pl.col("site_power_rise") >= p_step_strict)
            )
            .fill_null(False)
            .alias("reconnect_edge"),
        ]
    )
    disconnect_edges = (
        frame.filter(pl.col("disconnect_edge"))
        .select(
            [
                "site_id",
                "local_tstamp",
                "v10m_avg",
                "vinst_max",
                "site_power_drop",
                "disconnect_edge",
            ]
        )
        .with_columns(pl.lit("strict_10pct").alias("edge_source"))
        .sort("local_tstamp")
    )
    reconnect_edges = (
        frame.filter(pl.col("reconnect_edge"))
        .select(
            [
                "site_id",
                "local_tstamp",
                "v10m_avg",
                "vinst_max",
                "reconnect_edge",
            ]
        )
        .with_columns(pl.lit("strict_10pct").alias("edge_source"))
        .sort("local_tstamp")
    )
    return {
        "frame": frame,
        "disconnect_edges": disconnect_edges,
        "reconnect_edges": reconnect_edges,
    }


def classify_disconnects_as_los_or_ov1(
    edge_result,
    PRated,
    *,
    los_lo=251.1,
    los_hi_strict=259.0,
    los_hi_cap=260.3,
    eps=0.02,
):
    """Classify one day's disconnect edges and construct reconnect brackets."""
    frame = edge_result["frame"]
    disconnect_edges = edge_result["disconnect_edges"]
    reconnect_edges = edge_result["reconnect_edges"]
    if frame.is_empty():
        return {
            "records": pl.DataFrame(),
            "brackets": pl.DataFrame(),
            "reconnects": reconnect_edges,
        }

    timestamp_dtype = frame.schema["local_tstamp"]
    if disconnect_edges.is_empty():
        return {
            "records": pl.DataFrame(),
            "brackets": pl.DataFrame(),
            "reconnects": reconnect_edges,
        }

    reconnect_list = list(reconnect_edges.iter_rows(named=True))
    records: list[dict[str, Any]] = []
    brackets: list[dict[str, Any]] = []
    event_idx = 0

    for row in disconnect_edges.iter_rows(named=True):
        event_idx += 1
        tdisc = row["local_tstamp"]
        v10m = row["v10m_avg"]
        vinst = row["vinst_max"]
        edge_source = row["edge_source"]
        site_power_drop_kw = row["site_power_drop"]
        site_power_drop_pct = (
            None
            if PRated in [None, 0] or site_power_drop_kw is None
            else (float(site_power_drop_kw) / float(PRated)) * 100.0
        )
        mech = None
        threshold_voltage = None
        grey_non_sustained = False
        reconnect = next(
            (record for record in reconnect_list if record["local_tstamp"] > tdisc),
            None,
        )
        vinst_in_ov1_region = vinst is not None and (
            los_hi_strict <= vinst <= los_hi_cap
        )

        if v10m is not None and (los_lo <= v10m <= los_hi_strict):
            mech = "LOS"
            threshold_voltage = v10m
        elif vinst is not None and (vinst > los_hi_cap):
            mech = "OV1"
            threshold_voltage = vinst
        elif v10m is not None and (los_hi_strict < v10m <= los_hi_cap):
            if vinst_in_ov1_region:
                mech = "OV1"
                threshold_voltage = vinst
                grey_non_sustained = True
            else:
                mech = "LOS"
                threshold_voltage = v10m

        if mech is None or threshold_voltage is None:
            continue

        records.append(
            {
                "site_id": row["site_id"],
                "event_id": event_idx,
                "ts_disc": tdisc,
                "edge_source": edge_source,
                "mech": mech,
                "v_los_recorded": threshold_voltage if mech == "LOS" else None,
                "v_ov1_recorded": threshold_voltage if mech == "OV1" else None,
                "v10m_disc": v10m,
                "vinst_disc": vinst,
                "site_power_drop_kw": site_power_drop_kw,
                "site_power_drop_pct_rated": site_power_drop_pct,
                "grey_non_sustained": grey_non_sustained,
            }
        )

        if reconnect is None:
            continue
        vrec = reconnect["v10m_avg"] if mech == "LOS" else reconnect["vinst_max"]
        if vrec is None:
            continue
        brackets.append(
            {
                "site_id": row["site_id"],
                "event_id": event_idx,
                "edge_source": edge_source,
                "mech": mech,
                "ts_disc": tdisc,
                "ts_rec": reconnect["local_tstamp"],
                "L": vrec + eps,
                "U": threshold_voltage,
                "midpoint": (vrec + eps + threshold_voltage) / 2.0,
                "width": threshold_voltage - (vrec + eps),
            }
        )

    record_frame = (
        pl.DataFrame(records)
        if records
        else pl.DataFrame(
            schema={
                "site_id": pl.Int64,
                "event_id": pl.Int64,
                "ts_disc": timestamp_dtype,
                "edge_source": pl.Utf8,
                "mech": pl.Utf8,
                "v_los_recorded": pl.Float64,
                "v_ov1_recorded": pl.Float64,
                "v10m_disc": pl.Float64,
                "vinst_disc": pl.Float64,
                "site_power_drop_kw": pl.Float64,
                "site_power_drop_pct_rated": pl.Float64,
                "grey_non_sustained": pl.Boolean,
            }
        )
    ).with_columns(
        [
            pl.col("site_id").cast(pl.Int64),
            pl.col("event_id").cast(pl.Int64),
            pl.col("ts_disc").cast(timestamp_dtype),
            pl.col("edge_source").cast(pl.Utf8),
            pl.col("mech").cast(pl.Utf8),
            pl.col("v_los_recorded").cast(pl.Float64),
            pl.col("v_ov1_recorded").cast(pl.Float64),
            pl.col("v10m_disc").cast(pl.Float64),
            pl.col("vinst_disc").cast(pl.Float64),
            pl.col("site_power_drop_kw").cast(pl.Float64),
            pl.col("site_power_drop_pct_rated").cast(pl.Float64),
            pl.col("grey_non_sustained").cast(pl.Boolean),
        ]
    )
    bracket_frame = (
        pl.DataFrame(brackets)
        if brackets
        else pl.DataFrame(
            schema={
                "site_id": pl.Int64,
                "event_id": pl.Int64,
                "edge_source": pl.Utf8,
                "mech": pl.Utf8,
                "ts_disc": timestamp_dtype,
                "ts_rec": timestamp_dtype,
                "L": pl.Float64,
                "U": pl.Float64,
                "midpoint": pl.Float64,
                "width": pl.Float64,
            }
        )
    ).with_columns(
        [
            pl.col("site_id").cast(pl.Int64),
            pl.col("event_id").cast(pl.Int64),
            pl.col("edge_source").cast(pl.Utf8),
            pl.col("mech").cast(pl.Utf8),
            pl.col("ts_disc").cast(timestamp_dtype),
            pl.col("ts_rec").cast(timestamp_dtype),
            pl.col("L").cast(pl.Float64),
            pl.col("U").cast(pl.Float64),
            pl.col("midpoint").cast(pl.Float64),
            pl.col("width").cast(pl.Float64),
        ]
    )
    return {
        "records": record_frame,
        "brackets": bracket_frame,
        "reconnects": reconnect_edges,
    }


def learn_site_thresholds(
    records: pl.DataFrame,
    *,
    tau: float = 0.3,
    ov1_floor_offset: float = 0.5,
):
    """Learn the site LOS and OV1 thresholds from attributed disconnects."""
    evidence = {}
    for mechanism, voltage_column, default in (
        ("LOS", "v_los_recorded", 258.0),
        ("OV1", "v_ov1_recorded", 265.0),
    ):
        if records.is_empty() or voltage_column not in records.columns:
            values = []
        else:
            values = (
                records.filter(
                    (pl.col("mech") == mechanism)
                    & pl.col(voltage_column).is_not_null()
                )
                .get_column(voltage_column)
                .to_list()
            )
        values = sorted(float(value) for value in values if value is not None)
        voltage_range = None if not values else max(values) - min(values)

        winning_values = []
        right = 0
        for left, start_v in enumerate(values):
            right = max(right, left)
            while right < len(values) and values[right] <= start_v + 0.5:
                right += 1
            candidate = values[left:right]
            if len(candidate) > len(winning_values):
                winning_values = candidate

        winning_median = None
        if winning_values:
            midpoint = len(winning_values) // 2
            if len(winning_values) % 2 == 1:
                winning_median = winning_values[midpoint]
            else:
                winning_median = (
                    winning_values[midpoint - 1] + winning_values[midpoint]
                ) / 2.0

        learned = (
            len(winning_values) >= 3
            and voltage_range is not None
            and voltage_range <= 2.0
        )
        evidence[mechanism] = {
            "threshold": winning_median if learned else default,
            "basis": "learned" if learned else "default",
            "winning_window_count": len(winning_values),
            "winning_window_median_v": winning_median,
            "overall_range_v": voltage_range,
        }

    los = evidence["LOS"]
    ov1 = evidence["OV1"]

    los_anchor = float(los["threshold"])
    ov1_anchor = float(ov1["threshold"])
    delta_los = None if los["basis"] == "default" else los_anchor - 258.0
    delta_ov1 = None if ov1["basis"] == "default" else ov1_anchor - 265.0

    raw_thresholds = {
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
    confidence_info = {
        "threshold_confidence_tier": "not_used",
        "confidence_primary_mech": None,
        "confidence_event_count": records.height,
        "confidence_drop20_count": 0,
        "confidence_drop10_count": 0,
        "confidence_spread_v": None,
    }
    return {
        "raw_thresholds": raw_thresholds,
        "confidence_info": confidence_info,
    }


def run_phase_a_for_site(
    site_id,
    prepared_site_days,
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
    records_all = []
    brackets_all = []
    phase_a_days = []
    for prepared_day in prepared_site_days:
        edge_result = detect_edges(prepared_day["signal_frame"], PRated)
        outcome = classify_disconnects_as_los_or_ov1(
            edge_result,
            PRated,
            eps=eps,
        )
        phase_a_days.append(
            {
                "day": prepared_day["analysis_date"],
                "frame": edge_result["frame"],
                **outcome,
            }
        )
        if not outcome["records"].is_empty():
            records_all.append(
                outcome["records"].with_columns(
                    pl.lit(prepared_day["analysis_date"]).alias("event_day")
                )
            )
        if not outcome["brackets"].is_empty():
            brackets_all.append(
                outcome["brackets"].with_columns(
                    pl.lit(prepared_day["analysis_date"]).alias("event_day")
                )
            )

    site_records = (
        pl.concat(records_all, how="vertical") if records_all else pl.DataFrame()
    )
    site_brackets = (
        pl.concat(brackets_all, how="vertical") if brackets_all else pl.DataFrame()
    )

    threshold_result = learn_site_thresholds(
        site_records,
        tau=tau,
        ov1_floor_offset=delta_lower_daily_cap,
    )

    return {
        "raw_thresholds": threshold_result["raw_thresholds"],
        "confidence_info": threshold_result["confidence_info"],
        "records": site_records,
        "brackets": site_brackets,
        "day_outputs": phase_a_days,
    }
