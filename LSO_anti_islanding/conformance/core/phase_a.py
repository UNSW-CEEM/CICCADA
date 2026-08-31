"""Linear disconnect-edge detection, attribution, and threshold learning."""

from typing import Any

import polars as pl

MAX_DISCONNECT_EDGE_GAP_SECONDS = 300

SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA = {
    "site_id": pl.Int64,
    "los_threshold": pl.Float64,
    "los_lowest_disconnect_voltage": pl.Float64,
    "los_median_all_disconnect_voltages": pl.Float64,
    "los_lowest_reconnect_voltage": pl.Float64,
    "los_median_all_reconnect_voltages": pl.Float64,
    "ov1_threshold": pl.Float64,
    "ov1_lowest_disconnect_voltage": pl.Float64,
    "ov1_median_all_disconnect_voltages": pl.Float64,
    "ov1_lowest_reconnect_voltage": pl.Float64,
    "ov1_median_all_reconnect_voltages": pl.Float64,
}


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
                & (pl.col("dt_next_s").shift(1) <= MAX_DISCONNECT_EDGE_GAP_SECONDS)
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
            ]
        )
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
):
    """Classify one day's disconnect edges and pair the next reconnect voltage."""
    frame = edge_result["frame"]
    disconnect_edges = edge_result["disconnect_edges"]
    reconnect_edges = edge_result["reconnect_edges"]
    timestamp_dtype = frame.schema.get("local_tstamp", pl.Datetime)
    record_schema = {
        "site_id": pl.Int64,
        "event_id": pl.Int64,
        "ts_disc": timestamp_dtype,
        "edge_source": pl.Utf8,
        "mechanism": pl.Utf8,
        "v10m_disc": pl.Float64,
        "vinst_disc": pl.Float64,
        "reconnect_voltage": pl.Float64,
        "site_power_drop_kw": pl.Float64,
        "site_power_drop_pct_rated": pl.Float64,
        "grey_non_sustained": pl.Boolean,
    }
    if frame.is_empty() or disconnect_edges.is_empty():
        return pl.DataFrame(schema=record_schema)

    reconnect_list = list(reconnect_edges.iter_rows(named=True))
    records: list[dict[str, Any]] = []

    for event_idx, row in enumerate(
        disconnect_edges.iter_rows(named=True),
        start=1,
    ):
        tdisc = row["local_tstamp"]
        v10m = row["v10m_avg"]
        vinst = row["vinst_max"]
        site_power_drop_kw = row["site_power_drop"]
        site_power_drop_pct = (
            None
            if PRated in [None, 0] or site_power_drop_kw is None
            else (float(site_power_drop_kw) / float(PRated)) * 100.0
        )
        mechanism = None
        disconnect_voltage = None
        grey_non_sustained = False
        vinst_in_ov1_region = vinst is not None and (
            los_hi_strict <= vinst <= los_hi_cap
        )

        if v10m is not None and (los_lo <= v10m <= los_hi_strict):
            mechanism = "LOS"
            disconnect_voltage = v10m
        elif vinst is not None and (vinst > los_hi_cap):
            mechanism = "OV1"
            disconnect_voltage = vinst
        elif v10m is not None and (los_hi_strict < v10m <= los_hi_cap):
            if vinst_in_ov1_region:
                mechanism = "OV1"
                disconnect_voltage = vinst
                grey_non_sustained = True
            else:
                mechanism = "LOS"
                disconnect_voltage = v10m

        if mechanism is None or disconnect_voltage is None:
            continue

        reconnect = next(
            (record for record in reconnect_list if record["local_tstamp"] > tdisc),
            None,
        )
        reconnect_voltage = None
        if reconnect is not None:
            reconnect_voltage = (
                reconnect["v10m_avg"] if mechanism == "LOS" else reconnect["vinst_max"]
            )

        records.append(
            {
                "site_id": row["site_id"],
                "event_id": event_idx,
                "ts_disc": tdisc,
                "edge_source": row["edge_source"],
                "mechanism": mechanism,
                "v10m_disc": v10m,
                "vinst_disc": vinst,
                "reconnect_voltage": reconnect_voltage,
                "site_power_drop_kw": site_power_drop_kw,
                "site_power_drop_pct_rated": site_power_drop_pct,
                "grey_non_sustained": grey_non_sustained,
            }
        )

    return pl.DataFrame(records, schema=record_schema, strict=False)


def learn_site_thresholds(records: pl.DataFrame):
    """Learn thresholds and summarize all paired site voltages by mechanism."""
    site_voltages = {}
    for mechanism, voltage_column, prefix, default in (
        ("LOS", "v10m_disc", "los", 258.0),
        ("OV1", "vinst_disc", "ov1", 265.0),
    ):
        if records.is_empty():
            values = []
            reconnect_values = []
        else:
            mechanism_records = records.filter(pl.col("mechanism") == mechanism)
            values = mechanism_records.get_column(voltage_column).drop_nulls().to_list()
            reconnect_values = (
                mechanism_records.get_column("reconnect_voltage").drop_nulls().to_list()
            )

        values = sorted(float(value) for value in values)
        reconnect_values = sorted(float(value) for value in reconnect_values)
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
            middle_index = len(winning_values) // 2
            if len(winning_values) % 2 == 1:
                winning_median = winning_values[middle_index]
            else:
                winning_median = (
                    winning_values[middle_index - 1] + winning_values[middle_index]
                ) / 2.0

        all_disconnect_median = None
        if values:
            middle_index = len(values) // 2
            if len(values) % 2 == 1:
                all_disconnect_median = values[middle_index]
            else:
                all_disconnect_median = (
                    values[middle_index - 1] + values[middle_index]
                ) / 2.0

        all_reconnect_median = None
        if reconnect_values:
            middle_index = len(reconnect_values) // 2
            if len(reconnect_values) % 2 == 1:
                all_reconnect_median = reconnect_values[middle_index]
            else:
                all_reconnect_median = (
                    reconnect_values[middle_index - 1] + reconnect_values[middle_index]
                ) / 2.0

        learned = (
            len(winning_values) >= 3
            and voltage_range is not None
            and voltage_range <= 2.0
        )
        site_voltages.update(
            {
                f"{prefix}_threshold": winning_median if learned else default,
                f"{prefix}_lowest_disconnect_voltage": (
                    min(values) if values else None
                ),
                f"{prefix}_median_all_disconnect_voltages": all_disconnect_median,
                f"{prefix}_lowest_reconnect_voltage": (
                    min(reconnect_values) if reconnect_values else None
                ),
                f"{prefix}_median_all_reconnect_voltages": all_reconnect_median,
            }
        )

    return site_voltages


def run_phase_a_for_site(site_id, prepared_site_days, PRated):
    """Run Phase A for one site and produce thresholds plus voltage summaries."""
    records_all = []
    for prepared_day in prepared_site_days:
        edge_result = detect_edges(prepared_day["signal_frame"], PRated)
        day_records = classify_disconnects_as_los_or_ov1(edge_result, PRated)
        if not day_records.is_empty():
            records_all.append(
                day_records.with_columns(
                    pl.lit(prepared_day["analysis_date"]).alias("event_day")
                )
            )

    site_records = (
        pl.concat(records_all, how="vertical") if records_all else pl.DataFrame()
    )
    site_voltage_values = learn_site_thresholds(site_records)
    site_level_various_voltages = pl.DataFrame(
        [{"site_id": site_id, **site_voltage_values}],
        schema=SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA,
        strict=False,
    )

    return {
        "site_thresholds": site_level_various_voltages.select(
            ["site_id", "los_threshold", "ov1_threshold"]
        ),
        "site_level_various_voltages": site_level_various_voltages,
        "records": site_records,
    }
