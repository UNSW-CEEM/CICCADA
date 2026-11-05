"""Linear disconnect-edge detection, attribution, and threshold learning."""

import json
from statistics import median
from typing import Any

import polars as pl

SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA = {
    "site_id": pl.Int64,
    "los_threshold": pl.Float64,
    "los_lowest_disconnect_voltage": pl.Float64,
    "los_median_all_disconnect_voltages": pl.Float64,
    "los_lowest_reconnect_voltage": pl.Float64,
    "los_median_all_reconnect_voltages": pl.Float64,
    "los_cluster_type": pl.Utf8,
    "los_primary_cluster_count": pl.Int64,
    "los_outlier_voltages": pl.Utf8,
    "los_secondary_clusters": pl.Utf8,
    "ov1_threshold": pl.Float64,
    "ov1_lowest_disconnect_voltage": pl.Float64,
    "ov1_median_all_disconnect_voltages": pl.Float64,
    "ov1_lowest_reconnect_voltage": pl.Float64,
    "ov1_median_all_reconnect_voltages": pl.Float64,
    "ov1_cluster_type": pl.Utf8,
    "ov1_primary_cluster_count": pl.Int64,
    "ov1_outlier_voltages": pl.Utf8,
    "ov1_secondary_clusters": pl.Utf8,
}


def detect_edges(
    signal_frame: pl.DataFrame,
    PRated,
    *,
    max_disconnect_edge_gap_seconds=60,
):
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
            pl.col("vinst_max").shift(1).alias("vinst_max_previous"),
            (
                (~pl.col("is_disc").shift(1))
                & pl.col("is_disc")
                & (pl.col("site_power_drop") >= p_step_strict)
                & (pl.col("dt_next_s").shift(1) > 0)
                & (
                    pl.col("dt_next_s").shift(1)
                    <= max_disconnect_edge_gap_seconds
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
                "vinst_max_previous",
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


def attribute_edge_dynamics(
    v10m,
    vinst,
    vinst_previous,
    *,
    los_lo,
    los_hi_cap,
    ov1_candidate_lo,
    ov1_jump_min,
):
    """Attribute one disconnect using its instantaneous-voltage change."""
    vinst_jump = (
        None
        if vinst is None or vinst_previous is None
        else float(vinst) - float(vinst_previous)
    )

    if vinst is not None and vinst > los_hi_cap:
        return "OV1", vinst, False
    if (
        vinst is not None
        and vinst > ov1_candidate_lo
        and vinst_jump is not None
        and vinst_jump >= ov1_jump_min
    ):
        return "OV1", vinst, False
    if v10m is not None and los_lo <= v10m <= los_hi_cap:
        return "LOS", v10m, False
    return None, None, False


def classify_disconnects_as_los_or_ov1(
    edge_result,
    PRated,
    *,
    los_lo=251.1,  # 3% 3error
    # los_lo=244, # 3% 3error
    los_hi_strict=259.0,
    los_hi_cap=260.3,
    attribution_method="grey_band",
    ov1_candidate_lo=258.0,
    ov1_jump_min=2.0, # can be modified if you want to modify the OV jump
):
    """Classify one day's disconnect edges and pair the next reconnect voltage."""
    if attribution_method not in {"grey_band", "edge_dynamics"}:
        raise KeyError(f"Unknown Phase A attribution method: {attribution_method}")

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
        "disconnect_voltage": pl.Float64,
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

        if attribution_method == "grey_band":
            # uncomment the following block for inclduding grey band
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

            # comment below and uncoment above to implement grey band
            # if vinst is not None and vinst >= los_hi_strict:
            #     mechanism = "OV1"
            #     disconnect_voltage = vinst
            # elif v10m is not None and (los_lo <= v10m <= los_hi_strict):
            #     mechanism = "LOS"
            #     disconnect_voltage = v10m
            ####
        else:
            mechanism, disconnect_voltage, grey_non_sustained = attribute_edge_dynamics(
                v10m,
                vinst,
                row["vinst_max_previous"],
                los_lo=los_lo,
                los_hi_cap=los_hi_cap,
                ov1_candidate_lo=ov1_candidate_lo,
                ov1_jump_min=ov1_jump_min,
            )

        if mechanism is None or disconnect_voltage is None:
            mechanism = "unclassified"

        reconnect = next(
            (record for record in reconnect_list if record["local_tstamp"] > tdisc),
            None,
        )
        reconnect_voltage = None
        if reconnect is not None:
            if mechanism == "LOS":
                reconnect_voltage = reconnect["v10m_avg"]
            elif mechanism == "OV1":
                reconnect_voltage = reconnect["vinst_max"]

        records.append(
            {
                "site_id": row["site_id"],
                "event_id": event_idx,
                "ts_disc": tdisc,
                "edge_source": row["edge_source"],
                "mechanism": mechanism,
                "disconnect_voltage": disconnect_voltage,
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
    cluster_span_volts = 4.6
    comparison_tolerance = 1e-9
    site_voltages = {}
    for mechanism, prefix, default in (
        ("LOS", "los", 258.0),
        ("OV1", "ov1", 265.0),
    ):
        if records.is_empty():
            values = []
            reconnect_values = []
        else:
            mechanism_records = records.filter(pl.col("mechanism") == mechanism)
            values = (
                mechanism_records.get_column("disconnect_voltage")
                .drop_nulls()
                .to_list()
            )
            reconnect_values = (
                mechanism_records.get_column("reconnect_voltage").drop_nulls().to_list()
            )

        values = sorted(float(value) for value in values)
        reconnect_values = sorted(float(value) for value in reconnect_values)

        remaining_values = list(values)
        clusters = []
        while remaining_values:
            winning_left = 0
            winning_right = 1
            right = 0
            for left, start_v in enumerate(remaining_values):
                right = max(right, left)
                while (
                    right < len(remaining_values)
                    and remaining_values[right]
                    <= start_v + cluster_span_volts + comparison_tolerance
                ):
                    right += 1
                if right - left > winning_right - winning_left:
                    winning_left = left
                    winning_right = right

            clusters.append(remaining_values[winning_left:winning_right])
            remaining_values = (
                remaining_values[:winning_left]
                + remaining_values[winning_right:]
            )

        multi_value_clusters = [cluster for cluster in clusters if len(cluster) >= 2]
        primary_cluster = multi_value_clusters[0] if multi_value_clusters else []
        secondary_clusters = multi_value_clusters[1:]
        outlier_values = sorted(cluster[0] for cluster in clusters if len(cluster) == 1)

        if not primary_cluster:
            cluster_type = "no_cluster"
        elif secondary_clusters:
            cluster_type = "multimodal"
        elif outlier_values:
            cluster_type = "clustered_with_outliers"
        else:
            cluster_type = "clustered"

        secondary_cluster_summary = [
            {
                "count": len(cluster),
                "median": float(median(cluster)),
            }
            for cluster in secondary_clusters
        ]

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

        site_voltages.update(
            {
                f"{prefix}_threshold": (
                    float(median(primary_cluster)) if primary_cluster else default
                ),
                f"{prefix}_lowest_disconnect_voltage": (
                    min(values) if values else None
                ),
                f"{prefix}_median_all_disconnect_voltages": all_disconnect_median,
                f"{prefix}_lowest_reconnect_voltage": (
                    min(reconnect_values) if reconnect_values else None
                ),
                f"{prefix}_median_all_reconnect_voltages": all_reconnect_median,
                f"{prefix}_cluster_type": cluster_type,
                f"{prefix}_primary_cluster_count": len(primary_cluster),
                f"{prefix}_outlier_voltages": json.dumps(outlier_values),
                f"{prefix}_secondary_clusters": json.dumps(
                    secondary_cluster_summary,
                    separators=(",", ":"),
                ),
            }
        )

    return site_voltages


def run_phase_a_for_site(
    site_id,
    prepared_site_days,
    PRated,
    *,
    max_disconnect_edge_gap_seconds=60,
    attribution_method="grey_band",
):
    """Run Phase A for one site and produce thresholds plus voltage summaries."""
    records_all = []
    for prepared_day in prepared_site_days:
        edge_result = detect_edges(
            prepared_day["signal_frame"],
            PRated,
            max_disconnect_edge_gap_seconds=max_disconnect_edge_gap_seconds,
        )
        day_records = classify_disconnects_as_los_or_ov1(
            edge_result,
            PRated,
            attribution_method=attribution_method,
        )
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
            [
                "site_id",
                "los_threshold",
                "ov1_threshold",
                "los_lowest_disconnect_voltage",
                "ov1_lowest_disconnect_voltage",
                "los_cluster_type",
                "ov1_cluster_type",
            ]
        ),
        "site_level_various_voltages": site_level_various_voltages,
        "records": site_records,
    }
