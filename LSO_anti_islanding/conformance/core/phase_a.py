"""Linear disconnect-edge detection, attribution, and threshold learning."""

import json
from statistics import median
from typing import Any

import polars as pl

MAX_DISCONNECT_EDGE_GAP_SECONDS = 300

SITE_LEVEL_VARIOUS_VOLTAGES_SCHEMA = {
    "site_id": pl.Int64,
    "outlier_aware_thresholds": pl.Boolean,
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
    los_lo=251.1,  # 3% 3error
    # los_lo=244, # 3% 3error
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


def _partition_voltage_intervals(values, width):
    """Greedily partition sorted voltages into strongest fixed-width intervals."""
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
                and remaining_values[right] <= start_v + width
            ):
                right += 1
            if right - left > winning_right - winning_left:
                winning_left = left
                winning_right = right

        clusters.append(remaining_values[winning_left:winning_right])
        remaining_values = (
            remaining_values[:winning_left] + remaining_values[winning_right:]
        )

    return clusters


def learn_site_thresholds(
    records: pl.DataFrame,
    *,
    outlier_aware_thresholds=False,
):
    """Learn thresholds and summarize all paired site voltages by mechanism."""
    site_voltages = {
        "outlier_aware_thresholds": bool(outlier_aware_thresholds),
    }
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

        half_volt_clusters = _partition_voltage_intervals(values, 0.5)
        winning_values = half_volt_clusters[0] if half_volt_clusters else []
        winning_median = float(median(winning_values)) if winning_values else None
        all_disconnect_median = float(median(values)) if values else None
        all_reconnect_median = (
            float(median(reconnect_values)) if reconnect_values else None
        )

        range_gated_threshold_learned = (
            len(winning_values) >= 3
            and voltage_range is not None
            and voltage_range <= 2.0
        )

        primary_cluster = []
        remaining_clusters = []
        cluster_type = "no_cluster"

        if len(winning_values) >= 3:
            largest_count = len(winning_values)
            equally_largest_clusters = [
                cluster
                for cluster in half_volt_clusters
                if len(cluster) == largest_count
            ]
            combined_equal_clusters = sorted(
                value for cluster in equally_largest_clusters for value in cluster
            )

            if (
                len(equally_largest_clusters) > 1
                and combined_equal_clusters[-1] - combined_equal_clusters[0] <= 1.0
            ):
                # Equal leading clusters that fit within 1 V form one wide cluster.
                primary_cluster = combined_equal_clusters
                remaining_clusters = half_volt_clusters[len(equally_largest_clusters) :]
                cluster_type = "wide_cluster"
            else:
                # A tie outside 1 V is resolved conservatively to the lower median.
                selected_primary_index = min(
                    (
                        index
                        for index, cluster in enumerate(half_volt_clusters)
                        if len(cluster) == largest_count
                    ),
                    key=lambda index: float(median(half_volt_clusters[index])),
                )
                primary_cluster = half_volt_clusters[selected_primary_index]
                remaining_clusters = [
                    cluster
                    for index, cluster in enumerate(half_volt_clusters)
                    if index != selected_primary_index
                ]
        else:
            one_volt_clusters = _partition_voltage_intervals(values, 1.0)
            if one_volt_clusters and len(one_volt_clusters[0]) >= 3:
                primary_cluster = one_volt_clusters[0]
                remaining_values = sorted(
                    value for cluster in one_volt_clusters[1:] for value in cluster
                )
                remaining_clusters = _partition_voltage_intervals(
                    remaining_values,
                    0.5,
                )
                cluster_type = "wide_cluster"
            else:
                remaining_clusters = half_volt_clusters

        outlier_values = sorted(
            cluster[0] for cluster in remaining_clusters if len(cluster) == 1
        )
        secondary_clusters = [
            cluster for cluster in remaining_clusters if len(cluster) >= 2
        ]
        primary_median = float(median(primary_cluster)) if primary_cluster else None

        if primary_cluster and cluster_type != "wide_cluster":
            if secondary_clusters:
                maximum_separation = max(
                    abs(float(median(cluster)) - primary_median)
                    for cluster in secondary_clusters
                )
                cluster_type = (
                    "multimodal_large_separation"
                    if maximum_separation > 2.0
                    else "multimodal"
                )
            elif outlier_values:
                cluster_type = "clustered_with_outliers"
            else:
                cluster_type = "clustered"

        if outlier_aware_thresholds:
            threshold = primary_median if primary_median is not None else default
        elif range_gated_threshold_learned:
            threshold = winning_median
        else:
            threshold = default
        secondary_cluster_summary = [
            {
                "count": len(cluster),
                "median": float(median(cluster)),
            }
            for cluster in secondary_clusters
        ]

        site_voltages.update(
            {
                f"{prefix}_threshold": threshold,
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
    outlier_aware_thresholds=False,
):
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
    site_voltage_values = learn_site_thresholds(
        site_records,
        outlier_aware_thresholds=outlier_aware_thresholds,
    )
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
            ]
        ),
        "site_level_various_voltages": site_level_various_voltages,
        "records": site_records,
    }
