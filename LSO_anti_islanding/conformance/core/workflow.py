"""Dataset-neutral site and site-day preparation for conformance analysis."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import polars as pl
from config import MAX_PV_SITE_NET_CIRCUITS, REQUIRED_SITE_METADATA_ROWS
from core.check_pv_behaviour import CheckPVBehaviour
from core.site_day_preparation import (
    calculate_site_day_voltage_signals,
    extract_site_day,
    map_circuit_data_to_site,
    select_site_pv_data,
    trim_site_day_analysis_window,
)


@dataclass(frozen=True)
class DatasetDefinition:
    """Dataset-specific components supplied to the shared site workflow."""

    name: str
    load_inputs: Callable[[], dict[str, Any]]
    day_provider: Callable[[pl.DataFrame], list[tuple[Any, Any, Any]]]
    eligibility_function: Callable[[pl.DataFrame, pl.DataFrame], dict[str, Any]]
    output_dir: Path
    coverage_threshold: float
    exclusion_fields: tuple[str, ...]
    excluded_day_schema: dict[str, pl.DataType]


def build_workflow_inputs(site_details, circuit_details, all_data):
    """Build the standard input bundle consumed by the shared workflow."""
    required_site_columns = {"site_id", "capacity_kw"}
    required_circuit_columns = {"site_id", "c_id", "con_type"}
    missing_site_columns = required_site_columns.difference(site_details.columns)
    missing_circuit_columns = required_circuit_columns.difference(
        circuit_details.columns
    )
    if missing_site_columns:
        raise ValueError(
            "Site metadata is missing standard workflow columns: "
            f"{sorted(missing_site_columns)}"
        )
    if missing_circuit_columns:
        raise ValueError(
            "Circuit metadata is missing standard workflow columns: "
            f"{sorted(missing_circuit_columns)}"
        )

    site_metadata_rows = {
        row["site_id"]: int(row["site_metadata_rows"])
        for row in (
            site_details.group_by("site_id")
            .len()
            .rename({"len": "site_metadata_rows"})
            .to_dicts()
        )
    }
    pv_site_net_counts = {
        row["site_id"]: int(row["pv_site_net_count"])
        for row in (
            circuit_details.filter(pl.col("con_type") == "pv_site_net")
            .group_by("site_id")
            .len()
            .rename({"len": "pv_site_net_count"})
            .to_dicts()
        )
    }
    all_data_ldf = all_data if isinstance(all_data, pl.LazyFrame) else all_data.lazy()
    candidate_site_ids = (
        all_data_ldf.select("c_id")
        .unique()
        .join(
            circuit_details.select(["c_id", "site_id"]).unique().lazy(),
            on="c_id",
            how="inner",
        )
        .select("site_id")
        .unique()
        .collect()["site_id"]
        .to_list()
    )
    return {
        "site_details": site_details,
        "circuit_details": circuit_details,
        "all_data": all_data_ldf,
        "site_metadata_rows": site_metadata_rows,
        "pv_site_net_counts": pv_site_net_counts,
        "candidate_site_ids": candidate_site_ids,
    }


def _round_up_to_half_kw(value_kw):
    return math.ceil(value_kw * 2.0) / 2.0


def _metadata_capacity_kw(site_details, site_number):
    site_row = site_details.filter(pl.col("site_id") == site_number).select(
        "capacity_kw"
    )
    if site_row.is_empty():
        return None
    capacity_kw = site_row["capacity_kw"][0]
    if capacity_kw is None:
        return None
    try:
        capacity_kw = float(capacity_kw)
    except (TypeError, ValueError):
        return None
    return capacity_kw if capacity_kw > 0 else None


def _robust_observed_peak_kw(day_behaviours):
    site_power_frames = []
    for day_info in day_behaviours or []:
        behaviour = day_info.get("behaviour")
        if behaviour is None:
            continue
        df = behaviour.circuitData
        power_cols = [
            column
            for column in df.columns
            if column.startswith("power")
            and not column.endswith("_next")
            and not column.endswith("_logic")
        ]
        if not power_cols:
            continue
        complete_power = pl.all_horizontal(
            [pl.col(column).is_not_null() for column in power_cols]
        )
        site_power_frames.append(
            df.filter(complete_power).select(
                pl.sum_horizontal(
                    [
                        pl.col(column)
                        .cast(pl.Float64, strict=False)
                        .clip(lower_bound=0)
                        for column in power_cols
                    ]
                ).alias("site_power_kw")
            )
        )
    if not site_power_frames:
        return None
    site_power = pl.concat(site_power_frames, how="vertical").filter(
        pl.col("site_power_kw") > 0
    )
    if site_power.is_empty():
        return None
    sample_count = site_power.height
    top_n = min(sample_count, max(20, math.ceil(sample_count * 0.01)))
    return (
        site_power.sort("site_power_kw", descending=True)
        .head(top_n)
        .select(pl.col("site_power_kw").median())
        .item()
    )


def rated_capacity_of_pv(
    site_details,
    site_number,
    day_behaviours=None,
    metadata_tolerance=1.10,
    fallback_kw=5.0,
):
    """Apply the shared metadata/observed-power rated-capacity policy."""
    metadata_kw = _metadata_capacity_kw(site_details, site_number)
    robust_peak_kw = _robust_observed_peak_kw(day_behaviours)
    if metadata_kw is not None:
        if robust_peak_kw is None or robust_peak_kw <= metadata_kw * metadata_tolerance:
            return metadata_kw
        return _round_up_to_half_kw(robust_peak_kw)
    if robust_peak_kw is not None:
        return _round_up_to_half_kw(robust_peak_kw)
    return fallback_kw


def collect_site_days(site_number, inputs, definition):
    """Prepare every configured or discovered day for one site."""
    eligible_day_behaviours = []
    excluded_day_rows = []
    mapped_day_count = 0
    site_data = select_site_pv_data(
        inputs["all_data"],
        inputs["circuit_details"],
        site_number,
    )
    if site_data.is_empty():
        return {
            "eligible_day_behaviours": eligible_day_behaviours,
            "excluded_day_rows": excluded_day_rows,
            "mapped_day_count": mapped_day_count,
        }

    for day_key, start_day, end_day in definition.day_provider(site_data):
        site_day_long = extract_site_day(
            site_data,
            start_day,
            end_day,
        )
        if site_day_long.is_empty():
            continue

        mapped_day_count += 1
        wide = map_circuit_data_to_site(site_day_long, site_number)
        prepared_day_df = calculate_site_day_voltage_signals(
            wide,
            voltage_prefix="voltage_valid",
        )
        analysis_day_long = trim_site_day_analysis_window(site_day_long)
        analysis_day_df = trim_site_day_analysis_window(prepared_day_df)
        eligibility = definition.eligibility_function(
            analysis_day_long,
            analysis_day_df,
        )
        if eligibility["eligible"]:
            eligible_day_behaviours.append(
                {
                    "day": day_key,
                    "behaviour": CheckPVBehaviour(
                        analysis_day_df,
                        volCol="voltage_valid",
                    ),
                }
            )
            continue

        excluded_row = {
            "site_id": site_number,
            "day": day_key,
            "reason": eligibility["reason"],
        }
        excluded_row.update(
            {field: eligibility[field] for field in definition.exclusion_fields}
        )
        excluded_row["coverage_threshold_pct"] = definition.coverage_threshold * 100.0
        excluded_day_rows.append(excluded_row)

    return {
        "eligible_day_behaviours": eligible_day_behaviours,
        "excluded_day_rows": excluded_day_rows,
        "mapped_day_count": mapped_day_count,
    }


def prepare_site(site_number, inputs, definition):
    """Return one pipeline-ready site or the reason it must be skipped."""
    metadata_row_count = inputs["site_metadata_rows"].get(site_number, 0)
    pv_site_net_count = inputs["pv_site_net_counts"].get(site_number, 0)

    if metadata_row_count != REQUIRED_SITE_METADATA_ROWS:
        return {"site_id": site_number, "skip_reason": "not_single_inverter"}
    if pv_site_net_count == 0:
        return {"site_id": site_number, "skip_reason": "no_pv_site_net"}
    if pv_site_net_count > MAX_PV_SITE_NET_CIRCUITS:
        return {"site_id": site_number, "skip_reason": "more_than_3_pv_circuits"}

    site_day_result = collect_site_days(site_number, inputs, definition)
    base_result = {
        "site_id": site_number,
        "excluded_day_rows": site_day_result["excluded_day_rows"],
    }
    if site_day_result["mapped_day_count"] == 0:
        return {**base_result, "skip_reason": "no_day_data"}

    day_behaviours = site_day_result["eligible_day_behaviours"]
    if not day_behaviours:
        return {**base_result, "skip_reason": "no_eligible_days"}

    return {
        **base_result,
        "skip_reason": None,
        "day_behaviours": day_behaviours,
        "p_rated": rated_capacity_of_pv(
            inputs["site_details"],
            site_number,
            day_behaviours=day_behaviours,
        ),
    }
