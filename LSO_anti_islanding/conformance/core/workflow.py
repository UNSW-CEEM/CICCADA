"""Dataset-neutral site and site-day preparation for conformance analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import polars as pl
from config import MAX_PV_SITE_NET_CIRCUITS
from core.check_pv_behaviour import CheckPVBehaviour
from core.site_day_preparation import (
    calculate_site_day_voltage_signals,
    extract_site_day,
    map_circuit_data_to_site,
    select_site_pv_data,
    trim_site_day_analysis_window,
)


@dataclass(frozen=True)
class DatasetConformanceConfig:
    """Dataset-specific configuration for the shared conformance workflow."""

    name: str
    load_inputs: Callable[..., dict[str, Any]]
    day_provider: Callable[[pl.DataFrame], list[tuple[Any, Any, Any]]]
    eligibility_function: Callable[[pl.DataFrame, pl.DataFrame], dict[str, Any]]
    capacity_estimator: Callable[..., float | None] | None
    output_dir: Path
    coverage_threshold: float
    exclusion_fields: tuple[str, ...]
    excluded_day_schema: dict[str, pl.DataType]


def build_workflow_inputs(
    site_details,
    circuit_details,
    all_data,
    *,
    rated_capacity_column="capacity_kw",
):
    """Build the standard input bundle consumed by the shared workflow."""
    required_site_columns = {"site_id", rated_capacity_column}
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

    eligible_site_ids = set(
        site_details.get_column("site_id").drop_nulls().unique().to_list()
    )
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
        "eligible_site_ids": eligible_site_ids,
        "pv_site_net_counts": pv_site_net_counts,
        "candidate_site_ids": candidate_site_ids,
    }


def collect_site_days(site_number, inputs, workflow_config):
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

    for day_key, start_day, end_day in workflow_config.day_provider(site_data):
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
        eligibility = workflow_config.eligibility_function(
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
            {field: eligibility[field] for field in workflow_config.exclusion_fields}
        )
        excluded_row["coverage_threshold_pct"] = (
            workflow_config.coverage_threshold * 100.0
        )
        excluded_day_rows.append(excluded_row)

    return {
        "eligible_day_behaviours": eligible_day_behaviours,
        "excluded_day_rows": excluded_day_rows,
        "mapped_day_count": mapped_day_count,
    }


def prepare_site(site_number, inputs, workflow_config):
    """Return one pipeline-ready site or the reason it must be skipped."""
    pv_site_net_count = inputs["pv_site_net_counts"].get(site_number, 0)

    if site_number not in inputs["eligible_site_ids"]:
        return {"site_id": site_number, "skip_reason": "not_single_inverter"}
    if pv_site_net_count == 0:
        return {"site_id": site_number, "skip_reason": "no_pv_site_net"}
    if pv_site_net_count > MAX_PV_SITE_NET_CIRCUITS:
        return {"site_id": site_number, "skip_reason": "more_than_3_pv_circuits"}

    site_day_result = collect_site_days(site_number, inputs, workflow_config)
    base_result = {
        "site_id": site_number,
        "excluded_day_rows": site_day_result["excluded_day_rows"],
    }
    if site_day_result["mapped_day_count"] == 0:
        return {**base_result, "skip_reason": "no_day_data"}

    day_behaviours = site_day_result["eligible_day_behaviours"]
    if not day_behaviours:
        return {**base_result, "skip_reason": "no_eligible_days"}

    # The shared p_rated name contains P_rated for SAPN and S_rated for SolA.
    if workflow_config.capacity_estimator is None:
        site_capacity = (
            inputs["site_details"]
            .filter(pl.col("site_id") == site_number)
            .select("capacity_kw")
        )
        p_rated = None if site_capacity.is_empty() else site_capacity["capacity_kw"][0]
    else:
        p_rated = workflow_config.capacity_estimator(
            inputs["site_details"],
            site_number,
            day_behaviours=day_behaviours,
        )
    if p_rated is None:
        return {**base_result, "skip_reason": "missing_rated_capacity"}

    return {
        **base_result,
        "skip_reason": None,
        "day_behaviours": day_behaviours,
        "p_rated": p_rated,
    }
