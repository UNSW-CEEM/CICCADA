"""Load SAPN inputs and prepare eligible sites for the shared pipeline."""

import math
from pathlib import Path

import polars as pl

from config import (
    SAPN2022_DAY_COVERAGE_THRESHOLD,
    SAPN2022_DAY_END,
    SAPN2022_DAY_START,
    SAPN2022_EVENT_DAYS,
)
from core.check_pv_behaviour import CheckPVBehaviour
from core.site_day_preparation import (
    calculate_site_day_voltage_signals,
    map_circuit_data_to_site,
    trim_site_day_analysis_window,
)
from sapn2022_workflow.nov2022_site_day_extraction import extract_nov2022_site_day
from sapn2022_workflow.sapn_paths import (
    CIRCUIT_DETAILS_PATH,
    CLEANED_SITE_DATA_PATH,
    SITE_DETAILS_PATH,
)
from sapn2022_workflow.site_day_filtering import summarize_nov2022_day_eligibility


def load_cleaned_site_data(cleaned_path=CLEANED_SITE_DATA_PATH):
    cleaned_path = Path(cleaned_path)
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned site data at {cleaned_path}. "
            "Run run_sapn2022_preprocessing.py first."
        )
    return pl.scan_parquet(cleaned_path)


def build_site_selection_tables(site_details, circuit_details):
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
    return site_metadata_rows, pv_site_net_counts


def load_sapn2022_inputs():
    site_details = pl.read_csv(SITE_DETAILS_PATH)
    circuit_details = pl.read_csv(CIRCUIT_DETAILS_PATH)
    all_data = load_cleaned_site_data()
    site_metadata_rows, pv_site_net_counts = build_site_selection_tables(
        site_details, circuit_details
    )
    candidate_site_ids = (
        all_data.select("c_id")
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
        "all_data": all_data,
        "site_metadata_rows": site_metadata_rows,
        "pv_site_net_counts": pv_site_net_counts,
        "candidate_site_ids": candidate_site_ids,
    }


def _round_up_to_half_kw(value_kw):
    return math.ceil(value_kw * 2.0) / 2.0


def _metadata_capacity_kw(site_details, site_number):
    site_row = site_details.filter(pl.col("site_id") == site_number).select("ac_cap_w")
    if site_row.is_empty():
        return None
    ac_cap_w = site_row["ac_cap_w"][0]
    if ac_cap_w is None:
        return None
    try:
        ac_cap_kw = float(ac_cap_w) / 1000.0
    except (TypeError, ValueError):
        return None
    return ac_cap_kw if ac_cap_kw > 0 else None


def _robust_observed_peak_kw(day_behaviours):
    site_power_frames = []
    for day_info in day_behaviours or []:
        behaviour = day_info.get("behaviour")
        if behaviour is None:
            continue
        df = behaviour.circuitData
        power_cols = [
            c for c in df.columns
            if c.startswith("power")
            and not c.endswith("_next")
            and not c.endswith("_logic")
        ]
        if not power_cols:
            continue
        site_power_frames.append(
            df.select(
                pl.sum_horizontal([
                    pl.col(c).cast(pl.Float64, strict=False).fill_null(0).clip(lower_bound=0)
                    for c in power_cols
                ]).alias("site_power_kw")
            )
        )
    if not site_power_frames:
        return None, None
    site_power = pl.concat(site_power_frames, how="vertical").filter(
        pl.col("site_power_kw") > 0
    )
    if site_power.is_empty():
        return None, None
    sample_count = site_power.height
    top_n = min(sample_count, max(20, math.ceil(sample_count * 0.01)))
    top_slice = site_power.sort("site_power_kw", descending=True).head(top_n)
    return (
        top_slice.select(pl.col("site_power_kw").median()).item(),
        site_power.select(pl.col("site_power_kw").max()).item(),
    )


def rated_capacity_of_pv(
    site_details,
    site_number,
    day_behaviours=None,
    metadata_tolerance=1.10,
    fallback_kw=5.0,
):
    metadata_kw = _metadata_capacity_kw(site_details, site_number)
    robust_peak_kw, _ = _robust_observed_peak_kw(day_behaviours)
    if metadata_kw is not None:
        if robust_peak_kw is None or robust_peak_kw <= metadata_kw * metadata_tolerance:
            return metadata_kw
        return _round_up_to_half_kw(robust_peak_kw)
    if robust_peak_kw is not None:
        return _round_up_to_half_kw(robust_peak_kw)
    return fallback_kw


def collect_sapn2022_site_days(
    site_number,
    circuit_details,
    all_data,
    days_to_check=SAPN2022_EVENT_DAYS,
):
    eligible_day_behaviours = []
    excluded_day_rows = []
    mapped_day_count = 0
    local_timestamp_dtype = all_data.collect_schema()["local_tstamp"]
    local_timezone = (
        local_timestamp_dtype.time_zone
        if isinstance(local_timestamp_dtype, pl.Datetime)
        else None
    )

    for day in days_to_check:
        start_day = pl.datetime(
            2022,
            11,
            day,
            SAPN2022_DAY_START.hour,
            SAPN2022_DAY_START.minute,
            SAPN2022_DAY_START.second,
            time_zone=local_timezone,
        )
        end_day = pl.datetime(
            2022,
            11,
            day,
            SAPN2022_DAY_END.hour,
            SAPN2022_DAY_END.minute,
            SAPN2022_DAY_END.second,
            time_zone=local_timezone,
        )
        site_day_long = extract_nov2022_site_day(
            all_data,
            circuit_details,
            site_number,
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
        eligibility = summarize_nov2022_day_eligibility(
            analysis_day_long,
            analysis_day_df,
            coverage_threshold=SAPN2022_DAY_COVERAGE_THRESHOLD,
        )
        if eligibility["eligible"]:
            eligible_day_behaviours.append({
                "day": day,
                "behaviour": CheckPVBehaviour(
                    analysis_day_df,
                    volCol="voltage_valid",
                ),
            })
            continue

        excluded_day_rows.append({
            "site_id": site_number,
            "day": day,
            "reason": eligibility["reason"],
            "common_power_v10m_coverage_pct": eligibility[
                "common_power_v10m_coverage_pct"
            ],
            "rows_common_power_v10m": eligibility["rows_common_power_v10m"],
            "rows_with_power": eligibility["rows_with_power"],
            "rows_with_v10m": eligibility["rows_with_v10m"],
            "covered_seconds": eligibility["covered_seconds"],
            "window_seconds": eligibility["window_seconds"],
            "total_rows": eligibility["total_rows"],
            "coverage_threshold_pct": SAPN2022_DAY_COVERAGE_THRESHOLD * 100.0,
        })

    return {
        "eligible_day_behaviours": eligible_day_behaviours,
        "excluded_day_rows": excluded_day_rows,
        "mapped_day_count": mapped_day_count,
    }


def prepare_sapn2022_site(site_number, inputs):
    """Apply SAPN site/day policy and return one pipeline-ready site dictionary."""
    metadata_row_count = inputs["site_metadata_rows"].get(site_number, 0)
    pv_site_net_count = inputs["pv_site_net_counts"].get(site_number, 0)

    if metadata_row_count != 1:
        return {"site_id": site_number, "skip_reason": "not_single_inverter"}
    if pv_site_net_count == 0:
        return {"site_id": site_number, "skip_reason": "no_pv_site_net"}
    if pv_site_net_count > 3:
        return {"site_id": site_number, "skip_reason": "more_than_3_pv_circuits"}

    site_day_result = collect_sapn2022_site_days(
        site_number,
        inputs["circuit_details"],
        inputs["all_data"],
    )
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
