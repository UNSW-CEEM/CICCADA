"""Parallel site-level conformance processing for the analysis notebook."""

import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
from as_4777_conformance import (
    combined_conformance,
    combined_conformance_with_voltage_tolerance,
)
from csip_conformance import has_usable_csip_export_limit
from reporting import summarise_csip_filtered_volt_watt, summarise_site_conformance

_NETWORK_TIMEZONES = {
    "SAPN": "Australia/Adelaide",
    "AUSNET": "Australia/Melbourne",
    "CPU": "Australia/Melbourne",
}


def _limit_to_first_eligible_plot_days(
    site_results: pl.DataFrame,
    network: str,
    day_limit: int,
) -> pl.DataFrame:
    """Return the rows for the first eligible local plot days."""
    timezone_name = _NETWORK_TIMEZONES[network]
    local_date = (
        pl.col("timestamp")
        .dt.convert_time_zone(timezone_name)
        .dt.date()
        .alias("_local_date")
    )
    eligible_columns = (
        "passive_anti_islanding_conformant",
        "sustained_operation_conformant",
        "volt_watt_conformant",
        "volt_var_conformant",
    )
    eligible_dates = (
        site_results.with_columns(local_date)
        .filter(
            pl.col("timestamp")
            .dt.convert_time_zone(timezone_name)
            .dt.time()
            .is_between(dt.time(6), dt.time(18), closed="both")
        )
        .filter(
            pl.any_horizontal(
                [pl.col(column).is_not_null() for column in eligible_columns]
            )
        )
        .select("_local_date")
        .unique()
        .sort("_local_date")
        .head(day_limit)
    )
    return (
        site_results.with_columns(local_date)
        .join(eligible_dates, on="_local_date", how="semi")
        .drop("_local_date")
    )


def _process_site_conformance(
    site_metadata: dict[str, Any],
    dataset_directory: str,
    output_directory: str,
    create_plots: bool,
    apply_site_valid_filter: bool,
    include_csip_filtered_summary: bool,
    use_voltage_tolerance: bool,
    vw_require_measured_above_threshold: bool,
    volt_var_method: str,
    plot_day_limit: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, int]:
    """Calculate and optionally plot one site inside a worker process."""
    if create_plots:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from plotting import plot_site_conformance_days

    pseudonym = site_metadata["pseudonym"]
    telemetry_path = (
        Path(dataset_directory)
        / site_metadata["folder"]
        / f"{pseudonym}_telemetry_60sec.parquet"
    )
    site_telemetry = pl.read_parquet(telemetry_path).sort("timestamp")

    csip_export_limit_usable = False
    if site_metadata["solar_coupling"] == "dc_only":
        csip_export_limit_usable = has_usable_csip_export_limit(site_telemetry)
        site_telemetry = site_telemetry.with_columns(
            pl.col("csip_export_limit_w")
            .forward_fill()
            .alias("csip_export_limit_w_ffill")
        )

    if apply_site_valid_filter:
        site_tolerance_w = site_metadata["calc_capacity"] * 1000 * 0.04
        site_inv_power_diff_w = (
            site_telemetry["pw_dc_pv_power"]
            + site_telemetry["battery_power"]
            - site_telemetry["inverter_ac_real_power"]
        )
        site_solar_difference_w = (
            site_telemetry["pw_dc_pv_power"] - site_telemetry["site_solar_power"]
        )
        common_valid_mask = (
            site_inv_power_diff_w.is_not_null()
            & site_solar_difference_w.is_not_null()
            & (site_inv_power_diff_w.abs() <= site_tolerance_w)
        )
        if site_metadata["solar_coupling"] == "dc_only":
            solar_valid_mask = site_solar_difference_w.abs() <= site_tolerance_w
        elif site_metadata["solar_coupling"] == "ac_coupled":
            solar_valid_mask = site_solar_difference_w <= site_tolerance_w
        else:
            raise ValueError(
                "apply_site_valid_filter requires solar_coupling to be "
                "'dc_only' or 'ac_coupled', got "
                f"{site_metadata['solar_coupling']!r}"
            )
        site_valid_mask = common_valid_mask & solar_valid_mask
        site_telemetry = site_telemetry.filter(site_valid_mask)

    if use_voltage_tolerance:
        site_results = combined_conformance_with_voltage_tolerance(
            site_telemetry,
            site_metadata["calc_capacity"],
            site_metadata["solar_coupling"],
            csip_export_limit_usable=csip_export_limit_usable,
            vw_require_measured_above_threshold=(
                vw_require_measured_above_threshold
            ),
        )
    else:
        site_results = combined_conformance(
            site_telemetry,
            site_metadata["calc_capacity"],
            site_metadata["solar_coupling"],
            csip_export_limit_usable=csip_export_limit_usable,
            volt_var_method=volt_var_method,
        )
    site_results = site_results.with_columns(
        pl.lit(pseudonym).alias("pseudonym"),
        pl.lit(site_metadata["solar_coupling"]).alias("solar_coupling"),
    )
    site_summary = summarise_site_conformance(site_results)
    csip_filtered_summary = (
        summarise_csip_filtered_volt_watt(site_results)
        if include_csip_filtered_summary
        else None
    )

    plot_count = 0
    if create_plots:
        plot_results = site_results
        if plot_day_limit is not None:
            plot_results = _limit_to_first_eligible_plot_days(
                plot_results,
                site_metadata["folder"],
                plot_day_limit,
            )
        plot_count = plot_site_conformance_days(
            plot_results,
            site_summary,
            site_metadata,
            output_directory=output_directory,
        )

    return site_summary, csip_filtered_summary, plot_count


def run_site_conformance_batch(
    site_data: pl.DataFrame | list[dict[str, Any]],
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    max_workers: int = 2,
    create_plots: bool = True,
    apply_site_valid_filter: bool = False,
    include_csip_filtered_summary: bool = False,
    use_voltage_tolerance: bool = False,
    vw_require_measured_above_threshold: bool = True,
    volt_var_method: str = "standard",
    plot_day_limit: int | None = None,
    show_progress: bool = True,
) -> (
    tuple[list[pl.DataFrame], int]
    | tuple[list[pl.DataFrame], list[pl.DataFrame], int]
):
    """Process independent sites concurrently and preserve their input order.

    Set ``apply_site_valid_filter`` true to apply the
    inverter-battery-balance and solar-consistency mask before conformance assessment.
    Set ``include_csip_filtered_summary`` true to return separate CSIP-filtered
    Volt-Watt summaries.
    Set ``use_voltage_tolerance`` true to use the voltage-tolerance workflow.
    ``vw_require_measured_above_threshold`` applies only to that workflow.
    """
    site_rows = (
        site_data.to_dicts() if isinstance(site_data, pl.DataFrame) else list(site_data)
    )
    if not site_rows:
        if include_csip_filtered_summary:
            return [], [], 0
        return [], 0
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if volt_var_method not in {"standard", "min_cap_curve"}:
        raise ValueError(
            "volt_var_method must be 'standard' or 'min_cap_curve', "
            f"got {volt_var_method!r}"
        )

    ordered_results: list[
        tuple[pl.DataFrame, pl.DataFrame | None, int] | None
    ] = [None] * len(site_rows)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_indexes = {
            executor.submit(
                _process_site_conformance,
                site,
                str(dataset_directory),
                str(output_directory),
                create_plots,
                apply_site_valid_filter,
                include_csip_filtered_summary,
                use_voltage_tolerance,
                vw_require_measured_above_threshold,
                volt_var_method,
                plot_day_limit,
            ): index
            for index, site in enumerate(site_rows)
        }
        completed_count = 0
        for future in as_completed(future_indexes):
            index = future_indexes[future]
            ordered_results[index] = future.result()
            completed_count += 1
            if show_progress:
                print(
                    f"Completed {site_rows[index]['pseudonym']} "
                    f"({completed_count}/{len(site_rows)})"
                )

    completed_results = [result for result in ordered_results if result is not None]
    summaries = [summary for summary, _, _ in completed_results]
    total_plot_count = sum(plot_count for _, _, plot_count in completed_results)
    if include_csip_filtered_summary:
        csip_filtered_summaries = [
            summary
            for _, summary, _ in completed_results
            if summary is not None
        ]
        return summaries, csip_filtered_summaries, total_plot_count
    return summaries, total_plot_count
