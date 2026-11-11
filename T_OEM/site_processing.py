"""Parallel site-level conformance processing for the analysis notebook."""

import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
from as_4777_conformance import combined_conformance
from csip_conformance import has_usable_csip_export_limit
from reporting import summarise_site_conformance

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
    volt_var_method: str,
    plot_day_limit: int | None = None,
) -> tuple[pl.DataFrame, int]:
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

    csip_export_limit_usable = True
    if site_metadata["solar_coupling"] == "dc_only":
        csip_export_limit_usable = has_usable_csip_export_limit(site_telemetry)
        site_telemetry = site_telemetry.with_columns(
            pl.col("csip_export_limit_w")
            .forward_fill()
            .alias("csip_export_limit_w_ffill")
        )

    site_results = combined_conformance(
        site_telemetry,
        site_metadata["calc_capacity"],
        site_metadata["solar_coupling"],
        csip_export_limit_usable=csip_export_limit_usable,
        volt_var_method=volt_var_method,
    ).with_columns(
        pl.lit(pseudonym).alias("pseudonym"),
        pl.lit(site_metadata["solar_coupling"]).alias("solar_coupling"),
    )
    site_summary = summarise_site_conformance(site_results)

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

    return site_summary, plot_count


def run_site_conformance_batch(
    sites: pl.DataFrame | list[dict[str, Any]],
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    max_workers: int = 2,
    create_plots: bool = True,
    volt_var_method: str = "standard",
    plot_day_limit: int | None = None,
    show_progress: bool = True,
) -> tuple[list[pl.DataFrame], int]:
    """Process independent sites concurrently and preserve their input order."""
    site_rows = sites.to_dicts() if isinstance(sites, pl.DataFrame) else list(sites)
    if not site_rows:
        return [], 0
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if volt_var_method not in {"standard", "min_cap_curve"}:
        raise ValueError(
            "volt_var_method must be 'standard' or 'min_cap_curve', "
            f"got {volt_var_method!r}"
        )

    ordered_results: list[tuple[pl.DataFrame, int] | None] = [None] * len(
        site_rows
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_indexes = {
            executor.submit(
                _process_site_conformance,
                site,
                str(dataset_directory),
                str(output_directory),
                create_plots,
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
    summaries = [summary for summary, _ in completed_results]
    total_plot_count = sum(plot_count for _, plot_count in completed_results)
    return summaries, total_plot_count
