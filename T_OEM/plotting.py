"""Site-level plots for the coordinated AS/NZS 4777 and CSIP analysis."""

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Patch


def _calculate_day_conformance_percentage(
    conformant: int, eligible: int
) -> float | None:
    """Return the daily conformant percentage, or None when none are eligible."""
    if eligible == 0:
        return None
    return conformant / eligible * 100


def plot_site_conformance_days(
    site_conformance_results: pl.DataFrame,
    site_conformance_performance: pl.DataFrame,
    site_metadata: dict[str, object],
    output_directory: str | Path = "site_level_plots",
) -> int:
    """Save one site's 06:00-18:00 plots by applicable mechanism."""
    output_directory = Path(output_directory)
    performance = site_conformance_performance.row(0, named=True)
    network_timezones = (
        ("SAPN", "Australia/Adelaide"),
        ("AUSNET", "Australia/Melbourne"),
        ("CPU", "Australia/Melbourne"),
    )
    plot_count = 0

    for network, timezone_name in network_timezones:
        if site_metadata["folder"] != network:
            continue
        local_results = (
            site_conformance_results.lazy()
            .select(
                "timestamp",
                "pseudonym",
                "solar_coupling",
                "grid_voltage",
                "grid_voltage_10m_avg",
                "inverter_ac_real_power",
                "site_instant_power",
                (
                    pl.col("csip_export_limit_w_ffill")
                    if performance["solar_coupling"] == "dc_only"
                    else pl.lit(None, dtype=pl.Float64).alias(
                        "csip_export_limit_w_ffill"
                    )
                ),
                "passive_anti_islanding_conformant",
                "sustained_operation_conformant",
                "volt_watt_conformant",
                "csip_conformant",
            )
            .with_columns(
                pl.lit(site_metadata["calc_capacity"]).alias("calc_capacity"),
                pl.col("timestamp")
                .dt.convert_time_zone(timezone_name)
                .alias("local_timestamp"),
            )
            .filter(
                pl.col("local_timestamp")
                .dt.time()
                .is_between(dt.time(6), dt.time(18), closed="both")
            )
            .with_columns(pl.col("local_timestamp").dt.date().alias("local_date"))
        )
        eligible_site_days = (
            local_results.filter(
                pl.any_horizontal(
                    pl.col("passive_anti_islanding_conformant").is_not_null(),
                    pl.col("sustained_operation_conformant").is_not_null(),
                    pl.col("volt_watt_conformant").is_not_null(),
                )
            )
            .select("pseudonym", "local_date")
            .unique()
        )
        plot_rows = (
            local_results.join(
                eligible_site_days,
                on=["pseudonym", "local_date"],
                how="semi",
            )
            .sort("pseudonym", "local_date", "local_timestamp")
            .collect()
        )

        for group_key, site_day in plot_rows.group_by(
            ["pseudonym", "local_date"], maintain_order=True
        ):
            pseudonym, local_date = group_key
            passive_site_eligible = performance[
                "passive_anti_islanding_num_eligible_timestamps"
            ]
            passive_site_conformant = performance[
                "passive_anti_islanding_num_conformant_timestamps"
            ]
            passive_site_percentage = performance[
                "passive_anti_islanding_conformant_percentage"
            ]
            sustained_site_eligible = performance[
                "sustained_operation_num_eligible_timestamps"
            ]
            sustained_site_conformant = performance[
                "sustained_operation_num_conformant_timestamps"
            ]
            sustained_site_percentage = performance[
                "sustained_operation_conformant_percentage"
            ]
            volt_watt_site_eligible = performance[
                "volt_watt_num_eligible_timestamps"
            ]
            volt_watt_site_conformant = performance[
                "volt_watt_num_conformant_timestamps"
            ]
            volt_watt_site_percentage = performance[
                "volt_watt_conformant_percentage"
            ]
            csip_site_eligible = performance["csip_num_eligible_timestamps"]
            csip_site_conformant = performance[
                "csip_num_conformant_timestamps"
            ]
            csip_site_percentage = performance["csip_conformant_percentage"]
            capacity = site_day["calc_capacity"][0]

            site_day = site_day.sort("local_timestamp").with_columns(
                (
                    pl.col("timestamp")
                    .diff()
                    .dt.total_seconds()
                    .fill_null(0)
                    .gt(60)
                    .cast(pl.Int64)
                    .cum_sum()
                ).alias("plot_segment"),
                (pl.col("inverter_ac_real_power") / 1000).alias(
                    "inverter_ac_real_power_kw"
                ),
                (-pl.col("site_instant_power"))
                .clip(lower_bound=0)
                .truediv(1000)
                .alias("site_export_power_kw"),
                (pl.col("csip_export_limit_w_ffill") / 1000).alias(
                    "csip_export_limit_kw"
                ),
                pl.when(pl.col("volt_watt_conformant").is_not_null())
                .then(
                    pl.col("calc_capacity")
                    * pl.when(pl.col("grid_voltage") < 260.0)
                    .then(
                        1.0
                        - (pl.col("grid_voltage") - 253.0)
                        * (0.8 / 7.0)
                    )
                    .otherwise(0.2)
                    + 0.04 * pl.col("calc_capacity")
                )
                .otherwise(None)
                .alias("volt_watt_ceiling_kw"),
            )
            passive_eligible = int(
                site_day["passive_anti_islanding_conformant"].is_not_null().sum()
            )
            passive_conformant = int(
                site_day["passive_anti_islanding_conformant"].fill_null(False).sum()
            )
            sustained_eligible = int(
                site_day["sustained_operation_conformant"].is_not_null().sum()
            )
            sustained_conformant = int(
                site_day["sustained_operation_conformant"].fill_null(False).sum()
            )
            volt_watt_eligible = int(
                site_day["volt_watt_conformant"].is_not_null().sum()
            )
            volt_watt_conformant = int(
                site_day["volt_watt_conformant"].fill_null(False).sum()
            )
            csip_day_eligible = int(site_day["csip_conformant"].is_not_null().sum())
            csip_day_conformant = int(
                site_day["csip_conformant"].fill_null(False).sum()
            )
            passive_day_percentage = _calculate_day_conformance_percentage(
                passive_conformant, passive_eligible
            )
            sustained_day_percentage = _calculate_day_conformance_percentage(
                sustained_conformant, sustained_eligible
            )
            volt_watt_day_percentage = _calculate_day_conformance_percentage(
                volt_watt_conformant, volt_watt_eligible
            )
            csip_day_percentage = _calculate_day_conformance_percentage(
                csip_day_conformant, csip_day_eligible
            )
            save_directories = []
            for mechanism, day_eligible in (
                ("volt_watt", volt_watt_eligible),
                ("sustained_operation", sustained_eligible),
                ("passive_anti_islanding", passive_eligible),
            ):
                if day_eligible == 0:
                    continue
                mechanism_eligible = performance[
                    f"{mechanism}_num_eligible_timestamps"
                ]
                mechanism_percentage = performance[
                    f"{mechanism}_conformant_percentage"
                ]
                if mechanism_eligible < 20:
                    mechanism_status = "insufficient_evidence"
                elif mechanism_percentage >= 90:
                    mechanism_status = "conformant"
                else:
                    mechanism_status = "non_conformant"
                save_directories.append(
                    output_directory / mechanism / mechanism_status
                )

            figure, (voltage_axis, power_axis) = plt.subplots(
                2,
                1,
                figsize=(15, 9),
                sharex=True,
                height_ratios=(1, 1),
            )
            has_cessation_shading = False
            has_volt_watt_assessment = False
            has_volt_watt_ceiling = False
            has_volt_watt_conformant = False
            has_volt_watt_non_conformant = False
            for segment_number, segment in enumerate(
                site_day.partition_by("plot_segment", maintain_order=True)
            ):
                timestamps = segment["local_timestamp"].to_list()
                cessation_mask = (
                    segment["passive_anti_islanding_conformant"].is_not_null()
                    | segment["sustained_operation_conformant"].is_not_null()
                ).to_numpy()
                volt_watt_mask = (
                    segment["volt_watt_conformant"].is_not_null().to_numpy()
                )
                volt_watt_conformant_mask = (
                    segment["volt_watt_conformant"].fill_null(False).to_numpy()
                )
                volt_watt_non_conformant_mask = (
                    ~segment["volt_watt_conformant"].fill_null(True)
                ).to_numpy()
                displayed_volt_watt_conformant_mask = (
                    volt_watt_conformant_mask & ~cessation_mask
                )
                displayed_volt_watt_non_conformant_mask = (
                    volt_watt_non_conformant_mask & ~cessation_mask
                )
                segment_has_volt_watt = bool(volt_watt_mask.any())
                has_volt_watt_assessment = (
                    has_volt_watt_assessment or segment_has_volt_watt
                )
                has_volt_watt_conformant = has_volt_watt_conformant or bool(
                    displayed_volt_watt_conformant_mask.any()
                )
                has_volt_watt_non_conformant = (
                    has_volt_watt_non_conformant
                    or bool(displayed_volt_watt_non_conformant_mask.any())
                )
                has_cessation_shading = has_cessation_shading or bool(
                    cessation_mask.any()
                )
                for shading_mask, shading_color, shading_alpha, shading_axes in (
                    (
                        cessation_mask,
                        "#7c3aed",
                        0.18,
                        (voltage_axis, power_axis),
                    ),
                    (volt_watt_mask, "#0891b2", 0.18, (voltage_axis,)),
                    (
                        displayed_volt_watt_conformant_mask,
                        "#16a34a",
                        0.14,
                        (power_axis,),
                    ),
                    (
                        displayed_volt_watt_non_conformant_mask,
                        "#dc2626",
                        0.20,
                        (power_axis,),
                    ),
                ):
                    shading_start = None
                    for mask_index, applicable in enumerate(shading_mask):
                        if applicable and shading_start is None:
                            shading_start = timestamps[mask_index]
                        if shading_start is not None and (
                            not applicable or mask_index == len(shading_mask) - 1
                        ):
                            if applicable:
                                final_interval = (
                                    timestamps[mask_index] - timestamps[mask_index - 1]
                                    if mask_index
                                    else dt.timedelta(minutes=1)
                                )
                                shading_end = timestamps[mask_index] + final_interval
                            else:
                                shading_end = timestamps[mask_index]
                            for axis in shading_axes:
                                axis.axvspan(
                                    shading_start,
                                    shading_end,
                                    color=shading_color,
                                    alpha=shading_alpha,
                                    zorder=0,
                                    linewidth=0,
                                )
                            shading_start = None
                voltage_axis.plot(
                    timestamps,
                    segment["grid_voltage"].to_list(),
                    color="#b45309",
                    linewidth=1.2,
                    label=(
                        "Inverter grid voltage" if segment_number == 0 else "_nolegend_"
                    ),
                )
                voltage_axis.plot(
                    timestamps,
                    segment["grid_voltage_10m_avg"].to_list(),
                    color="#1a1a1a",
                    linewidth=1.8,
                    linestyle="--",
                    label=(
                        "10-minute rolling voltage"
                        if segment_number == 0
                        else "_nolegend_"
                    ),
                )
                power_axis.plot(
                    timestamps,
                    segment["inverter_ac_real_power_kw"].to_list(),
                    color="#1565c0",
                    linewidth=1.3,
                    label=(
                        "Inverter AC real power"
                        if segment_number == 0
                        else "_nolegend_"
                    ),
                )
                if segment_has_volt_watt:
                    power_axis.plot(
                        timestamps,
                        segment["volt_watt_ceiling_kw"].to_list(),
                        color="#d97706",
                        linewidth=1.6,
                        linestyle="-",
                        label=(
                            "Volt-Watt ceiling"
                            if not has_volt_watt_ceiling
                            else "_nolegend_"
                        ),
                    )
                    has_volt_watt_ceiling = True
                if performance["solar_coupling"] == "dc_only":
                    power_axis.plot(
                        timestamps,
                        segment["site_export_power_kw"].to_list(),
                        color="#374151",
                        linewidth=1.3,
                        label=(
                            "Site export power" if segment_number == 0 else "_nolegend_"
                        ),
                    )
                    power_axis.plot(
                        timestamps,
                        segment["csip_export_limit_kw"].to_list(),
                        color="#7c3aed",
                        linewidth=1.6,
                        linestyle=":",
                        drawstyle="steps-post",
                        label=(
                            "CSIP export limit" if segment_number == 0 else "_nolegend_"
                        ),
                    )

            voltage_axis.axhline(
                253,
                color="#6b7280",
                linewidth=1,
                linestyle=":",
                label="Volt-Watt threshold: 253 V",
            )
            voltage_axis.axhline(
                258,
                color="#111827",
                linewidth=1,
                linestyle="--",
                label="Sustained-operation threshold: 258 V",
            )
            voltage_axis.axhline(
                265,
                color="#dc2626",
                linewidth=1,
                linestyle="-.",
                label="Passive anti-islanding threshold: 265 V",
            )
            voltage_axis.set_ylabel("Voltage (V)", fontsize=12)
            power_axis.set_ylabel("Power (kW)", fontsize=12)
            power_axis.set_xlabel(f"Local time ({timezone_name})", fontsize=12)
            voltage_axis.grid(True, color="#e5e7eb", linewidth=0.8)
            power_axis.grid(True, color="#e5e7eb", linewidth=0.8)
            power_handles, power_labels = power_axis.get_legend_handles_labels()
            if has_volt_watt_conformant:
                power_handles.append(
                    Patch(facecolor="#16a34a", alpha=0.14, edgecolor="none")
                )
                power_labels.append("Volt-Watt conformant")
            if has_volt_watt_non_conformant:
                power_handles.append(
                    Patch(facecolor="#dc2626", alpha=0.20, edgecolor="none")
                )
                power_labels.append("Volt-Watt non-conformant")
            power_axis.legend(
                power_handles,
                power_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                frameon=False,
                ncol=3,
                fontsize=11,
            )

            voltage_handles, voltage_labels = voltage_axis.get_legend_handles_labels()
            if has_cessation_shading:
                voltage_handles.append(
                    Patch(facecolor="#7c3aed", alpha=0.18, edgecolor="none")
                )
                voltage_labels.append("Sustained operation / passive anti-islanding")
            if has_volt_watt_assessment:
                voltage_handles.append(
                    Patch(facecolor="#0891b2", alpha=0.18, edgecolor="none")
                )
                voltage_labels.append("Volt-Watt assessment period")
            voltage_axis.legend(
                voltage_handles,
                voltage_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                frameon=False,
                ncol=4,
                fontsize=11,
            )

            timezone = ZoneInfo(timezone_name)
            power_axis.set_xlim(
                dt.datetime.combine(local_date, dt.time(6), tzinfo=timezone),
                dt.datetime.combine(local_date, dt.time(18), tzinfo=timezone),
            )
            power_axis.xaxis.set_major_locator(
                mdates.HourLocator(interval=1, tz=timezone)
            )
            power_axis.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M", tz=timezone)
            )

            passive_site_percentage_text = (
                f"{passive_site_percentage:.1f}%"
                if passive_site_percentage is not None
                else "N/A"
            )
            sustained_site_percentage_text = (
                f"{sustained_site_percentage:.1f}%"
                if sustained_site_percentage is not None
                else "N/A"
            )
            volt_watt_site_percentage_text = (
                f"{volt_watt_site_percentage:.1f}%"
                if volt_watt_site_percentage is not None
                else "N/A"
            )
            csip_site_percentage_text = (
                f"{csip_site_percentage:.1f}%"
                if csip_site_percentage is not None
                else "N/A"
            )
            passive_day_percentage_text = (
                f"{passive_day_percentage:.1f}%"
                if passive_day_percentage is not None
                else "N/A"
            )
            sustained_day_percentage_text = (
                f"{sustained_day_percentage:.1f}%"
                if sustained_day_percentage is not None
                else "N/A"
            )
            volt_watt_day_percentage_text = (
                f"{volt_watt_day_percentage:.1f}%"
                if volt_watt_day_percentage is not None
                else "N/A"
            )
            csip_day_percentage_text = (
                f"{csip_day_percentage:.1f}%"
                if csip_day_percentage is not None
                else "N/A"
            )

            figure.suptitle(
                f"Site {pseudonym} | {local_date:%d/%m/%Y} | "
                f"Capacity: {capacity:.2f} kW\n"
                f"Site summary — Volt-Watt: "
                f"{volt_watt_site_conformant}/{volt_watt_site_eligible} "
                f"({volt_watt_site_percentage_text}) | "
                f"Sustained operation: "
                f"{sustained_site_conformant}/{sustained_site_eligible} "
                f"({sustained_site_percentage_text}) | "
                f"Passive anti-islanding: "
                f"{passive_site_conformant}/{passive_site_eligible} "
                f"({passive_site_percentage_text}) | "
                f"CSIP: {csip_site_conformant}/{csip_site_eligible} "
                f"({csip_site_percentage_text})\n"
                f"Day summary — Volt-Watt: "
                f"{volt_watt_conformant}/{volt_watt_eligible} "
                f"({volt_watt_day_percentage_text}) | "
                f"Sustained operation: "
                f"{sustained_conformant}/{sustained_eligible} "
                f"({sustained_day_percentage_text}) | "
                f"Passive anti-islanding: "
                f"{passive_conformant}/{passive_eligible} "
                f"({passive_day_percentage_text}) | "
                f"CSIP: {csip_day_conformant}/{csip_day_eligible} "
                f"({csip_day_percentage_text})",
                y=0.98,
                fontsize=14,
            )
            figure.autofmt_xdate()
            figure.tight_layout(h_pad=4.0)

            dc_suffix = "_dc" if performance["solar_coupling"] == "dc_only" else ""
            plot_filename = f"{pseudonym}_{local_date.isoformat()}{dc_suffix}.png"
            for save_directory in save_directories:
                save_directory.mkdir(parents=True, exist_ok=True)
                figure.savefig(
                    save_directory / plot_filename,
                    dpi=150,
                    bbox_inches="tight",
                )
            plt.close(figure)
            plot_count += 1

    return plot_count
