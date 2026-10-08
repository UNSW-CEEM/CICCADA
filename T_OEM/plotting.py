"""Site-level plots for the coordinated AS/NZS 4777 and CSIP analysis."""

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Patch


def plot_site_conformance_days(
    combined_conformance_results: pl.DataFrame,
    site_conformance_performance: pl.DataFrame,
    site_metadata: pl.DataFrame,
    output_directory: str | Path = "site_level_plots",
) -> int:
    """Save 06:00-18:00 local-time plots for AS/NZS-eligible site-days."""
    output_directory = Path(output_directory)
    performance_by_site = {
        row["pseudonym"]: row
        for row in site_conformance_performance.iter_rows(named=True)
    }
    network_timezones = (
        ("SAPN", "Australia/Adelaide"),
        ("AUSNET", "Australia/Melbourne"),
        ("CPU", "Australia/Melbourne"),
    )
    plot_count = 0

    for network, timezone_name in network_timezones:
        site_ids = (
            site_metadata.filter(pl.col("folder") == network)
            .get_column("pseudonym")
            .to_list()
        )
        local_results = (
            combined_conformance_results.lazy()
            .filter(pl.col("pseudonym").is_in(site_ids))
            .select(
                "timestamp",
                "pseudonym",
                "solar_coupling",
                "grid_voltage",
                "grid_voltage_10m_avg",
                "inverter_ac_real_power",
                "site_instant_power",
                "csip_export_limit_w_ffill",
                "passive_anti_islanding_conformant",
                "sustained_operation_conformant",
                "volt_watt_conformant",
                "csip_conformant",
                "as_4777_conformant",
            )
            .with_columns(
                pl.col("timestamp")
                .dt.convert_time_zone(timezone_name)
                .alias("local_timestamp")
            )
            .filter(
                pl.col("local_timestamp")
                .dt.time()
                .is_between(dt.time(6), dt.time(18), closed="both")
            )
            .with_columns(pl.col("local_timestamp").dt.date().alias("local_date"))
        )
        eligible_site_days = (
            local_results.filter(pl.col("as_4777_conformant").is_not_null())
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
            performance = performance_by_site[pseudonym]
            combined_status = performance["combined_conformant_standard_level"]
            if combined_status is True:
                status_directory = "conformant"
                combined_status_label = "Conformant"
            elif combined_status is False:
                status_directory = "non_conformant"
                combined_status_label = "Non-conformant"
            else:
                status_directory = "null_less_eligible_timestamps"
                combined_status_label = "N/A (insufficient eligible timestamps)"

            as_eligible = performance["as_4777_num_eligible_timestamps"]
            as_percentage = performance["as_4777_conformant_percentage"]
            if as_eligible < 20:
                as_status_label = "N/A (<20 eligible)"
            elif as_percentage >= 90:
                as_status_label = "Pass"
            else:
                as_status_label = "Fail"
            as_percentage_label = (
                f"{as_percentage:.2f}%" if as_percentage is not None else "N/A"
            )

            csip_eligible = performance["csip_num_eligible_timestamps"]
            csip_percentage = performance["csip_conformant_percentage"]
            if csip_eligible < 20:
                csip_status_label = "N/A (<20 eligible)"
            elif csip_percentage >= 90:
                csip_status_label = "Pass"
            else:
                csip_status_label = "Fail"
            csip_percentage_label = (
                f"{csip_percentage:.2f}%" if csip_percentage is not None else "N/A"
            )

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

            figure, (voltage_axis, power_axis) = plt.subplots(
                2,
                1,
                figsize=(15, 9),
                sharex=True,
                height_ratios=(1, 1),
            )
            has_cessation_shading = False
            has_volt_watt_shading = False
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
                has_cessation_shading = has_cessation_shading or bool(
                    cessation_mask.any()
                )
                has_volt_watt_shading = has_volt_watt_shading or bool(
                    volt_watt_mask.any()
                )
                for shading_mask, shading_color in (
                    (cessation_mask, "#7c3aed"),
                    (volt_watt_mask, "#0891b2"),
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
                            for axis in (voltage_axis, power_axis):
                                axis.axvspan(
                                    shading_start,
                                    shading_end,
                                    color=shading_color,
                                    alpha=0.18,
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
                if performance["solar_coupling"] == "dc_only":
                    power_axis.plot(
                        timestamps,
                        segment["site_export_power_kw"].to_list(),
                        color="#2e7d32",
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
            power_axis.legend(loc="best", ncol=3, fontsize=11)

            voltage_handles, voltage_labels = voltage_axis.get_legend_handles_labels()
            if has_cessation_shading:
                voltage_handles.append(
                    Patch(facecolor="#7c3aed", alpha=0.18, edgecolor="none")
                )
                voltage_labels.append("Sustained operation / passive anti-islanding")
            if has_volt_watt_shading:
                voltage_handles.append(
                    Patch(facecolor="#0891b2", alpha=0.18, edgecolor="none")
                )
                voltage_labels.append("Volt-Watt operation")
            figure.legend(
                voltage_handles,
                voltage_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.90),
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

            figure.suptitle(
                f"Site {pseudonym} | {local_date:%d/%m/%Y} | "
                f"Overall standard-level status: {combined_status_label}\n"
                f"Site totals - AS/NZS 4777: {as_percentage_label} "
                f"({as_eligible} eligible) - {as_status_label} | "
                f"CSIP: {csip_percentage_label} "
                f"({csip_eligible} eligible) - {csip_status_label}\n"
                f"Day conformant/eligible - Volt-Watt: "
                f"{volt_watt_conformant}/{volt_watt_eligible} | "
                f"Sustained operation: "
                f"{sustained_conformant}/{sustained_eligible} | "
                f"Passive anti-islanding: "
                f"{passive_conformant}/{passive_eligible} | "
                f"CSIP: {csip_day_conformant}/{csip_day_eligible}",
                y=0.98,
                fontsize=14,
            )
            figure.autofmt_xdate()
            figure.tight_layout(rect=(0, 0, 1, 0.82))

            save_directory = output_directory / status_directory
            save_directory.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                save_directory / f"{pseudonym}_{local_date.isoformat()}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(figure)
            plot_count += 1

    return plot_count
