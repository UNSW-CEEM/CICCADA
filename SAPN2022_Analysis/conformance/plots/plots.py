import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import datetime as dt
import zoneinfo
import polars as pl
import numpy as np
from helperFuncs import split_nonmixed_groups

METHOD_DISPLAY_LABEL_MAP = {
    "Default thresholds": "Default",
    "Original Phase A raw": "Original",
    "Current confidence-tier": "Tier based",
    "Old sweep method": "Old sweep",
    "High -> blended": "Blended",
    "default": "Default",
    "original_raw": "Original",
    "original": "Original",
    "confidence_tier": "Tier based",
    "tier_based": "Tier based",
    "old_sweep": "Old sweep",
    "high_blended": "Blended",
    "blended": "Blended",
}


def _normalize_method_display_label(method_label):
    if method_label is None:
        return None
    return METHOD_DISPLAY_LABEL_MAP.get(method_label, method_label)

# LOS Plots
def plotLosDataForSite(df, siteNumber, tz_name="Australia/Adelaide", 
                       savePlot = False, pathFolder = None, vPlot = None, pPlot = None,
                       behavior=None):
    """ Plot power and vmean_rolling_10m using local_tstamp column 
        vPlot: avg/ max
        pPlot: max/ sum
        this funcitonality has not been implemented but should be straight forward
    """
    tz = zoneinfo.ZoneInfo(tz_name)
    x = df["local_tstamp"]
    power_cols = [c for c in df.columns 
                if c.startswith("power") and not c.endswith("_next")]
    vmean_cols = [c for c in df.columns 
                if c.startswith("vmean_rolling_10m")]

    # Max voltage across phases
    df = df.with_columns(pl.max_horizontal(vmean_cols).alias("vmean_max"))
    # Sum of powers across phases
    df = df.with_columns(pl.sum_horizontal(power_cols).alias("power_sum"))

    x = df["local_tstamp"]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # =========================
    # TOP: Individual phases
    # =========================
    ax1 = ax_top
    ax2 = ax1.twinx()

    for phaseNum, power_col in enumerate(power_cols, start=1):
        ax1.plot(x, df[power_col], label=power_col)
    
    ax2.plot(x, df["vmean_max"], color="black", linestyle="--",
         label="Max Vmean (10m)")

    ax1.set_ylabel("Power")
    ax2.set_ylabel("Max Vmean (10-min rolling)")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(f"Site: {siteNumber} — Per Phase Power vs Max Voltage")

    # =========================
    # BOTTOM: Total Power
    # =========================
    ax3 = ax_bottom
    ax4 = ax3.twinx()

    ax3.plot(x, df["power_sum"], color="tab:blue", label="Total Power")

    ax4.plot(x, df["vmean_max"], color="black", linestyle="--", label="Max Vmean (10m)")

    ax3.set_ylabel("Total Power")
    ax4.set_ylabel("Max Vmean (10-min rolling)")
    ax3.grid(True, alpha=0.3)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="best")

    ax3.set_xlabel("Time")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M", tz=tz))

    fig.autofmt_xdate()
    plt.tight_layout()
    # plt.show()

    if savePlot==True:
        if pathFolder == None:
            raise ValueError ("Path Folder cannot be empty")
        else:
            day = df["local_tstamp"][0].day
            if behavior == None:
                # Save the figure to a file
                path = 'updated results/LOS/site level/'+pathFolder+'/Site:{} Power vs Vmean 10-min Day: {}'.format(siteNumber, day)
            else:
                path = 'updated results/LOS/site level/'+pathFolder+'/Site:{} Power vs Vmean 10-min Day: {} {}'.format(siteNumber, day, behavior)
            plt.savefig(path) # Saves as a PNG file
    plt.close()

# Islanding stats
def plotIslandingDataForSite(df, siteNumber,  tz_name="Australia/Adelaide", 
                    savePlot = False, pathFolder = None, vPlot = None, pPlot = None):
    """ Plot power and vmean_rolling_10m using local_tstamp column 
        vPlot: avg/ max
        pPlot: max/ sum
        this funcitonality has not been implemented but should be straight forward
    """
    tz = zoneinfo.ZoneInfo(tz_name)
    x = df["local_tstamp"]
    power_cols = [c for c in df.columns 
                if c.startswith("power") and not c.endswith("_next")]
    voltage_cols = [c for c in df.columns 
                if c.startswith("voltage") and not c.endswith("_next")]

    # Max voltage across phases
    df = df.with_columns(pl.max_horizontal(voltage_cols).alias("voltage_max"))
    # Sum of powers across phases
    df = df.with_columns(pl.sum_horizontal(power_cols).alias("power_sum"))

    x = df["local_tstamp"]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # =========================
    # TOP: Individual phases
    # =========================
    ax1 = ax_top
    ax2 = ax1.twinx()

    for phaseNum, power_col in enumerate(power_cols, start=1):
        ax1.plot(x, df[power_col], label=power_col)
    
    ax2.plot(x, df["voltage_max"], color="black", linestyle="--",
         label="Max Voltage")

    ax1.set_ylabel("Power")
    ax2.set_ylabel("Max Voltage")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(f"Site: {siteNumber} — Per Phase Power vs Max Voltage")

    # =========================
    # BOTTOM: Total Power
    # =========================
    ax3 = ax_bottom
    ax4 = ax3.twinx()

    ax3.plot(x, df["power_sum"], color="tab:blue", label="Total Power")

    ax4.plot(x, df["voltage_max"], color="black", linestyle="--", label="Max Vmean (10m)")

    ax3.set_ylabel("Total Power")
    ax4.set_ylabel("Max Voltage")
    ax3.grid(True, alpha=0.3)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="best")

    ax3.set_xlabel("Time")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M", tz=tz))

    fig.autofmt_xdate()
    plt.tight_layout()
    # plt.show()

    if savePlot==True:
        # Save the figure to a file
        day = df["local_tstamp"][0].day
        path = 'updated results/Islanding/Site:{} Power vs Vmax Day: {}'.format(siteNumber, day)
        plt.savefig(path) # Saves as a PNG file
    plt.close()

# plot voltage stats for sites with mixed behavior
def plot_disconnection_voltage_lines_for_mixed_sites(
    site_out: pl.DataFrame,
    site_summary: pl.DataFrame,
    title: str = "Disconnection Voltage — Min / Median / Max (Mixed Compliance Sites)",
    figsize=(14, 7),
    save_path: str | None = None
):
    """
    For Mixed sites only (from site_summary), plot per-site:
      - Lines (left Y): disc_all_min_v, disc_all_median_v, disc_all_max_v  [Volts]
      - (Optional) Line (right Y): disc_all_std_v  [Volts]  <-- commented below

    Requires in site_out:
      'site_id', 'disc_all_min_v', 'disc_all_median_v', 'disc_all_max_v'
      (optionally 'disc_all_std_v' if you enable the right axis)
    """

    # Filter to Mixed sites only
    mixed_ids = site_summary.select("site_id")
    df = site_out.join(mixed_ids, on="site_id", how="inner")

    # Keep rows with at least one of the metrics present
    needed = ["disc_all_min_v", "disc_all_median_v", "disc_all_max_v"]
    df = df.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in needed]))

    if df.height == 0:
        print("No data to plot for mixed sites.")
        return

    # Sort by median (desc) for a consistent left→right read
    df = df.sort("disc_all_median_v", descending=True, nulls_last=True)

    # Extract arrays
    site_ids = df.get_column("site_id").to_list()
    v_min    = df.get_column("disc_all_min_v").to_list()
    v_med    = df.get_column("disc_all_median_v").to_list()
    v_max    = df.get_column("disc_all_max_v").to_list()

    x = np.arange(len(site_ids))

    fig, ax_left = plt.subplots(figsize=figsize)

    # Lines (left Y): min / median / max
    ax_left.plot(x, v_min, color="#9ecae1", marker="o", linewidth=2, label="Min (V)")
    ax_left.plot(x, v_med, color="#4C78A8", marker="o", linewidth=2.5, label="Median (V)")
    ax_left.plot(x, v_max, color="#2ca02c", marker="o", linewidth=2, label="Max (V)")

    ax_left.set_ylabel("Voltage (V) — Min / Median / Max")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(site_ids, rotation=45, ha="right")
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.set_title(title)

    # OPTIONAL: add Std as a right-axis line (uncomment if desired)
    std_v = df.get_column("disc_all_std_v").to_list() if "disc_all_std_v" in df.columns else None
    if std_v is not None:
        ax_right = ax_left.twinx()
        ax_right.plot(x, std_v, color="#F58518", marker="s", linewidth=2, label="Std (V)")
        ax_right.set_ylabel("Std (V)")
        # Merge legends
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
    else:
        ax_left.legend(loc="best")

    # If not plotting std on right axis, show legend from left axis only:
    ax_left.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# plot circuit data
def plotData(df, circuitNumber, tz=None):
    "plot in lcoal timzeone"
    tz = zoneinfo.ZoneInfo("Australia/Adelaide")
    fig, ax = plt.subplots()
    ax.plot(df["local_tstamp"], df["power"])  # your data here
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d %H:%M", tz=tz)
    )
    plt.show()


def plot_site_compliance_day(
    df: pl.DataFrame,
    site_number,
    day_label,
    *,
    p_rated: float,
    los_threshold: float | None,
    los_threshold_p25: float | None = None,
    los_threshold_p10: float | None = None,
    los_threshold_min: float | None = None,
    ov1_threshold: float | None,
    delta_los_site: float | None,
    delta_los_p25_site: float | None = None,
    delta_los_p10_site: float | None = None,
    delta_los_min_site: float | None = None,
    delta_ov1_site: float | None,
    ov1_basis: str | None = None,
    overall_pass,
    pass_basis: str | None = None,
    day_summary: dict | None = None,
    force_draw_los_threshold: bool = False,
    force_draw_ov1_threshold: bool = False,
    compliance_threshold_pct: float = 90.0,
    tz_name: str = "Australia/Adelaide",
    method_label: str | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot a site-day using the combined compliance view:
      - top: per-channel power + site voltages
      - bottom: total site power + site voltages
    Voltage overlays:
      - V10m,avg
      - Vinst,max
      - LOS threshold used for Phase B
      - OV1 threshold used for Phase B
      - if neither threshold is learned, show default/fallback tested references
    """
    if df.is_empty():
        return

    tz = zoneinfo.ZoneInfo(tz_name)
    power_cols = [
        c for c in df.columns
        if c.startswith("power")
        and not c.endswith("_next")
        and not c.endswith("_logic")
    ]
    if not power_cols:
        return

    plot_df = df.sort("local_tstamp")
    if "site_power" not in plot_df.columns:
        plot_df = plot_df.with_columns(pl.sum_horizontal([pl.col(c) for c in power_cols]).alias("site_power"))

    x = plot_df["local_tstamp"].to_list()
    v10m_vals = plot_df["v10m_avg"].to_list() if "v10m_avg" in plot_df.columns else [None] * plot_df.height
    vinst_vals = plot_df["vinst_max"].to_list() if "vinst_max" in plot_df.columns else [None] * plot_df.height

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax_top_v = ax_top.twinx()
    ax_bottom_v = ax_bottom.twinx()

    for idx, power_col in enumerate(power_cols, start=1):
        ax_top.plot(x, plot_df[power_col].to_list(), linewidth=1.4, label=f"Power {idx}")

    ax_bottom.plot(x, plot_df["site_power"].to_list(), color="tab:blue", linewidth=1.8, label="Total Power")

    for v_ax in [ax_top_v, ax_bottom_v]:
        v_ax.plot(
            x,
            vinst_vals,
            color="tab:red",
            linestyle="-",
            linewidth=0.9,
            alpha=0.55,
            zorder=2,
            label="Vinst,max",
        )
        v_ax.plot(
            x,
            v10m_vals,
            color="#111111",
            linestyle="--",
            linewidth=2.1,
            zorder=3,
            label="V10m,avg",
        )

    thresholds_to_draw = []
    if los_threshold is not None and (delta_los_site is not None or force_draw_los_threshold):
        if pass_basis == "min_override":
            los_label = "LOS threshold (min override)"
        elif pass_basis == "p10_override":
            los_label = "LOS threshold (p10 override)"
        elif pass_basis == "p25_override":
            los_label = "LOS threshold (p25 override)"
        elif delta_los_site is not None:
            los_label = "LOS threshold"
        else:
            los_label = "LOS ref 258 V (default)"
        if "258 V" not in los_label:
            los_label = f"{los_label} ({float(los_threshold):.3f} V)"
        thresholds_to_draw.append((los_label, los_threshold, "#2ca02c", ":"))
    if ov1_threshold is not None and (delta_ov1_site is not None or force_draw_ov1_threshold):
        if ov1_basis == "ov1_records":
            ov1_label = "OV1 threshold (tested)"
        elif ov1_basis == "blended":
            ov1_label = "OV1 threshold (blended, tested)"
        elif ov1_basis == "los_fallback":
            ov1_label = "OV1 threshold (LOS fallback, tested)"
        else:
            ov1_label = "OV1 threshold (default, tested)"
        thresholds_to_draw.append((ov1_label, ov1_threshold, "#ff7f0e", "-."))
    if not thresholds_to_draw:
        thresholds_to_draw = [
            ("LOS ref 258 V (no Phase A threshold)", 258.0, "#2ca02c", ":"),
            ("OV1 threshold (default, tested)", 264.7, "#ff7f0e", "-."),
        ]

    for v_ax in [ax_top_v, ax_bottom_v]:
        for label, value, color, style in thresholds_to_draw:
            v_ax.axhline(value, color=color, linestyle=style, linewidth=1.4, alpha=0.9, label=label)

    overall_label = (
        "Compliant" if overall_pass is True else
        "Non-compliant" if overall_pass is False else
        "Unassessed"
    )
    if day_summary is None:
        day_label_text = "Day status unavailable"
    else:
        los_eligible = int(day_summary.get("los_eligible", 0) or 0)
        los_compliant = int(day_summary.get("los_compliant", 0) or 0)
        ov1_eligible = int(day_summary.get("ov1_eligible", 0) or 0)
        ov1_compliant = int(day_summary.get("ov1_compliant", 0) or 0)
        total_eligible = los_eligible + ov1_eligible
        total_compliant = los_compliant + ov1_compliant
        if total_eligible == 0:
            day_label_text = "Day: no eligible timestamps"
        else:
            day_pct = (total_compliant / total_eligible) * 100.0
            day_is_compliant = day_pct >= compliance_threshold_pct
            day_pct_text = f"{day_pct:.1f}%"
            if day_is_compliant:
                day_label_text = (
                    f"Day: compliant "
                    f"({day_pct_text}; LOS {los_compliant}/{los_eligible}, OV1 {ov1_compliant}/{ov1_eligible})"
                )
            else:
                day_label_text = (
                    f"Day: non-compliant "
                    f"({day_pct_text}; LOS {los_compliant}/{los_eligible}, OV1 {ov1_compliant}/{ov1_eligible})"
                )

    basis_text = "" if pass_basis in (None, "median", "unassessed") else f" | Basis: {pass_basis}"
    method_label = _normalize_method_display_label(method_label)
    method_text = "" if not method_label else f" | Method: {method_label}"
    title = f"Site: {site_number} Day: {day_label} — Site overall: {overall_label}{basis_text}{method_text} | {day_label_text}"

    ax_top.set_title(title)
    ax_top.set_ylabel("Per-channel Power (kW)")
    ax_bottom.set_ylabel("Total Power (kW)")
    ax_bottom.set_xlabel("Time")

    for p_ax in [ax_top, ax_bottom]:
        p_ax.set_ylim(0, max(0.1, float(p_rated)))
        p_ax.grid(True, alpha=0.25)

    all_voltage_vals = [v for v in [*v10m_vals, *vinst_vals] if v is not None]
    for _, value, _, _ in thresholds_to_draw:
        if value is not None:
            all_voltage_vals.append(value)
    if all_voltage_vals:
        v_min = np.floor((min(all_voltage_vals) - 1.0) / 2.0) * 2.0
        v_max = np.ceil((max(all_voltage_vals) + 1.0) / 2.0) * 2.0
    else:
        v_min, v_max = 248.0, 268.0

    for v_ax in [ax_top_v, ax_bottom_v]:
        v_ax.set_ylabel("Voltage (V)")
        v_ax.set_ylim(v_min, v_max)
        v_ax.yaxis.set_major_locator(MultipleLocator(2.0))
        v_ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        v_ax.grid(True, which="major", axis="y", alpha=0.18)

    day_start = dt.datetime(2022, 11, int(day_label), 6, 0, 0, tzinfo=tz)
    day_end = dt.datetime(2022, 11, int(day_label), 18, 0, 0, tzinfo=tz)
    ax_bottom.set_xlim(day_start, day_end)
    ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=tz))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))

    top_lines, top_labels = ax_top.get_legend_handles_labels()
    top_v_lines, top_v_labels = ax_top_v.get_legend_handles_labels()
    ax_top.legend(top_lines + top_v_lines, top_labels + top_v_labels, loc="upper left", ncol=2)

    bottom_lines, bottom_labels = ax_bottom.get_legend_handles_labels()
    bottom_v_lines, bottom_v_labels = ax_bottom_v.get_legend_handles_labels()
    ax_bottom.legend(
        bottom_lines + bottom_v_lines,
        bottom_labels + bottom_v_labels,
        loc="upper left",
        ncol=2,
    )

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_three_method_threshold_overlay_day(
    df: pl.DataFrame,
    site_number,
    day_label,
    *,
    p_rated: float,
    method_thresholds: list[dict],
    tz_name: str = "Australia/Adelaide",
    save_path: str | Path | None = None,
):
    """
    Plot a site-day using the standard two-panel compliance style, but overlay
    LOS thresholds for multiple methods on the same voltage axis.

    Expected method_thresholds entries:
      - label
      - los_threshold
      - status
      - color
    """
    if df.is_empty():
        return

    tz = zoneinfo.ZoneInfo(tz_name)
    power_cols = [
        c for c in df.columns
        if c.startswith("power")
        and not c.endswith("_next")
        and not c.endswith("_logic")
    ]
    if not power_cols:
        return

    plot_df = df.sort("local_tstamp")
    if "site_power" not in plot_df.columns:
        plot_df = plot_df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in power_cols]).alias("site_power")
        )

    x = plot_df["local_tstamp"].to_list()
    v10m_vals = (
        plot_df["v10m_avg"].to_list()
        if "v10m_avg" in plot_df.columns else
        [None] * plot_df.height
    )
    vinst_vals = (
        plot_df["vinst_max"].to_list()
        if "vinst_max" in plot_df.columns else
        [None] * plot_df.height
    )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax_top_v = ax_top.twinx()
    ax_bottom_v = ax_bottom.twinx()

    for idx, power_col in enumerate(power_cols, start=1):
        ax_top.plot(x, plot_df[power_col].to_list(), linewidth=1.4, label=f"Power {idx}")

    ax_bottom.plot(x, plot_df["site_power"].to_list(), color="tab:blue", linewidth=1.8, label="Total Power")

    threshold_groups: dict[float, list[dict]] = {}
    for method_info in method_thresholds:
        key = round(float(method_info["los_threshold"]), 6)
        threshold_groups.setdefault(key, []).append(method_info)

    for v_ax in [ax_top_v, ax_bottom_v]:
        v_ax.plot(
            x,
            vinst_vals,
            color="tab:red",
            linestyle="-",
            linewidth=0.9,
            alpha=0.55,
            zorder=2,
            label="Vinst,max",
        )
        v_ax.plot(
            x,
            v10m_vals,
            color="#111111",
            linestyle="--",
            linewidth=2.1,
            zorder=3,
            label="V10m,avg",
        )
        for threshold_key, grouped_methods in threshold_groups.items():
            threshold_value = float(grouped_methods[0]["los_threshold"])
            if len(grouped_methods) == 1:
                line_color = grouped_methods[0]["color"]
                line_label = f'{grouped_methods[0]["label"]} LOS {threshold_value:.3f} V'
            else:
                joined_labels = " / ".join(m["label"] for m in grouped_methods)
                line_color = "#7f7f7f"
                line_label = f"{joined_labels} LOS {threshold_value:.3f} V"
            v_ax.axhline(
                threshold_value,
                color=line_color,
                linestyle=":",
                linewidth=1.4,
                alpha=0.9,
                zorder=1,
                label=line_label,
            )

    method_status_parts = []
    for method_info in method_thresholds:
        compliant_ts = method_info.get("compliant_timestamps")
        eligible_ts = method_info.get("eligible_timestamps")
        if compliant_ts is not None and eligible_ts is not None:
            method_status_parts.append(
                f'{method_info["label"]}: {method_info["status"]} ({int(compliant_ts)}/{int(eligible_ts)} ts)'
            )
        else:
            method_status_parts.append(f'{method_info["label"]}: {method_info["status"]}')
    method_status_text = " | ".join(method_status_parts)
    title = f"Site: {site_number} Day: {day_label}\n{method_status_text}"

    ax_top.set_title(title)
    ax_top.set_ylabel("Per-channel Power (kW)")
    ax_bottom.set_ylabel("Total Power (kW)")
    ax_bottom.set_xlabel("Time")

    for p_ax in [ax_top, ax_bottom]:
        p_ax.set_ylim(0, max(0.1, float(p_rated)))
        p_ax.grid(True, alpha=0.25)

    all_voltage_vals = [v for v in [*v10m_vals, *vinst_vals] if v is not None]
    for method_info in method_thresholds:
        all_voltage_vals.append(float(method_info["los_threshold"]))
    if all_voltage_vals:
        v_min = np.floor((min(all_voltage_vals) - 1.0) / 2.0) * 2.0
        v_max = np.ceil((max(all_voltage_vals) + 1.0) / 2.0) * 2.0
    else:
        v_min, v_max = 248.0, 268.0

    for v_ax in [ax_top_v, ax_bottom_v]:
        v_ax.set_ylabel("Voltage (V)")
        v_ax.set_ylim(v_min, v_max)
        v_ax.yaxis.set_major_locator(MultipleLocator(2.0))
        v_ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        v_ax.grid(True, which="major", axis="y", alpha=0.18)

    day_start = dt.datetime(2022, 11, int(day_label), 6, 0, 0, tzinfo=tz)
    day_end = dt.datetime(2022, 11, int(day_label), 18, 0, 0, tzinfo=tz)
    ax_bottom.set_xlim(day_start, day_end)
    ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=tz))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))

    top_lines, top_labels = ax_top.get_legend_handles_labels()
    top_v_lines, top_v_labels = ax_top_v.get_legend_handles_labels()
    ax_top.legend(top_lines + top_v_lines, top_labels + top_v_labels, loc="upper left", ncol=2)

    bottom_lines, bottom_labels = ax_bottom.get_legend_handles_labels()
    bottom_v_lines, bottom_v_labels = ax_bottom_v.get_legend_handles_labels()
    ax_bottom.legend(
        bottom_lines + bottom_v_lines,
        bottom_labels + bottom_v_labels,
        loc="upper left",
        ncol=2,
    )

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_site_threshold_distribution(
    df: pl.DataFrame,
    *,
    title: str,
    sort_by: str = "median_v",
    descending: bool = False,
    save_path: str | Path | None = None,
):
    """
    Plot per-site min / median / max threshold values with std on the right axis.
    Expects columns:
      - site_id
      - min_v
      - median_v
      - max_v
      - std_v
    """
    if df.is_empty():
        return

    plot_df = df.sort(sort_by, descending=descending, nulls_last=True)
    site_ids = plot_df["site_id"].to_list()
    v_min = plot_df["min_v"].to_list()
    v_med = plot_df["median_v"].to_list()
    v_max = plot_df["max_v"].to_list()
    std_v = plot_df["std_v"].to_list() if "std_v" in plot_df.columns else None

    x = np.arange(len(site_ids))
    fig, ax_left = plt.subplots(figsize=(max(12, len(site_ids) * 0.18), 7))

    ax_left.plot(x, v_min, color="#9ecae1", marker="o", linewidth=1.8, label="Min (V)")
    ax_left.plot(x, v_med, color="#4C78A8", marker="o", linewidth=2.2, label="Median (V)")
    ax_left.plot(x, v_max, color="#2ca02c", marker="o", linewidth=1.8, label="Max (V)")

    ax_left.set_title(title)
    ax_left.set_ylabel("Voltage (V) — Min / Median / Max")
    ax_left.grid(axis="y", alpha=0.25)

    label_step = max(1, len(site_ids) // 40)
    tick_idx = x[::label_step]
    tick_labels = [str(site_ids[i]) for i in range(0, len(site_ids), label_step)]
    ax_left.set_xticks(tick_idx)
    ax_left.set_xticklabels(tick_labels, rotation=45, ha="right")

    if std_v is not None and any(v is not None for v in std_v):
        ax_right = ax_left.twinx()
        ax_right.plot(x, std_v, color="#F58518", marker="s", linewidth=1.8, label="Std (V)")
        ax_right.set_ylabel("Std (V)")
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
    else:
        ax_left.legend(loc="best")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_site_threshold_distribution_extremes(
    df: pl.DataFrame,
    *,
    title: str,
    save_path: str | Path | None = None,
    n_sites: int = 20,
    min_events: int = 3,
    highest_std: bool = True,
):
    """
    Plot the highest- or lowest-std sites only, with an event-count floor.
    """
    if df.is_empty():
        return

    plot_df = df
    if "n_events" in plot_df.columns:
        plot_df = plot_df.filter(pl.col("n_events") >= min_events)
    if plot_df.is_empty():
        return

    plot_df = plot_df.sort("std_v", descending=highest_std, nulls_last=True).head(n_sites)
    plot_site_threshold_distribution(
        plot_df,
        title=title,
        sort_by="std_v",
        descending=highest_std,
        save_path=save_path,
    )

# plots for non-mixed

import matplotlib.pyplot as plt
import numpy as np

def plot_lines_min_med_max_std_topN(
    df: pl.DataFrame,
    title: str,
    top_n: int = 10,
    sort_by: str = "disc_all_std_v",
    save_path: str | None = None
):
    """
    Line plot (Top‑N by `sort_by`):
      - Left Y: disc_all_min_v, disc_all_median_v, disc_all_max_v
      - Right Y: disc_all_std_v (if present)
    """
    if df.height == 0:
        print(f"No data to plot: {title}")
        return

    usable = ["disc_all_min_v", "disc_all_median_v", "disc_all_max_v"]
    df = df.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in usable]))

    if sort_by in df.columns:
        df = df.sort(sort_by, descending=True, nulls_last=True).head(top_n)

    if df.height == 0:
        print(f"No usable rows after filtering for: {title}")
        return

    site_ids = df.get_column("site_id").to_list()
    v_min    = df.get_column("disc_all_min_v").to_list()     if "disc_all_min_v"    in df.columns else [None]*len(site_ids)
    v_med    = df.get_column("disc_all_median_v").to_list()  if "disc_all_median_v" in df.columns else [None]*len(site_ids)
    v_max    = df.get_column("disc_all_max_v").to_list()     if "disc_all_max_v"    in df.columns else [None]*len(site_ids)
    has_std  = "disc_all_std_v" in df.columns
    std_v    = df.get_column("disc_all_std_v").to_list() if has_std else None

    x = np.arange(len(site_ids))
    plt.close('all')
    fig, ax_left = plt.subplots(figsize=(14, 6))

    # Left-axis lines
    if any(v is not None for v in v_min):
        ax_left.plot(x, v_min, color="#9ecae1", marker="o", linestyle="-", linewidth=2, label="Min (V)")
    if any(v is not None for v in v_med):
        ax_left.plot(x, v_med, color="#4C78A8", marker="o", linestyle="-", linewidth=2.5, label="Median (V)")
    if any(v is not None for v in v_max):
        ax_left.plot(x, v_max, color="#2ca02c", marker="o", linestyle="-", linewidth=2, label="Max (V)")

    ax_left.set_ylabel("Voltage (V) — Min / Median / Max")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(site_ids, rotation=45, ha="right")
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.set_title(title)

    # Right-axis std
    if has_std and any(v is not None for v in std_v):
        ax_right = ax_left.twinx()
        ax_right.plot(x, std_v, color="#F58518", marker="s", linestyle="-", linewidth=2, label="Std (V)")
        ax_right.set_ylabel("Std (V)")
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
    else:
        ax_left.legend(loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def plot_nonmixed_top10_lines_by_status(
    site_out: pl.DataFrame,
    site_summary_mixed: pl.DataFrame,
    voltage_stats_site_day: pl.DataFrame,
    top_n: int = 10,
    disconnect_pct_site_col: str | None = None,  # e.g., "disconnect_pct_site" (0–100). If None, fallback to transitions.
    T: float = 90.0,
    save_prefix: str | None = None
):
    """
    Produces two line plots (Top‑10 by std) for non‑mixed sites:
      - Always Compliant (AC) with disconnections
      - Always Non‑compliant (ANC) with disconnections
    Disconnection gate:
      - If disconnect_pct_site_col provided: require > 0
      - Else fallback: disc_all_count_transitions > 0
    """
    nonmixed_status = split_nonmixed_groups(voltage_stats_site_day, site_summary_mixed, T=T)
    base = site_out.join(nonmixed_status, on="site_id", how="inner")

    # Disconnection gate
    if disconnect_pct_site_col and (disconnect_pct_site_col in base.columns):
        base = base.filter(pl.col(disconnect_pct_site_col) > 0)
    elif "disc_all_count_transitions" in base.columns:
        base = base.filter(pl.col("disc_all_count_transitions") > 0)
    else:
        base = base.filter(pl.col("disc_all_std_v").is_not_null() & (pl.col("disc_all_std_v") > 0))

    # Split
    ac_df  = base.filter(pl.col("nonmixed_status") == "Always Compliant")
    anc_df = base.filter(pl.col("nonmixed_status") == "Always Non-compliant")

    # Plot 2: AC (Top‑10 by std)
    plot_lines_min_med_max_std_topN(
        ac_df,
        title=f"Non‑Mixed — Always Compliant (Top‑{top_n} by Std; disconnection > 0)",
        top_n=top_n,
        sort_by="disc_all_std_v",
        save_path=(f"{save_prefix}_nonmixed_AC_top{top_n}.png" if save_prefix else None)
    )

    # Plot 3: ANC (Top‑10 by std)
    plot_lines_min_med_max_std_topN(
        anc_df,
        title=f"Non‑Mixed — Always Non‑compliant (Top‑{top_n} by Std; disconnection > 0)",
        top_n=top_n,
        sort_by="disc_all_std_v",
        save_path=(f"{save_prefix}_nonmixed_ANC_top{top_n}.png" if save_prefix else None)
    )

    return ac_df, anc_df

def plot_std_hist_nonmixed_all(
    site_out: pl.DataFrame,
    site_summary_mixed: pl.DataFrame,
    voltage_stats_site_day: pl.DataFrame,
    disconnect_pct_site_col: str | None = None,
    bins: int = 30,
    T: float = 90.0,
    title: str = "Non‑Mixed — Distribution of Disconnection Std (V)",
    save_path: str | None = None
):
    nonmixed_status = split_nonmixed_groups(voltage_stats_site_day, site_summary_mixed, T=T)
    base = site_out.join(nonmixed_status, on="site_id", how="inner")

    # Disconnection gate
    if disconnect_pct_site_col and (disconnect_pct_site_col in base.columns):
        base = base.filter(pl.col(disconnect_pct_site_col) > 0)
    elif "disc_all_count_transitions" in base.columns:
        base = base.filter(pl.col("disc_all_count_transitions") > 0)
    else:
        base = base.filter(pl.col("disc_all_std_v").is_not_null() & (pl.col("disc_all_std_v") > 0))

    std_series = base.select(pl.col("disc_all_std_v")).drop_nulls()
    if std_series.height == 0:
        print("No std data for non‑mixed sites (post disconnection filter).")
        return

    std_vals = std_series.get_column("disc_all_std_v").to_list()

    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(std_vals, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Std (V)")
    ax.set_ylabel("Count of non‑mixed sites")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    med = float(np.median(std_vals))
    p90 = float(np.percentile(std_vals, 90))
    ax.axvline(med, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Median = {med:.2f} V")
    ax.axvline(p90, color="#F58518", linestyle="--", linewidth=1.5, label=f"90th pct = {p90:.2f} V")
    ax.legend(loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
