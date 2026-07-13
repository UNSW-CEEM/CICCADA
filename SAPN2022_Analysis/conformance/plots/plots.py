import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from pathlib import Path
import datetime as dt
import zoneinfo
import polars as pl
import numpy as np
from helperFuncs import split_nonmixed_groups

PLOT_COLORS = {
    "power_total": "#2e7d32",
    "power_channels": ["#2e7d32", "#2e7d32", "#2e7d32", "#2e7d32"],
    "voltage_inst": "#b45309",
    "voltage_avg": "#1a1a1a",
    "threshold_lso": "#1a1a1a",
    "threshold_ov1": "#c62828",
    "grid": "#ebebeb",
    "shade": "#7c3aed",
}


def _true_mask_spans(timestamps, mask):
    if not timestamps or not mask or len(timestamps) != len(mask):
        return []

    if len(timestamps) == 1:
        half_width = dt.timedelta(seconds=2.5)
        if mask[0]:
            return [(timestamps[0] - half_width, timestamps[0] + half_width)]
        return []

    boundaries = [timestamps[0] - ((timestamps[1] - timestamps[0]) / 2)]
    for idx in range(1, len(timestamps)):
        boundaries.append(timestamps[idx - 1] + ((timestamps[idx] - timestamps[idx - 1]) / 2))
    boundaries.append(timestamps[-1] + ((timestamps[-1] - timestamps[-2]) / 2))

    spans = []
    run_start = None
    for idx, is_active in enumerate(mask):
        if is_active and run_start is None:
            run_start = idx
        if run_start is not None and (not is_active or idx == len(mask) - 1):
            run_end = idx if is_active and idx == len(mask) - 1 else idx - 1
            spans.append((boundaries[run_start], boundaries[run_end + 1]))
            run_start = None
    return spans


def _power_trace_label(power_cols, idx=None):
    if len(power_cols) == 1:
        return "Power"
    if idx is None:
        return "Power"
    return f"Power {idx}"


def _format_plot_date(day_label):
    try:
        return dt.date(2022, 11, int(day_label)).strftime("%d/%m/%Y")
    except (TypeError, ValueError):
        return str(day_label)


def _day_status_label(day_compliant_ts, day_eligible_ts, threshold_pct=90.0):
    if day_eligible_ts is None or day_compliant_ts is None:
        return "unassessed", None

    day_eligible_ts = int(day_eligible_ts)
    day_compliant_ts = int(day_compliant_ts)
    if day_eligible_ts <= 0:
        return "unassessed", None

    day_pct = (float(day_compliant_ts) / float(day_eligible_ts)) * 100.0
    day_status = "conformant" if day_pct >= threshold_pct else "non-conformant"
    return day_status, day_pct

# LSO Plots
def plotLsoDataForSite(df, siteNumber, tz_name="Australia/Adelaide", 
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
                path = 'updated results/LSO/site level/'+pathFolder+'/Site:{} Power vs Vmean 10-min Day: {}'.format(siteNumber, day)
            else:
                path = 'updated results/LSO/site level/'+pathFolder+'/Site:{} Power vs Vmean 10-min Day: {} {}'.format(siteNumber, day, behavior)
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
    lso_threshold: float | None,
    ov1_threshold: float | None,
    overall_pass,
    day_summary: dict | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot a single site-day using a shared two-panel compliance layout.
    """
    if df.is_empty():
        return

    tz = zoneinfo.ZoneInfo("Australia/Adelaide")
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
    event_active = None
    if {"los_responsible", "ov1_responsible"}.issubset(set(plot_df.columns)):
        event_active = (
            plot_df["los_responsible"].fill_null(False).cast(pl.Boolean)
            | plot_df["ov1_responsible"].fill_null(False).cast(pl.Boolean)
        ).to_numpy()
    is_single_phase = len(power_cols) == 1

    if is_single_phase:
        fig, ax_main = plt.subplots(1, 1, figsize=(14, 5.5), sharex=True)
        plot_power_axes = [ax_main]
        voltage_axes = [ax_main.twinx()]
        title_axis = ax_main
        bottom_axis = ax_main
    else:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        plot_power_axes = [ax_top, ax_bottom]
        voltage_axes = [ax_top.twinx(), ax_bottom.twinx()]
        title_axis = ax_top
        bottom_axis = ax_bottom
    fig.patch.set_facecolor("white")

    for axis in plot_power_axes:
        axis.set_facecolor("white")
        if event_active is not None and bool(np.any(event_active)):
            axis.fill_between(
                x,
                0,
                1,
                where=event_active,
                transform=axis.get_xaxis_transform(),
                color=PLOT_COLORS["shade"],
                alpha=0.18,
                zorder=0,
                linewidth=0,
            )

    if is_single_phase:
        ax_main.plot(
            x,
            plot_df[power_cols[0]].to_list(),
            color=PLOT_COLORS["power_channels"][0],
            linewidth=1.25,
            alpha=0.95,
            zorder=4,
            label=_power_trace_label(power_cols),
        )
    else:
        for idx, power_col in enumerate(power_cols, start=1):
            ax_top.plot(
                x,
                plot_df[power_col].to_list(),
                color=PLOT_COLORS["power_channels"][(idx - 1) % len(PLOT_COLORS["power_channels"])],
                linewidth=1.25,
                alpha=0.95,
                zorder=4,
                label=_power_trace_label(power_cols, idx),
            )

        ax_bottom.plot(
            x,
            plot_df["site_power"].to_list(),
            color=PLOT_COLORS["power_total"],
            linewidth=2.2,
            zorder=4,
            label="Total Power",
        )

    for v_ax in voltage_axes:
        v_ax.plot(
            x,
            vinst_vals,
            color=PLOT_COLORS["voltage_inst"],
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
            label="Vinst(max)",
        )
        v_ax.plot(
            x,
            v10m_vals,
            color=PLOT_COLORS["voltage_avg"],
            linestyle="--",
            linewidth=1.9,
            zorder=3,
            label="V10m rolling avg",
        )

    thresholds_to_draw = []
    if lso_threshold is not None:
        thresholds_to_draw.append((
            f"LSO threshold: {float(lso_threshold):.1f} V",
            lso_threshold,
            PLOT_COLORS["threshold_lso"],
            ":",
        ))
    if ov1_threshold is not None:
        thresholds_to_draw.append((
            f"OV1 threshold: {float(ov1_threshold):.1f} V",
            ov1_threshold,
            PLOT_COLORS["threshold_ov1"],
            "-.",
        ))

    for v_ax in voltage_axes:
        for label, value, color, style in thresholds_to_draw:
            v_ax.axhline(value, color=color, linestyle=style, linewidth=1.5, alpha=0.95, label=label)

    overall_label = (
        "Conformant" if overall_pass is True else
        "Non-conformant" if overall_pass is False else
        "Unassessed"
    )
    if day_summary is None:
        day_label_text = "Day status unavailable"
    else:
        lso_eligible = int(day_summary.get("los_eligible", 0) or 0)
        lso_compliant = int(day_summary.get("los_compliant", 0) or 0)
        ov1_eligible = int(day_summary.get("ov1_eligible", 0) or 0)
        ov1_compliant = int(day_summary.get("ov1_compliant", 0) or 0)
        total_eligible = lso_eligible + ov1_eligible
        total_compliant = lso_compliant + ov1_compliant
        if total_eligible == 0:
            day_label_text = "No eligible timestamps"
        else:
            day_pct = (total_compliant / total_eligible) * 100.0
            day_state = "Day pass" if day_pct >= 90.0 else "Day fail"
            day_label_text = (
                f"{day_state} {day_pct:.1f}% | LSO {lso_compliant}/{lso_eligible} | "
                f"OV1 {ov1_compliant}/{ov1_eligible}"
            )

    plot_date = _format_plot_date(day_label)
    title = f"Site example | Date: {plot_date} | {overall_label}"
    if day_label_text:
        title = f"{title}\n{day_label_text}"

    title_axis.set_title(title, pad=12)
    if is_single_phase:
        ax_main.set_ylabel("Power (kW)")
        ax_main.set_xlabel("Time")
    else:
        ax_top.set_ylabel("Per-channel Power (kW)")
        ax_bottom.set_ylabel("Total Power (kW)")
        ax_bottom.set_xlabel("Time")

    for p_ax in plot_power_axes:
        p_ax.set_ylim(0, max(0.1, float(p_rated)))
        p_ax.grid(True, color=PLOT_COLORS["grid"], linewidth=0.8, alpha=0.9)
        p_ax.spines["top"].set_visible(False)
        p_ax.spines["right"].set_visible(False)

    all_voltage_vals = [v for v in [*v10m_vals, *vinst_vals] if v is not None]
    for _, value, _, _ in thresholds_to_draw:
        if value is not None:
            all_voltage_vals.append(value)
    if all_voltage_vals:
        v_min = np.floor((min(all_voltage_vals) - 1.0) / 2.0) * 2.0
        v_max = np.ceil((max(all_voltage_vals) + 1.0) / 2.0) * 2.0
    else:
        v_min, v_max = 248.0, 268.0

    for v_ax in voltage_axes:
        v_ax.set_ylabel("Voltage (V)")
        v_ax.set_ylim(v_min, v_max)
        v_ax.yaxis.set_major_locator(MultipleLocator(2.0))
        v_ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        v_ax.grid(True, which="major", axis="y", color=PLOT_COLORS["grid"], alpha=0.35)
        v_ax.spines["top"].set_visible(False)

    day_start = dt.datetime(2022, 11, int(day_label), 6, 0, 0, tzinfo=tz)
    day_end = dt.datetime(2022, 11, int(day_label), 18, 0, 0, tzinfo=tz)
    bottom_axis.set_xlim(day_start, day_end)
    bottom_axis.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=tz))
    bottom_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))

    if is_single_phase:
        lines, labels = ax_main.get_legend_handles_labels()
        v_lines, v_labels = voltage_axes[0].get_legend_handles_labels()
        if event_active is not None and bool(np.any(event_active)):
            v_lines = v_lines + [Patch(facecolor=PLOT_COLORS["shade"], alpha=0.18, edgecolor="none")]
            v_labels = v_labels + ["EVM event"]
        ax_main.legend(lines + v_lines, labels + v_labels, loc="upper left", ncol=2)
    else:
        top_lines, top_labels = ax_top.get_legend_handles_labels()
        top_v_lines, top_v_labels = voltage_axes[0].get_legend_handles_labels()
        if event_active is not None and bool(np.any(event_active)):
            top_v_lines = top_v_lines + [Patch(facecolor=PLOT_COLORS["shade"], alpha=0.18, edgecolor="none")]
            top_v_labels = top_v_labels + ["EVM event"]
        ax_top.legend(top_lines + top_v_lines, top_labels + top_v_labels, loc="upper left", ncol=2)

        bottom_lines, bottom_labels = ax_bottom.get_legend_handles_labels()
        bottom_v_lines, bottom_v_labels = voltage_axes[1].get_legend_handles_labels()
        if event_active is not None and bool(np.any(event_active)):
            bottom_v_lines = bottom_v_lines + [Patch(facecolor=PLOT_COLORS["shade"], alpha=0.18, edgecolor="none")]
            bottom_v_labels = bottom_v_labels + ["EVM event"]
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


def plot_method_threshold_overlay_day(
    df: pl.DataFrame,
    site_number,
    day_label,
    *,
    p_rated: float,
    method_thresholds: list[dict],
    method_event_overlays: list[dict] | None = None,
    comparison_event_mask: list[bool] | None = None,
    tz_name: str = "Australia/Adelaide",
    save_path: str | Path | None = None,
):
    """
    Plot a site-day using the comparison overlay layout and multi-method LSO
    thresholds on the same voltage axis.

    Expected method_thresholds entries:
      - label
      - lso_threshold
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

    overlay_spans = []
    if method_event_overlays is not None:
        for overlay_info in method_event_overlays:
            event_mask = overlay_info.get("event_mask")
            if event_mask is None:
                continue
            if len(event_mask) != len(x):
                raise ValueError("method_event_overlays mask length does not match plot frame length")
            event_spans = _true_mask_spans(x, event_mask)
            if not event_spans:
                continue
            overlay_spans.append({
                "label": overlay_info.get("label", "Method"),
                "color": overlay_info.get("color", PLOT_COLORS["shade"]),
                "alpha": float(overlay_info.get("alpha", 0.12)),
                "spans": event_spans,
            })
    else:
        event_active = comparison_event_mask
        if event_active is None and {"los_responsible", "ov1_responsible"}.issubset(set(plot_df.columns)):
            event_active = (
                plot_df["los_responsible"].fill_null(False).cast(pl.Boolean)
                | plot_df["ov1_responsible"].fill_null(False).cast(pl.Boolean)
            ).to_list()
        if event_active is not None and len(event_active) != len(x):
            raise ValueError("comparison_event_mask length does not match plot frame length")
        event_spans = _true_mask_spans(x, event_active) if event_active is not None else []
        if event_spans:
            overlay_spans.append({
                "label": "EVM event",
                "color": PLOT_COLORS["shade"],
                "alpha": 0.22,
                "spans": event_spans,
            })
    is_single_phase = len(power_cols) == 1

    if is_single_phase:
        fig, ax_main = plt.subplots(1, 1, figsize=(14, 5.5), sharex=True)
        plot_power_axes = [ax_main]
        voltage_axes = [ax_main.twinx()]
        title_axis = ax_main
        bottom_axis = ax_main
    else:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        plot_power_axes = [ax_top, ax_bottom]
        voltage_axes = [ax_top.twinx(), ax_bottom.twinx()]
        title_axis = ax_top
        bottom_axis = ax_bottom

    for axis in plot_power_axes:
        for overlay_info in overlay_spans:
            for span_start, span_end in overlay_info["spans"]:
                axis.axvspan(
                    span_start,
                    span_end,
                    color=overlay_info["color"],
                    alpha=overlay_info["alpha"],
                    zorder=0,
                    linewidth=0,
                )

    if is_single_phase:
        ax_main.plot(
            x,
            plot_df[power_cols[0]].to_list(),
            color=PLOT_COLORS["power_channels"][0],
            linewidth=1.4,
            zorder=4,
            label=_power_trace_label(power_cols),
        )
    else:
        for idx, power_col in enumerate(power_cols, start=1):
            ax_top.plot(
                x,
                plot_df[power_col].to_list(),
                color=PLOT_COLORS["power_channels"][(idx - 1) % len(PLOT_COLORS["power_channels"])],
                linewidth=1.4,
                zorder=4,
                label=_power_trace_label(power_cols, idx),
            )

        ax_bottom.plot(
            x,
            plot_df["site_power"].to_list(),
            color=PLOT_COLORS["power_total"],
            linewidth=1.8,
            zorder=4,
            label="Total Power",
        )

    for v_ax in voltage_axes:
        v_ax.plot(
            x,
            vinst_vals,
            color=PLOT_COLORS["voltage_inst"],
            linestyle="-",
            linewidth=0.9,
            alpha=0.55,
            zorder=2,
            label="Vinst,max",
        )
        v_ax.plot(
            x,
            v10m_vals,
            color=PLOT_COLORS["voltage_avg"],
            linestyle="--",
            linewidth=2.1,
            zorder=3,
            label="V10m,avg",
        )
        for method_info in method_thresholds:
            threshold_value = float(method_info["lso_threshold"])
            v_ax.axhline(
                threshold_value,
                color=method_info["color"],
                linestyle=":",
                linewidth=1.4,
                alpha=0.9,
                zorder=1,
                label=f'{method_info["label"]} LSO {threshold_value:.3f} V',
            )

    method_status_parts = []
    for method_info in method_thresholds:
        day_eligible_ts = method_info.get("day_eligible_timestamps")
        day_compliant_ts = method_info.get("day_compliant_timestamps")
        day_status, day_pct = _day_status_label(day_compliant_ts, day_eligible_ts)
        if day_pct is not None:
            method_status_parts.append(
                f'{method_info["label"]}: site {method_info["status"]} | '
                f'day {day_status} {day_pct:.1f}% '
                f'({int(day_compliant_ts)}/{int(day_eligible_ts)} ts)'
            )
        else:
            method_status_parts.append(
                f'{method_info["label"]}: site {method_info["status"]} | day {day_status}'
            )
    method_status_text = " | ".join(method_status_parts)
    plot_date = _format_plot_date(day_label)
    title = f"Site example | Date: {plot_date}"
    if method_status_text:
        title = f"{title}\n{method_status_text}"

    title_axis.set_title(title)
    if is_single_phase:
        ax_main.set_ylabel("Power (kW)")
        ax_main.set_xlabel("Time")
    else:
        ax_top.set_ylabel("Per-channel Power (kW)")
        ax_bottom.set_ylabel("Total Power (kW)")
        ax_bottom.set_xlabel("Time")

    for p_ax in plot_power_axes:
        p_ax.set_ylim(0, max(0.1, float(p_rated)))
        p_ax.grid(True, color=PLOT_COLORS["grid"], alpha=0.35)

    all_voltage_vals = [v for v in [*v10m_vals, *vinst_vals] if v is not None]
    for method_info in method_thresholds:
        all_voltage_vals.append(float(method_info["lso_threshold"]))
    if all_voltage_vals:
        v_min = np.floor((min(all_voltage_vals) - 1.0) / 2.0) * 2.0
        v_max = np.ceil((max(all_voltage_vals) + 1.0) / 2.0) * 2.0
    else:
        v_min, v_max = 248.0, 268.0

    for v_ax in voltage_axes:
        v_ax.set_ylabel("Voltage (V)")
        v_ax.set_ylim(v_min, v_max)
        v_ax.yaxis.set_major_locator(MultipleLocator(2.0))
        v_ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        v_ax.grid(True, which="major", axis="y", color=PLOT_COLORS["grid"], alpha=0.35)

    day_start = dt.datetime(2022, 11, int(day_label), 6, 0, 0, tzinfo=tz)
    day_end = dt.datetime(2022, 11, int(day_label), 18, 0, 0, tzinfo=tz)
    bottom_axis.set_xlim(day_start, day_end)
    bottom_axis.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=tz))
    bottom_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))

    if is_single_phase:
        lines, labels = ax_main.get_legend_handles_labels()
        v_lines, v_labels = voltage_axes[0].get_legend_handles_labels()
        if overlay_spans:
            v_lines = v_lines + [
                Patch(facecolor=overlay_info["color"], alpha=overlay_info["alpha"], edgecolor="none")
                for overlay_info in overlay_spans
            ]
            v_labels = v_labels + [f'{overlay_info["label"]} EVM window' for overlay_info in overlay_spans]
        ax_main.legend(lines + v_lines, labels + v_labels, loc="upper left", ncol=2)
    else:
        top_lines, top_labels = ax_top.get_legend_handles_labels()
        top_v_lines, top_v_labels = voltage_axes[0].get_legend_handles_labels()
        if overlay_spans:
            top_v_lines = top_v_lines + [
                Patch(facecolor=overlay_info["color"], alpha=overlay_info["alpha"], edgecolor="none")
                for overlay_info in overlay_spans
            ]
            top_v_labels = top_v_labels + [f'{overlay_info["label"]} EVM window' for overlay_info in overlay_spans]
        ax_top.legend(top_lines + top_v_lines, top_labels + top_v_labels, loc="upper left", ncol=2)

        bottom_lines, bottom_labels = ax_bottom.get_legend_handles_labels()
        bottom_v_lines, bottom_v_labels = voltage_axes[1].get_legend_handles_labels()
        if overlay_spans:
            bottom_v_lines = bottom_v_lines + [
                Patch(facecolor=overlay_info["color"], alpha=overlay_info["alpha"], edgecolor="none")
                for overlay_info in overlay_spans
            ]
            bottom_v_labels = bottom_v_labels + [f'{overlay_info["label"]} EVM window' for overlay_info in overlay_spans]
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
    comparison_event_mask: list[bool] | None = None,
    tz_name: str = "Australia/Adelaide",
    save_path: str | Path | None = None,
):
    """
    Backwards-compatible alias for the generalized multi-method overlay plot.
    """
    plot_method_threshold_overlay_day(
        df,
        site_number,
        day_label,
        p_rated=p_rated,
        method_thresholds=method_thresholds,
        comparison_event_mask=comparison_event_mask,
        tz_name=tz_name,
        save_path=save_path,
    )


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
