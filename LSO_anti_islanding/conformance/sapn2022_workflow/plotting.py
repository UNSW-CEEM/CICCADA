"""Active conformance plotting functions."""

import datetime as dt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

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
        boundaries.append(
            timestamps[idx - 1] + ((timestamps[idx] - timestamps[idx - 1]) / 2)
        )
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


def _format_plot_date(day_label, timestamps=None):
    if timestamps:
        return timestamps[0].strftime("%d/%m/%Y")
    if isinstance(day_label, (dt.date, dt.datetime)):
        return day_label.strftime("%d/%m/%Y")
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


def plot_site_compliance_day(
    df: pl.DataFrame,
    site_number,
    day_label,
    *,
    p_rated: float,
    lso_threshold: float | None,
    ov1_threshold: float | None,
    overall_pass,
    plot_no_responsible_timestamp_days: bool = False,
    save_path: str | Path | None = None,
):
    """
    Plot a single site-day using a shared two-panel compliance layout.
    """
    if df.is_empty():
        return

    power_cols = [
        c
        for c in df.columns
        if c.startswith("power")
        and not c.endswith("_next")
        and not c.endswith("_logic")
    ]
    if not power_cols:
        return

    los_responsible_count = int(df.get_column("los_responsible").sum() or 0)
    los_compliant_count = int(df.get_column("los_compliant").sum() or 0)
    ov1_responsible_count = int(df.get_column("ov1_responsible").sum() or 0)
    ov1_compliant_count = int(df.get_column("ov1_compliant").sum() or 0)
    total_responsible_count = los_responsible_count + ov1_responsible_count
    if total_responsible_count == 0 and not plot_no_responsible_timestamp_days:
        return

    plot_df = df.sort("local_tstamp")
    if "site_power" not in plot_df.columns:
        plot_df = plot_df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in power_cols]).alias("site_power")
        )

    x = plot_df["local_tstamp"].to_list()
    v10m_vals = (
        plot_df["v10m_avg"].to_list()
        if "v10m_avg" in plot_df.columns
        else [None] * plot_df.height
    )
    vinst_vals = (
        plot_df["vinst_max"].to_list()
        if "vinst_max" in plot_df.columns
        else [None] * plot_df.height
    )
    event_active = None
    disconnected_below_lso_ov1_threshold_mask = None
    if {"los_responsible", "ov1_responsible"}.issubset(set(plot_df.columns)):
        event_active = (
            plot_df["los_responsible"].fill_null(False).cast(pl.Boolean)
            | plot_df["ov1_responsible"].fill_null(False).cast(pl.Boolean)
        ).to_numpy()
        disconnected_below_lso_ov1_threshold_mask = (
            plot_df["is_disc"].fill_null(False).cast(pl.Boolean).to_numpy()
            & ~event_active
        )
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
        if disconnected_below_lso_ov1_threshold_mask is not None and bool(
            np.any(disconnected_below_lso_ov1_threshold_mask)
        ):
            axis.fill_between(
                x,
                0,
                1,
                where=disconnected_below_lso_ov1_threshold_mask,
                transform=axis.get_xaxis_transform(),
                color="#9ca3af",
                alpha=0.22,
                zorder=0,
                linewidth=0,
            )
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
                color=PLOT_COLORS["power_channels"][
                    (idx - 1) % len(PLOT_COLORS["power_channels"])
                ],
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
        thresholds_to_draw.append(
            (
                f"LOS threshold: {float(lso_threshold):.1f} V",
                lso_threshold,
                PLOT_COLORS["threshold_lso"],
                ":",
            )
        )
    if ov1_threshold is not None:
        thresholds_to_draw.append(
            (
                f"OV1 threshold: {float(ov1_threshold):.1f} V",
                ov1_threshold,
                PLOT_COLORS["threshold_ov1"],
                "-.",
            )
        )

    for v_ax in voltage_axes:
        for label, value, color, style in thresholds_to_draw:
            v_ax.axhline(
                value,
                color=color,
                linestyle=style,
                linewidth=1.5,
                alpha=0.95,
                label=label,
            )

    overall_label = (
        "Conformant"
        if overall_pass is True
        else "Non-conformant"
        if overall_pass is False
        else "Unassessed"
    )
    total_compliant_count = los_compliant_count + ov1_compliant_count
    if total_responsible_count == 0:
        day_label_text = "No responsible timestamps"
    else:
        day_pct = (total_compliant_count / total_responsible_count) * 100.0
        day_state = "Day pass" if day_pct >= 90.0 else "Day fail"
        day_label_text = (
            f"{day_state} {day_pct:.1f}% | "
            f"LOS {los_compliant_count}/{los_responsible_count} responsible | "
            f"OV1 {ov1_compliant_count}/{ov1_responsible_count} responsible"
        )

    plot_date = _format_plot_date(day_label, x)
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

    plot_day = x[0].date()
    plot_timezone = x[0].tzinfo
    day_start = dt.datetime.combine(
        plot_day,
        dt.time(6, 0),
        tzinfo=plot_timezone,
    )
    day_end = dt.datetime.combine(
        plot_day,
        dt.time(18, 0),
        tzinfo=plot_timezone,
    )
    bottom_axis.set_xlim(day_start, day_end)
    bottom_axis.xaxis.set_major_locator(
        mdates.MinuteLocator(byminute=[0, 30], tz=plot_timezone)
    )
    bottom_axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=plot_timezone)
    )

    if is_single_phase:
        lines, labels = ax_main.get_legend_handles_labels()
        v_lines, v_labels = voltage_axes[0].get_legend_handles_labels()
        if event_active is not None and bool(np.any(event_active)):
            v_lines = v_lines + [
                Patch(facecolor=PLOT_COLORS["shade"], alpha=0.18, edgecolor="none")
            ]
            v_labels = v_labels + ["Responsible timestamp"]
        if disconnected_below_lso_ov1_threshold_mask is not None and bool(
            np.any(disconnected_below_lso_ov1_threshold_mask)
        ):
            v_lines = v_lines + [
                Patch(facecolor="#9ca3af", alpha=0.22, edgecolor="none")
            ]
            v_labels = v_labels + ["Disconnected below threshold"]
        ax_main.legend(lines + v_lines, labels + v_labels, loc="upper left", ncol=2)
    else:
        top_lines, top_labels = ax_top.get_legend_handles_labels()
        top_v_lines, top_v_labels = voltage_axes[0].get_legend_handles_labels()
        if event_active is not None and bool(np.any(event_active)):
            top_v_lines = top_v_lines + [
                Patch(facecolor=PLOT_COLORS["shade"], alpha=0.18, edgecolor="none")
            ]
            top_v_labels = top_v_labels + ["Responsible timestamp"]
        if disconnected_below_lso_ov1_threshold_mask is not None and bool(
            np.any(disconnected_below_lso_ov1_threshold_mask)
        ):
            top_v_lines = top_v_lines + [
                Patch(facecolor="#9ca3af", alpha=0.22, edgecolor="none")
            ]
            top_v_labels = top_v_labels + ["Disconnected below threshold"]
        ax_top.legend(
            top_lines + top_v_lines, top_labels + top_v_labels, loc="upper left", ncol=2
        )

        bottom_lines, bottom_labels = ax_bottom.get_legend_handles_labels()
        bottom_v_lines, bottom_v_labels = voltage_axes[1].get_legend_handles_labels()
        if event_active is not None and bool(np.any(event_active)):
            bottom_v_lines = bottom_v_lines + [
                Patch(facecolor=PLOT_COLORS["shade"], alpha=0.18, edgecolor="none")
            ]
            bottom_v_labels = bottom_v_labels + ["Responsible timestamp"]
        if disconnected_below_lso_ov1_threshold_mask is not None and bool(
            np.any(disconnected_below_lso_ov1_threshold_mask)
        ):
            bottom_v_lines = bottom_v_lines + [
                Patch(facecolor="#9ca3af", alpha=0.22, edgecolor="none")
            ]
            bottom_v_labels = bottom_v_labels + ["Disconnected below threshold"]
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
    Plot a site-day using the comparison overlay layout and multi-method LOS
    thresholds on the same voltage axis.

    Expected method_thresholds entries:
      - label
      - lso_threshold
      - status
      - color
    """
    if df.is_empty():
        return

    power_cols = [
        c
        for c in df.columns
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
        if "v10m_avg" in plot_df.columns
        else [None] * plot_df.height
    )
    vinst_vals = (
        plot_df["vinst_max"].to_list()
        if "vinst_max" in plot_df.columns
        else [None] * plot_df.height
    )

    overlay_spans = []
    if method_event_overlays is not None:
        for overlay_info in method_event_overlays:
            event_mask = overlay_info.get("event_mask")
            if event_mask is None:
                continue
            if len(event_mask) != len(x):
                raise ValueError(
                    "method_event_overlays mask length does not match plot frame length"
                )
            event_spans = _true_mask_spans(x, event_mask)
            if not event_spans:
                continue
            overlay_spans.append(
                {
                    "label": overlay_info.get("label", "Method"),
                    "color": overlay_info.get("color", PLOT_COLORS["shade"]),
                    "alpha": float(overlay_info.get("alpha", 0.12)),
                    "spans": event_spans,
                }
            )
    else:
        event_active = comparison_event_mask
        if event_active is None and {"los_responsible", "ov1_responsible"}.issubset(
            set(plot_df.columns)
        ):
            event_active = (
                plot_df["los_responsible"].fill_null(False).cast(pl.Boolean)
                | plot_df["ov1_responsible"].fill_null(False).cast(pl.Boolean)
            ).to_list()
        if event_active is not None and len(event_active) != len(x):
            raise ValueError(
                "comparison_event_mask length does not match plot frame length"
            )
        event_spans = (
            _true_mask_spans(x, event_active) if event_active is not None else []
        )
        if event_spans:
            overlay_spans.append(
                {
                    "label": "Responsible timestamp",
                    "color": PLOT_COLORS["shade"],
                    "alpha": 0.22,
                    "spans": event_spans,
                }
            )
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
                color=PLOT_COLORS["power_channels"][
                    (idx - 1) % len(PLOT_COLORS["power_channels"])
                ],
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
                label=f"{method_info['label']} LOS {threshold_value:.3f} V",
            )

    method_status_parts = []
    for method_info in method_thresholds:
        day_eligible_ts = method_info.get("day_eligible_timestamps")
        day_compliant_ts = method_info.get("day_compliant_timestamps")
        day_status, day_pct = _day_status_label(day_compliant_ts, day_eligible_ts)
        if day_pct is not None:
            method_status_parts.append(
                f"{method_info['label']}: site {method_info['status']} | "
                f"day {day_status} {day_pct:.1f}% "
                f"({int(day_compliant_ts)}/{int(day_eligible_ts)} ts)"
            )
        else:
            method_status_parts.append(
                f"{method_info['label']}: site {method_info['status']} | day {day_status}"
            )
    method_status_text = " | ".join(method_status_parts)
    plot_date = _format_plot_date(day_label, x)
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

    plot_day = x[0].date()
    plot_timezone = x[0].tzinfo
    day_start = dt.datetime.combine(
        plot_day,
        dt.time(6, 0),
        tzinfo=plot_timezone,
    )
    day_end = dt.datetime.combine(
        plot_day,
        dt.time(18, 0),
        tzinfo=plot_timezone,
    )
    bottom_axis.set_xlim(day_start, day_end)
    bottom_axis.xaxis.set_major_locator(
        mdates.MinuteLocator(byminute=[0, 30], tz=plot_timezone)
    )
    bottom_axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=plot_timezone)
    )

    if is_single_phase:
        lines, labels = ax_main.get_legend_handles_labels()
        v_lines, v_labels = voltage_axes[0].get_legend_handles_labels()
        if overlay_spans:
            v_lines = v_lines + [
                Patch(
                    facecolor=overlay_info["color"],
                    alpha=overlay_info["alpha"],
                    edgecolor="none",
                )
                for overlay_info in overlay_spans
            ]
            v_labels = v_labels + [
                f"{overlay_info['label']} EVM window" for overlay_info in overlay_spans
            ]
        ax_main.legend(lines + v_lines, labels + v_labels, loc="upper left", ncol=2)
    else:
        top_lines, top_labels = ax_top.get_legend_handles_labels()
        top_v_lines, top_v_labels = voltage_axes[0].get_legend_handles_labels()
        if overlay_spans:
            top_v_lines = top_v_lines + [
                Patch(
                    facecolor=overlay_info["color"],
                    alpha=overlay_info["alpha"],
                    edgecolor="none",
                )
                for overlay_info in overlay_spans
            ]
            top_v_labels = top_v_labels + [
                f"{overlay_info['label']} EVM window" for overlay_info in overlay_spans
            ]
        ax_top.legend(
            top_lines + top_v_lines, top_labels + top_v_labels, loc="upper left", ncol=2
        )

        bottom_lines, bottom_labels = ax_bottom.get_legend_handles_labels()
        bottom_v_lines, bottom_v_labels = voltage_axes[1].get_legend_handles_labels()
        if overlay_spans:
            bottom_v_lines = bottom_v_lines + [
                Patch(
                    facecolor=overlay_info["color"],
                    alpha=overlay_info["alpha"],
                    edgecolor="none",
                )
                for overlay_info in overlay_spans
            ]
            bottom_v_labels = bottom_v_labels + [
                f"{overlay_info['label']} EVM window" for overlay_info in overlay_spans
            ]
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
    ax_left.plot(
        x, v_med, color="#4C78A8", marker="o", linewidth=2.2, label="Median (V)"
    )
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
        ax_right.plot(
            x, std_v, color="#F58518", marker="s", linewidth=1.8, label="Std (V)"
        )
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

    plot_df = plot_df.sort("std_v", descending=highest_std, nulls_last=True).head(
        n_sites
    )
    plot_site_threshold_distribution(
        plot_df,
        title=title,
        sort_by="std_v",
        descending=highest_std,
        save_path=save_path,
    )
