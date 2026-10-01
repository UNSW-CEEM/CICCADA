"""EBM-owned conformance plotting functions."""

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
    "power_channels": ["#1565C0", "#4A148C", "#A67C00"],
    "voltage_inst": "#b45309",
    "voltage_avg": "#1a1a1a",
    "threshold_los": "#1a1a1a",
    "threshold_ov1": "#c62828",
    "grid": "#ebebeb",
    "shade": "#7c3aed",
    "shade_extra": "#0891b2",
    "shade_missing_voltage": "#dc2626",
    "shade_below_threshold": "#9ca3af",
}


def _format_plot_date(day_label, timestamps=None):
    if timestamps:
        return timestamps[0].strftime("%d/%m/%Y")
    if isinstance(day_label, (dt.date, dt.datetime)):
        return day_label.strftime("%d/%m/%Y")
    return str(day_label)


def _boolean_mask(frame, column):
    return frame[column].fill_null(False).cast(pl.Boolean).to_numpy()


def _add_status_shading(axis, timestamps, masks):
    for mask, color, alpha in masks:
        if bool(np.any(mask)):
            axis.fill_between(
                timestamps,
                0,
                1,
                where=mask,
                transform=axis.get_xaxis_transform(),
                color=color,
                alpha=alpha,
                zorder=0,
                linewidth=0,
            )


def _valid_numeric(values):
    valid = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            valid.append(numeric)
    return valid


def plot_site_compliance_day(
    df: pl.DataFrame,
    site_number,
    day_label,
    *,
    p_rated: float,
    los_threshold: float | None,
    ov1_threshold: float | None,
    overall_category: str,
    los_lowest_disconnect_voltage: float | None = None,
    ov1_lowest_disconnect_voltage: float | None = None,
    plot_no_responsible_timestamp_days: bool = False,
    save_path: str | Path | None = None,
):
    """Plot one EBM site-day with power, voltage, and compliance context."""
    if df.is_empty():
        return
    power_columns = [
        column
        for column in df.columns
        if column.startswith("power")
        and not column.endswith("_next")
        and not column.endswith("_logic")
    ]
    if not power_columns:
        return

    counts = {
        column: int(df.get_column(column).sum() or 0)
        for column in (
            "los_responsible",
            "los_compliant",
            "los_disconnect_support_added",
            "ov1_responsible",
            "ov1_compliant",
            "ov1_disconnect_support_added",
        )
    }
    responsible_count = (
        counts["los_responsible"]
        + counts["los_disconnect_support_added"]
        + counts["ov1_responsible"]
        + counts["ov1_disconnect_support_added"]
    )
    compliant_count = (
        counts["los_compliant"]
        + counts["los_disconnect_support_added"]
        + counts["ov1_compliant"]
        + counts["ov1_disconnect_support_added"]
    )
    if responsible_count == 0 and not plot_no_responsible_timestamp_days:
        return

    plot_df = df.sort("local_tstamp")
    if "site_power_calculated" not in plot_df.columns:
        plot_df = plot_df.with_columns(
            pl.sum_horizontal([pl.col(column) for column in power_columns]).alias(
                "site_power_calculated"
            )
        )
    timestamps = plot_df["local_tstamp"].to_list()
    if not timestamps:
        return
    v10m_values = (
        plot_df["v10m_avg"].to_list()
        if "v10m_avg" in plot_df.columns
        else [None] * plot_df.height
    )
    vinst_values = (
        plot_df["vinst_max"].to_list()
        if "vinst_max" in plot_df.columns
        else [None] * plot_df.height
    )
    base_mask = _boolean_mask(plot_df, "los_responsible") | _boolean_mask(
        plot_df, "ov1_responsible"
    )
    support_mask = _boolean_mask(
        plot_df, "los_disconnect_support_added"
    ) | _boolean_mask(plot_df, "ov1_disconnect_support_added")
    missing_voltage_mask = _boolean_mask(plot_df, "disconnected_unknown_voltage")
    below_threshold_mask = _boolean_mask(plot_df, "disconnected_below_threshold")
    shading = [
        (below_threshold_mask, PLOT_COLORS["shade_below_threshold"], 0.22),
        (missing_voltage_mask, PLOT_COLORS["shade_missing_voltage"], 0.22),
        (base_mask, PLOT_COLORS["shade"], 0.18),
        (support_mask, PLOT_COLORS["shade_extra"], 0.18),
    ]

    single_phase = len(power_columns) == 1
    if single_phase:
        fig, power_axis = plt.subplots(figsize=(14, 5.5))
        power_axes = [power_axis]
        voltage_axes = [power_axis.twinx()]
        bottom_axis = power_axis
    else:
        fig, (channel_axis, total_axis) = plt.subplots(
            2, 1, figsize=(14, 9), sharex=True
        )
        power_axes = [channel_axis, total_axis]
        voltage_axes = [channel_axis.twinx(), total_axis.twinx()]
        bottom_axis = total_axis
    fig.patch.set_facecolor("white")
    for axis in power_axes:
        axis.set_facecolor("white")
        _add_status_shading(axis, timestamps, shading)

    if single_phase:
        power_axis.plot(
            timestamps,
            plot_df[power_columns[0]].to_list(),
            color=PLOT_COLORS["power_channels"][0],
            linewidth=1.25,
            label="Power",
            zorder=4,
        )
    else:
        for index, column in enumerate(power_columns, start=1):
            channel_axis.plot(
                timestamps,
                plot_df[column].to_list(),
                color=PLOT_COLORS["power_channels"][
                    (index - 1) % len(PLOT_COLORS["power_channels"])
                ],
                linewidth=1.25,
                label=f"Power {index}",
                zorder=4,
            )
        total_axis.plot(
            timestamps,
            plot_df["site_power_calculated"].to_list(),
            color=PLOT_COLORS["power_total"],
            linewidth=2.2,
            label="Total Power",
            zorder=4,
        )

    for voltage_axis in voltage_axes:
        voltage_axis.plot(
            timestamps,
            vinst_values,
            color=PLOT_COLORS["voltage_inst"],
            linewidth=1.2,
            alpha=0.85,
            label="Vinst(max)",
        )
        voltage_axis.plot(
            timestamps,
            v10m_values,
            color=PLOT_COLORS["voltage_avg"],
            linestyle="--",
            linewidth=1.9,
            label="V10m rolling avg",
        )

    threshold_lines = [
        ("LSO threshold", los_threshold, PLOT_COLORS["threshold_los"], ":", 1.5),
        ("OV1 threshold", ov1_threshold, PLOT_COLORS["threshold_ov1"], "-.", 1.5),
        (
            "LSO lowest",
            los_lowest_disconnect_voltage,
            PLOT_COLORS["threshold_los"],
            "--",
            1.1,
        ),
        (
            "OV1 lowest",
            ov1_lowest_disconnect_voltage,
            PLOT_COLORS["threshold_ov1"],
            "--",
            1.1,
        ),
    ]
    for voltage_axis in voltage_axes:
        for label, value, color, linestyle, linewidth in threshold_lines:
            if value is not None:
                voltage_axis.axhline(
                    float(value),
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=0.95,
                    label=f"{label}: {float(value):.1f} V",
                )

    category_label = {
        "compliant": "Conformant",
        "compliant_erratic": "Conformant (erratic)",
        "non_compliant": "Non-conformant",
        "unassessed": "Unassessed",
    }.get(overall_category, str(overall_category))
    if responsible_count:
        compliance_pct = compliant_count / responsible_count * 100.0
        status = "Pass" if compliance_pct >= 90.0 else "Fail"
        day_summary = (
            f"Day total: {status} {compliance_pct:.1f}% "
            f"({compliant_count}/{responsible_count})"
        )
    else:
        day_summary = "Day total: No responsible timestamps"
    detail_summary = (
        f"Base: LSO {counts['los_compliant']}/{counts['los_responsible']}, "
        f"OV1 {counts['ov1_compliant']}/{counts['ov1_responsible']}"
    )
    if counts["los_disconnect_support_added"] or counts["ov1_disconnect_support_added"]:
        detail_summary += (
            " | Additional compliant: "
            f"LSO {counts['los_disconnect_support_added']}, "
            f"OV1 {counts['ov1_disconnect_support_added']}"
        )
    fig.suptitle(
        f"EBM site {site_number} | Date: {_format_plot_date(day_label, timestamps)} "
        f"| {category_label}\n{day_summary}\n{detail_summary}",
        y=0.985,
    )

    if single_phase:
        power_axis.set_ylabel("Power (kW)")
        power_axis.set_xlabel("Time")
    else:
        channel_axis.set_ylabel("Per-channel Power (kW)")
        total_axis.set_ylabel("Total Power (kW)")
        total_axis.set_xlabel("Time")
    for axis in power_axes:
        axis.set_ylim(0, max(0.1, float(p_rated)))
        axis.grid(True, color=PLOT_COLORS["grid"], linewidth=0.8, alpha=0.9)
        axis.spines["top"].set_visible(False)

    voltage_range_values = _valid_numeric([*v10m_values, *vinst_values])
    voltage_range_values.extend(
        _valid_numeric([value for _, value, _, _, _ in threshold_lines])
    )
    if voltage_range_values:
        voltage_min = np.floor((min(voltage_range_values) - 1.0) / 2.0) * 2.0
        voltage_max = np.ceil((max(voltage_range_values) + 1.0) / 2.0) * 2.0
    else:
        voltage_min, voltage_max = 248.0, 268.0
    for axis in voltage_axes:
        axis.set_ylabel("Voltage (V)")
        axis.set_ylim(voltage_min, voltage_max)
        axis.yaxis.set_major_locator(MultipleLocator(2.0))
        axis.yaxis.set_minor_locator(MultipleLocator(1.0))
        axis.grid(True, which="major", axis="y", color=PLOT_COLORS["grid"], alpha=0.35)
        axis.spines["top"].set_visible(False)

    plot_day = timestamps[0].date()
    plot_timezone = timestamps[0].tzinfo
    bottom_axis.set_xlim(
        dt.datetime.combine(plot_day, dt.time(6, 0), tzinfo=plot_timezone),
        dt.datetime.combine(plot_day, dt.time(18, 0), tzinfo=plot_timezone),
    )
    bottom_axis.xaxis.set_major_locator(
        mdates.MinuteLocator(byminute=[0, 30], tz=plot_timezone)
    )
    bottom_axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=plot_timezone)
    )

    legend_entries = {}
    for axis in [*power_axes, voltage_axes[0]]:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            legend_entries.setdefault(label, handle)
    for label, mask, color, alpha in (
        ("EVM event", base_mask, PLOT_COLORS["shade"], 0.18),
        (
            "Additional responsible (lowest-disconnect criterion)",
            support_mask,
            PLOT_COLORS["shade_extra"],
            0.18,
        ),
        (
            "Disconnected (below threshold)",
            below_threshold_mask,
            PLOT_COLORS["shade_below_threshold"],
            0.22,
        ),
        (
            "Disconnected (missing voltage)",
            missing_voltage_mask,
            PLOT_COLORS["shade_missing_voltage"],
            0.22,
        ),
    ):
        if bool(np.any(mask)):
            legend_entries[label] = Patch(
                facecolor=color, alpha=alpha, edgecolor="none"
            )
    fig.legend(
        list(legend_entries.values()),
        list(legend_entries.keys()),
        loc="upper left",
        bbox_to_anchor=(0.02, 0.87 if single_phase else 0.91),
        frameon=False,
        ncol=4 if len(legend_entries) > 8 else 3,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.78 if single_phase else 0.86))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    """Plot per-site minimum, median, maximum, and standard deviation."""
    if df.is_empty():
        return
    plot_df = df.sort(sort_by, descending=descending, nulls_last=True)
    site_ids = plot_df["site_id"].to_list()
    x_values = np.arange(len(site_ids))
    fig, left_axis = plt.subplots(figsize=(max(12, len(site_ids) * 0.18), 7))
    left_axis.plot(
        x_values,
        plot_df["min_v"].to_list(),
        color="#9ecae1",
        marker="o",
        linewidth=1.8,
        label="Min (V)",
    )
    left_axis.plot(
        x_values,
        plot_df["median_v"].to_list(),
        color="#4C78A8",
        marker="o",
        linewidth=2.2,
        label="Median (V)",
    )
    left_axis.plot(
        x_values,
        plot_df["max_v"].to_list(),
        color="#2ca02c",
        marker="o",
        linewidth=1.8,
        label="Max (V)",
    )
    left_axis.set_title(title)
    left_axis.set_ylabel("Voltage (V) — Min / Median / Max")
    left_axis.grid(axis="y", alpha=0.25)
    label_step = max(1, len(site_ids) // 40)
    tick_indices = x_values[::label_step]
    left_axis.set_xticks(tick_indices)
    left_axis.set_xticklabels(
        [str(site_ids[index]) for index in range(0, len(site_ids), label_step)],
        rotation=45,
        ha="right",
    )
    std_values = plot_df["std_v"].to_list() if "std_v" in plot_df.columns else None
    if std_values is not None and any(value is not None for value in std_values):
        right_axis = left_axis.twinx()
        right_axis.plot(
            x_values,
            std_values,
            color="#F58518",
            marker="s",
            linewidth=1.8,
            label="Std (V)",
        )
        right_axis.set_ylabel("Std (V)")
        left_handles, left_labels = left_axis.get_legend_handles_labels()
        right_handles, right_labels = right_axis.get_legend_handles_labels()
        left_axis.legend(
            left_handles + right_handles, left_labels + right_labels, loc="best"
        )
    else:
        left_axis.legend(loc="best")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    """Plot the highest- or lowest-variation EBM sites."""
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
