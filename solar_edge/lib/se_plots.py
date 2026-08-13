"""
Figures for the SolarEdge fleet EDA.
====================================

Deliverable D6. Every function takes a frame produced by ``se_queries`` and
returns a matplotlib Figure. No querying, no aggregation — if a number needs
computing it belongs in ``se_queries``, so that what is plotted is exactly what
can be tabulated.

AS/NZS 4777.2 set-points are drawn from the shared config, never hard-coded, so
the reference lines cannot drift from the thresholds the analysis uses.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from solar_edge.config import se_config as C

__all__ = [
    "plot_voltage_distribution",
    "plot_voltvar_signature",
    "plot_derating_by_voltage",
    "plot_diurnal_profile",
    "plot_capacity_distribution",
    "plot_monthly_coverage",
    "plot_reactive_character",
    "plot_q_categories",
    "plot_q_impact_distribution",
    "plot_sign_flip",
]

_A = C.as4777()
C_V, C_P, C_Q = "#b45309", "#2a7f62", "#4709b2"
C_ACCENT, C_GREY = "#e8702a", "#8a8a8a"


def _vvar_bands(ax, ymin=None, ymax=None):
    """Shade the AS/NZS 4777.2 Australia A response regions."""
    ax.axvspan(_A["VVAR"]["V2"], _A["VVAR"]["V3"], color=C_GREY, alpha=0.10, lw=0,
               label=f"Volt-VAr deadband ({_A['VVAR']['V2']:.0f}-{_A['VVAR']['V3']:.0f} V)")
    ax.axvspan(_A["VVAR"]["V3"], _A["VW"]["V1"], color=C_Q, alpha=0.08, lw=0,
               label=f"Volt-VAr absorb only ({_A['VVAR']['V3']:.0f}-{_A['VW']['V1']:.0f} V)")
    ax.axvspan(_A["VW"]["V1"], _A["VVAR"]["V4"], color=C_ACCENT, alpha=0.14, lw=0,
               label=f"Volt-Watt + Volt-VAr overlap ({_A['VW']['V1']:.0f}-{_A['VVAR']['V4']:.0f} V)")
    ax.axvline(_A["VW"]["V1"], color=C_ACCENT, ls="--", lw=1.0)


def plot_voltage_distribution(frame, figsize=(11, 4)):
    """Where the fleet actually sits relative to the response thresholds."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    _vvar_bands(ax)
    ax.bar(frame.v_bin, frame.n_intervals / 1e6, width=0.9, color=C_V, alpha=0.85)
    ax.set_xlabel("Site voltage (V)")
    ax.set_ylabel("Intervals (millions)")
    ax.set_title("Voltage distribution against AS/NZS 4777.2 Australia A thresholds")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_xlim(frame.v_bin.min(), frame.v_bin.max())
    fig.tight_layout()
    return fig


def plot_voltvar_signature(frame, figsize=(11, 4.6)):
    """
    Median reactive power against voltage.

    In the CICCADA generator convention a conforming inverter absorbs (Q < 0)
    above 240 V, so the line should slope DOWNWARD. This is the plot that
    validates the D2 sign flip — read it before trusting D9 or D13.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    _vvar_bands(ax)
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.plot(frame.v_bin, frame.median_Q_kvar, color=C_Q, lw=1.9, marker="o",
            ms=3.5, label="Median Q (kvar)")
    ax.set_xlabel("Site voltage (V)")
    ax.set_ylabel("Median Q (kvar)   —   negative = absorbing")
    ax.set_title("Volt-VAr signature: Q should fall as voltage rises")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_reactive_character(frame, figsize=(11, 4.2)):
    """
    Volt-VAr response or fixed power factor?

    Left: |Q| against voltage — rising means a genuine response.
    Right: Q/P against voltage — flat means a fixed power factor, and any apparent
    Q-vs-V slope is just the confound that voltage rises with export.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=130, sharex=True)
    for cohort, group in frame.groupby("cohort"):
        colour = C_Q if cohort.startswith("single") else C_ACCENT
        axes[0].plot(group.v_bin, group.med_abs_Q_kvar, marker="o", ms=3.5,
                     lw=1.8, color=colour, label=cohort)
        axes[1].plot(group.v_bin, group.med_Q_over_P, marker="o", ms=3.5,
                     lw=1.8, color=colour, label=cohort)
    axes[0].set_ylabel("Median |Q| (kvar)")
    axes[0].set_title("Magnitude of reactive power")
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.6)
    axes[1].set_ylabel("Median Q / P")
    axes[1].set_title("Reactive fraction — flat implies fixed power factor")
    for ax in axes:
        ax.set_xlabel("Site voltage (V)")
        ax.axvline(_A["VVAR"]["V3"], color=C_GREY, ls=":", lw=1.0)
        ax.axvline(_A["VW"]["V1"], color=C_ACCENT, ls="--", lw=1.0)
        ax.legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_derating_by_voltage(frame, figsize=(11, 4.2)):
    """Inverter-reported derating rate against voltage."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    _vvar_bands(ax)
    ax.plot(frame.v_bin, frame.pct_derating, color=C_ACCENT, lw=2.0, marker="o", ms=3.5,
            label="Intervals with derating_active (%)")
    ax.set_xlabel("Site voltage (V)")
    ax.set_ylabel("Derating active (% of intervals)")
    ax.set_title("SolarEdge derating flag against voltage")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_diurnal_profile(frame, figsize=(11, 3.6)):
    """Fleet mean power by hour in the AEST frame. Should peak at hour 12."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.bar(frame.hour_aest, frame.mean_P_kW, width=0.85, color=C_P, alpha=0.9)
    peak = int(frame.loc[frame.mean_P_kW.idxmax(), "hour_aest"])
    ax.axvline(peak, color=C_ACCENT, ls="--", lw=1.2, label=f"peak hour = {peak}")
    ax.set_xlabel("Hour of day (AEST, fixed UTC+10)")
    ax.set_ylabel("Mean P (kW)")
    ax.set_title("Fleet diurnal profile — timezone resolution check")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_capacity_distribution(frame, figsize=(9, 3.6)):
    """Distribution of the s_99 empirical apparent-power limit."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.bar(frame.s_99_bin, frame.n_sites, width=0.9, color=C_P, alpha=0.9)
    ax.set_xlabel("s_99 empirical apparent-power limit (kVA)")
    ax.set_ylabel("Sites")
    ax.set_title("Capacity proxy — no nameplate exists in this delivery")
    fig.tight_layout()
    return fig


def plot_q_categories(summary, figsize=(10, 4.0)):
    """
    Volt-VAr conformance categories as a share of assessable intervals, by cohort.

    The three left-hand bars are the reduced non-conformance set. `Q_near_conformant`
    is deliberately NOT part of it — those inverters deliver 90–110% of required
    reactive power.
    """
    from solar_edge.lib.se_conformance import Q_CATEGORIES, REDUCED_NONCONF

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    width = 0.8 / max(len(summary), 1)
    x = np.arange(len(Q_CATEGORIES))

    for offset, (_, row) in enumerate(summary.iterrows()):
        denom = row.capability_assessable_intervals or 1
        heights = [100 * row[f"{c}_intervals"] / denom for c in Q_CATEGORIES]
        ax.bar(x + offset * width, heights, width=width * 0.92, label=row.cohort)

    ax.set_xticks(x + width * (len(summary) - 1) / 2)
    ax.set_xticklabels([c.replace("Q_", "").replace("_", "\n") for c in Q_CATEGORIES],
                       fontsize=8)
    ax.set_ylabel("% of capability-assessable intervals")
    ax.set_title("Volt-VAr Q_impact categories")
    for i, cat in enumerate(Q_CATEGORIES):
        if cat in REDUCED_NONCONF:
            ax.axvspan(i - 0.42, i + 0.42 + width * (len(summary) - 1),
                       color=C_ACCENT, alpha=0.07, lw=0, zorder=0)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_q_impact_distribution(frame, figsize=(10, 3.8)):
    """
    Distribution of the signed Q_impact ratio.

    1.0 means the inverter sat exactly on the nearest permitted band edge; 0 means
    no response; below 0 means the wrong direction. A fleet clustered near 0 is
    responding weakly regardless of what the category counts say.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    for cohort, group in frame.groupby("cohort"):
        share = 100 * group.n_intervals / group.n_intervals.sum()
        colour = C_Q if cohort.startswith("single") else C_ACCENT
        ax.plot(group.q_impact_bin, share, lw=1.8, marker="o", ms=3,
                color=colour, label=cohort)
    for x, label in ((0.0, "no response"), (1.0, "at required edge")):
        ax.axvline(x, color=C_GREY, ls="--", lw=1.0)
        ax.text(x, ax.get_ylim()[1] * 0.92, f" {label}", fontsize=7.5, color=C_GREY)
    ax.set_xlabel("Q_impact  (measured / required, signed)")
    ax.set_ylabel("% of assessable intervals")
    ax.set_title("Volt-VAr response strength")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_sign_flip(flip, figsize=(10, 3.8)):
    """Three-phase Volt-VAr categories scored as stored versus sign-flipped."""
    from solar_edge.lib.se_conformance import Q_CATEGORIES

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    x = np.arange(len(Q_CATEGORIES))
    for offset, (_, row) in enumerate(flip.iterrows()):
        denom = row.assessable_intervals or 1
        heights = [100 * row[c] / denom for c in Q_CATEGORIES]
        ax.bar(x + offset * 0.4, heights, width=0.37,
               label=f"{row.scenario}  ({row.reduced_nonconf_pct}% reduced non-conf)")
    ax.set_xticks(x + 0.2)
    ax.set_xticklabels([c.replace("Q_", "").replace("_", "\n") for c in Q_CATEGORIES],
                       fontsize=8)
    ax.set_ylabel("% of assessable intervals")
    ax.set_title("Three-phase cohort: the cost of the sign decision")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_monthly_coverage(frame, figsize=(10, 3.4)):
    """Reporting sites and rows per month."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    x = np.arange(len(frame))
    ax.bar(x, frame.n_rows / 1e6, color=C_GREY, alpha=0.55, label="Rows (millions)")
    ax2 = ax.twinx()
    ax2.plot(x, frame.n_sites, color=C_ACCENT, lw=2.0, marker="o", ms=4,
             label="Reporting sites")
    ax.set_xticks(x)
    ax.set_xticklabels(frame.dt_month, rotation=45, ha="right")
    ax.set_ylabel("Rows (millions)")
    ax2.set_ylabel("Reporting sites", color=C_ACCENT)
    ax2.tick_params(axis="y", colors=C_ACCENT)
    ax.set_title("Coverage by AEST month")
    fig.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(0.98, 0.16))
    fig.tight_layout()
    return fig
