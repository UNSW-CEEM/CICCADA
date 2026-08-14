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

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from solar_edge.config import se_config as C

__all__ = [
    "plot_voltage_distribution",
    "plot_voltvar_signature",
    "plot_derating_by_voltage",
    "plot_diurnal_profile",
    "plot_capacity_distribution",
    "plot_monthly_coverage",
    "plot_reactive_character",
    "plot_voltvar_spread",
    "plot_site_response_histogram",
    "plot_q_categories",
    "plot_conformance_dashboard",
    "plot_site_verdicts",
    "plot_breakdown_rates",
    "plot_postcode_map",
    "postcode_centroids",
    "plot_q_impact_distribution",
    "plot_sign_flip",
]

_A = C.as4777()
C_V, C_P, C_Q = "#b45309", "#2a7f62", "#4709b2"
C_ACCENT, C_GREY = "#e8702a", "#8a8a8a"
C_NA_GREY = "#9e9e9e"   # "not assessable" -- absent, not passing


def _done(fig):
    """
    Close the figure before returning it.

    The notebooks call ``display(fig)``. With the inline backend an OPEN figure is
    ALSO auto-rendered when the cell finishes, so every plot appeared twice.
    Closing it suppresses the automatic render; ``display(fig)`` still works,
    because it draws from the figure object rather than the pyplot state.
    """
    plt.close(fig)
    return fig


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
    return _done(fig)


def plot_voltvar_signature(frame, figsize=(11, 4.6)):
    """
    DEPRECATED -- use ``plot_voltvar_spread`` instead.

    A single median line over pooled intervals is misleading on this fleet for two
    reasons: it interval-weights the sites, so a heavily-reporting site counts many
    times over; and it collapses a bimodal population -- most sites near zero, a
    minority absorbing strongly -- into a point that sits between the two groups
    and describes neither.

    Kept only so existing references do not break. It emits a warning.
    """
    import warnings

    warnings.warn(
        "plot_voltvar_signature pools intervals across sites and hides the "
        "bimodality; use plot_voltvar_spread (per-site quantiles) instead.",
        stacklevel=2,
    )
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
    return _done(fig)


def plot_reactive_character(frame, figsize=(13, 4.2)):
    """
    Three panels.

    Left: median |Q| -- the numerator. Rising means the inverter IS doing more.
    Middle: median P -- the denominator, shown so the confound is visible.
    Right: median POWER FACTOR, cos(phi) = P / sqrt(P^2 + Q^2).

    Power factor is the correct normalised quantity. An earlier version plotted
    Q/P here and called it power factor; Q/P is tan(phi), a different thing, and it
    is confounded on top of that -- site voltage rises when export rises, so P
    grows faster than |Q| and the ratio falls even while the response strengthens.
    tan(phi) is still in the table as ``med_tan_phi`` for reference.

    Power factor near 1.0 means the inverter is moving almost no reactive power.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=130, sharex=True)
    for cohort, group in frame.groupby("cohort"):
        colour = C_Q if cohort.startswith("single") else C_ACCENT
        for ax, col in zip(axes, ("med_abs_Q_kvar", "med_P_kW", "med_power_factor")):
            ax.plot(group.v_bin, group[col], marker="o", ms=3.5, lw=1.8,
                    color=colour, label=cohort)

    axes[0].set_ylabel("Median |Q| (kvar)")
    axes[0].set_title("Numerator: reactive magnitude\n(rising = doing more)", fontsize=9.5)
    axes[1].set_ylabel("Median P (kW)")
    axes[1].set_title("Denominator: active power\n(rises faster)", fontsize=9.5)
    axes[2].axhline(1.0, color="black", lw=0.8, alpha=0.6)
    axes[2].set_ylabel("Median power factor, cos(phi) = P / S")
    axes[2].set_title("Power factor\n(1.0 = no reactive power at all)", fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("Site voltage (V)")
        ax.axvline(_A["VVAR"]["V3"], color=C_GREY, ls=":", lw=1.0)
        ax.axvline(_A["VW"]["V1"], color=C_ACCENT, ls="--", lw=1.0)
        ax.legend(fontsize=7.5, framealpha=0.9)
        ax.grid(color="#ebebeb", lw=0.5)
    fig.tight_layout()
    return _done(fig)


def plot_voltvar_spread(spread, figsize=(12, 4.4)):
    """
    Reactive response ACROSS SITES, as a distribution rather than a single line.

    Each site contributes one median per voltage bin; the bands are quantiles of
    those site medians, so every site counts once regardless of how much data it
    reported.

    This replaces the pooled median line. That line sat entirely above zero and
    read as "the fleet supplies reactive power", which is an average over two
    populations that behave oppositely. Here the p10 edge dives negative while
    p25-p90 stay flat and positive -- a minority absorbing strongly, the majority
    doing nothing. No single line can show that.
    """
    cohorts = list(spread.cohort.unique())
    fig, axes = plt.subplots(1, max(len(cohorts), 1), figsize=figsize, dpi=130,
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, cohort in zip(axes, cohorts):
        g = spread[spread.cohort == cohort].sort_values("v_bin")
        ax.axvspan(_A["VVAR"]["V2"], _A["VVAR"]["V3"], color=C_GREY, alpha=0.10, lw=0,
                   label=f"Volt-VAr deadband ({_A['VVAR']['V2']:.0f}-{_A['VVAR']['V3']:.0f} V)")
        ax.axvline(_A["VW"]["V1"], color=C_ACCENT, ls="--", lw=1.2,
                   label=f"{_A['VW']['V1']:.0f} V - Volt-Watt starts")
        ax.axhline(0, color="black", lw=0.9)
        ax.fill_between(g.v_bin, g.p10, g.p90, color=C_Q, alpha=0.15, lw=0,
                        label="p10-p90 of sites")
        ax.fill_between(g.v_bin, g.p25, g.p75, color=C_Q, alpha=0.32, lw=0,
                        label="p25-p75 of sites")
        ax.plot(g.v_bin, g.p50, color=C_Q, lw=2.0, marker="o", ms=3,
                label="median site")
        ax.set_xlabel("Site voltage (V)")
        ax.set_title(f"{cohort}\n{int(g.n_sites.max()):,} sites at peak bin", fontsize=9.5)
        ax.grid(color="#ebebeb", lw=0.5)
        ax.legend(fontsize=7.5, framealpha=0.9, loc="lower left")

    axes[0].set_ylabel("Site median Q (kvar)\nnegative = absorbing")
    fig.tight_layout()
    return _done(fig)


def plot_site_response_histogram(response, figsize=(12, 4.0), clip=2.0):
    """
    One number per site: how far its median Q moves from the deadband to the
    upper Volt-VAr ramp.

    In the generator convention a conforming inverter absorbs more as voltage
    rises, so responders sit LEFT of zero (shaded). Everything piled at zero is a
    site that does not respond at all.

    This is the cleanest answer to "how many of my inverters actually do
    Volt-VAr" -- a question a pooled median cannot address, because it averages the
    two groups into a number that describes neither.
    """
    cohorts = list(response.cohort.unique())
    fig, axes = plt.subplots(1, max(len(cohorts), 1), figsize=figsize, dpi=130,
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, cohort in zip(axes, cohorts):
        g = response[response.cohort == cohort]
        vals = g.delta_q_kvar.clip(-clip, clip)
        ax.axvspan(-clip, -0.05, color=C_P, alpha=0.10, lw=0)
        ax.hist(vals, bins=60, color=C_Q, alpha=0.85)
        ax.axvline(0, color="black", lw=1.0)
        ax.axvline(-0.05, color=C_P, ls="--", lw=1.2)
        n_resp = int((g.delta_q_kvar < -0.05).sum())
        ax.set_title(f"{cohort}\n{n_resp} of {len(g)} sites respond "
                     f"({100 * n_resp / max(len(g), 1):.0f}%)", fontsize=9.5)
        ax.set_xlabel("delta median Q, deadband -> high voltage (kvar)")
        ax.grid(color="#ebebeb", lw=0.5, axis="y")

    axes[0].set_ylabel("Sites")
    fig.suptitle("<- absorbing more (conforming)      |      no response / adverse ->",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    return _done(fig)


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
    return _done(fig)


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
    return _done(fig)


def plot_capacity_distribution(frame, figsize=(9, 3.6), title="Capacity proxy"):
    """Distribution of the s_99 empirical apparent-power limit."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.bar(frame.s_99_bin, frame.n_sites, width=0.9, color=C_P, alpha=0.9)
    ax.set_xlabel("s_99 empirical apparent-power limit (kVA)")
    ax.set_ylabel("Sites")
    ax.set_title(title)
    fig.tight_layout()
    return _done(fig)


def plot_q_categories(summary, figsize=(12, 4.4)):
    """
    Volt-VAr conformance categories, one panel per cohort.

    Shares a y-axis so the two are directly comparable, and shades the three
    reduced-non-conformance categories. `Q_near_conformant` is deliberately NOT
    among them — those inverters deliver 90–110% of required reactive power.
    """
    from solar_edge.lib.se_conformance import Q_CATEGORIES, REDUCED_NONCONF

    n = max(len(summary), 1)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=130, sharey=True, squeeze=False)
    axes = axes[0]
    labels = [c.replace("Q_", "").replace("_", "\n") for c in Q_CATEGORIES]
    x = np.arange(len(Q_CATEGORIES))

    for ax, (_, row) in zip(axes, summary.iterrows()):
        denom = row.capability_assessable_intervals or 1
        heights = [100 * row[f"{c}_intervals"] / denom for c in Q_CATEGORIES]
        colours = [C_ACCENT if c in REDUCED_NONCONF else C_P for c in Q_CATEGORIES]
        bars = ax.bar(x, heights, width=0.72, color=colours, alpha=0.9)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%",
                    ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{row.cohort}\n{int(row.n_sites):,} sites | "
                     f"{int(denom):,} assessable intervals\n"
                     f"reduced non-conformance {row.reduced_nonconf_pct:.1f}%",
                     fontsize=9.5)
        ax.grid(color="#ebebeb", lw=0.5, axis="y")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("% of capability-assessable intervals")
    fig.legend(handles=[Patch(color=C_ACCENT, alpha=0.9,
                              label="Considered non-conformance (adverse + inactive + shortfall)"),
                        Patch(color=C_P, alpha=0.9, label="not counted as non-conformant")],
               fontsize=8, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return _done(fig)


def plot_conformance_dashboard(measures, figsize=(12.5, 12.0)):
    """
    The Volt-VAr result in all four currencies, one row per measure, one column
    per cohort. Extends ``plot_q_categories``, which is the first row here.

    Milestone 3 reported Volt-VAr as intervals, as sites, as kVArh and as
    kVArh/kW/h. They are not interchangeable and the figure exists so the
    disagreement is visible rather than a choice buried in a caption:

    * rows 1-2 differ when non-conformance is CONCENTRATED -- a high interval
      share with a low site share means a handful of inverters misbehaving
      constantly, not a fleet-wide problem;
    * rows 1 and 3 differ when severity and frequency come apart -- many small
      misses versus few large ones;
    * row 4 removes system size and reporting volume from row 3, so a fleet that
      merely contains larger inverters no longer looks worse behaved.

    Rows 1, 2 and 4 share a y-axis across cohorts and are directly comparable.
    Row 3 does NOT -- kVArh totals depend on how many sites are in each cohort,
    so comparing raw heights across panels would compare fleet sizes. Read row 4
    for the cohort comparison.
    """
    from solar_edge.lib.se_conformance import MEASURES, Q_CATEGORIES, REDUCED_NONCONF

    cohorts = list(dict.fromkeys(measures.cohort))
    keys = list(MEASURES)
    labels = [c.replace("Q_", "").replace("_", "\n") for c in Q_CATEGORIES]
    x = np.arange(len(Q_CATEGORIES))
    colours = [C_ACCENT if c in REDUCED_NONCONF else C_P for c in Q_CATEGORIES]

    fig, axes = plt.subplots(len(keys), max(len(cohorts), 1), figsize=figsize,
                             dpi=130, squeeze=False)

    for r, key in enumerate(keys):
        title, subtitle = MEASURES[key]
        # kVArh totals scale with cohort size, so a shared axis would invite a
        # comparison of fleet sizes dressed up as a comparison of behaviour.
        share = key != "kvarh"
        top = max(1e-9, float(measures[key].max())) * 1.22 if share else None

        for c, cohort in enumerate(cohorts):
            ax = axes[r][c]
            g = measures[measures.cohort == cohort].set_index("category")
            heights = [float(g.loc[cat, key]) for cat in Q_CATEGORIES]
            bars = ax.bar(x, heights, width=0.72, color=colours, alpha=0.9)

            hi = max(heights) if max(heights) > 0 else 1.0
            for bar, h in zip(bars, heights):
                fmt = format(h, ",.0f") if key == "kvarh" else (
                    format(h, ".3f") if key == "kvarh_per_kw_per_h"
                    else format(h, ".1f") + "%")
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03 * hi, fmt,
                        ha="center", fontsize=7.2)

            ax.set_ylim(0, top if share else hi * 1.22)
            ax.set_xticks(x)
            ax.set_xticklabels(labels if r == len(keys) - 1 else [], fontsize=7.8)
            ax.grid(color="#ebebeb", lw=0.5, axis="y")
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(title + "\n(" + subtitle + ")", fontsize=8.5)
            elif share:
                ax.tick_params(labelleft=False)
            if r == 0:
                row = g.iloc[0]
                ax.set_title(cohort + "\n" + format(int(row.n_sites), ",")
                             + " sites | "
                             + format(int(row.assessable_intervals), ",")
                             + " assessable intervals",
                             fontsize=10, fontweight="bold")

    fig.legend(handles=[
        Patch(color=C_ACCENT, alpha=0.9,
              label="Counted as non-conformance (adverse + inactive + shortfall)"),
        Patch(color=C_P, alpha=0.9, label="Not counted as non-conformant")],
        fontsize=8.5, loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=(0, 0.022, 1, 1))
    return _done(fig)


def plot_site_verdicts(verdicts, config=None, figsize=(11.0, 8.0),
                       measures=None, mode="Volt-VAr", denominator=None):
    """
    The 10% site rule, in the same four currencies as the category dashboard.

    ``plot_conformance_dashboard`` splits non-conformance across five Q_impact
    categories. This collapses them to the binary the standard's reporting
    actually turns on -- conformant vs non-conformant on REDUCED non-conformance
    (adverse + inactive + significant shortfall) -- and shows the same four
    denominators:

    * **% of sites** -- one vote per site, the 10% rule itself;
    * **% of assessable intervals** -- the intervals those sites contribute;
    * **kVArh** -- the reactive-energy shortfall attributable to each verdict;
    * **kVArh/kW/h** -- that energy normalised by rating and assessable time.

    The four disagree in a specific and useful way here. Non-conformant sites are
    a minority by headcount but hold most of the intervals and nearly all of the
    missing kVArh, because a site fails the 10% rule by misbehaving persistently.
    A verdict count alone hides that concentration; this figure is the argument
    for not reporting the site rate on its own.

    ``verdicts`` is the frame from ``se_conformance.site_verdict_measures`` or
    ``voltwatt_verdict_measures``. Pass the matching ``measures`` mapping
    (``VERDICT_MEASURES`` / ``VW_VERDICT_MEASURES``) so the axis labels and number
    formats follow the units -- kVArh missing for Volt-VAr, kWh over the ceiling
    for Volt-Watt.
    """
    from solar_edge.lib.se_conformance import VERDICT_MEASURES

    measures = measures or VERDICT_MEASURES
    order = ["conformant", "non-conformant",
             "not assessable", "not exposed", "not supported"]
    colours = {"conformant": C_P, "non-conformant": C_ACCENT,
               "not assessable": C_NA_GREY, "not exposed": C_NA_GREY,
               "not supported": C_NA_GREY}
    cohorts = list(dict.fromkeys(verdicts.cohort))
    keys = list(measures)

    fig, axes = plt.subplots(len(keys), max(len(cohorts), 1), figsize=figsize,
                             dpi=130, squeeze=False)

    for r, key in enumerate(keys):
        title, subtitle = measures[key]
        share = key not in ("kvarh", "kwh")
        top = max(1e-9, float(verdicts[key].max())) * 1.24 if share else None

        for c, cohort in enumerate(cohorts):
            ax = axes[r][c]
            g = verdicts[verdicts.cohort == cohort].set_index("verdict")
            present = [v for v in order if v in g.index]
            x = np.arange(len(present))
            heights = [float(g.loc[v, key]) for v in present]
            bars = ax.bar(x, heights, width=0.62,
                          color=[colours[v] for v in present], alpha=0.9)

            hi = max(heights) if max(heights) > 0 else 1.0
            for bar, h in zip(bars, heights):
                fmt = format(h, ",.0f") if key in ("kvarh", "kwh") else (
                    format(h, ".3f") if key.endswith("_per_kw_per_h")
                    else format(h, ".1f") + "%")
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03 * hi, fmt,
                        ha="center", fontsize=7.8)

            ax.set_ylim(0, top if share else hi * 1.24)
            ax.set_xticks(x)
            ax.set_xticklabels([v.replace("-", "-\n") for v in present]
                               if r == len(keys) - 1 else [], fontsize=8)
            ax.grid(color="#ebebeb", lw=0.5, axis="y")
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(title, fontsize=9)
                ax.annotate(subtitle, xy=(-0.155, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center",
                            fontsize=7.2, color=C_GREY)
            elif share:
                ax.tick_params(labelleft=False)
            if r == 0:
                n = int(g.n_sites.sum())
                ax.set_title(cohort + "\n" + format(n, ",") + " sites",
                             fontsize=10, fontweight="bold")

    thr = (config.site_nonconf_threshold if config is not None else 0.10) * 100
    if denominator is None:
        denominator = ("reduced non-conformance = adverse + inactive + shortfall"
                       if mode == "Volt-VAr" else "P above the ceiling + 4%")
    fig.suptitle(f"{mode} site verdicts on the {thr:.0f}% rule ({denominator})",
                 fontsize=10.5, y=1.005)
    fig.tight_layout(h_pad=1.4, rect=(0.02, 0, 1, 0.985))
    return _done(fig)


def plot_breakdown_rates(site_pct, interval_pct, by, interval_col,
                         label=None, figsize=(11.5, 4.4), min_sites=1):
    """
    A breakdown dimension seen both ways at once: sites on the left, intervals on
    the right, same groups, same order.

    The two disagree whenever non-conformance is concentrated. A group where 40%
    of *sites* fail but 80% of *intervals* are non-conformant contains a few
    heavily observed bad actors; the reverse means many sites each failing
    occasionally. Reporting one number per group hides which.

    ``n_sites`` is annotated on every bar because these rates are ratios over
    small denominators in the tail -- a 100% group with three sites is not a
    finding, and the figure should make that impossible to miss.
    """
    site_col = next((c for c in site_pct.columns
                     if c.startswith("pct_sites_non")), None)
    if site_col is None:
        raise ValueError(f"no non-conformant column in {list(site_pct.columns)}")

    left = site_pct[site_pct.n_sites >= min_sites].copy()
    right = interval_pct[interval_pct.n_sites >= min_sites].copy()
    merged = left[[by, "n_sites", site_col]].merge(
        right[[by, interval_col]], on=by, how="inner")
    if isinstance(merged[by].dtype, pd.CategoricalDtype):
        merged = merged.sort_values(by)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=130, sharey=True)
    y = np.arange(len(merged))
    panels = [
        (axes[0], site_col, "% of SITES non-conformant", C_ACCENT),
        (axes[1], interval_col, "% of INTERVALS non-conformant", C_Q),
    ]
    for ax, col, title, colour in panels:
        vals = merged[col].astype(float).fillna(0)
        ax.barh(y, vals, color=colour, alpha=0.85, height=0.66)
        for yi, (v, n) in enumerate(zip(vals, merged.n_sites)):
            ax.text(v + max(vals.max(), 1) * 0.015, yi,
                    f"{v:.1f}%  (n={int(n):,})", va="center", fontsize=7.4)
        ax.set_xlim(0, max(vals.max(), 1) * 1.28)
        ax.set_title(title, fontsize=9.5)
        ax.grid(color="#ebebeb", lw=0.5, axis="x")
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([str(v) for v in merged[by]], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.suptitle(label or by, fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _done(fig)


def postcode_centroids(con, shapefile=None) -> pd.DataFrame:
    """
    Postcode -> representative point, from the store if D4 attached geography,
    otherwise straight from the POA-2021 shapefile.

    ``representative_point()`` rather than the geometric centroid: a centroid can
    fall outside a concave or multi-part polygon, and Australian postcodes include
    plenty of both.
    """
    from solar_edge.config import se_config as C

    got = con.execute(
        """
        SELECT postcode,
               any_value(centroid_lat) AS lat,
               any_value(centroid_lon) AS lon
        FROM se_site
        WHERE centroid_lat IS NOT NULL AND postcode IS NOT NULL
        GROUP BY postcode
        """
    ).df()
    if len(got):
        return got

    import geopandas as gpd

    path = pathlib.Path(shapefile or C.POA_SHAPEFILE)
    if not path.exists():
        raise FileNotFoundError(
            f"No centroids in se_site and no shapefile at {path}.\n"
            "  Run notebook 04 section 2 (attach_geography) first."
        )
    poa = gpd.read_file(path)[["POA_CODE21", "geometry"]].to_crs("EPSG:4326")
    pts = poa.geometry.representative_point()
    return pd.DataFrame({"postcode": poa.POA_CODE21.astype(str),
                         "lat": pts.y.values, "lon": pts.x.values})


def plot_postcode_map(con, frame, value_col="pct_reduced_nonconf", *,
                      min_sites=5, label=None, title=None, shapefile=None,
                      context=True, extent=None, figsize=(10.5, 9.0),
                      cmap="RdYlGn_r", vmin=None, vmax=None):
    """
    Conformance by postcode, as a **bubble map** rather than a choropleth.

    That is a deliberate choice, not a convenience. Australian postcode areas span
    more than four orders of magnitude -- 4702 covers ~50,000 km^2, a Sydney
    postcode a couple. Filling polygons by rate hands almost the entire visual
    field to a handful of enormous rural postcodes carrying a handful of sites,
    while the dense metropolitan postcodes where most of the fleet actually lives
    shrink to invisible specks. The reader's eye then weights the map by land
    area, which is the one variable that carries no information here.

    So: one marker per postcode at its representative point, **area proportional
    to the number of sites** and **colour to the rate**. Both variables the reader
    should be weighting are then encoded explicitly, and the postcode outlines are
    drawn faintly underneath only as geographic context.

    Postcodes below ``min_sites`` are drawn as small hollow grey markers rather
    than dropped. A 100% rate over two sites is not a finding, but silently
    deleting it would misrepresent coverage -- the reader should see where the
    fleet is thin.

    ``label`` names the colour scale, ``title`` heads the figure; zoomed panels
    want a different title but the same scale label. The colour limits are taken
    from the WHOLE frame, not from what falls inside ``extent``, so a set of metro
    zooms stays comparable with each other and with the national view. Rescaling
    per panel would let a well-behaved city look as red as a badly behaved one.
    """
    from solar_edge.config import se_config as C

    data = frame.copy()
    data["postcode"] = data.postcode.astype(str)
    centroids = postcode_centroids(con, shapefile)
    centroids["postcode"] = centroids.postcode.astype(str)
    data = data.merge(centroids, on="postcode", how="left")

    missing = int(data.lat.isna().sum())
    data = data.dropna(subset=["lat", "lon", value_col])
    if data.empty:
        raise ValueError(f"No postcode in {value_col} could be located.")

    big = data[data.n_sites >= min_sites]
    small = data[data.n_sites < min_sites]

    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    if context:
        try:
            import geopandas as gpd

            path = pathlib.Path(shapefile or C.POA_SHAPEFILE)
            if path.exists():
                poa = gpd.read_file(path)[["POA_CODE21", "geometry"]].to_crs("EPSG:4326")
                poa = poa[poa.POA_CODE21.astype(str).isin(set(data.postcode))]
                poa.boundary.plot(ax=ax, linewidth=0.35, color="#c9c9c9", zorder=1)
        except Exception as exc:            # context is optional, never fatal
            print(f"(postcode outlines skipped: {exc})")

    # Area, not radius, proportional to site count -- perceived size is area.
    size = 18 + 5.5 * big.n_sites.clip(upper=big.n_sites.quantile(0.98))
    sc = ax.scatter(big.lon, big.lat, s=size, c=big[value_col], cmap=cmap,
                    vmin=vmin if vmin is not None else float(big[value_col].min()),
                    vmax=vmax if vmax is not None else float(big[value_col].max()),
                    edgecolor="#333333", linewidth=0.4, alpha=0.9, zorder=3)
    if len(small):
        ax.scatter(small.lon, small.lat, s=10, facecolor="none",
                   edgecolor=C_NA_GREY, linewidth=0.5, alpha=0.7, zorder=2,
                   label=f"< {min_sites} sites (not ranked)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.02)
    cbar.set_label(label or value_col, fontsize=9)

    # Count what the reader can actually see, not the national total.
    if extent:
        in_view = big[(big.lon.between(extent[0], extent[1]))
                      & (big.lat.between(extent[2], extent[3]))]
    else:
        in_view = big

    if extent:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    else:
        pad = 1.0
        ax.set_xlim(data.lon.min() - pad, data.lon.max() + pad)
        ax.set_ylim(data.lat.min() - pad, data.lat.max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(float(data.lat.mean()))))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(color="#f0f0f0", lw=0.5)
    ax.set_axisbelow(True)

    heading = title or label or value_col
    shown = (f"{len(in_view):,} postcodes shown with >= {min_sites} sites"
             if extent else
             f"{len(big):,} postcodes with >= {min_sites} sites")
    ax.set_title(f"{heading}\nmarker area = sites in postcode  |  {shown}"
                 + (f"  |  {missing} unlocated" if missing else ""),
                 fontsize=10.5)
    fig.tight_layout()
    return _done(fig)


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
    return _done(fig)


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
    return _done(fig)


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
    return _done(fig)
