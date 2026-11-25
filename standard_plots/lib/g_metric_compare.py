"""
Family G: the same intervals counted three ways (Volt-VAr metrics side by side).
================================================================================

    G.0  flip_table                 one row per site: tag under metric a and b, and
                                    where the site's failures come from (per zone)
    G.0  pick_examples              a few sites per change class, for figures
    G.1  plot_metric_comparison_day one site-day, one column per metric
    G.2  plot_qv_metric_comparison  one site, several days, one Q-V panel per metric

Dataset-agnostic: the figures take the standard_plots tidy frame + info dict
(sources.load_local / load_aws / load_month, or an OEM site_frame), and the
tables take a one-row-per-site frame from shared.vvar_metrics.conformance_metrics
(all three `<metric>_verdict` columns). `key` names the site column
('site_id' for Solar Analytics, 'site_alias' for OEM).

The band is identical in every column; only what is counted changes.

Zones (shared.as4777_curves.ZONE_LABELS):
    A  response required              (P >= 20%, V < 220 or V > 240 V)
    B  sufficient power, voltage deadband (P >= 20%, 220-240 V)
    C  low power, voltage exposed     (P < 20%, V < 220 or V > 240 V)
    D  low power, voltage deadband    (P < 20%, 220-240 V)
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from standard_plots.lib import a_site_day as A
from standard_plots.lib import b_qv_response as B
from standard_plots.lib import palette as PAL

__all__ = [
    "METRICS",
    "SHORT",
    "ZONE_NAMES",
    "flip_table",
    "pick_examples",
    "plot_metric_comparison_day",
    "plot_qv_metric_comparison",
]

#: Display order everywhere: legacy first (the original results), then the two new ones.
METRICS = ("legacy_conformance_rate", "nonconformance_frequency", "remote_conformance_rate")
SHORT = {
    "legacy_conformance_rate": "legacy",
    "nonconformance_frequency": "frequency",
    "remote_conformance_rate": "remote",
}
ZONES = ("A", "B", "C", "D")
#: Column-friendly zone names (used in flip_table column names).
ZONE_NAMES = {
    "A": "A_response_required",
    "B": "B_sufficient_power_voltage_deadband",
    "C": "C_low_power_voltage_exposed",
    "D": "D_low_power_voltage_deadband",
}


def _wrapped_summary(vv, width: int = 78) -> str:
    """standard_plots metric summary, wrapped to fit a narrow panel."""
    return "\n".join(
        textwrap.fill(line.replace("  ·  ", " · "), width)
        for line in A.metric_summary(vv).split("\n")
    )


# ═══════════════════════════════════════════════════════════════════════════
# G.0 Which sites change tag
# ═══════════════════════════════════════════════════════════════════════════
def flip_table(metrics: pd.DataFrame, a: str, b: str, key: str = "site_id",
               extra=()) -> pd.DataFrame:
    """
    One row per site: tag under metric `a` and `b`, a `change` label
    ('legacy conformant -> remote non-conformant', ..., 'same: conformant', ...),
    the three non-conforming shares, voltage exposure, coverage, and the share of
    the site's failures in each zone (failures_in_zone_<zone name>_%).
    `metrics`: one row per site with the conformance_metrics columns.
    `extra`: further columns to keep (e.g. metadata merged in).
    """
    m = metrics.reset_index() if key not in metrics.columns else metrics.copy()
    ta, tb = m[f"{a}_verdict"], m[f"{b}_verdict"]
    m["change"] = np.where(
        ta == tb,
        "same: " + ta.astype(str),
        f"{SHORT[a]} " + ta.astype(str) + f" -> {SHORT[b]} " + tb.astype(str),
    )
    fails = m[[f"fail_{z}" for z in ZONES]]
    tot = fails.sum(axis=1).replace(0, np.nan)
    for z in ZONES:
        m[f"failures_in_zone_{ZONE_NAMES[z]}_%"] = (100 * m[f"fail_{z}"] / tot).round(1)
    keep = (
        [key, *extra, "change", f"{a}_verdict", f"{b}_verdict",
         "legacy_nonconformance_rate", "nonconformance_frequency",
         "remote_nonconformance_rate", "voltage_exposure", "coverage",
         "n_valid", "n_A"]
        + [f"failures_in_zone_{ZONE_NAMES[z]}_%" for z in ZONES]
    )
    seen = []
    for c in keep:
        if c in m.columns and c not in seen:
            seen.append(c)
    return m[seen]


def pick_examples(flips: pd.DataFrame, n_changed: int = 4, n_same: int = 2,
                  seed: int = 0) -> pd.DataFrame:
    """
    Up to `n_changed` sites per changed class and `n_same` per unchanged class,
    preferring sites with many zone A (response required) intervals, so the
    figures have something to show. Changed classes first.
    """
    picks = []
    for change, g in flips.groupby("change"):
        n = n_same if change.startswith("same:") else n_changed
        pool = g[g.n_A >= g.n_A.quantile(0.5)] if len(g) > n else g
        picks.append(pool.sample(min(n, len(pool)), random_state=seed))
    if not picks:
        return flips.head(0)
    out = pd.concat(picks, ignore_index=True)
    order = out.change.str.startswith("same:")
    return (out.assign(_o=order).sort_values(["_o", "change"])
            .drop(columns="_o").reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════
# G.1 / G.2 Side-by-side figures (standard_plots building blocks)
# ═══════════════════════════════════════════════════════════════════════════
def plot_metric_comparison_day(df, info, metrics=METRICS, tol=None, failure_definition=None):
    """
    One site-day, one column per metric. Top (shared): voltage and active
    power. Per column: the Q panel (accepted band; hatched = not counted by
    that metric; zone strip) and the outside-band bars with that metric's
    count, rate and site tag. The band is identical in every column; only the
    counting changes.
    """
    day = A.prepare(df, info["rating_kva"], info)
    vvs = [A.assess_voltvar(day, m, tol, failure_definition) for m in metrics]
    n = len(metrics)
    fig = plt.figure(figsize=(6.2 * n, 13))
    gs = fig.add_gridspec(4, n, height_ratios=[1.3, 1.3, 2.4, 1.2], hspace=0.12, wspace=0.28)
    a_v = fig.add_subplot(gs[0, :])
    a_p = fig.add_subplot(gs[1, :], sharex=a_v)
    A.panel_voltage(a_v, day, "voltvar")
    A.panel_active_power(a_p, day, gate=True)
    for ax in (a_v, a_p):
        ax.tick_params(labelbottom=False)
    for j, (m, vv) in enumerate(zip(metrics, vvs)):
        a_q = fig.add_subplot(gs[2, j], sharex=a_v)
        a_nc = fig.add_subplot(gs[3, j], sharex=a_v)
        A.panel_reactive_power(a_q, day, vv)
        A.panel_nc_voltvar(a_nc, day, vv)
        a_q.set_title(A.METRIC_LABELS[m], fontsize=10, loc="left")
        a_q.tick_params(labelbottom=False)
        A._time_axis(a_nc)
        if j > 0:  # one legend per row is enough
            for ax in (a_q, a_nc):
                if ax.get_legend():
                    ax.get_legend().remove()
                ax.set_ylabel("")
        for t in a_nc.texts:  # summary box: wrapped to the column width
            t.set_text(_wrapped_summary(vv))
            t.set_fontsize(6.8)
    fig.suptitle(
        f"Same intervals, three ways of counting\n{info.get('label', '')}  ·  {info.get('date', '')}"
        f"  ·  rating {info['rating_kva']:g} kVA ({info.get('rating_basis', '')})",
        x=0.06, ha="left", y=0.995, fontsize=10.5,
    )
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.06, right=0.96)
    return fig


def plot_qv_metric_comparison(df, info, metrics=METRICS, tol=None, failure_definition=None):
    """
    One site, several days (e.g. a month), one Q-V panel per metric with the
    standard_plots Q-V axes (curve, tolerance band, dashed 44% line). Points:
    counted failure (non-conformant colour), counted pass (conformant colour),
    not counted by that metric (grey ring). The box gives that metric's count.
    """
    day = A.prepare(df, info["rating_kva"], info)
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.6), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        vv = A.assess_voltvar(day, m, tol, failure_definition)
        tol_used = vv.attrs["tol"]
        B.qv_axes(ax, window=False, cap=True, tol=tol_used)
        V, Q = day.v.to_numpy(), vv.q.to_numpy()
        valid, counted, fail = (vv[c].to_numpy(bool) for c in ("valid", "counted", "failure"))
        nc = valid & ~counted
        ax.scatter(V[nc], Q[nc], s=7, facecolors="none", edgecolors=PAL.NOT_COUNTED,
                   lw=0.5, alpha=0.5, zorder=2)
        ok = counted & ~fail
        ax.scatter(V[ok], Q[ok], s=8, color=PAL.STATUS["conformant"], alpha=0.45, lw=0, zorder=3)
        ax.scatter(V[fail], Q[fail], s=8, color=PAL.STATUS["nonconformant"], alpha=0.55, lw=0,
                   zorder=4)
        ax.set_title(A.METRIC_LABELS[m], fontsize=10, loc="left")
        ax.text(
            0.02, 0.02, _wrapped_summary(vv), transform=ax.transAxes, fontsize=6.8,
            va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"),
        )
        if ax is not np.atleast_1d(axes)[0]:
            ax.set_ylabel("")
    handles = [
        Line2D([], [], ls="", marker="o", color=PAL.STATUS["nonconformant"], label="Counted: failure"),
        Line2D([], [], ls="", marker="o", color=PAL.STATUS["conformant"], label="Counted: pass"),
        Line2D([], [], ls="", marker="o", mfc="none", mec=PAL.NOT_COUNTED,
               label="Not counted by this metric"),
        Line2D([], [], color=PAL.REFERENCE["required"], ls="--", label="Volt-VAr curve"),
        Patch(facecolor=PAL.REFERENCE["band"], alpha=0.25, label="± tolerance"),
        Line2D([], [], color=PAL.REFERENCE["threshold"], ls="--",
               label="Fig. 2.1 minimum, 20–60% P (−44%)"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        f"Q–V, every interval, three ways of counting\n{info.get('label', '')}  ·  {info.get('date', '')}",
        x=0.04, ha="left", y=0.995, fontsize=10.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.05, right=0.98, wspace=0.08)
    return fig
