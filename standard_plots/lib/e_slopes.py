"""
Family E: response-strength distributions. How steep each site's (or site-month's)
Volt-VAr response is, against the slope the standard requires.

    E.1  plot_slope_hist    histogram of slopes; several slope columns overlaid, or one
                            panel per group on a shared x axis
    E.2  plot_slope_box     box and strip per group (install year, month, cohort ...)
    E.3  plot_slope_ecdf    cumulative distribution per group: what share is at least
                            as steep as any given slope

Shape (all figures)
-------------------
Slope in % of rating per volt, always on the same fixed range (-8 to +3 %/V) so figures
are comparable. The required slope (-3.33 %/V: -60 % over 240-258 V) is a black dashed
line; the pass band (+/-20 %, -4.00 to -2.67 %/V) is shaded grey; 0 %/V (no response) is
a thin solid line. Each panel states n and the share inside the band. Slopes outside the
range are clipped to its edge and counted.

Input
-----
Any table with slope column(s) in %/V: sources.slope_verdicts() (slope = net meter,
slope_truth = PV circuits; notebook 13), or one value per site from
b_qv_response.fit_slope.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import palette as C
from .b_qv_response import REQUIRED_SLOPE, SLOPE_TOL, SLOPE_WINDOW

SLOPE_RANGE = (-8.0, 3.0)
BAND = (REQUIRED_SLOPE * (1 + SLOPE_TOL), REQUIRED_SLOPE * (1 - SLOPE_TOL))      # (-4.00, -2.67)
SLOPE_LABELS = {"slope": "Net meter (blind slope)", "slope_truth": "PV circuits (reference slope)"}
SLOPE_SOURCE = {"slope": "net_meter", "slope_truth": "pv_circuit"}


def in_band(x) -> pd.Series:
    x = pd.Series(x, dtype=float)
    return (x >= BAND[0]) & (x <= BAND[1])


def _colours(cols):
    if len(cols) == 1:
        return {cols[0]: C.QUANTITY["Q"]}
    if all(c in SLOPE_SOURCE for c in cols):
        return {c: C.SOURCE[SLOPE_SOURCE[c]]["color"] for c in cols}
    seq = mpl.color_sequences.get(C.GROUP_CYCLE, mpl.color_sequences["tab10"])
    return {c: seq[i % len(seq)] for i, c in enumerate(cols)}


def _references(ax, vertical: bool = True):
    """Pass band, required slope and zero, on the slope axis (x if vertical lines, else y)."""
    span, line = (ax.axvspan, ax.axvline) if vertical else (ax.axhspan, ax.axhline)
    span(*BAND, color=C.REFERENCE["band"], alpha=C.BAND_ALPHA, lw=0, zorder=0)
    line(REQUIRED_SLOPE, color=C.REFERENCE["required"], ls="--", lw=1.2, zorder=4)
    line(0, color=C.REFERENCE["zero"], lw=0.6, zorder=1)


def _ref_handles():
    return [Line2D([], [], color=C.REFERENCE["required"], ls="--", lw=1.2,
                   label=f"Required slope ({REQUIRED_SLOPE:.2f} %/V)"),
            Patch(facecolor=C.REFERENCE["band"], alpha=0.25,
                  label=f"Pass band ±{SLOPE_TOL*100:.0f}% ({BAND[1]:.2f} to {BAND[0]:.2f})")]


def _clip(x):
    return np.clip(pd.Series(x, dtype=float).dropna(), *SLOPE_RANGE)


def _title(fig, title, info, unit):
    info = info or {}
    sub = "  ·  ".join(p for p in (info.get("label", ""), info.get("date", ""), f"unit: {unit}",
                                   f"slope of Q (% of rating) on V over {SLOPE_WINDOW[0]:.0f}–{SLOPE_WINDOW[1]:.0f} V") if p)
    fig.suptitle(f"{title}\n{sub}", x=0.02, ha="left", y=0.995, fontsize=10.5)


def _group_values(df, by, order):
    vals = order or sorted(df[by].dropna().unique(), key=lambda v: (not isinstance(v, (int, float, np.integer)), v))
    return list(vals)


# =============================================================================
# Figures
# =============================================================================
def plot_slope_hist(df: pd.DataFrame, cols="slope", by: str | None = None, info: dict | None = None,
                    unit: str = "site-months", bin_width: float = 0.2, order=None, labels: dict | None = None,
                    density: bool = True, title: str | None = None):
    """
    E.1 Histogram of slopes. Several `cols` (e.g. ['slope', 'slope_truth']) are overlaid
    as outlines in the data-source colours; with `by`, one panel per group, shared x.
    density=True scales each histogram to its own n so groups of different size compare.
    """
    cols = [cols] if isinstance(cols, str) else list(cols)
    groups = [None] if by is None else _group_values(df, by, order)
    col = _colours(cols)
    bins = np.arange(SLOPE_RANGE[0], SLOPE_RANGE[1] + bin_width, bin_width)
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2.2 * len(groups) + 1.6 if by else 5.2),
                             sharex=True, squeeze=False)
    for ax, g in zip(axes[:, 0], groups):
        sub = df if g is None else df[df[by] == g]
        _references(ax)
        notes = []
        for c in cols:
            x = _clip(sub[c])
            if not len(x):
                continue
            filled = len(cols) == 1
            ax.hist(x, bins=bins, density=density, color=col[c], alpha=0.75 if filled else 1,
                    histtype="stepfilled" if filled else "step", lw=1.6, zorder=3)
            name = (labels or {}).get(c, SLOPE_LABELS.get(c, c)) if len(cols) > 1 else ""
            notes.append(f"{name + ': ' if name else ''}n = {len(x):,}, {100 * in_band(x).mean():.0f}% in band")
        ax.text(0.99, 0.95, "\n".join(notes), transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"))
        ax.set_ylabel("Density" if density else "Count")
        if g is not None:
            ax.set_title(f"{by} = {g}", loc="left", fontsize=9.5)
        ax.set_xlim(*SLOPE_RANGE)
    axes[-1, 0].set_xlabel("Slope (% of rating per V)")
    handles = _ref_handles()
    if len(cols) > 1:
        handles = [Line2D([], [], color=col[c], lw=1.6, label=(labels or {}).get(c, SLOPE_LABELS.get(c, c)))
                   for c in cols] + handles
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=8)
    _title(fig, title or "Distribution of Volt-VAr response slopes", info, unit)
    fig.subplots_adjust(top=1 - 0.95 / fig.get_figheight(), bottom=0.6 / fig.get_figheight(), hspace=0.35,
                        left=0.08, right=0.98)
    return fig


def plot_slope_box(df: pd.DataFrame, col: str = "slope", by: str = "install_year", info: dict | None = None,
                   unit: str = "site-months", order=None, min_n: int = 10, max_points: int = 300,
                   title: str | None = None, group_label: str | None = None, seed: int = 0):
    """
    E.2 Box (median, 25-75 %, whiskers 5-95 %) and a jittered strip of up to `max_points`
    points per group. Above each box: the share inside the pass band; below: n.
    """
    rng = np.random.default_rng(seed)
    groups = [g for g in _group_values(df, by, order) if df.loc[df[by] == g, col].notna().sum() >= min_n]
    data = [_clip(df.loc[df[by] == g, col]) for g in groups]
    fig, ax = plt.subplots(figsize=(min(14, max(7, 0.6 * len(groups) + 2.5)), 6))
    _references(ax, vertical=False)
    for i, x in enumerate(data):
        pts = x.sample(min(len(x), max_points), random_state=seed) if len(x) else x
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(pts)), pts, s=5, color=C.QUANTITY["Q"], alpha=0.25, lw=0, zorder=2)
    ax.boxplot(data, positions=range(len(groups)), widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="none", edgecolor=C.REFERENCE["zero"], lw=1.1),
               medianprops=dict(color=C.REFERENCE["zero"], lw=2), whiskerprops=dict(lw=1),
               capprops=dict(lw=1), zorder=5)
    for i, x in enumerate(data):
        ax.text(i, SLOPE_RANGE[1] - 0.15, f"{100 * in_band(x).mean():.0f}%", ha="center", va="top", fontsize=7.5,
                color=C.STATUS["conformant"])
        ax.text(i, SLOPE_RANGE[0] + 0.1, f"{len(x):,}", ha="center", va="bottom", fontsize=6.5, rotation=90,
                color=C.REFERENCE["text"])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], rotation=0 if len(groups) < 14 else 45)
    ax.set_ylim(*SLOPE_RANGE)
    ax.set_ylabel("Slope (% of rating per V)")
    ax.set_xlabel(group_label or by)
    ax.grid(axis="x", visible=False)
    handles = _ref_handles() + [Line2D([], [], ls="", marker="o", color=C.QUANTITY["Q"], alpha=0.5,
                                       label=f"{SLOPE_LABELS.get(col, col)} (≤ {max_points} per group)")]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    _title(fig, title or f"Volt-VAr response slope by {group_label or by}  ·  % above = share in band, numbers below = n",
           info, unit)
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.08, right=0.98)
    return fig


def plot_slope_ecdf(df: pd.DataFrame, col: str = "slope", by: str = "cohort", info: dict | None = None,
                    unit: str = "site-months", order=None, min_n: int = 20, title: str | None = None,
                    group_label: str | None = None):
    """
    E.3 Cumulative share of units with a slope at or below x (i.e. at least as steep),
    one line per group. Where a line crosses the band edges reads off the shares
    too steep, in band and too shallow.
    """
    groups = [g for g in _group_values(df, by, order) if df.loc[df[by] == g, col].notna().sum() >= min_n]
    seq = mpl.color_sequences.get(C.GROUP_CYCLE, mpl.color_sequences["tab10"])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    _references(ax)
    for i, g in enumerate(groups):
        x = np.sort(_clip(df.loc[df[by] == g, col]).to_numpy())
        y = np.arange(1, len(x) + 1) / len(x) * 100
        ax.step(x, y, where="post", color=seq[i % len(seq)], lw=1.8,
                label=f"{g}  (n = {len(x):,}; {100 * in_band(x).mean():.0f}% in band)")
    ax.set_xlim(*SLOPE_RANGE)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Slope (% of rating per V)")
    ax.set_ylabel(f"Share of {unit} with slope ≤ x (%)")
    leg1 = ax.legend(loc="upper left", fontsize=8, title=group_label or by, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=_ref_handles(), loc="lower right", fontsize=8)
    _title(fig, title or f"Cumulative distribution of Volt-VAr response slopes ({SLOPE_LABELS.get(col, col)})",
           info, unit)
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.08, right=0.98)
    return fig
