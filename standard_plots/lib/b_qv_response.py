"""
Family B: Q-V response. Reactive power against voltage, against the AS/NZS 4777.2
Volt-VAr requirement.

    B.1  plot_qv_scatter   one site, several days: every interval as a point, binned median,
                           and the slope fitted over the slope-method window
    B.2  plot_qv_curve     one site, several days: one median curve per day, and the overall
                           median with its interquartile range
    B.3  plot_qv_fleet     many sites: spread of the per-site median curves (from
                           sources.qv_profiles), optionally one panel per group

Shared axes (all figures)
-------------------------
x = voltage, 205-265 V; y = Q as % of rating, generator convention (+ supplying,
- absorbing), -70 to +55 %. Behind the data: the Volt-VAr curve (black dashed) with its
+/-4 % band, set-point voltages (207, 220, 240, 258 V) dotted, and the Figure 2.1
absorbing minimum for 20-60 % P (-44 %) only if cap=True. The slope-method window
(242-253 V) is shaded lightly.

Input
-----
B.1 and B.2 take the same tidy frame as family A (t_aest, V, P_kw, Q_kvar, optional
phase), covering any number of days, plus the loader's info dict. Three-phase sites
use the mean phase voltage and the sum of the phase Q and P (as in family A and the
3L3P method). Only assessable intervals (|P_total| >= 20 % of rating) count towards
medians and fits; the rest are drawn faint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "bms_sa_review"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.ciccada_config import AS4777                     # noqa: E402
from shared.as4777_curves import vvar_required_q            # noqa: E402

from . import palette as C                                   # noqa: E402
from .a_site_day import assess_voltvar, metadata_lines, prepare   # noqa: E402

VV, TOL, QCAP, QIMP = AS4777["VVAR"], AS4777["TOL_FRAC"], AS4777["QCAP"], AS4777["QIMP"]

# Outside-band categories, as in the Stage 2 table (build_conformance_voltvar.py): Q_impact
# = signed |Q| / |nearest permitted band edge|, for assessable intervals outside the band.
CATEGORIES = [("Q_adverse", "adverse"), ("Q_inactive", "inactive"),
              ("Q_significant_shortfall", "significant shortfall"),
              ("Q_near_conformant", "near-conformant"), ("Q_major_surplus", "major surplus")]

V_RANGE = (205.0, 265.0)
Q_RANGE = (-70.0, 55.0)

# Slope method (notebooks 05-13): OLS of Q (% of rating) on V inside the window.
SLOPE_WINDOW = (242.0, 253.0)
REQUIRED_SLOPE = -VV["Q4"] * 100 / (VV["V4"] - VV["V3"])      # -3.333 %/V
SLOPE_TOL = 0.20                                               # pass band +/-20 % of the required slope
MIN_FIT_N, MIN_FIT_SPREAD = 20, 5.0


# =============================================================================
# 1. Data
# =============================================================================
def qv_points(df: pd.DataFrame, rating_kva: float) -> pd.DataFrame:
    """
    One row per interval: V (mean of phases), P and Q (sum, % of rating), assessable, the
    Figure-2.1-capped permitted band (as family A), inside/outside, Q_impact and the
    outside-band category, plus hour and date.
    """
    day = prepare(df, rating_kva)
    vv = assess_voltvar(day)
    pts = pd.DataFrame({"V": day.v, "p_pct": day.p_pct, "q_pct": day.q_pct,
                        "q_lo": vv.q_min_final, "q_hi": vv.q_max_final}, index=day.t).dropna()
    pts["assessable"] = pts.p_pct.abs() >= QCAP["P_MIN"] * 100
    pts["inside"] = (pts.q_pct >= pts.q_lo) & (pts.q_pct <= pts.q_hi)
    # Q_impact against the nearest band edge (same rule as shared.as4777_curves.q_impact_nearest_edge)
    near_hi = (pts.q_pct - pts.q_hi).abs() <= (pts.q_pct - pts.q_lo).abs()
    edge = np.where(near_hi, pts.q_hi, pts.q_lo)
    sym = (pts.q_hi + pts.q_lo).abs() <= 1e-9
    direction = np.where(sym, 1.0, np.sign(edge) * np.sign(pts.q_pct))
    pts["q_impact"] = direction * pts.q_pct.abs() / (np.abs(edge) + 1e-9)
    qi = pts.q_impact
    cat = np.select([qi < QIMP["thr1"], qi <= QIMP["thr2"], qi < QIMP["thr3"], qi <= QIMP["thr4"]],
                    ["Q_adverse", "Q_inactive", "Q_significant_shortfall", "Q_near_conformant"], "Q_major_surplus")
    pts["category"] = np.where(~pts.assessable, "Q_not_assessable", np.where(pts.inside, "Q_conformant", cat))
    pts["hour"] = pts.index.hour + pts.index.minute / 60
    pts["date"] = pts.index.date
    return pts


def vv_counts(pts: pd.DataFrame) -> dict:
    """Assessable intervals inside / outside the permitted band, and outside-band counts per category."""
    a = pts[pts.assessable]
    out = {"assessable": len(a), "total": len(pts), "inside": int(a.inside.sum()), "outside": int((~a.inside).sum())}
    out.update({k: int((a.category == k).sum()) for k, _ in CATEGORIES})
    return out


def _counts_text(pts: pd.DataFrame) -> str:
    c = vv_counts(pts)
    pct = lambda x: f"{100 * x / c['assessable']:.0f}%" if c["assessable"] else "–"
    line1 = (f"Assessable {c['assessable']:,} of {c['total']:,} intervals  ·  inside band {c['inside']:,} "
             f"({pct(c['inside'])})  ·  outside {c['outside']:,} ({pct(c['outside'])})")
    line2 = "Outside band:  " + "  ·  ".join(f"{lab} {c[k]:,} ({pct(c[k])})" for k, lab in CATEGORIES)
    return line1 + "\n" + line2


def binned_median(pts: pd.DataFrame, bin_v: float = 1.0, min_n: int = 5) -> pd.DataFrame:
    """Median and interquartile range of Q per voltage bin (assessable intervals only)."""
    a = pts[pts.assessable]
    b = (np.floor(a.V / bin_v) * bin_v + bin_v / 2).rename("v_bin")
    g = a.groupby(b)["q_pct"]
    out = pd.DataFrame({"q_med": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75), "n": g.size()})
    return out[out.n >= min_n].reset_index()


def fit_slope(pts: pd.DataFrame, window: tuple[float, float] = SLOPE_WINDOW) -> dict:
    """OLS of Q (% of rating) on V over the window, assessable intervals only; verdict against the pass band."""
    a = pts[pts.assessable & (pts.V >= window[0]) & (pts.V < window[1])]
    out = {"n": len(a), "spread": float(a.V.max() - a.V.min()) if len(a) else 0.0,
           "slope": np.nan, "intercept": np.nan, "r2": np.nan, "status": "too few points"}
    if len(a) < MIN_FIT_N:
        return out
    if out["spread"] < MIN_FIT_SPREAD:
        out["status"] = "narrow voltage range"
        return out
    slope, icpt = np.polyfit(a.V, a.q_pct, 1)
    resid = a.q_pct - (slope * a.V + icpt)
    r2 = 1 - resid.var() / a.q_pct.var() if a.q_pct.var() > 0 else np.nan
    lo, hi = REQUIRED_SLOPE * (1 + SLOPE_TOL), REQUIRED_SLOPE * (1 - SLOPE_TOL)
    status = "in band" if lo <= slope <= hi else ("too steep" if slope < lo else "too shallow")
    out.update(slope=float(slope), intercept=float(icpt), r2=float(r2), status=status)
    return out


# =============================================================================
# 2. Shared axes
# =============================================================================
def qv_axes(ax, window: bool = True, cap: bool = False, legend_extra=None):
    """Requirement, references and fixed ranges for any Q-V panel."""
    v = np.linspace(*V_RANGE, 241)
    req = np.array([vvar_required_q(x, 100.0) for x in v])
    ax.fill_between(v, req - TOL * 100, req + TOL * 100, color=C.REFERENCE["band"], alpha=C.BAND_ALPHA,
                    lw=0, zorder=1)
    ax.plot(v, req, color=C.REFERENCE["required"], lw=1.3, ls="--", zorder=4)
    for x in (VV["V1"], VV["V2"], VV["V3"], VV["V4"]):
        ax.axvline(x, color=C.REFERENCE["threshold"], lw=0.7, ls=":", zorder=1)
    if cap:
        ax.axhline(-QCAP["Q_FLAT"] * 100, color=C.REFERENCE["threshold"], lw=0.8, ls=(0, (1, 3)), zorder=1)
    if window:
        ax.axvspan(*SLOPE_WINDOW, color=C.REFERENCE["night"], zorder=0)
    ax.axhline(0, color=C.REFERENCE["zero"], lw=0.5, zorder=1)
    ax.set_xlim(*V_RANGE)
    ax.set_ylim(*Q_RANGE)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Reactive power (% of rating, + supply / − absorb)")
    handles = [
        Line2D([], [], color=C.REFERENCE["required"], ls="--", lw=1.3, label="Volt-VAr curve (AS/NZS 4777.2)"),
        Patch(facecolor=C.REFERENCE["band"], alpha=0.25, label=f"±{TOL*100:.0f}% tolerance"),
    ]
    if cap:
        handles.append(Line2D([], [], color=C.REFERENCE["threshold"], ls=(0, (1, 3)),
                              label=f"Fig. 2.1 minimum, 20–60% P (−{QCAP['Q_FLAT']*100:.0f}%)"))
    if window:
        handles.append(Patch(facecolor=C.REFERENCE["night"], edgecolor="none",
                             label=f"Slope window {SLOPE_WINDOW[0]:.0f}–{SLOPE_WINDOW[1]:.0f} V"))
    return handles + (legend_extra or [])


def _title(fig, title, info, extra=""):
    n = info.get("site", {}).get("n_pv_circuits")
    ph = "" if n is None else ("  ·  3-phase (mean V, total Q)" if n == 3 else "  ·  single-phase")
    fig.suptitle(f"{title}\n{info.get('label', '')}  ·  {info.get('date', '')}"
                 f"{'  ·  rating ' + format(info['rating_kva'], 'g') + ' kVA' if 'rating_kva' in info else ''}{ph}{extra}",
                 x=0.07, ha="left", y=0.995, fontsize=10.5)


def _print_meta(df, info, show_meta):
    if show_meta and df is not None:
        print("\n".join(metadata_lines(prepare(df, info["rating_kva"], info))[:2]))


def _fit_text(fit: dict) -> str:
    if np.isnan(fit["slope"]):
        return f"Slope {SLOPE_WINDOW[0]:.0f}–{SLOPE_WINDOW[1]:.0f} V: not fitted ({fit['status']}, n = {fit['n']})"
    lo, hi = REQUIRED_SLOPE * (1 - SLOPE_TOL), REQUIRED_SLOPE * (1 + SLOPE_TOL)
    return (f"Slope {SLOPE_WINDOW[0]:.0f}–{SLOPE_WINDOW[1]:.0f} V: {fit['slope']:.2f} %/V  ·  "
            f"required {REQUIRED_SLOPE:.2f} (band {hi:.2f} to {lo:.2f})  ·  {fit['status']}  ·  "
            f"n = {fit['n']}, R² = {fit['r2']:.2f}")


def _median_line(ax, med):
    """Binned median: Q colour with a thin black edge so it reads over a cloud of Q-coloured points."""
    ax.plot(med.v_bin, med.q_med, color=C.QUANTITY["Q"], lw=2.4, zorder=6, solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=4.0, foreground=C.REFERENCE["zero"]), pe.Normal()])


def _median_handle(label: str = "Median per 1 V bin (assessable)"):
    return Line2D([], [], color=C.QUANTITY["Q"], lw=2.4, label=label,
                  path_effects=[pe.Stroke(linewidth=4.0, foreground=C.REFERENCE["zero"]), pe.Normal()])


def _info_box(ax, lines):
    """Fit and counts, printed under the x axis so they never cover the data or the curve."""
    ax.text(0.0, -0.11, "\n".join(lines), transform=ax.transAxes, fontsize=7.8, va="top", ha="left",
            linespacing=1.45, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC"))
    ax.figure.subplots_adjust(bottom=0.2)


# =============================================================================
# 3. Figures
# =============================================================================
def plot_qv_scatter(df, info: dict, colour_by: str = "assessable", fit: bool = True, show_meta: bool = True):
    """
    B.1 Every interval as a point, the binned median (assessable only) and, if `fit`,
    the OLS slope over the slope window with its verdict.

    colour_by: 'assessable' (Q colour vs faint grey), 'category' (inside the band in the
               conformant colour, outside by Volt-VAr category), 'hour' (time of day,
               cyclic map) or 'P' (active power as % of rating).
    The box at the bottom gives the slope fit, the assessable intervals inside and outside
    the permitted band (Figure 2.1 capped, as family A), and the outside-band intervals
    per category (Stage 2 Q_impact thresholds); percentages are of assessable intervals.
    """
    _print_meta(df, info, show_meta)
    pts = qv_points(df, info["rating_kva"])
    fig, ax = plt.subplots(figsize=(10, 8))
    handles = qv_axes(ax)
    na, ok = pts[~pts.assessable], pts[pts.assessable]
    ax.scatter(na.V, na.q_pct, s=7, facecolors="none", edgecolors=C.STATUS["not_assessable"], lw=0.5,
               alpha=0.35, zorder=2)
    handles.append(Line2D([], [], ls="", marker="o", mfc="none", mec=C.STATUS["not_assessable"],
                          label=f"Not assessable (|P| < {QCAP['P_MIN']*100:.0f}%)"))
    if colour_by == "assessable":
        ax.scatter(ok.V, ok.q_pct, s=8, color=C.QUANTITY["Q"], alpha=0.35, lw=0, zorder=3)
        handles.append(Line2D([], [], ls="", marker="o", color=C.QUANTITY["Q"], label="Assessable interval"))
    elif colour_by == "category":
        cats = [("Q_conformant", "inside band")] + CATEGORIES
        for k, lab in cats:
            sub = ok[ok.category == k]
            if len(sub):
                ax.scatter(sub.V, sub.q_pct, s=8, color=C.VOLTVAR_CATEGORY[k], alpha=0.5, lw=0, zorder=3)
                handles.append(Line2D([], [], ls="", marker="o", color=C.VOLTVAR_CATEGORY[k], label=lab.capitalize()))
    else:
        col, cmap, lab, lim = {"hour": ("hour", C.CMAP["hour"], "Hour of day (AEST)", (0, 24)),
                               "P": ("p_pct", C.CMAP["count"], "Active power (% of rating)", (20, 100))}[colour_by]
        sc = ax.scatter(ok.V, ok.q_pct, s=9, c=ok[col], cmap=cmap, vmin=lim[0], vmax=lim[1], alpha=0.7, lw=0, zorder=3)
        fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04).set_label(lab)
    med = binned_median(pts)
    if len(med):
        _median_line(ax, med)
        handles.append(_median_handle())
    lines = [_counts_text(pts)]
    if fit:
        f = fit_slope(pts)
        if not np.isnan(f["slope"]):
            x = np.array(SLOPE_WINDOW)
            ax.plot(x, f["slope"] * x + f["intercept"], color=C.REFERENCE["fit"], lw=2.4, ls=(0, (5, 2)), zorder=7)
            handles.append(Line2D([], [], color=C.REFERENCE["fit"], lw=2.4, ls=(0, (5, 2)),
                                  label=f"Fitted slope, {SLOPE_WINDOW[0]:.0f}–{SLOPE_WINDOW[1]:.0f} V ({f['status']})"))
        lines.insert(0, _fit_text(f))
    _info_box(ax, lines)
    ax.legend(handles=handles, loc="upper right", ncol=1)
    _title(fig, "Q–V response, every interval", info, f"  ·  {pts.date.nunique()} days")
    fig.subplots_adjust(top=0.9, bottom=0.2)
    return fig


def plot_qv_curve(df, info: dict, show_meta: bool = True, min_n: int = 5):
    """
    B.2 One thin median curve per day (coloured by date) and the overall median with its
    interquartile range: is the response the same every day?
    """
    _print_meta(df, info, show_meta)
    pts = qv_points(df, info["rating_kva"])
    fig, ax = plt.subplots(figsize=(10, 8))
    handles = qv_axes(ax)
    days = sorted(pts.date.unique())
    cmap = plt.get_cmap(C.CMAP["year"])
    drawn = 0
    for i, d in enumerate(days):
        m = binned_median(pts[pts.date == d], min_n=3)
        if len(m) >= 3:
            ax.plot(m.v_bin, m.q_med, color=cmap(i / max(len(days) - 1, 1)), lw=0.9, alpha=0.8, zorder=3)
            drawn += 1
    med = binned_median(pts, min_n=min_n)
    if len(med):
        ax.fill_between(med.v_bin, med.q25, med.q75, color=C.QUANTITY["Q"], alpha=0.2, lw=0, zorder=2)
        ax.plot(med.v_bin, med.q_med, color=C.QUANTITY["Q"], lw=2.6, zorder=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(len(days) - 1, 1)))
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.04)
    ticks = np.linspace(0, max(len(days) - 1, 1), min(len(days), 5)).round().astype(int)
    cb.set_ticks(ticks)
    cb.set_ticklabels([pd.Timestamp(days[t]).strftime("%d %b") for t in ticks])
    cb.set_label("Day")
    handles += [Line2D([], [], color=cmap(0.6), lw=0.9, label=f"Daily median ({drawn} days)"),
                Line2D([], [], color=C.QUANTITY["Q"], lw=2.6, label="All days: median"),
                Patch(facecolor=C.QUANTITY["Q"], alpha=0.2, label="All days: 25–75%")]
    ax.legend(handles=handles, loc="upper right")
    _info_box(ax, [_fit_text(fit_slope(pts)), _counts_text(pts)])
    _title(fig, "Q–V response curve, day by day", info, f"  ·  {len(days)} days")
    fig.subplots_adjust(top=0.9, bottom=0.2)
    return fig


def plot_qv_fleet(profiles: pd.DataFrame, info: dict, group: str | None = None, min_sites: int = 10,
                  group_order=None, group_labels: dict | None = None):
    """
    B.3 Spread of per-site median curves (sources.qv_profiles): per voltage bin, the
    median across sites and the 25-75 % and 10-90 % ranges. With `group` (a column of
    `profiles`, e.g. a slope-method verdict), one panel per group on shared axes.
    The strip under each panel counts the sites contributing to each bin.
    """
    groups = [None] if group is None else (group_order or sorted(profiles[group].dropna().unique(), key=str))
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n if n > 1 else 10, 7.5), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [5, 1], "hspace": 0.06, "wspace": 0.05}, squeeze=False)
    for k, g in enumerate(groups):
        ax, axn = axes[0, k], axes[1, k]
        sub = profiles if g is None else profiles[profiles[group] == g]
        handles = qv_axes(ax, window=True)
        q = sub.groupby("v_bin")["q_med"]
        st = pd.DataFrame({"n": q.size(), "p10": q.quantile(0.1), "p25": q.quantile(0.25), "p50": q.median(),
                           "p75": q.quantile(0.75), "p90": q.quantile(0.9)})
        st = st[st.n >= min_sites]
        if len(st):
            ax.fill_between(st.index, st.p10, st.p90, color=C.QUANTITY["Q"], alpha=0.12, lw=0, zorder=2)
            ax.fill_between(st.index, st.p25, st.p75, color=C.QUANTITY["Q"], alpha=0.25, lw=0, zorder=2)
            ax.plot(st.index, st.p50, color=C.QUANTITY["Q"], lw=2.6, zorder=6)
        allq = sub.groupby("v_bin").size()
        axn.bar(allq.index, allq.values, width=0.9, color=C.OTHER, zorder=2)
        axn.bar(st.index, st.n, width=0.9, color=C.QUANTITY["Q"], alpha=0.6, zorder=3)
        axn.set_xlabel("Voltage (V)")
        ax.set_xlabel("")
        if k == 0:
            axn.set_ylabel("Sites")
        else:
            ax.set_ylabel("")
        name = "All sites" if g is None else (group_labels or {}).get(g, f"{group} = {g}")
        ax.set_title(f"{name}  ·  {sub.site_id.nunique():,} sites", fontsize=10, loc="left")
        if k == n - 1:
            handles += [Line2D([], [], color=C.QUANTITY["Q"], lw=2.6, label="Median across sites"),
                        Patch(facecolor=C.QUANTITY["Q"], alpha=0.37, label="25–75% of sites"),
                        Patch(facecolor=C.QUANTITY["Q"], alpha=0.12, label="10–90% of sites")]
            ax.legend(handles=handles, loc="upper right", fontsize=7.5)
    fig.suptitle(f"Q–V response across sites\n{info.get('label', '')}  ·  {info.get('date', '')}  ·  "
                 f"per-site median Q per 1 V bin, assessable intervals; bins with ≥ {min_sites} sites",
                 x=0.07, ha="left", y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.98)
    return fig
