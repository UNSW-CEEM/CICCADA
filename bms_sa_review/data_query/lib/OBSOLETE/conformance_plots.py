"""
conformance_plots.py  —  Visualisation for the conformance analysis
===================================================================

All plotting lifted out of `02_conformance_curtailment_analysis.ipynb`.
Every function takes DataFrames + explicit params. No notebook globals.

ONE IMPORTANT CHANGE
--------------------
Notebook cell 43 had a LOCAL re-implementation of the Volt-VAr curve
(`_q_curve_norm`). That was the FIFTH copy of that curve in the codebase, and
copies drift. It now comes from `as4777_curves.vvar_required_q` — the same code
that scored `conformance_voltvar_v2`. So the reference curve drawn on the plot
is guaranteed to be the curve the table was judged against, which is the whole
point of having a keystone.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bms_sa_review.data_query.OBSOLETEciccada_config import AS4777, FIXED_OFFSET
from as4777_curves import vvar_required_q

# House palette
C_MAIN, C_GHOST, C_GRID = "#7c3aed", "#c4b5fd", "#eeeeee"


# ═════════════════════════════════════════════════════════════
# 1. Per-site non-conforming fraction histograms   [cells 13/14]
# ═════════════════════════════════════════════════════════════

def plot_nonconf_distribution(panels, thresh=AS4777["SITE_CONF_THRESH"],
                              figsize=(11, 3.2)):
    """
    Histogram of each site's non-conforming fraction, one panel per mechanism.

    `panels` : list of (label, site_df) — site_df needs a `nonconf_frac` column.

    The red line is the 10%-rule threshold. Sites to its right are non-conformant.
    The SHAPE matters more than the rate: a spike at zero with a long tail means
    a few bad actors; a broad hump means a systemic problem.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, dpi=120)
    if len(panels) == 1:
        axes = [axes]

    for ax, (name, d) in zip(axes, panels):
        ax.hist(d["nonconf_frac"].clip(0, 1) * 100, bins=40,
                edgecolor="k", linewidth=0.3, color=C_MAIN, alpha=0.85)
        ax.axvline(thresh * 100, color="red", ls="--", lw=1,
                   label=f"{thresh*100:.0f}% threshold")
        ax.set_title(f"{name}: per-site non-conforming fraction", fontsize=9.5)
        ax.set_xlabel("% of evaluated intervals non-conforming", fontsize=8)
        ax.set_ylabel("sites", fontsize=8)
        ax.grid(color=C_GRID, lw=0.5)
        ax.legend(fontsize=7.5)

    plt.tight_layout()
    plt.show()
    return fig


# ═════════════════════════════════════════════════════════════
# 2. Volt-VAr band breakdown                       [cell 16 output]
# ═════════════════════════════════════════════════════════════

def plot_vvar_bands(breakdown_df, figsize=(9, 3.6)):
    """
    Horizontal bar of the five Q_impact bands as a share of evaluated intervals.

    This is the plot I'd put in the paper INSTEAD of any "reduced non-conformance"
    number: it needs no interpretive choice, so R5 can't contaminate it.
    """
    d = breakdown_df.drop(index="total_evaluated", errors="ignore").copy()

    pretty = {
        "adverse__lt_m010":            "Adverse (wrong direction)",
        "inactive__m010_to_010":       "Inactive (no response)",
        "shortfall__010_to_090":       "Significant shortfall (0.1-0.9)",
        "near_conformant__090_to_110": "Near-conformant (0.9-1.1)",
        "surplus__gt_110":             "Over-response (>1.1)",
    }
    colors = {
        "adverse__lt_m010":            "#b91c1c",
        "inactive__m010_to_010":       "#ea580c",
        "shortfall__010_to_090":       "#f59e0b",
        "near_conformant__090_to_110": "#65a30d",
        "surplus__gt_110":             "#0284c7",
    }

    labels = [pretty.get(i, i) for i in d.index]
    vals   = d["pct_of_evaluated"].values.astype(float)
    cols   = [colors.get(i, C_MAIN) for i in d.index]

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    bars = ax.barh(range(len(labels)), vals, color=cols, alpha=0.9)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("% of evaluated intervals", fontsize=9)
    ax.set_title("Volt-VAr response by Q_impact band", fontsize=10, weight="bold")
    ax.grid(axis="x", color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    return fig


# ═════════════════════════════════════════════════════════════
# 3. Non-conformance by group                      [cells 25/26]
# ═════════════════════════════════════════════════════════════

def plot_nonconf_by_group(df, by_label, title_suffix="", figsize=(8, 4)):
    """
    % of sites non-conformant per group (state / DNSP / OEM), 10%-rule.

    The ghost bar behind is the much looser "any NC" rate. The gap between them
    is the interesting part: a group where they're close has sites that fail
    persistently; a group where they're far apart has sites that fail rarely.

    `df` : output of conformance_metrics.breakdown()
    """
    d = df.dropna(subset=[by_label]).copy()
    d[by_label] = d[by_label].astype(str).str.replace("nan", "Unknown", regex=False)
    d = d.sort_values("pct_nonconf_10pct", ascending=True)

    labels = d[by_label].tolist()
    v10    = d["pct_nonconf_10pct"].values.astype(float)
    vany   = d["pct_any_nonconf"].values.astype(float)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.barh(range(len(labels)), vany, color=C_GHOST, alpha=0.50,
            label="Any NC (>=1 interval)")
    bars = ax.barh(range(len(labels)), v10, color=C_MAIN, alpha=0.88,
                   label="Non-conformant (>=10% of eligible)")

    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=7.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("% of eligible sites", fontsize=9)
    ax.set_title(f"Volt-Watt non-conformance by {by_label}{title_suffix}",
                 fontsize=10, weight="bold")
    ax.grid(axis="x", color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="lower right")

    plt.tight_layout()
    plt.show()
    return fig


# ═════════════════════════════════════════════════════════════
# 4. OEM ranking within each DNSP                  [cell 28]
# ═════════════════════════════════════════════════════════════

def plot_nonconf_by_dnsp_oem(df, top_n=8, min_sites=10, figsize=None,
                             sort_by="pct_nonconf_10pct"):
    """
    One subplot per DNSP, OEMs ranked by non-conformance rate within it.

    Separating by DNSP matters: OEM market share is not uniform across networks,
    and network voltage conditions differ, so a raw fleet-wide OEM ranking
    confounds "bad inverter" with "bad network".

    `df` : conformance_metrics.breakdown(vw_report, meta, by=["dnsp","oem"])
    """
    import math

    valid = (
        df[df["n_sites"] >= min_sites]
        .groupby("dnsp")["oem"].count()
        .loc[lambda s: s > 0]
        .index.tolist()
    )
    dnsps = sorted(valid)
    n     = len(dnsps)
    dropped = df["dnsp"].nunique() - n
    if dropped > 0:
        print(f"{dropped} DNSP(s) dropped: no OEM with >= {min_sites} sites.")
    if n == 0:
        print("Nothing to plot.")
        return None

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    figsize = figsize or (5.2 * ncols, 2.8 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=120)
    axes = np.atleast_1d(axes).ravel()

    for ax, dnsp in zip(axes, dnsps):
        d = (df[(df["dnsp"] == dnsp) & (df["n_sites"] >= min_sites)]
             .sort_values(sort_by, ascending=False)
             .head(top_n)
             .sort_values(sort_by, ascending=True))

        labels = [f"{o}  (n={int(k)})" for o, k in zip(d["oem"], d["n_sites"])]
        ax.barh(range(len(d)), d["pct_any_nonconf"].values.astype(float),
                color=C_GHOST, alpha=0.5)
        bars = ax.barh(range(len(d)), d[sort_by].values.astype(float),
                       color=C_MAIN, alpha=0.88)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=6.5)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(dnsp, fontsize=9, weight="bold")
        ax.set_xlabel("% non-conformant", fontsize=7.5)
        ax.grid(axis="x", color=C_GRID, lw=0.5)
        ax.set_axisbelow(True)

    for ax in axes[n:]:
        ax.axis("off")

    fig.legend(handles=[
        mpatches.Patch(color=C_MAIN,  alpha=0.88, label="Non-conformant (>=10%)"),
        mpatches.Patch(color=C_GHOST, alpha=0.50, label="Any NC (>=1 interval)"),
    ], loc="lower center", ncol=2, fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
    return fig


# ═════════════════════════════════════════════════════════════
# 5. Volt-VAr scatter vs the reference curve       [cells 43-49]
# ═════════════════════════════════════════════════════════════

def _reference_curve(v_grid, s_rated=1.0):
    """
    The Volt-VAr required-Q curve, straight from the keystone.

    This is the change that matters: `as4777_curves.vvar_required_q` is the SAME
    function whose SQL twin scored conformance_voltvar_v2. The curve you see is
    the curve the data was judged against.
    """
    return np.array([vvar_required_q(v, s_rated) for v in v_grid])


def plot_vvar_scatter(df, title="", normalise=True, v_lim=(200, 270),
                      pt_alpha=0.15, pt_size=3, colour_by=None,
                      figsize=(9, 5.5)):
    """
    Q vs V scatter for a cohort of intervals, with the required-Q curve and its
    +/-4% tolerance band overlaid.

    `df`        : output of conformance_queries.fetch_vvar_intervals()
                  needs: V, Q_kvar, ac_capacity_kw [, oem / state to colour by]
    `normalise` : True  -> Q as a fraction of nameplate (comparable across sizes)
                  False -> Q in kvar (physical, but a 15 kW system dwarfs a 3 kW one)
    `colour_by` : column name, e.g. 'oem' — one colour per group.

    READING IT: points should hug the curve. A horizontal cloud at Q = 0 across
    the 240-258 V band is a fleet that isn't responding at all.
    """
    if df.empty:
        print("No intervals to plot.")
        return None

    d = df.copy()
    d = d[(d["V"] >= v_lim[0]) & (d["V"] <= v_lim[1])]
    if d.empty:
        print("No intervals inside the voltage window.")
        return None

    y = (d["Q_kvar"] / d["ac_capacity_kw"]) if normalise else d["Q_kvar"]
    ylab = ("Q / nameplate  [- = absorbing]" if normalise
            else "Q (kvar)  [- = absorbing]")

    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    # Reference curve + tolerance band (normalised units only —
    # in kvar the band is site-specific and can't be drawn as one line)
    v_grid = np.linspace(v_lim[0], v_lim[1], 600)
    if normalise:
        q_ref = _reference_curve(v_grid, 1.0)
        tol   = AS4777["TOL_FRAC"]
        ax.fill_between(v_grid, q_ref - tol, q_ref + tol,
                        color="k", alpha=0.10, lw=0,
                        label=f"+/-{tol*100:.0f}% tolerance band")
        ax.plot(v_grid, q_ref, color="k", lw=1.6, zorder=5,
                label="AS/NZS 4777.2 required Q")

    # Scatter
    if colour_by and colour_by in d.columns:
        groups = d[colour_by].fillna("Unknown").astype(str)
        for g in sorted(groups.unique()):
            m = groups == g
            ax.scatter(d.loc[m, "V"], y[m], s=pt_size, alpha=pt_alpha, label=str(g))
    else:
        ax.scatter(d["V"], y, s=pt_size, alpha=pt_alpha, color=C_MAIN)

    # Set-point guides
    vv = AS4777["VVAR"]
    for vx, lab in [(vv["V1"], "V1"), (vv["V2"], "V2"),
                    (vv["V3"], "V3"), (vv["V4"], "V4")]:
        if v_lim[0] <= vx <= v_lim[1]:
            ax.axvline(vx, ls=":", color="grey", lw=0.8, zorder=1)
            ax.text(vx, ax.get_ylim()[1], f" {lab}", fontsize=7,
                    va="top", color="grey")
    ax.axhline(0, color="grey", lw=0.5)

    ax.set_xlabel("Voltage (V)", fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title or "Volt-VAr response vs the required curve",
                 fontsize=10, weight="bold")
    ax.grid(color=C_GRID, lw=0.5)
    ax.set_axisbelow(True)

    leg = ax.legend(fontsize=7.5, loc="lower left", framealpha=0.92,
                    markerscale=3)
    for h in leg.legend_handles:
        try:
            h.set_alpha(1.0)
        except Exception:
            pass

    plt.tight_layout()
    plt.show()
    return fig


def plot_vvar_by_month(df_by_month, normalise=True, v_lim=(200, 270),
                       pt_alpha=0.15, pt_size=3, ncols=4, figsize=None):
    """
    Small-multiples: one Volt-VAr scatter per month.

    `df_by_month` : dict {month_int: intervals_df}

    Seasonal structure is the thing to look for — voltage is higher in summer
    (more PV export), so the 240-258 V band is more populated, and a fleet that
    responds correctly should show a visibly stronger absorbing tail then.
    """
    import math

    months = sorted(df_by_month.keys())
    months = [m for m in months if not df_by_month[m].empty]
    if not months:
        print("No monthly data to plot.")
        return None

    n = len(months)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    figsize = figsize or (3.6 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=110,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    v_grid = np.linspace(v_lim[0], v_lim[1], 400)
    q_ref  = _reference_curve(v_grid, 1.0)
    tol    = AS4777["TOL_FRAC"]

    for ax, m in zip(axes, months):
        d = df_by_month[m]
        d = d[(d["V"] >= v_lim[0]) & (d["V"] <= v_lim[1])]
        y = (d["Q_kvar"] / d["ac_capacity_kw"]) if normalise else d["Q_kvar"]

        ax.scatter(d["V"], y, s=pt_size, alpha=pt_alpha, color=C_MAIN)
        if normalise:
            ax.fill_between(v_grid, q_ref - tol, q_ref + tol,
                            color="k", alpha=0.10, lw=0)
            ax.plot(v_grid, q_ref, color="k", lw=1.3, zorder=5)

        ax.axhline(0, color="grey", lw=0.4)
        ax.set_title(f"Month {m:02d}  (n={len(d):,})", fontsize=8.5)
        ax.grid(color=C_GRID, lw=0.4)
        ax.set_axisbelow(True)

    for ax in axes[n:]:
        ax.axis("off")

    fig.supxlabel("Voltage (V)", fontsize=9)
    fig.supylabel("Q / nameplate" if normalise else "Q (kvar)", fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig
