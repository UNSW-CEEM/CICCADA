"""
Family F: curtailment concentration. How estimated curtailed energy is spread across
sites and groups. Style follows oem_analysis/notebooks/99_publication_figures.ipynb
(Figure 4) and se_plots.plot_curtailment_lorenz.

    F.1  plot_curtailment_summary   Lorenz curve per mechanism, with a bar chart of total
                                    curtailment by mechanism between them
                                    (Volt-VAr | totals | Volt-Watt, widths 40/20/40 %)
         plot_lorenz                one Lorenz panel (used by F.1; accepts ax=)
    F.2  plot_energy_by_group       curtailed energy per group: total (MWh), share of the
                                    group's generation, and share of sites affected

Lorenz curve
------------
Affected sites (curtailed energy > 0) ranked largest first; x = cumulative % of affected
sites, y = cumulative % of total curtailment. Ranked this way the curve bows ABOVE the
grey dashed equality line and reads directly as "the top X % of sites account for Y %".
Markers and labels at the top 1, 5, 10 and 20 %; the top 20 % is shaded. Gini in the
panel title (2A - 1 in this orientation; 0 = equal loss everywhere, 1 = one site).

Colours
-------
palette.MECHANISM: Volt-VAr-induced curtailment violet, Volt-Watt curtailment indigo
(the publication colours). Bars and Lorenz curves use the mechanism colour.

Input
-----
One row per site with energy columns in kWh: <mechanism>_kwh (voltvar_kwh, voltwatt_kwh),
optionally generation_kwh and grouping columns (state, cohort ...).
sources.se_curtailment() (SolarEdge, local store, Volt-Watt only),
sources.sa_curtailment_aws() (Solar Analytics, Athena) or sources.load_synthetic_curtailment().
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import palette as C

NAMES = {"voltvar": "Volt-VAr", "voltwatt": "Volt-Watt"}
TOPS = (1, 5, 10, 20)


def _mechs(df, mechanisms):
    ms = mechanisms or [m for m in NAMES if f"{m}_kwh" in df.columns]
    return [m for m in ms if f"{m}_kwh" in df.columns]


def _affected(df, mechanism):
    e = df[f"{mechanism}_kwh"]
    return e[e > 0].sort_values(ascending=False).to_numpy(dtype=float)


def concentration(df: pd.DataFrame, mechanism: str = "voltwatt", tops=(1, 5, 10, 20, 50)) -> pd.DataFrame:
    """Share of the total held by the top X % of affected sites (sites with energy > 0)."""
    e = _affected(df, mechanism)
    rows = []
    for t in tops:
        k = max(1, int(round(len(e) * t / 100))) if len(e) else 0
        rows.append({"top_pct": t, "n_sites": k, "kwh": e[:k].sum(),
                     "pct_of_total": 100 * e[:k].sum() / e.sum() if len(e) else np.nan})
    return pd.DataFrame(rows)


def gini(df: pd.DataFrame, mechanism: str = "voltwatt") -> float:
    """Gini coefficient of curtailed energy across affected sites (0 = equal, -> 1 = concentrated)."""
    e = _affected(df, mechanism)
    if not len(e):
        return np.nan
    x = np.concatenate([[0.0], np.arange(1, len(e) + 1) / len(e)])
    y = np.concatenate([[0.0], np.cumsum(e) / e.sum()])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(2 * trapz(y, x) - 1)


def plot_lorenz(df: pd.DataFrame, mechanism: str = "voltwatt", ax=None, title: str | None = None):
    """One Lorenz panel for one mechanism (descending ranking, top 20 % shaded, top 1/5/10/20 % marked)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
    e = _affected(df, mechanism)
    colour = C.MECHANISM[mechanism]
    ax.axvspan(0, 20, color=colour, alpha=0.07, zorder=0)
    ax.plot([0, 100], [0, 100], color=C.REFERENCE["threshold"], ls="--", lw=1.0, label="perfect equality", zorder=1)
    if len(e):
        x = np.concatenate([[0.0], np.arange(1, len(e) + 1) / len(e) * 100])
        y = np.concatenate([[0.0], np.cumsum(e) / e.sum() * 100])
        ax.plot(x, y, color=colour, lw=2.2, zorder=3, label=f"{len(e):,} affected sites")
        for q in TOPS:
            share = float(np.interp(q, x, y))
            ax.plot([q], [share], "o", ms=5, color=colour, zorder=4)
            ax.annotate(f"top {q}%: {share:.0f}%", xy=(q, share), xytext=(6, -11), textcoords="offset points",
                        fontsize=8, color=C.REFERENCE["text"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("cumulative % of affected sites (ranked by curtailment)")
    ax.set_ylabel("cumulative % of total estimated curtailment")
    g = gini(df, mechanism)
    ax.set_title(f"{title or NAMES[mechanism] + ' curtailment concentration'}\nGini = {g:.3f}", fontsize=10)
    ax.grid(color=C.REFERENCE["grid"], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    return fig if fig is not None else ax


def plot_curtailment_summary(df: pd.DataFrame, mechanisms=None, info: dict | None = None, title: str | None = None):
    """
    F.1 Lorenz curve per mechanism with a bar chart of total curtailment by mechanism in
    between: Volt-VAr | totals | Volt-Watt (widths 40/20/40 %). With one mechanism only:
    totals | Lorenz (25/75 %). Bars are labelled with kWh and the number of affected sites.
    """
    ms = _mechs(df, mechanisms)
    if len(ms) == 2:
        fig, (ax_l, ax_bar, ax_r) = plt.subplots(1, 3, figsize=(16, 5.4),
                                                 gridspec_kw={"width_ratios": [0.4, 0.2, 0.4]})
        lorenz_axes = {"voltvar": ax_l, "voltwatt": ax_r}
    else:
        fig, (ax_bar, ax_r) = plt.subplots(1, 2, figsize=(10, 5.4), gridspec_kw={"width_ratios": [0.25, 0.75]})
        lorenz_axes = {ms[0]: ax_r}
    totals = {m: df[f"{m}_kwh"].clip(lower=0).sum() for m in ms}
    ax_bar.bar([NAMES[m] for m in ms], list(totals.values()), color=[C.MECHANISM[m] for m in ms], width=0.55)
    for i, m in enumerate(ms):
        n_aff = int((df[f"{m}_kwh"] > 0).sum())
        ax_bar.text(i, totals[m], f"{totals[m]:,.0f} kWh\n{n_aff:,} sites", ha="center", va="bottom", fontsize=9)
    ax_bar.set_ylabel("Estimated curtailment (kWh)")
    ax_bar.set_title("Curtailment by response mode", fontsize=10)
    ax_bar.grid(color=C.REFERENCE["grid"], lw=0.5, axis="y")
    ax_bar.grid(axis="x", visible=False)
    ax_bar.set_axisbelow(True)
    ax_bar.margins(y=0.15)
    ax_bar.set_xlim(-0.75, len(ms) - 0.25)
    for m, ax in lorenz_axes.items():
        plot_lorenz(df, m, ax=ax)
    info = info or {}
    sub = "  ·  ".join(p for p in (info.get("label", ""), info.get("date", "")) if p)
    if title or sub:
        fig.suptitle("\n".join(p for p in (title, sub) if p), x=0.02, ha="left", y=0.995, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if (title or sub) else 1))
    return fig


def plot_energy_by_group(df: pd.DataFrame, by: str, mechanisms=None, info: dict | None = None, sort: str = "total",
                         title: str | None = None, group_label: str | None = None):
    """
    F.2 Three panels sharing the group axis: total curtailed energy (MWh), curtailed
    energy as % of the group's generation (needs generation_kwh), and the share of sites
    affected. Mechanisms side by side in their colours. Sorted by total of the first
    mechanism ('total'), by 'label', or as given (None).
    """
    ms = _mechs(df, mechanisms)
    g = df.groupby(by)
    tab = pd.DataFrame({"n": g.size()})
    for m in ms:
        tab[f"{m}_mwh"] = g[f"{m}_kwh"].sum() / 1000
        tab[f"{m}_aff"] = g[f"{m}_kwh"].apply(lambda e: 100 * (e > 0).mean())
        if "generation_kwh" in df:
            tab[f"{m}_pct"] = 100 * g[f"{m}_kwh"].sum() / g["generation_kwh"].sum()
    if sort == "total":
        tab = tab.sort_values(f"{ms[0]}_mwh")
    elif sort == "label":
        tab = tab.sort_index(ascending=False)
    panels = [("mwh", "Curtailed energy (MWh)")]
    if "generation_kwh" in df:
        panels.append(("pct", "Share of generation (%)"))
    panels.append(("aff", "Sites affected (%)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels) + 1.5, 0.45 * len(tab) * max(1, 0.7 * len(ms)) + 2),
                             sharey=True)
    y = np.arange(len(tab))
    h = 0.8 / len(ms)
    for ax, (key, lab) in zip(np.atleast_1d(axes), panels):
        for j, m in enumerate(ms):
            yy = y + ((len(ms) - 1) / 2 - j) * h
            v = tab[f"{m}_{key}"]
            ax.barh(yy, v, height=h * 0.92, color=C.MECHANISM[m], zorder=2)
            for yi, vi in zip(yy, v):
                ax.text(vi, yi, f" {vi:.0f}%" if key == "aff" else f" {vi:,.3g}", va="center", fontsize=7,
                        color=C.REFERENCE["text"])
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.18)
    axes = np.atleast_1d(axes)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{i}  (n = {int(n):,})" for i, n in zip(tab.index, tab.n)])
    if len(ms) > 1:
        axes[-1].legend(handles=[Line2D([], [], color=C.MECHANISM[m], lw=6, label=NAMES[m]) for m in ms],
                        loc="lower right", fontsize=8)
    info = info or {}
    sub = "  ·  ".join(p for p in (info.get("label", ""), info.get("date", "")) if p)
    fig.suptitle(f"{title or 'Estimated curtailment by ' + (group_label or by)}\n{sub}", x=0.02, ha="left",
                 y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=1 - 0.95 / fig.get_figheight(), bottom=0.65 / fig.get_figheight(), left=0.2, right=0.97,
                        wspace=0.08)
    return fig
