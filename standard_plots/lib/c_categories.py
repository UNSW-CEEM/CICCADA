"""
Family C: category breakdown. How intervals (or sites) split across the Volt-VAr
and Volt-Watt outcome categories.

    C.1  plot_category_bars   100 % stacked horizontal bars, one per group (cohort, network,
                              manufacturer, install era ...), in intervals or in sites
    C.2  plot_category_days   one site, day by day: 100 % stacked vertical bars per day
    C.3  plot_joint_bars      site verdicts on both modes at once (both / Volt-VAr only /
                              Volt-Watt only / neither), one bar per group

Shape (all figures)
-------------------
Categories always in the same order, worst -> best -> not judged, left to right
(bottom to top for C.2), with the legend in that order. Colours from
palette.VOLTVAR_CATEGORY / STATUS / JOINT. The number of intervals or sites behind each
bar is printed at its end.

Categories (same rules as families A and B; see sources.category_counts)
------------------------------------------------------------------------
Volt-VAr   adverse, inactive, significant shortfall, near-conformant (all outside the
           Figure-2.1-capped band, split by Q_impact), conformant (inside the band, outside
           the 220-240 V deadband), major surplus; not judged: not exposed (inside the
           band in the deadband), not assessable (|P| < 20 % of rating).
Volt-Watt  above the ceiling (non-conformant), at or below the ceiling (conformant);
           not judged: not exposed (V <= 253 V).

Site verdicts (unit="sites", C.3): a site conforms with a mode if at most 10 % of its
judged intervals are non-conforming (AS4777["SITE_CONF_THRESH"]); sites with fewer than
`min_judged` judged intervals are "not assessable".
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "bms_sa_review"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.ciccada_config import AS4777                     # noqa: E402

from . import palette as C                                   # noqa: E402
from .a_site_day import assess_voltwatt, prepare             # noqa: E402
from .b_qv_response import qv_points                         # noqa: E402

SITE_THRESH = AS4777["SITE_CONF_THRESH"]
STANDARD_MANDATORY = pd.Timestamp("2021-12-18")

# (column suffix, label, colour), plotting order
VV = [("adverse", "Adverse", C.VOLTVAR_CATEGORY["Q_adverse"]),
      ("inactive", "Inactive", C.VOLTVAR_CATEGORY["Q_inactive"]),
      ("significant_shortfall", "Significant shortfall", C.VOLTVAR_CATEGORY["Q_significant_shortfall"]),
      ("near_conformant", "Near-conformant", C.VOLTVAR_CATEGORY["Q_near_conformant"]),
      ("conformant", "Conformant", C.VOLTVAR_CATEGORY["Q_conformant"]),
      ("major_surplus", "Major surplus", C.VOLTVAR_CATEGORY["Q_major_surplus"]),
      ("not_exposed", "Not exposed (deadband)", C.VOLTVAR_CATEGORY["Q_not_exposed"]),
      ("not_assessable", "Not assessable (|P| < 20%)", C.VOLTVAR_CATEGORY["Q_not_assessable"])]
VW = [("nonconformant", "Above ceiling", C.VOLTWATT_CATEGORY["nonconformant"]),
      ("conformant", "At or below ceiling", C.VOLTWATT_CATEGORY["conformant"]),
      ("not_exposed", "Not exposed (V ≤ 253 V)", C.VOLTWATT_CATEGORY["not_exposed"])]
SITES = [("nonconformant", "Non-conformant site", C.STATUS["nonconformant"]),
         ("conformant", "Conformant site", C.STATUS["conformant"]),
         ("not_assessable", "Not assessable", C.STATUS["not_assessable"])]
JOINT = [("neither", "Neither", C.JOINT["neither"]),
         ("vw_only", "Volt-Watt only", C.JOINT["Volt-Watt only"]),
         ("vv_only", "Volt-VAr only", C.JOINT["Volt-VAr only"]),
         ("both", "Both conformant", C.JOINT["both conformant"])]
NOT_JUDGED = {"not_exposed", "not_assessable"}
HATCHED = {"not_assessable"}
VV_NONCONF = ["adverse", "inactive", "significant_shortfall", "near_conformant", "major_surplus"]
GROUP_NAMES = {"cohort": "cohort", "dnsp": "network (DNSP)", "oem": "manufacturer", "install_era": "install date",
               "install_year": "install year", "state": "state", "month": "month"}


def _cats(mode):
    return {"voltvar": VV, "voltwatt": VW, "sites": SITES, "joint": JOINT}[mode]


# =============================================================================
# 1. Preparing tables
# =============================================================================
def add_groups(counts: pd.DataFrame, top_oem: int = 8, top_dnsp: int = 8, anonymise: bool = False) -> pd.DataFrame:
    """
    Add grouping columns to a category_counts table: install_era (before / after the
    2020 standard became mandatory, 18 Dec 2021), install_year, oem and dnsp (largest
    `top_*` kept, the rest 'Other'; missing 'Unknown'). anonymise=True replaces names
    with 'Manufacturer 1..', 'Network 1..' in order of size.
    """
    d = counts.copy()
    inst = pd.to_datetime(d.get("pv_install_date"))
    d["install_year"] = inst.dt.year.astype("Int64")
    d["install_era"] = np.where(inst.isna(), "Unknown",
                                np.where(inst >= STANDARD_MANDATORY, "After 18 Dec 2021", "Before 18 Dec 2021"))
    for col, src, top, word in [("oem", "manufacturer", top_oem, "Manufacturer"), ("dnsp", "dnsp_name", top_dnsp, "Network")]:
        s = d[src].fillna("Unknown").astype(str)
        keep = s[s != "Unknown"].value_counts().index[:top]
        s = s.where(s.isin(keep) | (s == "Unknown"), "Other")
        if anonymise:
            names = {k: f"{word} {i + 1}" for i, k in enumerate(keep)}
            s = s.map(lambda x: names.get(x, x))
        d[col] = s
    return d


def site_verdicts(counts: pd.DataFrame, min_judged: int = 12, vv_nonconf=None) -> pd.DataFrame:
    """
    Per site and mode: share of judged intervals non-conforming, and the site verdict
    (10 % rule). Volt-VAr non-conforming = every outside-band category by default; pass
    e.g. vv_nonconf=["adverse", "inactive", "significant_shortfall"] for a stricter reading
    that tolerates near-misses and over-response.
    """
    d = counts.copy()
    nonconf = list(vv_nonconf or VV_NONCONF)
    vv_j = d[[f"vv_{c}" for c in VV_NONCONF + ["conformant"]]].sum(axis=1)
    vv_nc = d[[f"vv_{c}" for c in nonconf]].sum(axis=1)
    vw_j = d["vw_nonconformant"] + d["vw_conformant"]
    for m, j, nc in [("vv", vv_j, vv_nc), ("vw", vw_j, d["vw_nonconformant"])]:
        d[f"{m}_judged"] = j
        d[f"{m}_nc_share"] = nc / j.where(j > 0)
        d[f"{m}_verdict"] = np.select([j < min_judged, d[f"{m}_nc_share"] <= SITE_THRESH],
                                      ["not_assessable", "conformant"], "nonconformant")
    return d


def category_table(counts: pd.DataFrame, by: str | None, mode: str = "voltvar", unit: str = "intervals",
                   judged_only: bool = True, min_judged: int = 12, vv_nonconf=None) -> pd.DataFrame:
    """
    Rows = groups (`by`, or one row 'All sites'), columns = categories in plotting order
    (counts), plus 'n'. mode 'voltvar' / 'voltwatt'; unit 'intervals' or 'sites'
    (site verdicts); mode 'joint' always counts sites judged on both modes.
    judged_only drops the not-exposed / not-assessable categories (intervals only).
    """
    d = counts.assign(_all="All sites") if by is None else counts
    key = "_all" if by is None else by
    if mode == "joint" or unit == "sites":
        v = site_verdicts(d, min_judged, vv_nonconf)
        if mode == "joint":
            v = v[(v.vv_verdict != "not_assessable") & (v.vw_verdict != "not_assessable")]
            vv_ok, vw_ok = v.vv_verdict == "conformant", v.vw_verdict == "conformant"
            v["cat"] = np.select([vv_ok & vw_ok, vv_ok, vw_ok], ["both", "vv_only", "vw_only"], "neither")
        else:
            v["cat"] = v[f"{'vv' if mode == 'voltvar' else 'vw'}_verdict"]
        cats = [c for c, _, _ in _cats("joint" if mode == "joint" else "sites")]
        t = v.groupby([key, "cat"]).size().unstack(fill_value=0).reindex(columns=cats, fill_value=0)
    else:
        pre = "vv_" if mode == "voltvar" else "vw_"
        cats = [c for c, _, _ in _cats(mode) if not (judged_only and c in NOT_JUDGED)]
        t = d.groupby(key)[[pre + c for c in cats]].sum()
        t.columns = cats
    t["n"] = t.sum(axis=1)
    t.index.name = by or ""
    return t[t.n > 0]


# =============================================================================
# 2. Figures
# =============================================================================
def _legend_handles(cats):
    return [Patch(facecolor=col, hatch="///" if c in HATCHED else None, edgecolor="white", label=lab)
            for c, lab, col in cats]


def plot_category_bars(table: pd.DataFrame, mode: str = "voltvar", unit: str = "intervals", info: dict | None = None,
                       sort: str | None = "worst", title: str | None = None, label_min: float = 6.0):
    """
    C.1 100 % stacked horizontal bars from category_table. sort: 'worst' (largest
    non-conforming share at the top), 'n' (largest first), 'label', or None (table order;
    use it for ordered groups such as install year). Segments of at least `label_min` %
    are labelled with their share.
    """
    kind = "joint" if mode == "joint" else ("sites" if unit == "sites" else mode)
    cats = [c for c in _cats(kind) if c[0] in table.columns]
    share = table[[c for c, _, _ in cats]].div(table.n, axis=0) * 100
    bad = [c for c, _, _ in cats if c in VV_NONCONF + ["nonconformant", "neither"]]
    if sort == "worst":
        order = share[bad].sum(axis=1).sort_values().index
    elif sort == "n":
        order = table.n.sort_values().index
    elif sort == "label":
        order = sorted(share.index, key=str, reverse=True)
    else:
        order = share.index[::-1]
    share, n = share.loc[order], table.n.loc[order]

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(share) + 2.2))
    y = np.arange(len(share))
    left = np.zeros(len(share))
    for c, lab, col in cats:
        w = share[c].to_numpy()
        ax.barh(y, w, left=left, color=col, edgecolor="white", lw=0.6, height=0.72,
                hatch="///" if c in HATCHED else None, zorder=2)
        for yi, (l, wi) in enumerate(zip(left, w)):
            if wi >= label_min:
                dark = col in (C.VOLTVAR_CATEGORY["Q_adverse"], C.VOLTVAR_CATEGORY["Q_inactive"],
                               C.VOLTVAR_CATEGORY["Q_conformant"], C.VOLTVAR_CATEGORY["Q_major_surplus"],
                               C.STATUS["conformant"], C.STATUS["nonconformant"])
                ax.text(l + wi / 2, yi, f"{wi:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if dark else C.REFERENCE["zero"], zorder=3)
        left += w
    what = "sites" if (unit == "sites" or mode == "joint") else "intervals"
    for yi, v in enumerate(n):
        ax.text(101, yi, f"n = {int(v):,}", va="center", fontsize=8, color=C.REFERENCE["text"])
    ax.set_yticks(y)
    ax.set_yticklabels([str(i) for i in share.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Share of {what} (%)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=_legend_handles(cats), loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(len(cats), 4),
              frameon=False, fontsize=8)
    name = {"voltvar": "Volt-VAr", "voltwatt": "Volt-Watt", "joint": "Volt-VAr and Volt-Watt"}[mode]
    info = info or {}
    grp = GROUP_NAMES.get(table.index.name, table.index.name) or "group"
    head = title or (f"{name} outcome by {grp}, {what}" if table.index.name else f"{name} outcome, {what}")
    sub = "  ·  ".join(x for x in (info.get("label", ""), info.get("date", "")) if x)
    if unit == "sites" or mode == "joint":
        sub += f"  ·  site conforms if ≤ {SITE_THRESH*100:.0f}% of its judged intervals are non-conforming"
    fig.suptitle(f"{head}\n{sub}", x=0.02, ha="left", y=0.995, fontsize=10.5)
    fig.subplots_adjust(left=0.2, right=0.9, top=1 - 1.0 / fig.get_figheight(), bottom=0.6 / fig.get_figheight())
    return fig


def interval_categories(df: pd.DataFrame, rating_kva: float) -> pd.DataFrame:
    """Per interval of one site: Volt-VAr and Volt-Watt category (same names as category_counts)."""
    pts = qv_points(df, rating_kva)
    vv_map = {"Q_adverse": "adverse", "Q_inactive": "inactive", "Q_significant_shortfall": "significant_shortfall",
              "Q_near_conformant": "near_conformant", "Q_major_surplus": "major_surplus",
              "Q_conformant": "conformant", "Q_not_assessable": "not_assessable"}
    vv = pts.category.map(vv_map)
    dead = (pts.V >= AS4777["VVAR"]["V2"]) & (pts.V <= AS4777["VVAR"]["V3"])
    vv = vv.where(~((vv == "conformant") & dead), "not_exposed")
    day = prepare(df, rating_kva)
    w = assess_voltwatt(day).reindex(pts.index)
    vw = np.where(~w.exposed, "not_exposed", np.where(w.nc_pct > 0, "nonconformant", "conformant"))
    return pd.DataFrame({"vv": vv, "vw": vw, "date": pts.date}, index=pts.index)


def plot_category_days(df: pd.DataFrame, info: dict, judged_only: bool = True):
    """
    C.2 One site, day by day: two panels (Volt-VAr, Volt-Watt) of 100 % stacked bars per
    day. With judged_only, each bar is the day's judged intervals, and the number above
    it is how many there were (0 = nothing to judge that day).
    """
    ic = interval_categories(df, info["rating_kva"])
    days = sorted(ic.date.unique())
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, (col, cats, name) in zip(axes, [("vv", VV, "Volt-VAr"), ("vw", VW, "Volt-Watt")]):
        cats = [c for c in cats if not (judged_only and c[0] in NOT_JUDGED)]
        t = ic.groupby(["date", col]).size().unstack(fill_value=0).reindex(index=days, fill_value=0)
        t = t.reindex(columns=[c for c, _, _ in cats], fill_value=0)
        n = t.sum(axis=1)
        share = t.div(n.where(n > 0), axis=0).fillna(0) * 100
        x = np.arange(len(days))
        bottom = np.zeros(len(days))
        for c, lab, colr in cats:
            ax.bar(x, share[c], bottom=bottom, color=colr, width=0.8, edgecolor="white", lw=0.4,
                   hatch="///" if c in HATCHED else None, zorder=2)
            bottom += share[c].to_numpy()
        for xi, v in zip(x, n):
            ax.text(xi, 101.5, f"{int(v)}", ha="center", va="bottom", fontsize=6.5, color=C.REFERENCE["text"])
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"Share of {'judged ' if judged_only else ''}intervals (%)")
        ax.set_title(f"{name}  ·  number above each bar = {'judged ' if judged_only else ''}intervals that day",
                     loc="left", fontsize=9.5, pad=14)
        ax.legend(handles=_legend_handles(cats)[::-1], loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False,
                  fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(np.arange(len(days)))
    axes[-1].set_xticklabels([pd.Timestamp(d).strftime("%d %b") for d in days], rotation=90, fontsize=7.5)
    n_ph = info.get("site", {}).get("n_pv_circuits")
    ph = "  ·  3-phase (mean V, total P and Q)" if n_ph == 3 else ("  ·  single-phase" if n_ph == 1 else "")
    fig.suptitle(f"Outcome by day\n{info.get('label', '')}  ·  {info.get('date', '')}  ·  "
                 f"rating {info['rating_kva']:g} kVA{ph}", x=0.02, ha="left", y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=0.86, right=0.82, left=0.07, bottom=0.12)
    return fig


def plot_joint_bars(counts: pd.DataFrame, by: str | None, info: dict | None = None, sort: str | None = "worst",
                    min_judged: int = 12, vv_nonconf=None):
    """C.3 Site verdicts on both modes at once, for sites judged on both (≥ `min_judged` intervals each)."""
    t = category_table(counts, by, mode="joint", min_judged=min_judged, vv_nonconf=vv_nonconf)
    return plot_category_bars(t, mode="joint", info=info, sort=sort)
