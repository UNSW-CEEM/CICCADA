"""
Family C: category breakdown. How intervals (or sites) split across the Volt-VAr
and Volt-Watt outcome categories.

    C.1  plot_category_bars   100 % stacked horizontal bars, one per group (cohort, network,
                              manufacturer, install era ...), in intervals or in sites
    C.2  plot_category_days   one site, day by day: 100 % stacked vertical bars per day
    C.3  plot_joint_bars      site tags on both modes at once (both / Volt-VAr only /
                              Volt-Watt only / neither), one bar per group

Shape (all figures)
-------------------
Categories always in the same order, worst -> best, left to right (bottom to top for
C.2), with the legend in that order. Colours from palette.VVAR_INTERVAL / STATUS /
JOINT. The number of intervals or sites behind each bar is printed at its end.

Categories (bms_sa_review/shared, as families A and B; see sources.category_counts)
------------------------------------------------------------------------------------
Volt-VAr   per zone (A response required, B sufficient power + voltage deadband,
           C low power + voltage exposed, D low power + voltage deadband): adverse,
           inactive, significant shortfall, voltage deadband violation, near-conformant,
           pass, over-response. `zones=` picks which zones a bar covers (default A).
           Pass and violation are treated the same way in every zone (no category is
           dropped from the bar).
Volt-VAr   legacy breakdown (mode="voltvar_legacy"): the pre-rework categories
legacy     (P >= 20% at any voltage; nearest-edge Q_impact; low power not assessable).
Volt-Watt  above the ceiling (non-conformant), at or below the ceiling (conformant);
           not judged: not exposed (V <= 253 V).

Site tags (unit="sites", C.3, and site_verdicts for family D)
--------------------------------------------------------------
metric= 'remote_conformance_rate' (default), 'nonconformance_frequency' or
'legacy_conformance_rate', computed by shared.vvar_metrics.conformance_metrics: a
site is conformant if its non-conforming share <= that metric's threshold (10%),
'not assessable' below that metric's minimum. Voltage exposure is carried along.
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

from shared.as4777_curves import ZONE_LABELS  # noqa: E402
from shared.ciccada_config import AS4777, CONFORMANCE  # noqa: E402
from shared.vvar_metrics import (  # noqa: E402
    conformance_metrics,
    failure_categories,
    metric_columns,
)

from . import palette as C  # noqa: E402
from .a_site_day import METRIC_LABELS, assess_voltwatt, prepare  # noqa: E402
from .b_qv_response import qv_points  # noqa: E402

SITE_THRESH = AS4777["SITE_CONF_THRESH"]
STANDARD_MANDATORY = pd.Timestamp("2021-12-18")

# (column suffix, label, colour), plotting order
VV = [
    ("adverse", "Adverse", C.VVAR_INTERVAL["adverse"]),
    ("inactive", "Inactive", C.VVAR_INTERVAL["inactive"]),
    ("significant_shortfall", "Significant shortfall", C.VVAR_INTERVAL["significant_shortfall"]),
    (
        "voltage_deadband_violation",
        "Voltage deadband violation",
        C.VVAR_INTERVAL["voltage_deadband_violation"],
    ),
    ("near_conformant", "Near-conformant", C.VVAR_INTERVAL["near_conformant"]),
    ("pass", "Pass", C.VVAR_INTERVAL["pass"]),
    ("over_response", "Over-response", C.VVAR_INTERVAL["over_response"]),
]
VV_LEGACY = [
    ("adverse", "Adverse", C.VOLTVAR_CATEGORY["Q_adverse"]),
    ("inactive", "Inactive", C.VOLTVAR_CATEGORY["Q_inactive"]),
    (
        "significant_shortfall",
        "Significant shortfall",
        C.VOLTVAR_CATEGORY["Q_significant_shortfall"],
    ),
    ("near_conformant", "Near-conformant", C.VOLTVAR_CATEGORY["Q_near_conformant"]),
    ("inside_band", "Inside band", C.VOLTVAR_CATEGORY["Q_conformant"]),
    ("major_surplus", "Major surplus", C.VOLTVAR_CATEGORY["Q_major_surplus"]),
    (
        "not_assessable",
        "Not assessable (P < 20%)",
        C.VOLTVAR_CATEGORY["Q_not_assessable"],
    ),
]
VW = [
    ("nonconformant", "Above ceiling", C.VOLTWATT_CATEGORY["nonconformant"]),
    ("conformant", "At or below ceiling", C.VOLTWATT_CATEGORY["conformant"]),
    ("not_exposed", "Not exposed (V ≤ 253 V)", C.VOLTWATT_CATEGORY["not_exposed"]),
]
SITES = [
    ("nonconformant", "Non-conformant site", C.STATUS["nonconformant"]),
    ("conformant", "Conformant site", C.STATUS["conformant"]),
    ("not_assessable", "Not assessable", C.STATUS["not_assessable"]),
]
JOINT = [
    ("neither", "Neither", C.JOINT["neither"]),
    ("vw_only", "Volt-Watt only", C.JOINT["Volt-Watt only"]),
    ("vv_only", "Volt-VAr only", C.JOINT["Volt-VAr only"]),
    ("both", "Both conformant", C.JOINT["both conformant"]),
]
NOT_JUDGED = {"not_exposed", "not_assessable"}
HATCHED = {"not_assessable"}
#: Categories counted as "bad" when sorting bars worst-first.
VV_NONCONF = list(failure_categories(CONFORMANCE["FAILURE_DEFINITION_DEFAULT"]))
BAD = set(VV_NONCONF) | {"nonconformant", "neither"}
GROUP_NAMES = {
    "cohort": "cohort",
    "dnsp": "network (DNSP)",
    "oem": "manufacturer",
    "install_era": "install date",
    "install_year": "install year",
    "state": "state",
    "month": "month",
}


def _cats(mode):
    return {
        "voltvar": VV,
        "voltvar_legacy": VV_LEGACY,
        "voltwatt": VW,
        "sites": SITES,
        "joint": JOINT,
    }[mode]


# =============================================================================
# 1. Preparing tables
# =============================================================================
def add_groups(
    counts: pd.DataFrame, top_oem: int = 8, top_dnsp: int = 8, anonymise: bool = False
) -> pd.DataFrame:
    """
    Add grouping columns to a category_counts table: install_era (before / after the
    2020 standard became mandatory, 18 Dec 2021), install_year, oem and dnsp (largest
    `top_*` kept, the rest 'Other'; missing 'Unknown'). anonymise=True replaces names
    with 'Manufacturer 1..', 'Network 1..' in order of size.
    """
    d = counts.copy()
    inst = pd.to_datetime(d.get("pv_install_date"))
    d["install_year"] = inst.dt.year.astype("Int64")
    d["install_era"] = np.where(
        inst.isna(),
        "Unknown",
        np.where(inst >= STANDARD_MANDATORY, "After 18 Dec 2021", "Before 18 Dec 2021"),
    )
    for col, src, top, word in [
        ("oem", "manufacturer", top_oem, "Manufacturer"),
        ("dnsp", "dnsp_name", top_dnsp, "Network"),
    ]:
        s = d[src].fillna("Unknown").astype(str)
        keep = s[s != "Unknown"].value_counts().index[:top]
        s = s.where(s.isin(keep) | (s == "Unknown"), "Other")
        if anonymise:
            names = {k: f"{word} {i + 1}" for i, k in enumerate(keep)}
            s = s.map(lambda x: names.get(x, x))
        d[col] = s
    return d


def site_verdicts(
    counts: pd.DataFrame,
    metric: str | None = None,
    failure_definition: str | None = None,
    min_judged: int = 12,
    **metric_options,
) -> pd.DataFrame:
    """
    Per site: all three Volt-VAr metrics and their site tags
    (shared.vvar_metrics.conformance_metrics), plus convenience columns for the
    selected `metric` and for Volt-Watt:

      vv_verdict / vw_verdict    'conformant' / 'nonconformant' / 'not_assessable'
      vv_ok / vw_ok              True / False / NA (for family D rate plots)
      <metric>_ok                the same for each of the three Volt-VAr metrics
      vv_nc_share, vv_judged     non-conforming share and its denominator
      voltage_exposure           zone A intervals / valid intervals

    metric: 'remote_conformance_rate' (default), 'nonconformance_frequency' or
    'legacy_conformance_rate'. min_judged: minimum judged intervals for the
    Volt-Watt tag and for the legacy Volt-VAr tag (as before the rework).
    metric_options go to conformance_metrics (e.g. frequency_threshold=0.05,
    frequency_min_valid_intervals=288, min_zone_a_intervals=20).
    """
    metric = metric or CONFORMANCE["METRIC_DEFAULT"]
    fdef = failure_definition or CONFORMANCE["FAILURE_DEFINITION_DEFAULT"]
    metric_options.setdefault("legacy_min_intervals", min_judged)
    d = conformance_metrics(counts, failure_definition=fdef, **metric_options)
    cols = metric_columns(metric)
    judged = {
        "nonconformance_frequency": d["n_valid"],
        "remote_conformance_rate": d["remote_denominator"],
        "legacy_conformance_rate": d.get("n_legacy_total"),
    }[metric]
    d["vv_metric"] = metric
    d["vv_judged"] = judged
    d["vv_nc_share"] = d[cols["share"]]
    d["vv_verdict"] = d[cols["verdict"]].map(
        {
            "conformant": "conformant",
            "non-conformant": "nonconformant",
            "not assessable": "not_assessable",
        }
    )
    vw_j = d["vw_nonconformant"] + d["vw_conformant"]
    d["vw_judged"] = vw_j
    d["vw_nc_share"] = d["vw_nonconformant"] / vw_j.where(vw_j > 0)
    d["vw_verdict"] = np.select(
        [vw_j < min_judged, d["vw_nc_share"] <= SITE_THRESH],
        ["not_assessable", "conformant"],
        "nonconformant",
    )
    for m in ("vv", "vw"):
        v = d[f"{m}_verdict"]
        d[f"{m}_ok"] = (v == "conformant").where(v != "not_assessable")
    # one True / False / NA column per Volt-VAr metric, to compare them in family D
    for name in METRIC_LABELS:
        vcol = metric_columns(name)["verdict"]
        if vcol in d:
            d[f"{name}_ok"] = (d[vcol] == "conformant").where(d[vcol] != "not assessable")
    return d


def category_table(
    counts: pd.DataFrame,
    by: str | None,
    mode: str = "voltvar",
    unit: str = "intervals",
    judged_only: bool = True,
    min_judged: int = 12,
    metric: str | None = None,
    zones=("A",),
    failure_definition: str | None = None,
    **metric_options,
) -> pd.DataFrame:
    """
    Rows = groups (`by`, or one row 'All sites'), columns = categories in plotting order
    (counts), plus 'n'.
    mode 'voltvar' (intervals in `zones`, default zone A - response required),
    'voltvar_legacy' (pre-rework breakdown), 'voltwatt', or 'joint'.
    unit 'intervals' or 'sites' (site tags under `metric`); mode 'joint' always counts
    sites judged on both modes. judged_only drops Volt-Watt not exposed and legacy
    not assessable (intervals only).
    """
    d = counts.assign(_all="All sites") if by is None else counts
    key = "_all" if by is None else by
    if mode == "joint" or unit == "sites":
        v = site_verdicts(d, metric, failure_definition, min_judged, **metric_options)
        if mode == "joint":
            v = v[
                (v.vv_verdict != "not_assessable") & (v.vw_verdict != "not_assessable")
            ]
            vv_ok, vw_ok = v.vv_verdict == "conformant", v.vw_verdict == "conformant"
            v["cat"] = np.select(
                [vv_ok & vw_ok, vv_ok, vw_ok], ["both", "vv_only", "vw_only"], "neither"
            )
        else:
            v["cat"] = v[f"{'vw' if mode == 'voltwatt' else 'vv'}_verdict"]
        cats = [c for c, _, _ in _cats("joint" if mode == "joint" else "sites")]
        t = (
            v.groupby([key, "cat"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=cats, fill_value=0)
        )
    elif mode == "voltvar":
        zones = tuple(zones)
        cats = [c for c, _, _ in VV]
        t = pd.DataFrame(
            {c: d[[f"n_{z}_{c}" for z in zones]].sum(axis=1) for c in cats}
        ).groupby(d[key]).sum()
        t.attrs["zones"] = zones
    else:
        pre = "legacy_" if mode == "voltvar_legacy" else "vw_"
        cats = [c for c, _, _ in _cats(mode) if not (judged_only and c in NOT_JUDGED)]
        t = d.groupby(key)[[pre + c for c in cats]].sum()
        t.columns = cats
    t["n"] = t.sum(axis=1)
    t.index.name = by or ""
    t.attrs.update(metric=metric or CONFORMANCE["METRIC_DEFAULT"])
    return t[t.n > 0]



# =============================================================================
# 2. Figures
# =============================================================================
def _legend_handles(cats):
    return [
        Patch(
            facecolor=col,
            hatch="///" if c in HATCHED else None,
            edgecolor="white",
            label=lab,
        )
        for c, lab, col in cats
    ]


def plot_category_bars(
    table: pd.DataFrame,
    mode: str = "voltvar",
    unit: str = "intervals",
    info: dict | None = None,
    sort: str | None = "worst",
    title: str | None = None,
    label_min: float = 6.0,
):
    """
    C.1 100 % stacked horizontal bars from category_table. sort: 'worst' (largest
    non-conforming share at the top), 'n' (largest first), 'label', or None (table order;
    use it for ordered groups such as install year). Segments of at least `label_min` %
    are labelled with their share.
    """
    kind = "joint" if mode == "joint" else ("sites" if unit == "sites" else mode)
    cats = [c for c in _cats(kind) if c[0] in table.columns]
    share = table[[c for c, _, _ in cats]].div(table.n, axis=0) * 100
    bad = [c for c, _, _ in cats if c in BAD]
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
        ax.barh(
            y,
            w,
            left=left,
            color=col,
            edgecolor="white",
            lw=0.6,
            height=0.72,
            hatch="///" if c in HATCHED else None,
            zorder=2,
        )
        for yi, (l, wi) in enumerate(zip(left, w)):
            if wi >= label_min:
                dark = col in (
                    C.VOLTVAR_CATEGORY["Q_adverse"],
                    C.VOLTVAR_CATEGORY["Q_inactive"],
                    C.VOLTVAR_CATEGORY["Q_conformant"],
                    C.VOLTVAR_CATEGORY["Q_major_surplus"],
                    C.VVAR_INTERVAL["voltage_deadband_violation"],
                    C.STATUS["conformant"],
                    C.STATUS["nonconformant"],
                )
                ax.text(
                    l + wi / 2,
                    yi,
                    f"{wi:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if dark else C.REFERENCE["zero"],
                    zorder=3,
                )
        left += w
    what = "sites" if (unit == "sites" or mode == "joint") else "intervals"
    for yi, v in enumerate(n):
        ax.text(
            101,
            yi,
            f"n = {int(v):,}",
            va="center",
            fontsize=8,
            color=C.REFERENCE["text"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([str(i) for i in share.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Share of {what} (%)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=_legend_handles(cats),
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=min(len(cats), 4),
        frameon=False,
        fontsize=8,
    )
    name = {
        "voltvar": "Volt-VAr",
        "voltvar_legacy": "Volt-VAr (legacy categories)",
        "voltwatt": "Volt-Watt",
        "joint": "Volt-VAr and Volt-Watt",
    }[mode]
    info = info or {}
    grp = GROUP_NAMES.get(table.index.name, table.index.name) or "group"
    head = title or (
        f"{name} outcome by {grp}, {what}"
        if table.index.name
        else f"{name} outcome, {what}"
    )
    sub = "  ·  ".join(x for x in (info.get("label", ""), info.get("date", "")) if x)
    if unit == "sites" or mode == "joint":
        metric = table.attrs.get("metric", CONFORMANCE["METRIC_DEFAULT"])
        sub += (
            f"  ·  site tag: {METRIC_LABELS[metric].lower()}, conformant if ≤ "
            f"{SITE_THRESH*100:.0f}% of its judged intervals are non-conforming"
        )
    elif mode == "voltvar" and table.attrs.get("zones"):
        sub += "  ·  zones: " + ", ".join(ZONE_LABELS[z] for z in table.attrs["zones"])
    fig.suptitle(f"{head}\n{sub}", x=0.02, ha="left", y=0.995, fontsize=10.5)
    fig.subplots_adjust(
        left=0.2,
        right=0.9,
        top=1 - 1.0 / fig.get_figheight(),
        bottom=0.6 / fig.get_figheight(),
    )
    return fig


def interval_categories(
    df: pd.DataFrame, rating_kva: float, tol: float | None = None
) -> pd.DataFrame:
    """Per interval of one site: Volt-VAr zone and category (shared rules), Volt-Watt category."""
    pts = qv_points(df, rating_kva, tol)
    day = prepare(df, rating_kva)
    w = assess_voltwatt(day).reindex(pts.index)
    vw = np.where(
        ~w.exposed, "not_exposed", np.where(w.nc_pct > 0, "nonconformant", "conformant")
    )
    return pd.DataFrame(
        {"vv": pts.category, "zone": pts.zone, "vw": vw, "date": pts.date},
        index=pts.index,
    )



def plot_category_days(
    df: pd.DataFrame,
    info: dict,
    judged_only: bool = True,
    zones=("A",),
    tol: float | None = None,
):
    """
    C.2 One site, day by day: two panels (Volt-VAr, Volt-Watt) of 100 % stacked bars per
    day. Volt-VAr bars cover the intervals in `zones` (default A - response required);
    Volt-Watt bars, with judged_only, the day's judged (exposed) intervals. The number
    above each bar is how many intervals it covers (0 = nothing that day).
    """
    ic = interval_categories(df, info["rating_kva"], tol)
    zones = tuple(zones)
    days = sorted(ic.date.unique())
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"hspace": 0.35}
    )
    for ax, (col, cats, name) in zip(
        axes, [("vv", VV, "Volt-VAr"), ("vw", VW, "Volt-Watt")]
    ):
        cats = [c for c in cats if not (judged_only and c[0] in NOT_JUDGED)]
        src = ic[ic.zone.isin(zones)] if col == "vv" else ic
        t = (
            src.groupby(["date", col])
            .size()
            .unstack(fill_value=0)
            .reindex(index=days, fill_value=0)
        )
        t = t.reindex(columns=[c for c, _, _ in cats], fill_value=0)
        n = t.sum(axis=1)
        share = t.div(n.where(n > 0), axis=0).fillna(0) * 100
        x = np.arange(len(days))
        bottom = np.zeros(len(days))
        for c, lab, colr in cats:
            ax.bar(
                x,
                share[c],
                bottom=bottom,
                color=colr,
                width=0.8,
                edgecolor="white",
                lw=0.4,
                hatch="///" if c in HATCHED else None,
                zorder=2,
            )
            bottom += share[c].to_numpy()
        for xi, v in zip(x, n):
            ax.text(
                xi,
                101.5,
                f"{int(v)}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=C.REFERENCE["text"],
            )
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"Share of {'judged ' if judged_only else ''}intervals (%)")
        what = (
            "intervals in " + ", ".join(ZONE_LABELS[z] for z in zones)
            if col == "vv"
            else f"{'judged ' if judged_only else ''}intervals"
        )
        ax.set_title(
            f"{name}  ·  number above each bar = {what} that day",
            loc="left",
            fontsize=9.5,
            pad=14,
        )
        ax.legend(
            handles=_legend_handles(cats)[::-1],
            loc="upper left",
            bbox_to_anchor=(1.005, 1),
            frameon=False,
            fontsize=8,
        )
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(np.arange(len(days)))
    axes[-1].set_xticklabels(
        [pd.Timestamp(d).strftime("%d %b") for d in days], rotation=90, fontsize=7.5
    )
    n_ph = info.get("site", {}).get("n_pv_circuits")
    ph = (
        "  ·  3-phase (mean V, total P and Q)"
        if n_ph == 3
        else ("  ·  single-phase" if n_ph == 1 else "")
    )
    fig.suptitle(
        f"Outcome by day\n{info.get('label', '')}  ·  {info.get('date', '')}  ·  "
        f"rating {info['rating_kva']:g} kVA{ph}",
        x=0.02,
        ha="left",
        y=0.995,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.86, right=0.82, left=0.07, bottom=0.12)
    return fig


def plot_joint_bars(
    counts: pd.DataFrame,
    by: str | None,
    info: dict | None = None,
    sort: str | None = "worst",
    min_judged: int = 12,
    metric: str | None = None,
    failure_definition: str | None = None,
    **metric_options,
):
    """C.3 Site tags on both modes at once, for sites assessable on both (Volt-VAr under `metric`)."""
    t = category_table(
        counts,
        by,
        mode="joint",
        min_judged=min_judged,
        metric=metric,
        failure_definition=failure_definition,
        **metric_options,
    )
    return plot_category_bars(t, mode="joint", info=info, sort=sort)
