"""
Family D: conformance rate by group or over time. What share of sites (or site-months)
conform, broken down by install year, network, manufacturer, cohort or month.

    D.1  plot_rate_by_year    vertical bars per install year, line at the date
                              AS/NZS 4777.2:2020 became mandatory (18 Dec 2021)
    D.2  plot_rate_by_group   horizontal bars, unordered groups sorted by rate,
                              dotted line at the overall rate
    D.3  plot_rate_over_time  rate per calendar month, one line per series (e.g. cohort)
    D.4  plot_rate_heatmap    group x year grid of rates; cells with too few sites hatched

Shape (all figures)
-------------------
y (or x) = % conformant, 0-100. Every bar or point carries its n and a 95 % confidence
interval (error bar or band). The interval answers "how precisely does this group's
share pin down the underlying rate?": treating the units in the group as a sample, an
interval built this way contains the true rate in about 95 of 100 samples. It is
computed with the Wilson score method (Wilson 1927), which, unlike the textbook
p +/- 1.96 sqrt(p(1-p)/n), never leaves 0-100 % and stays reliable for small groups and
rates near 0 or 100 % (Brown, Cai & DasGupta 2001). Caveat: site-months of the same site
are not independent, so intervals on site-months are narrower than they should be;
site-level intervals do not have this problem. Several verdict columns at once (e.g. the net-meter and the PV-circuit slope
verdicts) are drawn side by side, coloured as data sources (palette.SOURCE) when the
names match, otherwise from palette.GROUP_CYCLE. One verdict column alone is drawn in the
conformant colour.

Input
-----
Any table with one row per unit (site or site-month), boolean verdict column(s) and the
grouping columns: sources.slope_verdicts() / slope_sites() (slope method, notebook 13),
or c_categories.site_verdicts(...) on point-wise category counts. NA verdicts are left
out of both numerator and denominator.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import palette as C

STANDARD_MANDATORY = pd.Timestamp("2021-12-18")
Z95 = 1.959964

#: Labels and source styles for the slope-method verdict columns.
VERDICT_LABELS = {"passed": "Net meter (blind slope)", "pass_truth": "PV circuits (reference slope)",
                  "conformant": "Conformant", "vv_ok": "Volt-VAr", "vw_ok": "Volt-Watt"}
VERDICT_SOURCE = {"passed": "net_meter", "pass_truth": "pv_circuit"}


# =============================================================================
# 1. Rates
# =============================================================================
def wilson(k, n, z: float = Z95):
    """Wilson score interval for k successes out of n (arrays ok). Returns (low, high) as fractions."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z ** 2 / n
        mid = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def rate_table(df: pd.DataFrame, by, ok="passed") -> pd.DataFrame:
    """
    Rate per group. `by`: column name, list of columns, or None (overall); `ok`: verdict
    column or list of them. Long format: group column(s), verdict, n, k, rate, lo, hi
    (rates in %).
    """
    oks = [ok] if isinstance(ok, str) else list(ok)
    keys = [] if by is None else ([by] if isinstance(by, str) else list(by))
    parts = []
    for o in oks:
        d = df[df[o].notna()].copy()
        d["_ok"] = d[o].astype(bool)
        g = d.groupby(keys, observed=True)["_ok"] if keys else d.assign(_k="All").groupby("_k")["_ok"]
        t = g.agg(n="size", k="sum").reset_index()
        lo, hi = wilson(t.k, t.n)
        t["rate"], t["lo"], t["hi"] = 100 * t.k / t.n, 100 * lo, 100 * hi
        t["verdict"] = o
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    return out.drop(columns=["_k"], errors="ignore")


def _colours(oks):
    if len(oks) == 1:
        return {oks[0]: C.STATUS["conformant"]}
    if all(o in VERDICT_SOURCE for o in oks):
        return {o: C.SOURCE[VERDICT_SOURCE[o]]["color"] for o in oks}
    seq = mpl.color_sequences.get(C.GROUP_CYCLE, mpl.color_sequences["tab10"])
    return {o: seq[i % len(seq)] for i, o in enumerate(oks)}


def _subtitle(info, unit):
    info = info or {}
    parts = [info.get("label", ""), info.get("date", ""), f"unit: {unit}",
             "error bars: 95 % confidence interval (Wilson score)"]
    return "  ·  ".join(p for p in parts if p)


def _verdict_legend(oks, colours, labels):
    if len(oks) == 1:
        return None
    return [Patch(facecolor=colours[o], label=(labels or {}).get(o, VERDICT_LABELS.get(o, o))) for o in oks]


# =============================================================================
# 2. Figures
# =============================================================================
def plot_rate_by_year(df: pd.DataFrame, ok="passed", year: str = "install_year", info: dict | None = None,
                      unit: str = "site-months", min_n: int = 10, labels: dict | None = None,
                      title: str | None = None, mandatory: pd.Timestamp | None = STANDARD_MANDATORY):
    """D.1 Rate per install year (ordered x), with the date the standard became mandatory marked."""
    oks = [ok] if isinstance(ok, str) else list(ok)
    t = rate_table(df.dropna(subset=[year]), year, oks)
    t = t[t.n >= min_n]
    years = sorted(t[year].astype(int).unique())
    col = _colours(oks)
    w = 0.8 / len(oks)
    fig, ax = plt.subplots(figsize=(min(13, max(8, 0.5 * len(years) * max(1, len(oks) * 0.7) + 2)), 5.6))
    for j, o in enumerate(oks):
        s = t[t.verdict == o].set_index(t[t.verdict == o][year].astype(int)).reindex(years)
        x = np.arange(len(years)) + (j - (len(oks) - 1) / 2) * w
        ax.bar(x, s.rate, width=w * 0.92, color=col[o], zorder=2)
        ax.errorbar(x, s.rate, yerr=[(s.rate - s.lo).clip(lower=0), (s.hi - s.rate).clip(lower=0)], fmt="none", ecolor=C.REFERENCE["zero"],
                    elinewidth=0.8, capsize=2, zorder=3)
        for xi, (r, n) in enumerate(zip(s.rate, s.n)):
            if pd.notna(n):
                ax.text(x[xi], -3, f"{int(n):,}", ha="center", va="top", fontsize=6.5, rotation=90,
                        color=C.REFERENCE["text"])
    if mandatory is not None:
        # position between the bars of the years either side of the date, scaled by day of year
        frac = (mandatory - pd.Timestamp(mandatory.year, 1, 1)).days / 365
        xm = np.interp(mandatory.year + frac - 0.5, years, np.arange(len(years)))
        ax.axvline(xm, color=C.REFERENCE["required"], ls="--", lw=1.1, zorder=4)
        ax.text(xm + 0.08, 98, f"AS/NZS 4777.2:2020 mandatory\nfrom {mandatory:%d %b %Y} →", va="top", fontsize=8)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(-14, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.spines["bottom"].set_position(("data", 0))
    ax.tick_params(axis="x", length=0, pad=34)
    ax.set_xlabel("Install year")
    ax.set_ylabel(f"Conformant {unit} (%)")
    ax.grid(axis="x", visible=False)
    ax.text(-0.01, -3, "n", transform=ax.get_yaxis_transform(), fontsize=7.5, color=C.REFERENCE["text"],
            ha="right", va="top")
    leg = _verdict_legend(oks, col, labels)
    if leg:
        ax.legend(handles=leg, loc="upper left")
    fig.suptitle(f"{title or 'Conformance rate by install year'}\n{_subtitle(info, unit)}",
                 x=0.02, ha="left", y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.98)
    return fig


def plot_rate_by_group(df: pd.DataFrame, by: str, ok="passed", info: dict | None = None, unit: str = "sites",
                       min_n: int = 20, sort: str | None = "rate", labels: dict | None = None,
                       title: str | None = None, group_label: str | None = None):
    """
    D.2 Rate per unordered group (network, manufacturer, cohort ...), horizontal bars,
    sorted by rate (the first verdict column) unless sort=None / 'n' / 'label'. Groups
    with fewer than `min_n` units are left out (count printed in the subtitle). The
    dotted line is the overall rate of the first verdict column.
    """
    oks = [ok] if isinstance(ok, str) else list(ok)
    t = rate_table(df, by, oks)
    small = t[(t.verdict == oks[0]) & (t.n < min_n)]
    keep = t[(t.verdict == oks[0]) & (t.n >= min_n)].set_index(by)
    order = {"rate": keep.rate.sort_values().index, "n": keep.n.sort_values().index,
             "label": sorted(keep.index, key=str, reverse=True)}.get(sort, keep.index[::-1])
    col = _colours(oks)
    h = 0.8 / len(oks)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(order) * max(1, 0.7 * len(oks)) + 2))
    y = np.arange(len(order))
    for j, o in enumerate(oks):
        s = t[t.verdict == o].set_index(by).reindex(order)
        yy = y + ((len(oks) - 1) / 2 - j) * h
        ax.barh(yy, s.rate, height=h * 0.92, color=col[o], zorder=2)
        ax.errorbar(s.rate, yy, xerr=[(s.rate - s.lo).clip(lower=0), (s.hi - s.rate).clip(lower=0)], fmt="none", ecolor=C.REFERENCE["zero"],
                    elinewidth=0.8, capsize=2, zorder=3)
        for yi, (hi, n) in zip(yy, zip(s.hi, s.n)):
            if pd.notna(n):
                ax.text(hi + 1, yi, f"n = {int(n):,}", va="center", fontsize=7, color=C.REFERENCE["text"])
    overall = rate_table(df, None, oks[0]).rate.iloc[0]
    ax.axvline(overall, color=C.REFERENCE["threshold"], ls=":", lw=1.2, zorder=1)
    ax.text(overall, len(order) - 0.4, f" all: {overall:.1f}%", fontsize=8, color=C.REFERENCE["text"], va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([str(i) for i in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Conformant {unit} (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    leg = _verdict_legend(oks, col, labels)
    if leg:
        ax.legend(handles=leg, loc="lower right")
    sub = _subtitle(info, unit) + (f"  ·  {len(small)} groups with n < {min_n} not shown" if len(small) else "")
    fig.suptitle(f"{title or 'Conformance rate by ' + (group_label or by)}\n{sub}", x=0.02, ha="left",
                 y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=1 - 0.95 / fig.get_figheight(), bottom=0.65 / fig.get_figheight(), left=0.18, right=0.97)
    return fig


def plot_rate_over_time(df: pd.DataFrame, ok="passed", series: str | None = None, info: dict | None = None,
                        unit: str = "site-months", min_n: int = 20, labels: dict | None = None,
                        title: str | None = None):
    """
    D.3 Rate per calendar month (needs year and month columns), one line per value of
    `series` (e.g. cohort) or per verdict column, each with its Wilson band. Months with
    fewer than `min_n` units are not drawn.
    """
    d = df.assign(_date=pd.to_datetime(dict(year=df.year, month=df.month, day=1)))
    oks = [ok] if isinstance(ok, str) else list(ok)
    if series is not None and len(oks) > 1:
        raise ValueError("Use either several verdict columns or a series column, not both")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    if series is None:
        t = rate_table(d, "_date", oks)
        lines = [(o, t[t.verdict == o]) for o in oks]
        col = _colours(oks)
        names = {o: (labels or {}).get(o, VERDICT_LABELS.get(o, o)) for o in oks}
    else:
        t = rate_table(d, ["_date", series], oks[0])
        vals = sorted(t[series].dropna().unique(), key=str)
        seq = mpl.color_sequences.get(C.GROUP_CYCLE, mpl.color_sequences["tab10"])
        col = {v: seq[i % len(seq)] for i, v in enumerate(vals)}
        lines = [(v, t[t[series] == v]) for v in vals]
        names = {v: str(v) for v in vals}
    for key, s in lines:
        s = s[s.n >= min_n].sort_values("_date")
        ax.fill_between(s._date, s.lo, s.hi, color=col[key], alpha=0.12, lw=0)
        ax.plot(s._date, s.rate, color=col[key], lw=1.8, marker="o", ms=3.5,
                label=f"{names[key]} (n = {int(s.n.sum()):,})")
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"Conformant {unit} (%)")
    ax.set_xlabel("Month")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b\n%Y"))
    ax.legend(loc="upper left")
    fig.suptitle(f"{title or 'Conformance rate by month'}\n{_subtitle(info, unit).replace('bars:', 'bands:')}",
                 x=0.02, ha="left", y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.07, right=0.98)
    return fig


def plot_rate_heatmap(df: pd.DataFrame, row: str, col: str = "install_year", ok="passed", info: dict | None = None,
                      unit: str = "sites", min_n: int = 10, max_rows: int | None = 12, title: str | None = None,
                      row_label: str | None = None, col_label: str | None = None, show_n: bool = True,
                      mandatory: pd.Timestamp | None = STANDARD_MANDATORY):
    """
    D.4 Rate for every row-group x column-group cell (e.g. manufacturer x install year) on
    a fixed 0-100 % ramp (palette.CMAP['rate']), so heatmaps from different runs compare.
    Each cell shows its rate and, below it, (n). Cells with fewer than `min_n` units are
    blank grey and hatched. White lines separate the cells, a black frame closes the grid,
    and a dashed line between install years marks when AS/NZS 4777.2:2020 became mandatory.
    Rows are the `max_rows` largest groups, sorted by their overall rate (highest first).
    """
    t = rate_table(df.dropna(subset=[row, col]), [row, col], ok)
    tot = rate_table(df.dropna(subset=[row]), row, ok).set_index(row)
    rows = tot.n.sort_values(ascending=False).index[: max_rows or len(tot)]
    rows = list(tot.loc[rows].rate.sort_values(ascending=False).index)
    cols = sorted(t[col].unique())
    rate = t.pivot(index=row, columns=col, values="rate").reindex(index=rows, columns=cols)
    n = t.pivot(index=row, columns=col, values="n").reindex(index=rows, columns=cols)
    data = rate.where(n >= min_n)
    col_names = [str(int(c)) if isinstance(c, (int, float, np.integer, np.floating)) and float(c).is_integer() else str(c)
                 for c in cols]

    cmap = plt.get_cmap(C.CMAP["rate"]).copy()
    cmap.set_bad(C.REFERENCE["night"])
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 3.5, 0.45 * len(rows) + 2.4))
    im = ax.imshow(np.ma.masked_invalid(data.to_numpy(dtype=float)), cmap=cmap,
                   norm=mpl.colors.Normalize(0, 100), aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v, nij = data.iat[i, j], n.iat[i, j]
            if pd.isna(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                                           edgecolor=C.REFERENCE["hatch"], linewidth=0))
                continue
            ax.text(j, i, f"{v:.0f}" + (f"\n({int(nij):,})" if show_n else ""), ha="center", va="center",
                    fontsize=6.5, color="white" if v > 60 else C.REFERENCE["text_dark"])

    if mandatory is not None:
        first_after = mandatory.year + (1 if mandatory.month == 12 else 0)
        if first_after in [int(c) for c in cols if str(c).isdigit() or isinstance(c, (int, np.integer, float))]:
            x = [int(c) for c in cols].index(first_after) - 0.5
            ax.axvline(x, color=C.REFERENCE["required"], ls="--", lw=1.4)
            ax.text(x + 0.1, -0.62, f"AS/NZS 4777.2:2020 mandatory from {mandatory:%d %b %Y} →", fontsize=7.5,
                    color=C.REFERENCE["required"], va="bottom")

    ax.set_xticks(range(len(cols)), col_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), [str(r) for r in rows], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(C.REFERENCE["zero"])
        ax.spines[side].set_linewidth(1.0)
    ax.set_xlabel(col_label or col)
    ax.set_ylabel(row_label or row)
    detail = "  ·  ".join(p for p in ((info or {}).get("label", ""), (info or {}).get("date", "")) if p)
    ax.set_title(f"{title or 'Conformance rate by ' + (row_label or row) + ' and ' + (col_label or col)}\n"
                 f"(% of {unit} conformant; (n) = {unit}{'; ' + detail if detail else ''})",
                 fontsize=10, fontweight="bold", pad=22, loc="center")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"Conformant {unit} (%)", fontsize=8)
    ax.legend(handles=[Patch(facecolor=C.REFERENCE["night"], edgecolor=C.REFERENCE["hatch"], hatch="///",
                             label=f"< {min_n} {unit}")],
              loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fig


def site_level(site_months: pd.DataFrame, ok: str = "passed", rule: str = "all",
               keep=("cohort", "oem", "dnsp", "install_year", "install_era", "rating_kw")) -> pd.DataFrame:
    """
    One row per site from site-month verdicts, as notebook 13: rule 'all' = conformant only
    if it passes in every assessable month; 'half' = in at least 50 % of them. NA months
    are ignored; sites with no assessable month get NA. `keep` columns are carried over.
    """
    d = site_months[site_months[ok].notna()]
    g = d.groupby("site_id")[ok].agg(n_months="size", share_pass=lambda x: x.astype(bool).mean())
    g["conformant"] = g.share_pass == 1.0 if rule == "all" else g.share_pass >= 0.5
    cols = [c for c in keep if c in site_months.columns]
    return g.join(site_months.groupby("site_id")[cols].first()).reset_index()
