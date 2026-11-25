"""
How the Volt-VAr site tag depends on the metric, and what missing nights do.
============================================================================

Notebook 03, sections 3c and 3d. Logic only (the notebook orchestrates).

3c  Sites whose tag changes between two metrics (e.g. legacy conformant but
    non-conformant on non-conformance frequency), with a handful of unchanged
    sites for comparison, and side-by-side figures showing how each metric
    counts the same intervals: a day plot and a month Q-V plot per site.
    Figures are built from standard_plots (a_site_day / b_qv_response panels,
    palette, assess_voltvar), so they follow the CICCADA plotting standards.
3d  Why the OEM non-conformance frequency and voltage exposure read high:
    OEM telemetry has (almost) no night intervals. Measured two ways:
      oem_hourly_coverage            rows per AEST hour vs expected
      night_effect_on_solar_analytics  the same metrics on Solar Analytics
                                     (which has nights), full day vs only the
                                     intervals an inverter would report.
    The metric definitions are identical in every run.

Nothing here writes to disk.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from oem_analysis.config import se_config as C
from oem_analysis.lib import se_contract as contract
from oem_analysis.lib import se_params

C.bootstrap_sys_path()
from standard_plots.lib import a_site_day as A  # noqa: E402
from standard_plots.lib import b_qv_response as B  # noqa: E402
from standard_plots.lib import palette as PAL  # noqa: E402

__all__ = [
    "METRICS",
    "flip_table",
    "pick_examples",
    "site_frame",
    "best_day",
    "plot_metric_comparison_day",
    "plot_qv_metric_comparison",
    "oem_hourly_coverage",
    "plot_hourly_coverage",
    "night_effect_on_solar_analytics",
]

#: Display order everywhere: legacy first (the original results), then the two new ones.
METRICS = ("legacy_conformance_rate", "nonconformance_frequency", "remote_conformance_rate")
SHORT = {
    "legacy_conformance_rate": "legacy",
    "nonconformance_frequency": "frequency",
    "remote_conformance_rate": "remote",
}
ZONES = ("A", "B", "C", "D")


def _wrapped_summary(vv, width: int = 78) -> str:
    """standard_plots metric summary, wrapped to fit a narrow panel."""
    return "\n".join(
        textwrap.fill(line.replace("  ·  ", " · "), width) for line in A.metric_summary(vv).split("\n")
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3c. Which sites change tag
# ═══════════════════════════════════════════════════════════════════════════
def flip_table(metrics: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """
    One row per site: tag under metric `a` and `b`, a `change` label
    ('<a> conformant -> <b> non-conformant', ..., 'same: conformant', ...),
    the three non-conforming shares, voltage exposure, coverage, and the share
    of the site's failures in each zone (where they come from).
    `metrics` = se_conformance.voltvar_metrics(...) (one row per site).
    """
    m = metrics.reset_index() if "site_alias" not in metrics.columns else metrics.copy()
    ta, tb = m[f"{a}_verdict"], m[f"{b}_verdict"]
    m["change"] = np.where(
        ta == tb,
        "same: " + ta.astype(str),
        f"{SHORT[a]} " + ta.astype(str) + f" -> {SHORT[b]} " + tb.astype(str),
    )
    fails = m[[f"fail_{z}" for z in ZONES]]
    tot = fails.sum(axis=1).replace(0, np.nan)
    for z in ZONES:
        m[f"failures_in_zone_{z}_%"] = (100 * m[f"fail_{z}"] / tot).round(1)
    keep = (
        ["site_alias", "state", "is_three_phase", "change",
         f"{a}_verdict", f"{b}_verdict",
         "legacy_nonconformance_rate", "nonconformance_frequency",
         "remote_nonconformance_rate", "voltage_exposure", "coverage",
         "n_valid", "n_A"]
        + [f"failures_in_zone_{z}_%" for z in ZONES]
    )
    return m[[c for c in keep if c in m.columns]]


def pick_examples(
    flips: pd.DataFrame,
    zone_counts: pd.DataFrame | None = None,
    n_changed: int = 4,
    n_same: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Example sites from flip_table: up to `n_changed` per changed class and
    `n_same` per unchanged class, preferring sites with many zone A intervals
    (so the figures have something to show). With `zone_counts`
    (voltvar_zone_counts), adds `month`, the site's month with most zone A.
    """
    picks = []
    for change, g in flips.groupby("change"):
        n = n_same if change.startswith("same:") else n_changed
        pool = g[g.n_A >= g.n_A.quantile(0.5)] if len(g) > n else g
        picks.append(pool.sample(min(n, len(pool)), random_state=seed))
    out = pd.concat(picks, ignore_index=True)
    if zone_counts is not None:
        best = (
            zone_counts.sort_values("n_A", ascending=False)
            .drop_duplicates("site_alias")[["site_alias", "month"]]
        )
        out = out.merge(best, on="site_alias", how="left")
    order = out.change.str.startswith("same:")
    return out.assign(_o=order).sort_values(["_o", "change"]).drop(columns="_o").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# OEM site data in the standard_plots input contract
# ═══════════════════════════════════════════════════════════════════════════
def site_frame(con, config, site_alias: str, start, end) -> tuple[pd.DataFrame, dict]:
    """
    One OEM site, AEST [start, end), as the standard_plots tidy frame
    (t_aest, V, P_kw, Q_kvar) plus the loader-style info dict. Uses the same
    conventions as the metrics: config.voltage_aggregation, Q orientation
    (config.reactive_orientation) and the rating basis (config.rating_basis).
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    q = contract.q_expr(config, "i")
    rating = contract.capacity_column(config.rating_basis)
    df = con.execute(
        f"""
        SELECT i.ts_aest AS t_aest, {v} AS V, i.P_kW AS P_kw, {q} AS Q_kvar,
               {rating} AS rating, s.state, s.is_three_phase
        FROM se_interval i {contract.cohort_join_sql('i')}
        WHERE i.site_alias = ? AND i.ts_aest >= ? AND i.ts_aest < ?
        ORDER BY i.ts_aest
        """,
        [site_alias, pd.Timestamp(start), pd.Timestamp(end)],
    ).df()
    if df.empty:
        raise ValueError(f"no rows for {site_alias} in [{start}, {end})")
    s = df.iloc[0]
    days = pd.date_range(start, end, freq="D", inclusive="left")
    info = {
        "rating_kva": float(s.rating),
        "label": f"OEM site {site_alias}",
        "date": str(days[0].date()) if len(days) == 1 else f"{days[0].date()} to {days[-1].date()}",
        "source": "OEM inverter telemetry (local store)",
        "rating_basis": f"{config.rating_basis} (empirical; no nameplate in this dataset)",
        "site": {"state": s.state, "n_pv_circuits": 3 if s.is_three_phase else 1},
    }
    return df[["t_aest", "V", "P_kw", "Q_kvar"]], info


def best_day(con, config, site_alias: str, month: str) -> str:
    """The AEST day in `month` with the most zone A intervals (ties: most rows)."""
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    rating = contract.capacity_column(config.rating_basis)
    d = con.execute(
        f"""
        SELECT CAST(i.ts_aest AS DATE) AS day,
               count(*) FILTER (WHERE abs(i.P_kW) >= 0.2 * {rating}
                                AND ({v} > 240 OR {v} < 220)) AS n_a,
               count(*) AS n
        FROM se_interval i {contract.cohort_join_sql('i')}
        WHERE i.site_alias = ? AND i.dt_month = ?
        GROUP BY 1 ORDER BY n_a DESC, n DESC LIMIT 1
        """,
        [site_alias, month],
    ).df()
    return str(d.day.iloc[0])


# ═══════════════════════════════════════════════════════════════════════════
# Side-by-side figures (standard_plots building blocks)
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
        ax.scatter(V[nc], Q[nc], s=7, facecolors="none", edgecolors=PAL.NOT_COUNTED, lw=0.5, alpha=0.5, zorder=2)
        ok = counted & ~fail
        ax.scatter(V[ok], Q[ok], s=8, color=PAL.STATUS["conformant"], alpha=0.45, lw=0, zorder=3)
        ax.scatter(V[fail], Q[fail], s=8, color=PAL.STATUS["nonconformant"], alpha=0.55, lw=0, zorder=4)
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
        Line2D([], [], ls="", marker="o", mfc="none", mec=PAL.NOT_COUNTED, label="Not counted by this metric"),
        Line2D([], [], color=PAL.REFERENCE["required"], ls="--", label="Volt-VAr curve"),
        Patch(facecolor=PAL.REFERENCE["band"], alpha=0.25, label="± tolerance"),
        Line2D([], [], color=PAL.REFERENCE["threshold"], ls="--", label="Fig. 2.1 minimum, 20–60% P (−44%)"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        f"Q–V, every interval, three ways of counting\n{info.get('label', '')}  ·  {info.get('date', '')}",
        x=0.04, ha="left", y=0.995, fontsize=10.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.05, right=0.98, wspace=0.08)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3d. What the missing nights do
# ═══════════════════════════════════════════════════════════════════════════
def oem_hourly_coverage(con, config=None) -> pd.DataFrame:
    """
    Per AEST hour: OEM rows as % of the 12 expected per site-day (on days the
    site reported at all), and % of those rows that are valid (V, P, Q present).
    Same cohort as the metrics.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    return con.execute(
        f"""
        WITH r AS (
            SELECT i.site_alias, i.ts_aest, i.P_kW, i.Q_kvar, {v} AS V
            FROM se_interval i {contract.cohort_join_sql('i')}
            WHERE {contract.cohort_where_sql(config)}
        ),
        sd AS (SELECT count(*) AS n FROM (SELECT DISTINCT site_alias, CAST(ts_aest AS DATE) FROM r))
        SELECT hour(ts_aest) AS hour_aest,
               round(100.0 * count(*) / (12 * (SELECT n FROM sd)), 1) AS rows_pct_of_expected,
               round(100.0 * avg(CASE WHEN P_kW IS NOT NULL AND Q_kvar IS NOT NULL
                                       AND V > 0 AND V < 300 THEN 1 ELSE 0 END), 1) AS valid_pct_of_rows
        FROM r GROUP BY 1 ORDER BY 1
        """
    ).df()


def plot_hourly_coverage(cov: pd.DataFrame):
    """Bars: OEM rows per AEST hour as % of expected."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(cov.hour_aest, cov.rows_pct_of_expected, color=PAL.QUANTITY["Q"], width=0.8, zorder=2)
    ax.axhline(100, color=PAL.REFERENCE["threshold"], ls=":", lw=1)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Hour of day (AEST, UTC+10)")
    ax.set_ylabel("Rows reported\n(% of expected)")
    ax.set_title("OEM telemetry coverage by hour: the inverter stops reporting when it sleeps", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def night_effect_on_solar_analytics(
    month: str = "2025-01",
    n_sites: int | None = 1000,
    keep: str = "P > 0.01 * S",
    seed: int = 1,
    **category_counts_options,
) -> dict:
    """
    The same three metrics on Solar Analytics (local store, which has nights),
    computed twice: on every interval, and on only the intervals an inverter's
    own telemetry would report (`keep`, default P > 1% of rating). Returns
    tables: 'summary', 'frequency_ratio', 'hidden_low_power_failures',
    'frequency_tags', 'tag_changes', and the two per-site frames ('full',
    'awake'). n_sites: random sample of sites (None = all; about a minute per run).
    """
    import duckdb
    from standard_plots.lib import c_categories, sources

    ids = None
    if n_sites:
        meta = duckdb.query(
            f"SELECT DISTINCT site_id FROM read_parquet('{sources.local_store_dir().as_posix()}/ami_site_metadata.parquet')"
        ).df()
        ids = meta.site_id.sample(min(n_sites, len(meta)), random_state=seed).tolist()
    runs = {}
    for name, filt in [("full day", None), ("inverter awake only", keep)]:
        c = sources.category_counts(
            month, site_ids=ids, keep_intervals=filt, with_metadata=False, **category_counts_options
        )
        runs[name] = c_categories.site_verdicts(c).set_index("site_id")
    rows = []
    for name, v in runs.items():
        def conf(col):
            judged = v[col] != "not assessable"
            return 100 * (v[col] == "conformant").sum() / max(judged.sum(), 1)

        rows.append({
            "data": name,
            "sites": len(v),
            "median coverage %": 100 * v.coverage.median(),
            "median voltage exposure %": 100 * v.voltage_exposure.median(),
            "median non-conformance frequency %": 100 * v.nonconformance_frequency.median(),
            "frequency: conformant sites %": conf("nonconformance_frequency_verdict"),
            "remote: conformant sites %": conf("remote_conformance_rate_verdict"),
            "legacy: conformant sites %": conf("legacy_conformance_rate_verdict"),
        })
    a = runs["full day"]
    b = runs["inverter awake only"].reindex(a.index)
    both = a.index[b.n_valid.notna().to_numpy()]
    a, b = a.loc[both], b.loc[both]
    ratio = (b.nonconformance_frequency / a.nonconformance_frequency).replace([np.inf], np.nan)
    hidden = 1 - (b.fail_C + b.fail_D) / (a.fail_C + a.fail_D).replace(0, np.nan)
    return {
        "summary": pd.DataFrame(rows).round(1),
        "frequency_ratio": ratio.describe(percentiles=[0.25, 0.5, 0.75]).round(2),
        "hidden_low_power_failures": (100 * hidden).describe(percentiles=[0.25, 0.5, 0.75]).round(1),
        "frequency_tags": pd.crosstab(
            a.nonconformance_frequency_verdict.rename("full day"),
            b.nonconformance_frequency_verdict.rename("inverter awake only"),
        ),
        "tag_changes": pd.Series(
            {m: int((a[f"{m}_verdict"] != b[f"{m}_verdict"]).sum()) for m in METRICS},
            name=f"sites whose tag changes (of {len(both)})",
        ),
        "full": a,
        "awake": b,
    }
