"""
Notebook 06: validate the final blind Volt-VAr slope test against
  (a) the same slope test applied DIRECTLY to ground truth (separate PV circuit), and
  (b) the original Solar Analytics (SA) analysis (legacy Athena table `conformance_voltvar`).

The fitting itself is `ami_fleet_vvar` (same code as notebook 07), so 06 and 07 cannot drift.
This module only adds: the notebook-05 cohort sample, the SA query/verdict, and the
three-way comparison (agreement metrics, disagreements, plots).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet

KEYS = ["site_id", "year", "month"]
SOURCE_LABELS = {"blind": "Blind (AMI, final method)", "truth": "Ground truth (direct)", "sa": "Original SA"}
PAIRS = [("blind", "truth"), ("truth", "sa"), ("blind", "sa")]   # (tested, reference)


# =========================================================================================
# Cohort
# =========================================================================================
def sample_cohort(site_meta: pd.DataFrame, n_sites: int | None = 100, random_state: int = 42,
                  rating_column: str = "ac_capacity_kw") -> pd.DataFrame:
    """
    Same random cohort as notebook 05, Section 7: sites with 1 load phase and 1 PV phase,
    then `n_sites` drawn with `random_state`. `n_sites=None` -> the WHOLE cohort (no sampling).
    Pass `site_meta` loaded the same way as in 05 (row order matters for pandas .sample).
    Returns site_id, rating_kw and site attributes.
    """
    cohort = site_meta.loc[(site_meta.n_load_phases == 1) & (site_meta.n_pv_phases == 1)]
    if n_sites is not None:
        cohort = cohort.loc[cohort["site_id"].sample(n=min(n_sites, len(cohort)), random_state=random_state).index]
    return (cohort
            .rename(columns={rating_column: "rating_kw"})
            [["site_id", "rating_kw", "flex_export_detected", "manufacturer", "dnsp_name", "pv_install_date"]]
            .reset_index(drop=True))


# =========================================================================================
# Blind + ground-truth fits for one scenario (voltage band etc.)
# =========================================================================================
def run_scenario(out_dir: Path, months: list[str], sites: pd.DataFrame, settings: dict, *,
                 min_n_window: int, min_v_spread: float, **run_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit `months` for both sources with `settings` (v_lo, v_hi, p_min_frac, night_hours),
    writing to out_dir/{meter_baseline,ground_truth}. Finished months are skipped (resume).
    Returns (blind, truth), restricted to `months`, with `fit_status` added.
    """
    results = []
    for source in Fleet.SOURCES:
        Fleet.run_months(months, Path(out_dir) / source, sites, source=source, **run_kwargs, **settings)
        res = Fleet.load_results(Path(out_dir) / source)
        res = res.loc[res.site_id.isin(sites.site_id)]          # only the current cohort's sites
        res = res.loc[(res.year.astype(str) + "-" + res.month.map("{:02d}".format)).isin(months)]
        results.append(Fleet.add_fit_status(res, min_n_window=min_n_window, min_v_spread=min_v_spread))
    return tuple(results)


# =========================================================================================
# Original SA analysis
# =========================================================================================
def sa_voltvar_sql(site_ids, months: list[str]) -> str:
    """
    Athena SQL for the ORIGINAL SA Volt-VAr results, one row per site-month. Queries the
    literal legacy table `conformance_voltvar` -- NOT TABLES["conformance_voltvar"], which
    points at the newer v2 table.
    """
    ids = ", ".join(str(int(s)) for s in site_ids)
    periods = " OR ".join(f"(year = {int(m[:4])} AND month = {int(m[5:])})" for m in months)
    return f"""
        SELECT site_id, year, month,
               SUM(total_count)                      AS sa_total_count,
               SUM(nonconformance_voltvar_red_count) AS sa_red_count
        FROM conformance_voltvar
        WHERE site_id IN ({ids}) AND ({periods})
        GROUP BY site_id, year, month
    """


def add_sa_verdict(sa: pd.DataFrame, red_threshold: float = 0.10) -> pd.DataFrame:
    """SA's own rule: conformant when red non-conformant intervals <= `red_threshold` of the total."""
    d = sa.astype({"site_id": "int64", "year": "int64", "month": "int64",   # Athena may return strings
                   "sa_total_count": "float64", "sa_red_count": "float64"})
    d["sa_red_frac"] = d.sa_red_count / d.sa_total_count.where(d.sa_total_count > 0)
    d["passed"] = (d.sa_red_frac <= red_threshold).astype("boolean").mask(d.sa_red_frac.isna())
    return d


# =========================================================================================
# Three-way comparison
# =========================================================================================
def combine_sources(blind_pf: pd.DataFrame, truth_pf: pd.DataFrame, sa: pd.DataFrame) -> pd.DataFrame:
    """
    One row per site-month seen by ANY source (outer join), with each source's verdict
    (`pass_blind`, `pass_truth`, `pass_sa`; NA = not assessable there) and key numbers.
    `blind_pf` / `truth_pf`: ok fits with a `passed` column (ConformancePlot.add_slope_pass_fail).
    """
    def pick(df, cols, tag):
        return df[KEYS + cols].rename(columns={c: f"{c}_{tag}" for c in cols})

    fit_cols = ["slope", "r_squared", "n_window", "passed"]
    out = (pick(blind_pf, fit_cols, "blind")
           .merge(pick(truth_pf, fit_cols, "truth"), on=KEYS, how="outer")
           .merge(pick(sa, ["sa_total_count", "sa_red_frac", "passed"], "sa"), on=KEYS, how="outer"))
    out = out.rename(columns={f"passed_{t}": f"pass_{t}" for t in SOURCE_LABELS})
    for t in SOURCE_LABELS:
        out[f"pass_{t}"] = out[f"pass_{t}"].astype("boolean")
    return out


def coverage_table(combined: pd.DataFrame) -> pd.DataFrame:
    """How many site-months each combination of sources could assess."""
    has = combined[[f"pass_{t}" for t in SOURCE_LABELS]].notna()
    has.columns = list(SOURCE_LABELS)
    return has.value_counts().rename("n_site_months").reset_index()


def classification_metrics(pred: pd.Series, actual: pd.Series) -> dict:
    """Confusion counts + accuracy / precision / recall / specificity / F1 (NaN where undefined)."""
    pred, actual = pred.astype(bool), actual.astype(bool)
    tp, fp = int((pred & actual).sum()), int((pred & ~actual).sum())
    fn, tn = int((~pred & actual).sum()), int((~pred & ~actual).sum())
    div = lambda a, b: a / b if b else np.nan
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {"n": tp + fp + fn + tn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": div(tp + tn, tp + fp + fn + tn), "precision": precision, "recall": recall,
            "specificity": div(tn, tn + fp), "f1": div(2 * precision * recall, precision + recall)}


def _pair_frame(combined, tested, reference, common_only):
    cols = [f"pass_{t}" for t in SOURCE_LABELS] if common_only else [f"pass_{tested}", f"pass_{reference}"]
    return combined.dropna(subset=cols)


def agreement_table(combined: pd.DataFrame, *, positive_is_fail: bool = False,
                    common_only: bool = False, pairs=PAIRS) -> pd.DataFrame:
    """
    Metrics for each (tested, reference) pair. Positive class = conformant (pass) unless
    `positive_is_fail`. `common_only`: use only site-months all three sources could assess.
    """
    rows = []
    for tested, reference in pairs:
        d = _pair_frame(combined, tested, reference, common_only)
        pred, actual = d[f"pass_{tested}"].astype(bool), d[f"pass_{reference}"].astype(bool)
        if positive_is_fail:
            pred, actual = ~pred, ~actual
        rows.append({"tested": SOURCE_LABELS[tested], "reference": SOURCE_LABELS[reference],
                     "pass_rate_tested_%": 100 * d[f"pass_{tested}"].astype(bool).mean(),
                     "pass_rate_reference_%": 100 * d[f"pass_{reference}"].astype(bool).mean(),
                     **classification_metrics(pred, actual)})
    return pd.DataFrame(rows)


def disagreements(combined: pd.DataFrame, tested: str, reference: str, *,
                  positive_is_fail: bool = False) -> pd.DataFrame:
    """Site-months where `tested` and `reference` disagree, labelled false_positive / false_negative."""
    d = _pair_frame(combined, tested, reference, common_only=False).copy()
    pred, actual = d[f"pass_{tested}"].astype(bool), d[f"pass_{reference}"].astype(bool)
    if positive_is_fail:
        pred, actual = ~pred, ~actual
    d["outcome"] = np.select([pred & ~actual, ~pred & actual], ["false_positive", "false_negative"], "agree")
    return d.loc[d.outcome != "agree"].sort_values(["outcome", *KEYS])


# =========================================================================================
# Plots
# =========================================================================================
def plot_confusion_matrices(
    combined: pd.DataFrame,
    *,
    common_only: bool = False,
    pairs=PAIRS,
):
    """One pass/fail confusion matrix per pair."""
    fig, axes = plt.subplots(
        1, len(pairs),
        figsize=(4.2 * len(pairs), 3.8),
        dpi=130,
    )

    for ax, (tested, reference) in zip(np.atleast_1d(axes), pairs):
        d = _pair_frame(combined, tested, reference, common_only)

        cm = (
            pd.crosstab(
                d[f"pass_{tested}"].astype(bool),
                d[f"pass_{reference}"].astype(bool),
            )
            .reindex(
                index=[True, False],
                columns=[True, False],
                fill_value=0,
            )
        )

        values = cm.to_numpy()
        total = values.sum()
        percentages = values / max(total, 1) * 100

        ax.imshow(
            values,
            cmap="Blues",
            vmin=0,
            vmax=max(values.max(), 1),
        )

        for i in range(2):
            for j in range(2):
                count = values[i, j]
                pct = percentages[i, j]

                color = (
                    "white"
                    if count > 0.6 * values.max()
                    else "#1f2937"
                )

                # Percentage
                ax.text(
                    j,
                    i - 0.08,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )

                # Count below, in parentheses and smaller
                ax.text(
                    j,
                    i + 0.13,
                    f"({count:,})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=color,
                )

        ax.set_xticks([0, 1], ["Pass", "Fail"])
        ax.set_yticks([0, 1], ["Pass", "Fail"])
        ax.set_xlabel(SOURCE_LABELS[reference], fontsize=9)
        ax.set_ylabel(SOURCE_LABELS[tested], fontsize=9)

        agreement = np.trace(values) / max(total, 1)

        ax.set_title(
            f"Agreement {agreement:.1%} (n={total:,})",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def plot_site_month(points: dict, fits: dict, *, v_lo: float, v_hi: float, title: str = ""):
    """
    One panel per source for one site-month: all intervals (grey), the fitted window (blue),
    the source's OLS line (orange dashed) and the required ramp (amber).
    `points` / `fits`: {label: fetch_site_points frame} / {label: that source's fit row}.
    """
    fig, axes = plt.subplots(1, len(points), figsize=(6.5 * len(points), 5.5), dpi=130,
                             sharey=True, squeeze=False)
    v_line = np.linspace(v_lo, v_hi, 50)
    for ax, (label, pts) in zip(axes[0], points.items()):
        fit, win = fits[label], pts.loc[pts.in_window]
        ax.scatter(pts.V, pts.q_pct, s=3, alpha=0.15, color="#9ca3af", label="All intervals")
        ax.scatter(win.V, win.q_pct, s=5, alpha=0.4, color="#2a78d6", label=f"Fitted window (n={len(win):,})")
        ax.plot(v_line, Fleet.REQUIRED_SLOPE * (v_line - Fleet.V_LO), color="#f59e0b", linewidth=2,
                label=f"Required: {Fleet.REQUIRED_SLOPE:.2f} %/V")
        if pd.notna(fit.get("slope")):
            ax.plot(v_line, fit["slope"] * v_line + fit["intercept"], color="#eb6834", linewidth=2,
                    linestyle="--", label=f"OLS: {fit['slope']:.2f} %/V, R²={fit['r_squared']:.2f}")
        ax.axvspan(v_lo, v_hi, color="#2a78d6", alpha=0.04)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{label}  [{fit.get('fit_status', '')}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("Voltage (V)")
        ax.grid(color="#ebebeb", linewidth=0.5)
        ax.legend(fontsize=7, loc="lower left")
    axes[0][0].set_ylabel("Reactive power (% of rating)\n+ supplying, − absorbing")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def response_fan(curves: pd.DataFrame, min_curves_per_bin: int = 5) -> pd.DataFrame:
    """
    Spread of site-month response curves (Fleet.fetch_response_curves) per voltage bin:
    10/25/50/75/90th percentiles across site-months (each counts once) and `n` curves.
    Bins with fewer than `min_curves_per_bin` curves are dropped.
    """
    g = curves.groupby("v_bin_mid")["q_pct_median"]
    fan = g.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    fan["n"] = g.size()
    return fan.loc[fan.n >= min_curves_per_bin]


def plot_cohort_response_and_verdicts(curves: dict, combined: pd.DataFrame, *, v_lo: float, v_hi: float,
                                      tolerance_pct: float, min_curves_per_bin: int = 5,
                                      colors: dict | None = None, title: str = ""):
    """
    (A) Response-curve fans per source ({label: curves frame}) against the required ramp.
    (B) Blind vs ground-truth slope per site-month with the slope-test pass band shaded,
        marked by whether the two verdicts agree (`combined` from combine_sources).
    """
    colors = colors or dict(zip(curves, ["#2a78d6", "#eb6834"]))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=130,
                                     gridspec_kw={"width_ratios": [1.3, 1]}, layout="constrained")
    v_line = np.linspace(v_lo, v_hi, 50)

    # (A) median line + 25-75% and 10-90% bands across site-months
    for label, cv in curves.items():
        fan, c = response_fan(cv, min_curves_per_bin), colors[label]
        n_curves = cv[KEYS].drop_duplicates().shape[0]
        ax_a.fill_between(fan.index, fan[0.10], fan[0.90], color=c, alpha=0.10, linewidth=0)
        ax_a.fill_between(fan.index, fan[0.25], fan[0.75], color=c, alpha=0.25, linewidth=0)
        ax_a.plot(fan.index, fan[0.50], color=c, linewidth=2.5, marker="o", markersize=4,
                  label=f"{label}: median ({n_curves:,} site-months)")
    ax_a.plot(v_line, Fleet.REQUIRED_SLOPE * (v_line - Fleet.V_LO), color="black", linestyle="--",
              linewidth=1.8, label=f"Required ramp ({Fleet.REQUIRED_SLOPE:.2f} %/V)")
    ax_a.fill_between([], [], color="#9ca3af", alpha=0.35, label="Shaded: 25–75% / 10–90% of site-months")
    ax_a.axhline(0, color="black", linewidth=0.5)
    ax_a.set(xlim=(v_lo, v_hi), xlabel="Voltage (V)", title="(A) Volt-VAr response across the cohort",
             ylabel="Reactive power (% of rating)\n+ supplying, − absorbing")
    ax_a.legend(fontsize=7.5, loc="lower left")
    ax_a.grid(color="#ebebeb", linewidth=0.5)

    # (B) slope vs slope; marker size shrinks for large cohorts
    req, tol = Fleet.REQUIRED_SLOPE, tolerance_pct / 100
    band = sorted([req * (1 - tol), req * (1 + tol)])
    both = combined.dropna(subset=["pass_blind", "pass_truth"])
    agree = both.pass_blind == both.pass_truth
    size = 28 if len(both) < 300 else 8
    slopes = both[["slope_blind", "slope_truth"]]
    lim = (min(slopes.quantile(0.005).min(), band[0]) - 0.5, max(slopes.quantile(0.995).max(), 0) + 0.5)
    ax_b.axvspan(*band, color="#1baf7a", alpha=0.10, linewidth=0)
    ax_b.axhspan(*band, color="#1baf7a", alpha=0.10, linewidth=0, label=f"Pass band (±{tolerance_pct}%)")
    ax_b.plot(lim, lim, color="black", linewidth=0.8, label="1:1")
    ax_b.scatter(both.loc[agree, "slope_truth"], both.loc[agree, "slope_blind"], s=size, color="#2a78d6",
                 alpha=0.6, edgecolor="white", linewidth=0.3, label=f"Same verdict (n={agree.sum():,})")
    ax_b.scatter(both.loc[~agree, "slope_truth"], both.loc[~agree, "slope_blind"], s=size * 2, marker="X",
                 color="#eb6834", edgecolor="white", linewidth=0.3, label=f"Different verdict (n={(~agree).sum():,})")
    ax_b.set(xlim=lim, ylim=lim, aspect="equal", xlabel="Ground-truth slope (%/V)", ylabel="Blind slope (%/V)",
             title=f"(B) Slope verdicts: {agree.mean():.0%} agreement (n={len(both):,})")
    ax_b.legend(fontsize=7.5, loc="upper left")
    ax_b.grid(color="#ebebeb", linewidth=0.5)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    return fig
