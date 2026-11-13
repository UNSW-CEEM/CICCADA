"""
Notebook 12: look at individual 3L3P site-months where the net-meter verdict and the PV-circuit verdict disagree.

For one site-month (three-phase inverter, test on the total of the three phases against the average voltage):
  left   Q_tot (% of rating) vs V_avg for the whole month: fitted intervals of both sources, both OLS lines and the
         AS/NZS 4777.2:2020 Volt-VAr curve (Australia A); the chosen day's net-meter intervals are outlined;
  right  one day (AEST): total active power, total reactive power (measured vs required at the PV-circuit V_avg)
         and the three house-phase voltages with the PV-circuit average. Shaded: the net-meter fit window.

The intervals and window rules are rebuilt in pandas from `Fleet._estimate_sql` rows and must reproduce the stored
fits (a refit check is printed).
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet
from bms_sa_review.ami_analysis.lib import ami_phase1p_plots as P1Plots

KEYS = ["site_id", "year", "month"]
BAND = P1Plots.BAND
COLORS = P1Plots.COLORS
PHASE_COLORS = ["#D85A30", "#b5832a", "#8a4fa3"]


def add_outcome(main: pd.DataFrame) -> pd.DataFrame:
    """TP/FP/FN/TN (positive = conformant), slope error and the side of the band the net-meter slope fell on."""
    d = main.copy()
    p, a = d.pass_blind.astype(bool), d.pass_truth.astype(bool)
    d["outcome"] = np.select([p & a, p & ~a, ~p & a], ["TP", "FP", "FN"], default="TN")
    d["slope_err"] = d.slope_blind - d.slope_truth
    d["meter_side"] = np.select([d.slope_blind > BAND[1], d.slope_blind < BAND[0]],
                                ["too_shallow", "too_steep"], default="in_band")
    return d


def outcome_overview(main: pd.DataFrame) -> pd.DataFrame:
    d = add_outcome(main)
    return (d.groupby(["outcome", "meter_side"])
            .agg(site_months=("site_id", "size"), sites=("site_id", "nunique"),
                 median_slope_meter=("slope_blind", "median"), median_slope_pv=("slope_truth", "median"),
                 median_abs_slope_err=("slope_err", lambda x: x.abs().median()))
            .round(3))


def select_cases(main: pd.DataFrame, outcome: str = "FN", n: int | None = 6, seed: int = 42,
                 how: str = "random") -> pd.DataFrame:
    """Site-months with `outcome`; how = "random" | "worst" (largest |slope error|) | "borderline"."""
    d = add_outcome(main)
    d = d[d.outcome == outcome]
    if how == "worst":
        d = d.reindex(d.slope_err.abs().sort_values(ascending=False).index)
    elif how == "borderline":
        edge = np.minimum((d.slope_blind - BAND[0]).abs(), (d.slope_blind - BAND[1]).abs())
        d = d.reindex(edge.sort_values().index)
    elif how == "random" and n is not None:
        d = d.sample(n=min(n, len(d)), random_state=seed)
    d = d if n is None else d.head(n)
    cols = KEYS + ["rating_kw", "slope_blind", "slope_truth", "slope_err", "meter_side", "pass_blind", "pass_truth",
                   "outcome"]
    return d[cols].reset_index(drop=True)


def fetch_site_month(site_id: int, year: int, month: int, rating_kw: float, *, night_hours=(0, 4),
                     p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = 242, v_hi: float = 253,
                     v_plausible=(150.0, 300.0)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (wide, long): `long` has one row per circuit and timestamp for both sources (tag m = house phase, pv = PV circuit);
    `wide` one row per timestamp with totals (p_m, q_m, p_pv, q_pv in kW/kvar and % of rating), V_avg per source,
    completeness and the fit-window flags exactly as in `ami_phase3p_vvar.total_fit_sql`.
    """
    ym = f"{int(year)}-{int(month):02d}"
    site = pd.DataFrame({"site_id": [int(site_id)], "rating_kw": [float(rating_kw)]})
    parts = []
    con = Fleet.connect("2GB", 2)
    try:
        con.register("_sites", site)
        for tag, source in [("m", "meter_baseline"), ("pv", "ground_truth")]:
            d = con.execute(f"""
                WITH {Fleet._estimate_sql(ym, source, night_hours)}
                SELECT circuit_id, t_stamp, V, p_est, q_est FROM est
            """).df()
            d["tag"] = tag
            parts.append(d)
    finally:
        con.close()
    long = pd.concat(parts, ignore_index=True)
    t = pd.to_datetime(long.t_stamp)
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    long["t_aest"] = t + pd.Timedelta(hours=10)
    long["V_ok"] = long.V.where(long.V.between(*v_plausible))

    g = long.groupby(["tag", "t_aest"])
    agg = g.agg(n_c=("circuit_id", "size"), n_p=("p_est", "count"), n_q=("q_est", "count"), n_v=("V_ok", "count"),
                p=("p_est", "sum"), q=("q_est", "sum"), v_avg=("V_ok", "mean")).reset_index()
    agg["complete"] = (agg.n_c == 3) & (agg.n_p == 3) & (agg.n_q == 3) & (agg.n_v == 3)
    wide = agg.pivot(index="t_aest", columns="tag", values=["p", "q", "v_avg", "complete"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for tag in ["m", "pv"]:
        for c in [f"p_{tag}", f"q_{tag}", f"v_avg_{tag}"]:
            if c not in wide:
                wide[c] = np.nan
        if f"complete_{tag}" not in wide:
            wide[f"complete_{tag}"] = False
        wide[f"complete_{tag}"] = wide[f"complete_{tag}"].astype("boolean").fillna(False).astype(bool)
        for c in [f"p_{tag}", f"q_{tag}", f"v_avg_{tag}"]:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")
        gate = wide[f"p_{tag}"] if tag == "m" else wide[f"p_{tag}"].abs()
        v = wide[f"v_avg_{tag}"]
        wide[f"win_{tag}"] = (wide[f"complete_{tag}"] & (v >= v_lo) & (v < v_hi)
                              & (gate >= p_min_frac * rating_kw) & wide[f"q_{tag}"].notna()).fillna(False)
    sign = np.sign(wide.loc[wide.win_pv, "p_pv"].median()) if wide.win_pv.any() else 1.0
    wide["p_pv_gen"] = wide.p_pv * (sign if sign != 0 else 1.0)
    for c in ["p_m", "q_m", "p_pv_gen", "q_pv"]:
        wide[f"{c}_pct"] = 100 * wide[c] / rating_kw
    wide["q_req_pv_pct"] = P1Plots.required_q_pct(wide.v_avg_pv)
    wide["day"] = wide.t_aest.dt.normalize()
    return wide, long


def plot_case(case: pd.Series, *, date=None, night_hours=(0, 4), p_min_frac: float = Fleet.P_MIN_FRAC,
              v_lo: float = 242, v_hi: float = 253, show_check: bool = True):
    """One figure for one site-month (a row of `select_cases`). Returns (fig, wide)."""
    rating = float(case.rating_kw)
    w, long = fetch_site_month(case.site_id, case.year, case.month, rating, night_hours=night_hours,
                               p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
    if date is not None:
        day = pd.Timestamp(date).normalize()
    else:
        both = w[w.win_m & w.win_pv]
        src = both if len(both) else (w[w.win_m] if w.win_m.any() else w)
        day = src.day.value_counts().idxmax()
    dd = w[w.day == day].set_index("t_aest").reindex(
        pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left"))
    wm, wp = w[w.win_m], w[w.win_pv]
    b_m, a_m = P1Plots.ols(wm.v_avg_m, wm.q_m_pct)
    b_p, a_p = P1Plots.ols(wp.v_avg_pv, wp.q_pv_pct)

    fig = plt.figure(figsize=(16, 9), dpi=110)
    gs = GridSpec(3, 2, width_ratios=[1, 1.6], hspace=0.12, wspace=0.18, figure=fig)
    ax_qv = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_v = fig.add_subplot(gs[2, 1], sharex=ax_p)

    v_line = np.linspace(v_lo - 4, v_hi + 3, 100)
    ax_qv.axvspan(v_lo, v_hi, color="#9ca3af", alpha=0.10, lw=0, label=f"Fit window [{v_lo}, {v_hi}) V")
    ax_qv.scatter(wp.v_avg_pv, wp.q_pv_pct, s=7, alpha=0.35, color=COLORS["pv"], lw=0,
                  label=f"PV circuits (total), fitted (n={len(wp):,})")
    ax_qv.scatter(wm.v_avg_m, wm.q_m_pct, s=7, alpha=0.35, color=COLORS["meter"], lw=0,
                  label=f"Net meter (total), fitted (n={len(wm):,})")
    on_day = w.day == day
    ax_qv.scatter(w.loc[on_day & w.win_m, "v_avg_m"], w.loc[on_day & w.win_m, "q_m_pct"], s=22, facecolor="none",
                  edgecolor="black", lw=0.6, label=f"{day.date()} (net meter)")
    ax_qv.plot(v_line, P1Plots.required_q_pct(v_line), color=COLORS["required"], lw=2.2,
               label="Required curve (AS/NZS 4777.2)")
    for b, a, c, lab in [(b_p, a_p, COLORS["pv"], "PV"), (b_m, a_m, COLORS["meter"], "Net meter")]:
        if np.isfinite(b):
            ax_qv.plot(v_line, a + b * v_line, color=c, lw=2, ls="--", label=f"{lab} OLS: {b:.2f} %/V")
    ax_qv.axhline(0, color="black", lw=0.5)
    ax_qv.set(xlabel="Average phase voltage (V)", ylabel="Total reactive power (% of rating)\n+ supplying, − absorbing",
              xlim=(v_line[0], v_line[-1]), title=f"Whole month. Pass band [{BAND[0]:.3f}, {BAND[1]:.3f}] %/V")
    ax_qv.legend(fontsize=7.5, loc="lower left")
    ax_qv.grid(color="#ebebeb", lw=0.5)

    t = dd.index
    shade = dd.win_m.astype("boolean").fillna(False).to_numpy(bool)
    for ax in (ax_p, ax_q, ax_v):
        ax.fill_between(t, 0, 1, where=shade, transform=ax.get_xaxis_transform(), color=COLORS["meter"],
                        alpha=0.08, lw=0, step="mid")
        ax.grid(color="#ebebeb", lw=0.5)
    ax_p.plot(t, dd.p_pv_gen_pct, color=COLORS["pv"], lw=1.2, label="PV circuits, total P (generation +)")
    ax_p.plot(t, dd.p_m_pct, color=COLORS["meter"], lw=1.2, label="Net meter, total P_est")
    ax_p.axhline(100 * p_min_frac, color="black", ls=":", lw=0.9, label=f"Gate {100 * p_min_frac:.0f}% of rating")
    ax_p.set_ylabel("Active power\n(% of rating)")
    ax_q.plot(t, dd.q_pv_pct, color=COLORS["pv"], lw=1.2, label="PV circuits, total Q")
    ax_q.plot(t, dd.q_m_pct, color=COLORS["meter"], lw=1.2, label="Net meter, total Q_est")
    ax_q.plot(t, dd.q_req_pv_pct, color=COLORS["required"], lw=1.6, label="Required Q at PV-circuit V_avg")
    ax_q.axhline(0, color="black", lw=0.5)
    ax_q.set_ylabel("Reactive power\n(% of rating)")

    lm = long[(long.tag == "m") & (long.t_aest.dt.normalize() == day)]
    for k, (cid, g) in enumerate(sorted(lm.groupby("circuit_id"), key=lambda x: x[0])):
        g = g.set_index("t_aest").V_ok.reindex(t)
        ax_v.plot(t, g, color=PHASE_COLORS[k % 3], lw=0.8, alpha=0.8, label=f"House phase L{k + 1}")
    ax_v.plot(t, dd.v_avg_pv, color=COLORS["pv"], lw=1.4, label="PV circuits, average V")
    for v, ls in [(240, ":"), (v_lo, "-"), (v_hi, "-"), (258, ":")]:
        ax_v.axhline(v, color="#6b7280", ls=ls, lw=0.7)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_xlabel(f"{day.date()} (AEST). Shaded: intervals in the net-meter fit window")
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in (ax_p, ax_q, ax_v):
        ax.legend(fontsize=7.5, loc="upper left", ncol=4)
    plt.setp(ax_p.get_xticklabels(), visible=False)
    plt.setp(ax_q.get_xticklabels(), visible=False)
    fig.suptitle(f"Site {case.site_id}, {int(case.year)}-{int(case.month):02d} ({case.outcome}): net meter "
                 f"{case.slope_blind:.2f} %/V ({'pass' if case.pass_blind else 'fail'}) vs PV circuits "
                 f"{case.slope_truth:.2f} %/V ({'pass' if case.pass_truth else 'fail'}); rating {rating:g} kW",
                 fontsize=11, fontweight="bold")
    if show_check:
        print(f"refit slope: net meter {b_m:.3f} (stored {case.slope_blind:.3f}), PV {b_p:.3f} "
              f"(stored {case.slope_truth:.3f})")
    return fig, w
