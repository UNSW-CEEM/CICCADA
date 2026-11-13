"""
Notebook 11: look at individual site-months where the net-meter verdict and the PV-circuit verdict disagree.

For one site-month (3L1P, net meter tested on the reference phase):
  left   Q (% of rating) vs V for the whole month: the fitted intervals of both sources, both OLS lines and
         the AS/NZS 4777.2:2020 Volt-VAr curve (Australia A); the chosen day's intervals are outlined;
  right  one day (AEST): active power, reactive power (measured vs required by the curve at the measured
         voltage) and voltage, for the net-meter phase and the PV circuit. Shaded: the net-meter fit window.

Data and window rules are exactly those of the fits (`ami_fleet_vvar._estimate_sql` / `_window_sql`).
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet
from shared.ciccada_config import AS4777

KEYS = ["site_id", "year", "month"]
_VV = AS4777["VVAR"]
BAND_TOL = 0.20                                                       # +-20 % of the required slope
BAND = sorted([Fleet.REQUIRED_SLOPE * (1 + BAND_TOL), Fleet.REQUIRED_SLOPE * (1 - BAND_TOL)])
COLORS = {"meter": "#D85A30", "pv": "#1f4e79", "required": "#1baf7a"}


# =========================================================================================
# Which site-months
# =========================================================================================
def add_outcome(main: pd.DataFrame) -> pd.DataFrame:
    """
    `main` from notebook 11 section 4 (pass_blind = net meter on the reference phase, pass_truth = PV
    circuit). Adds `outcome` (TP/FP/FN/TN, positive = conformant), `slope_err` (meter - PV) and
    `meter_side`: where the net-meter slope fell relative to the band [-4.000, -2.667] %/V.
    """
    d = main.copy()
    p, a = d.pass_blind.astype(bool), d.pass_truth.astype(bool)
    d["outcome"] = np.select([p & a, p & ~a, ~p & a], ["TP", "FP", "FN"], default="TN")
    d["slope_err"] = d.slope_blind - d.slope_truth
    d["meter_side"] = np.select([d.slope_blind > BAND[1], d.slope_blind < BAND[0]],
                                ["too_shallow", "too_steep"], default="in_band")
    return d


def outcome_overview(main: pd.DataFrame) -> pd.DataFrame:
    """Counts per outcome and, for FN/FP, the side of the band and the size of the slope error."""
    d = add_outcome(main)
    return (d.groupby(["outcome", "meter_side"])
            .agg(site_months=("site_id", "size"), sites=("site_id", "nunique"),
                 median_slope_meter=("slope_blind", "median"), median_slope_pv=("slope_truth", "median"),
                 median_abs_slope_err=("slope_err", lambda x: x.abs().median()))
            .round(3))


def select_cases(main: pd.DataFrame, outcome: str = "FN", n: int | None = 6, seed: int = 42,
                 how: str = "random") -> pd.DataFrame:
    """
    Site-months with `outcome`. `how`: "random" (seeded sample of n), "worst" (largest |slope error|) or
    "borderline" (net-meter slope closest to the band edge). n=None returns all of them.
    """
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
    cols = KEYS + ["ref_phase", "rating_kw_oracle", "slope_blind", "slope_truth", "slope_err", "meter_side",
                   "pass_blind", "pass_truth", "outcome"]
    return d[cols].reset_index(drop=True)


# =========================================================================================
# Data for one site-month
# =========================================================================================
def required_q_pct(v) -> np.ndarray:
    """AS/NZS 4777.2:2020 Volt-VAr curve (Australia A), % of rating, + supplying / - absorbing."""
    xs = [_VV["V1"], _VV["V2"], _VV["V3"], _VV["V4"]]
    ys = [100 * _VV["Q1"], 0.0, 0.0, -100 * _VV["Q4"]]
    return np.interp(np.asarray(v, float), xs, ys)          # flat beyond V1 and V4


def fetch_site_month(site_id: int, year: int, month: int, meter_circuit_id: int, rating_kw: float, *,
                     night_hours=(0, 4), p_min_frac: float = Fleet.P_MIN_FRAC,
                     v_lo: float = 242, v_hi: float = 253) -> pd.DataFrame:
    """
    Every 5-min interval of one site-month, both sources side by side on t_stamp:
      meter (net-meter circuit `meter_circuit_id`): V_m, p_m, q_m (= base - net, kW / kvar), win_m
      pv    (PV circuit, its own reading):          V_pv, p_pv, q_pv, win_pv
    `win_*` = the interval is in that source's fit window. Adds % of rating, AEST time and the required Q
    at each voltage. PV active power is flipped if needed so that generation is positive.
    """
    ym = f"{int(year)}-{int(month):02d}"
    site = pd.DataFrame({"site_id": [int(site_id)], "rating_kw": [float(rating_kw)]})
    parts = {}
    con = Fleet.connect("2GB", 2)
    try:
        con.register("_sites", site)
        for tag, source in [("m", "meter_baseline"), ("pv", "ground_truth")]:
            circ = f"AND e.circuit_id = {int(meter_circuit_id)}" if tag == "m" else ""
            parts[tag] = con.execute(f"""
                WITH {Fleet._estimate_sql(ym, source, night_hours)}
                SELECT e.t_stamp, e.V AS V_{tag}, e.p_est AS p_{tag}, e.q_est AS q_{tag},
                       coalesce({Fleet._window_sql(source, p_min_frac, v_lo, v_hi)}, false) AS win_{tag}
                FROM est e JOIN _sites s USING (site_id)
                WHERE true {circ}
            """).df()
    finally:
        con.close()
    d = parts["m"].merge(parts["pv"], on="t_stamp", how="outer").sort_values("t_stamp")
    t = pd.to_datetime(d.t_stamp)
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    d["t_aest"] = t + pd.Timedelta(hours=10)                 # same +10 h shift as the night baseline
    for c in ["win_m", "win_pv"]:
        d[c] = d[c].astype("boolean").fillna(False).astype(bool)
    sign = np.sign(d.loc[d.win_pv, "p_pv"].median()) if d.win_pv.any() else 1.0
    d["p_pv"] = d.p_pv * (sign if sign != 0 else 1.0)
    for c in ["p_m", "q_m", "p_pv", "q_pv"]:
        d[f"{c}_pct"] = 100 * d[c] / rating_kw
    d["q_req_m_pct"], d["q_req_pv_pct"] = required_q_pct(d.V_m), required_q_pct(d.V_pv)
    d["day"] = d.t_aest.dt.normalize()
    return d.reset_index(drop=True)


def ols(x, y) -> tuple[float, float]:
    """(slope, intercept); NaN if fewer than 3 points or no voltage spread."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def pick_day(d: pd.DataFrame) -> pd.Timestamp:
    """The AEST day with the most intervals in both fit windows (else in the net-meter window)."""
    both = d[d.win_m & d.win_pv]
    src = both if len(both) else d[d.win_m]
    if src.empty:
        src = d
    return src.day.value_counts().idxmax()


# =========================================================================================
# Plot
# =========================================================================================
def _gapped(day_df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Reindex one day to a 5-min grid so lines break at missing intervals."""
    grid = pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left")
    return day_df.drop_duplicates("t_aest").set_index("t_aest").reindex(grid)


def plot_case(case: pd.Series, *, date=None, night_hours=(0, 4), p_min_frac: float = Fleet.P_MIN_FRAC,
              v_lo: float = 242, v_hi: float = 253, show_table: bool = True):
    """
    One figure for one site-month (a row of `select_cases`). `date`: an AEST day (str/Timestamp) to show on
    the right; default = the day with the most intervals in both fit windows. Returns (fig, data).
    """
    rating = float(case.rating_kw_oracle)
    d = fetch_site_month(case.site_id, case.year, case.month, case.ref_phase, rating,
                         night_hours=night_hours, p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
    day = pd.Timestamp(date).normalize() if date is not None else pick_day(d)
    dd = _gapped(d[d.day == day], day)

    wm, wp = d[d.win_m], d[d.win_pv]
    b_m, a_m = ols(wm.V_m, wm.q_m_pct)
    b_p, a_p = ols(wp.V_pv, wp.q_pv_pct)

    fig = plt.figure(figsize=(16, 9), dpi=110)
    gs = GridSpec(3, 2, width_ratios=[1, 1.6], hspace=0.12, wspace=0.18, figure=fig)
    ax_qv = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_v = fig.add_subplot(gs[2, 1], sharex=ax_p)

    # ---- Q vs V, whole month --------------------------------------------------------------------
    v_line = np.linspace(v_lo - 4, v_hi + 3, 100)
    ax_qv.axvspan(v_lo, v_hi, color="#9ca3af", alpha=0.10, lw=0, label=f"Fit window [{v_lo}, {v_hi}) V")
    ax_qv.scatter(wp.V_pv, wp.q_pv_pct, s=7, alpha=0.35, color=COLORS["pv"], lw=0,
                  label=f"PV circuit, fitted (n={len(wp):,})")
    ax_qv.scatter(wm.V_m, wm.q_m_pct, s=7, alpha=0.35, color=COLORS["meter"], lw=0,
                  label=f"Net meter (ref phase), fitted (n={len(wm):,})")
    on_day = d.day == day
    ax_qv.scatter(d.loc[on_day & d.win_m, "V_m"], d.loc[on_day & d.win_m, "q_m_pct"], s=22, facecolor="none",
                  edgecolor="black", lw=0.6, label=f"{day.date()} (net meter)")
    ax_qv.plot(v_line, required_q_pct(v_line), color=COLORS["required"], lw=2.2, label="Required curve (AS/NZS 4777.2)")
    for b, a, c, lab in [(b_p, a_p, COLORS["pv"], "PV"), (b_m, a_m, COLORS["meter"], "Net meter")]:
        if np.isfinite(b):
            ax_qv.plot(v_line, a + b * v_line, color=c, lw=2, ls="--", label=f"{lab} OLS: {b:.2f} %/V")
    ax_qv.axhline(0, color="black", lw=0.5)
    ax_qv.set(xlabel="Voltage (V)", ylabel="Reactive power (% of rating)\n+ supplying, − absorbing",
              xlim=(v_line[0], v_line[-1]),
              title=f"Whole month. Pass band [{BAND[0]:.3f}, {BAND[1]:.3f}] %/V")
    ax_qv.legend(fontsize=7.5, loc="lower left")
    ax_qv.grid(color="#ebebeb", lw=0.5)

    # ---- One day: P, Q, V -----------------------------------------------------------------------
    t = dd.index
    shade = dd.win_m.astype("boolean").fillna(False).to_numpy(bool)
    for ax in (ax_p, ax_q, ax_v):
        ax.fill_between(t, 0, 1, where=shade, transform=ax.get_xaxis_transform(), color=COLORS["meter"],
                        alpha=0.08, lw=0, step="mid")
        ax.grid(color="#ebebeb", lw=0.5)

    ax_p.plot(t, dd.p_pv_pct, color=COLORS["pv"], lw=1.2, label="PV circuit P (generation +)")
    ax_p.plot(t, dd.p_m_pct, color=COLORS["meter"], lw=1.2, label="Net meter P_est = base − net")
    ax_p.axhline(100 * p_min_frac, color="black", ls=":", lw=0.9, label=f"Gate {100 * p_min_frac:.0f}% of rating")
    ax_p.set_ylabel("Active power\n(% of rating)")

    ax_q.plot(t, dd.q_pv_pct, color=COLORS["pv"], lw=1.2, label="PV circuit Q (measured)")
    ax_q.plot(t, dd.q_m_pct, color=COLORS["meter"], lw=1.2, label="Net meter Q_est = base − net")
    ax_q.plot(t, dd.q_req_pv_pct, color=COLORS["required"], lw=1.6, label="Required Q at PV-circuit V")
    ax_q.axhline(0, color="black", lw=0.5)
    ax_q.set_ylabel("Reactive power\n(% of rating)")

    ax_v.plot(t, dd.V_pv, color=COLORS["pv"], lw=1.0, label="PV circuit V")
    ax_v.plot(t, dd.V_m, color=COLORS["meter"], lw=1.0, ls="--", label="Net meter (ref phase) V")
    for v, ls in [(_VV["V3"], ":"), (v_lo, "-"), (v_hi, "-"), (_VV["V4"], ":")]:
        ax_v.axhline(v, color="#6b7280", ls=ls, lw=0.7)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_xlabel(f"{day.date()} (AEST). Shaded: intervals in the net-meter fit window")
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in (ax_p, ax_q, ax_v):
        ax.legend(fontsize=7.5, loc="upper left", ncol=3)
    plt.setp(ax_p.get_xticklabels(), visible=False)
    plt.setp(ax_q.get_xticklabels(), visible=False)

    fig.suptitle(f"Site {case.site_id}, {int(case.year)}-{int(case.month):02d} ({case.outcome}): net meter "
                 f"{case.slope_blind:.2f} %/V ({'pass' if case.pass_blind else 'fail'}) vs PV circuit "
                 f"{case.slope_truth:.2f} %/V ({'pass' if case.pass_truth else 'fail'}); rating {rating:g} kW, "
                 f"ref phase {int(case.ref_phase)}", fontsize=11, fontweight="bold")

    if show_table:
        # Check: refitted slopes must equal the stored ones (same intervals as the notebook fits)
        print(f"refit slope: net meter {b_m:.3f} (stored {case.slope_blind:.3f}), "
              f"PV {b_p:.3f} (stored {case.slope_truth:.3f}); days in month with fitted net-meter intervals: "
              f"{wm.day.nunique()}")
    return fig, d