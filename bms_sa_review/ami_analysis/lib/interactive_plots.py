"""
Interactive daily plots for notebook 11 (3L1P): one site-month, with a day selector.

  left   Q (% of rating) vs V for the whole month: fitted intervals of the net meter (tested phase) and the
         PV circuit, both OLS lines and the AS/NZS 4777.2:2020 Volt-VAr curve; the selected day's fitted
         intervals are outlined (circles = net meter, squares = PV circuit)
  right  the selected AEST day: active power, reactive power and voltage

Data and window rules are those of the fits (via `ami_phase1p_plots.fetch_site_month`). The month is loaded
once; changing the day only redraws.

Usage (notebook 11, after section 4):
    from bms_sa_review.ami_analysis.lib import interactive_plots as IPlots
    cases = IPlots.P1Plots.select_cases(main, outcome="FN", n=None, how="worst")
    data = IPlots.explore_case(cases.iloc[0], ev=ev)
"""
from __future__ import annotations

from functools import lru_cache

import ipywidgets as widgets
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.gridspec import GridSpec

from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet
from bms_sa_review.ami_analysis.lib import ami_phase1p_plots as P1Plots
from bms_sa_review.ami_analysis.lib import ami_phase3p_plots as P3Plots


NIGHT_HOURS = (0, 4)
V_LO, V_HI = 242, 253


# =========================================================================================
# Metadata
# =========================================================================================
@lru_cache(maxsize=1)
def _site_meta() -> pd.DataFrame:
    return pd.read_parquet(Fleet.Config.store_path("ami_site_metadata_residential"))


@lru_cache(maxsize=1)
def _circuit_meta() -> pd.DataFrame:
    return pd.read_parquet(Fleet.Config.store_path("ami_circuit_metadata"),
                           columns=["site_id", "circuit_id", "circuit_type", "device_type"])


def case_metadata(case: pd.Series, ev: pd.DataFrame | None = None) -> pd.Series:
    """Site metadata, monitoring device and (if `ev` = notebook 10 site evidence is given) notebook 10
    evidence for one site-month (a row of `P1Plots.select_cases`)."""
    sid = int(case.site_id)
    s = _site_meta()
    s = s[s.site_id == sid]
    s = s.iloc[0] if len(s) else pd.Series(dtype=object)
    devices = _circuit_meta()
    devices = devices[devices.site_id == sid]
    m = {
        "site": sid,
        "month": f"{int(case.year)}-{int(case.month):02d}",
        "state / DNSP / postcode": f"{s.get('state')} / {s.get('dnsp_name')} / {s.get('postcode')}",
        "manufacturer": s.get("manufacturer"),
        "model": s.get("model"),
        "AC / DC capacity (kW)": f"{s.get('ac_capacity_kw')} / {s.get('dc_capacity_kw')}",
        "export limit (kW)": s.get("export_limit_kw"),
        "inverters": s.get("inverter_count"),
        "PV installed": str(s.get("pv_install_date"))[:10],
        "house / PV channels": f"{s.get('n_load_phases')}L{s.get('n_pv_phases')}P",
        "monitoring device": ", ".join(sorted(devices.device_type.dropna().unique())) or "unknown",
        "PV circuit(s)": ", ".join(str(c) for c in devices.loc[devices.circuit_type == "pv_site_net", "circuit_id"]),
        "tested house phase": int(case.ref_phase) if pd.notna(case.get("ref_phase")) else "total of 3 phases",
        "net meter slope (%/V)": f"{case.slope_blind:.3f} ({'pass' if case.pass_blind else 'fail'})",
        "PV circuit slope (%/V)": f"{case.slope_truth:.3f} ({'pass' if case.pass_truth else 'fail'})",
        "outcome": case.get("outcome"),
    }
    if ev is not None and (ev.site_id == sid).any():
        e = ev[ev.site_id == sid].iloc[0]
        m.update({
            "nb10 model hint": e.get("model_hint"),
            "nb10 PV / rating": round(float(e.pv_over_rating), 2) if pd.notna(e.get("pv_over_rating")) else None,
            "nb10 voltage ref (min dv_ratio)": round(float(e.dv_ratio_min), 1) if pd.notna(e.get("dv_ratio_min")) else None,
            "nb10 exclusion": e.get("exclusion_reason"),
        })
    return pd.Series(m, name="value")


# =========================================================================================
# Figure
# =========================================================================================
def draw_case(case: pd.Series, d: pd.DataFrame, day, meta: pd.Series | None = None, *,
              p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = V_LO, v_hi: float = V_HI):
    """Same figure as `P1Plots.plot_case`, drawn from already-loaded data `d` (from `fetch_site_month`)
    for the AEST day `day`. Returns the figure."""
    C, VV, BAND = P1Plots.COLORS, P1Plots._VV, P1Plots.BAND
    day = pd.Timestamp(day).normalize()
    dd = P1Plots._gapped(d[d.day == day], day)
    wm, wp = d[d.win_m], d[d.win_pv]
    b_m, a_m = P1Plots.ols(wm.V_m, wm.q_m_pct)
    b_p, a_p = P1Plots.ols(wp.V_pv, wp.q_pv_pct)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = GridSpec(3, 2, width_ratios=[1, 1.6], hspace=0.12, wspace=0.18, figure=fig)
    ax_qv = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_v = fig.add_subplot(gs[2, 1], sharex=ax_p)

    # Left: whole month, selected day outlined
    v_line = np.linspace(v_lo - 4, v_hi + 3, 100)
    ax_qv.axvspan(v_lo, v_hi, color="#9ca3af", alpha=0.10, lw=0, label=f"Fit window [{v_lo}, {v_hi}) V")
    ax_qv.scatter(wp.V_pv, wp.q_pv_pct, s=7, alpha=0.3, color=C["pv"], lw=0,
                  label=f"PV circuit, fitted (n={len(wp):,})")
    ax_qv.scatter(wm.V_m, wm.q_m_pct, s=7, alpha=0.3, color=C["meter"], lw=0,
                  label=f"Net meter, fitted (n={len(wm):,})")
    on = d.day == day
    ax_qv.scatter(d.loc[on & d.win_m, "V_m"], d.loc[on & d.win_m, "q_m_pct"], s=26, facecolor="none",
                  edgecolor="black", lw=0.7, label=f"{day.date()} net meter (n={int((on & d.win_m).sum())})")
    ax_qv.scatter(d.loc[on & d.win_pv, "V_pv"], d.loc[on & d.win_pv, "q_pv_pct"], s=26, marker="s",
                  facecolor="none", edgecolor="#6250d6", lw=0.7,
                  label=f"{day.date()} PV circuit (n={int((on & d.win_pv).sum())})")
    ax_qv.plot(v_line, P1Plots.required_q_pct(v_line), color=C["required"], lw=2.2,
               label="Required curve (AS/NZS 4777.2)")
    for b, a, c, lab in [(b_p, a_p, C["pv"], "PV"), (b_m, a_m, C["meter"], "Net meter")]:
        if np.isfinite(b):
            ax_qv.plot(v_line, a + b * v_line, color=c, lw=2, ls="--", label=f"{lab} OLS: {b:.2f} %/V")
    ax_qv.axhline(0, color="black", lw=0.5)
    ax_qv.set(xlabel="Voltage (V)", ylabel="Reactive power (% of rating)\n+ supplying, − absorbing",
              xlim=(v_line[0], v_line[-1]), title=f"Whole month. Pass band [{BAND[0]:.3f}, {BAND[1]:.3f}] %/V")
    ax_qv.legend(fontsize=7, loc="lower left")
    ax_qv.grid(color="#ebebeb", lw=0.5)

    # Right: the selected day
    t = dd.index
    shade = dd.win_m.astype("boolean").fillna(False).to_numpy(bool)
    for ax in (ax_p, ax_q, ax_v):
        ax.fill_between(t, 0, 1, where=shade, transform=ax.get_xaxis_transform(), color=C["meter"],
                        alpha=0.08, lw=0, step="mid")
        ax.grid(color="#ebebeb", lw=0.5)
    ax_p.plot(t, dd.p_pv_pct, color=C["pv"], lw=1.2, label="PV circuit P (generation +)")
    ax_p.plot(t, dd.p_m_pct, color=C["meter"], lw=1.2, label="Net meter P_est = base − net")
    ax_p.axhline(100 * p_min_frac, color="black", ls=":", lw=0.9, label=f"Gate {100 * p_min_frac:.0f}% of rating")
    ax_p.set_ylabel("Active power\n(% of rating)")
    ax_q.plot(t, dd.q_pv_pct, color=C["pv"], lw=1.2, label="PV circuit Q (measured)")
    ax_q.plot(t, dd.q_m_pct, color=C["meter"], lw=1.2, label="Net meter Q_est = base − net")
    ax_q.plot(t, dd.q_req_pv_pct, color=C["required"], lw=1.6, label="Required Q at PV-circuit V")
    ax_q.axhline(0, color="black", lw=0.5)
    ax_q.set_ylabel("Reactive power\n(% of rating)")
    ax_v.plot(t, dd.V_pv, color=C["pv"], lw=1.0, label="PV circuit V")
    ax_v.plot(t, dd.V_m, color=C["meter"], lw=1.0, ls="--", label="Net meter (tested phase) V")
    for v, ls in [(VV["V3"], ":"), (v_lo, "-"), (v_hi, "-"), (VV["V4"], ":")]:
        ax_v.axhline(v, color="#6b7280", ls=ls, lw=0.7)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_xlabel(f"{day.date()} (AEST). Shaded: intervals in the net-meter fit window")
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in (ax_p, ax_q, ax_v):
        ax.legend(fontsize=7, loc="upper left", ncol=3)
    plt.setp(ax_p.get_xticklabels(), visible=False)
    plt.setp(ax_q.get_xticklabels(), visible=False)

    who = "" if meta is None else f" | {meta['manufacturer']} {meta['model']}, {meta['monitoring device']}"
    fig.suptitle(f"Site {case.site_id}, {int(case.year)}-{int(case.month):02d} ({case.get('outcome')}): net meter "
                 f"{case.slope_blind:.2f} %/V ({'pass' if case.pass_blind else 'fail'}) vs PV circuit "
                 f"{case.slope_truth:.2f} %/V ({'pass' if case.pass_truth else 'fail'}); rating "
                 f"{float(case.rating_kw_oracle):g} kW{who}", fontsize=10.5, fontweight="bold")
    return fig


# =========================================================================================
# Interactive view
# =========================================================================================
def explore_case(case: pd.Series, ev: pd.DataFrame | None = None, date=None, *, night_hours=NIGHT_HOURS,
                 p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = V_LO, v_hi: float = V_HI,
                 show_meta: bool = True) -> pd.DataFrame:
    """
    Metadata table + figure with a day selector (dropdown, Prev / Next) for one site-month (a row of
    `P1Plots.select_cases`). `date`: start day (AEST); default = the day with the most intervals in both fit
    windows. Returns the month's interval data.
    """
    d = P1Plots.fetch_site_month(case.site_id, case.year, case.month, case.ref_phase, float(case.rating_kw_oracle),
                                 night_hours=night_hours, p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
    meta = case_metadata(case, ev)
    if show_meta:
        display(meta.to_frame())

    per_day = d.groupby("day").agg(n_m=("win_m", "sum"), n_pv=("win_pv", "sum"))
    days = list(per_day.index)
    labels = [(f"{x:%Y-%m-%d} ({x:%a})  fitted: net {int(r.n_m):3d} | PV {int(r.n_pv):3d}", x)
              for x, r in per_day.iterrows()]
    start = pd.Timestamp(date).normalize() if date is not None else P1Plots.pick_day(d)
    pick = widgets.Dropdown(options=labels, value=start if start in days else days[0], description="Day",
                            layout=widgets.Layout(width="420px"))
    prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
    nxt = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
    out = widgets.Output()

    def render(*_):
        with out:
            clear_output(wait=True)
            draw_case(case, d, pick.value, meta, p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
            plt.show()

    def step(k: int):
        i = days.index(pick.value) + k
        if 0 <= i < len(days):
            pick.value = days[i]

    pick.observe(render, names="value")
    prev.on_click(lambda _: step(-1))
    nxt.on_click(lambda _: step(1))
    display(widgets.HBox([prev, pick, nxt]), out)
    render()
    return d


# =========================================================================================
# Three-phase inverters (notebook 12): total of the three phases against the average voltage
# =========================================================================================


def draw_case_3p(case: pd.Series, w: pd.DataFrame, long: pd.DataFrame, day, meta: pd.Series | None = None, *,
                 p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = V_LO, v_hi: float = V_HI):
    """Same figure as `P3Plots.plot_case`, drawn from already-loaded data (`w`, `long` from
    `P3Plots.fetch_site_month`) for the AEST day `day`. Returns the figure."""
    C, BAND = P3Plots.COLORS, P3Plots.BAND
    day = pd.Timestamp(day).normalize()
    t_grid = pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left")
    dd = w[w.day == day].drop_duplicates("t_aest").set_index("t_aest").reindex(t_grid)
    wm, wp = w[w.win_m], w[w.win_pv]
    b_m, a_m = P1Plots.ols(wm.v_avg_m, wm.q_m_pct)
    b_p, a_p = P1Plots.ols(wp.v_avg_pv, wp.q_pv_pct)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = GridSpec(3, 2, width_ratios=[1, 1.6], hspace=0.12, wspace=0.18, figure=fig)
    ax_qv = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_v = fig.add_subplot(gs[2, 1], sharex=ax_p)

    # Left: whole month, selected day outlined (circles = net meter, squares = PV circuits)
    v_line = np.linspace(v_lo - 4, v_hi + 3, 100)
    ax_qv.axvspan(v_lo, v_hi, color="#9ca3af", alpha=0.10, lw=0, label=f"Fit window [{v_lo}, {v_hi}) V")
    ax_qv.scatter(wp.v_avg_pv, wp.q_pv_pct, s=7, alpha=0.3, color=C["pv"], lw=0,
                  label=f"PV circuits (total), fitted (n={len(wp):,})")
    ax_qv.scatter(wm.v_avg_m, wm.q_m_pct, s=7, alpha=0.3, color=C["meter"], lw=0,
                  label=f"Net meter (total), fitted (n={len(wm):,})")
    on = w.day == day
    ax_qv.scatter(w.loc[on & w.win_m, "v_avg_m"], w.loc[on & w.win_m, "q_m_pct"], s=26, facecolor="none",
                  edgecolor="black", lw=0.7, label=f"{day.date()} net meter (n={int((on & w.win_m).sum())})")
    ax_qv.scatter(w.loc[on & w.win_pv, "v_avg_pv"], w.loc[on & w.win_pv, "q_pv_pct"], s=26, marker="s",
                  facecolor="none", edgecolor="#6250d6", lw=0.7,
                  label=f"{day.date()} PV circuits (n={int((on & w.win_pv).sum())})")
    ax_qv.plot(v_line, P1Plots.required_q_pct(v_line), color=C["required"], lw=2.2,
               label="Required curve (AS/NZS 4777.2)")
    for b, a, c, lab in [(b_p, a_p, C["pv"], "PV"), (b_m, a_m, C["meter"], "Net meter")]:
        if np.isfinite(b):
            ax_qv.plot(v_line, a + b * v_line, color=c, lw=2, ls="--", label=f"{lab} OLS: {b:.2f} %/V")
    ax_qv.axhline(0, color="black", lw=0.5)
    ax_qv.set(xlabel="Average phase voltage (V)",
              ylabel="Total reactive power (% of rating)\n+ supplying, − absorbing",
              xlim=(v_line[0], v_line[-1]), title=f"Whole month. Pass band [{BAND[0]:.3f}, {BAND[1]:.3f}] %/V")
    ax_qv.legend(fontsize=7, loc="lower left")
    ax_qv.grid(color="#ebebeb", lw=0.5)

    # Right: the selected day
    t = dd.index
    shade = dd.win_m.astype("boolean").fillna(False).to_numpy(bool)
    for ax in (ax_p, ax_q, ax_v):
        ax.fill_between(t, 0, 1, where=shade, transform=ax.get_xaxis_transform(), color=C["meter"],
                        alpha=0.08, lw=0, step="mid")
        ax.grid(color="#ebebeb", lw=0.5)
    ax_p.plot(t, dd.p_pv_gen_pct, color=C["pv"], lw=1.2, label="PV circuits, total P (generation +)")
    ax_p.plot(t, dd.p_m_pct, color=C["meter"], lw=1.2, label="Net meter, total P_est")
    ax_p.axhline(100 * p_min_frac, color="black", ls=":", lw=0.9, label=f"Gate {100 * p_min_frac:.0f}% of rating")
    ax_p.set_ylabel("Active power\n(% of rating)")
    ax_q.plot(t, dd.q_pv_pct, color=C["pv"], lw=1.2, label="PV circuits, total Q")
    ax_q.plot(t, dd.q_m_pct, color=C["meter"], lw=1.2, label="Net meter, total Q_est")
    ax_q.plot(t, dd.q_req_pv_pct, color=C["required"], lw=1.6, label="Required Q at PV-circuit V_avg")
    ax_q.axhline(0, color="black", lw=0.5)
    ax_q.set_ylabel("Reactive power\n(% of rating)")
    lm = long[(long.tag == "m") & (long.t_aest.dt.normalize() == day)]
    for k, (cid, g) in enumerate(sorted(lm.groupby("circuit_id"), key=lambda x: x[0])):
        g = g.drop_duplicates("t_aest").set_index("t_aest").V_ok.reindex(t)
        ax_v.plot(t, g, color=P3Plots.PHASE_COLORS[k % 3], lw=0.8, alpha=0.8, label=f"House phase L{k + 1} ({cid})")
    ax_v.plot(t, dd.v_avg_pv, color=C["pv"], lw=1.4, label="PV circuits, average V")
    for v, ls in [(240, ":"), (v_lo, "-"), (v_hi, "-"), (258, ":")]:
        ax_v.axhline(v, color="#6b7280", ls=ls, lw=0.7)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_xlabel(f"{day.date()} (AEST). Shaded: intervals in the net-meter fit window")
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in (ax_p, ax_q, ax_v):
        ax.legend(fontsize=7, loc="upper left", ncol=4)
    plt.setp(ax_p.get_xticklabels(), visible=False)
    plt.setp(ax_q.get_xticklabels(), visible=False)

    who = "" if meta is None else f" | {meta['manufacturer']} {meta['model']}, {meta['monitoring device']}"
    fig.suptitle(f"Site {case.site_id}, {int(case.year)}-{int(case.month):02d} ({case.get('outcome')}): net meter "
                 f"{case.slope_blind:.2f} %/V ({'pass' if case.pass_blind else 'fail'}) vs PV circuits "
                 f"{case.slope_truth:.2f} %/V ({'pass' if case.pass_truth else 'fail'}); rating "
                 f"{float(case.rating_kw):g} kW{who}", fontsize=10.5, fontweight="bold")
    return fig


def explore_case_3p(case: pd.Series, ev: pd.DataFrame | None = None, date=None, *, night_hours=NIGHT_HOURS,
                    p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = V_LO, v_hi: float = V_HI,
                    show_meta: bool = True) -> pd.DataFrame:
    """Notebook 12 version of `explore_case`, for a row of `P3Plots.select_cases`. Returns the month's
    timestamp table (totals per source)."""
    w, long = P3Plots.fetch_site_month(case.site_id, case.year, case.month, float(case.rating_kw),
                                       night_hours=night_hours, p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
    meta = case_metadata(case, ev)
    if show_meta:
        display(meta.to_frame())

    per_day = w.groupby("day").agg(n_m=("win_m", "sum"), n_pv=("win_pv", "sum"))
    days = list(per_day.index)
    labels = [(f"{x:%Y-%m-%d} ({x:%a})  fitted: net {int(r.n_m):3d} | PV {int(r.n_pv):3d}", x)
              for x, r in per_day.iterrows()]
    if date is not None:
        start = pd.Timestamp(date).normalize()
    else:
        both = w[w.win_m & w.win_pv]
        src = both if len(both) else (w[w.win_m] if w.win_m.any() else w)
        start = src.day.value_counts().idxmax()
    pick = widgets.Dropdown(options=labels, value=start if start in days else days[0], description="Day",
                            layout=widgets.Layout(width="420px"))
    prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
    nxt = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
    out = widgets.Output()

    def render(*_):
        with out:
            clear_output(wait=True)
            draw_case_3p(case, w, long, pick.value, meta, p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
            plt.show()

    def step(k: int):
        i = days.index(pick.value) + k
        if 0 <= i < len(days):
            pick.value = days[i]

    pick.observe(render, names="value")
    prev.on_click(lambda _: step(-1))
    nxt.on_click(lambda _: step(1))
    display(widgets.HBox([prev, pick, nxt]), out)
    render()
    return w


# =========================================================================================
# Three-phase explorer (notebook 12): case navigation, per-phase split, error panel, daily slopes
# =========================================================================================
# What the figure shows for one site-month of a three-phase inverter:
#   left   (a) Q_tot vs V_avg for the month, net-meter points coloured by the chosen mode; the day outlined
#          (b) e = net-meter Q - PV Q vs V_avg on the intervals in both fit windows; its OLS slope is beta_e,
#              the part of the net-meter slope that comes from household reactive power
#          (c) daily slopes (net meter and PV circuits) for every day of the month; the selected day marked
#   right  the selected day, per phase and in total: P, Q, e (= household reactive power) and V
# House phases are L1-L3 in circuit-id order; each PV circuit is paired with the house phase whose voltage it
# tracks most closely over the month (mean |dV|, as notebook 10).

PHASE_PALETTE = ["#e69f00", "#56b4e9", "#cc79a7"]          # L1, L2, L3 (distinct from the total / PV / curve colours)
PV_OFF_FRAC = 0.05                                          # PV total |P| below this x rating = "PV off" (notebook 12)
DAY_START, DAY_END = 7, 18                                  # daylight hours (AEST) for "outside daylight"
MIN_DAY_POINTS, MIN_DAY_SPREAD = 10, 2.0                    # a daily slope needs >= 10 points over >= 2 V
COLOUR_MODES = ["source", "hour of day", "PV output", "PV off"]
DETAILS_3P = ("vvar_phase", "3l3p_vvar", "site_month_details_3l3p.parquet")
_DETAIL_COLS = {"min_share_pv": "PV phase share, smallest (1/3 = balanced)",
                "max_share_pv": "PV phase share, largest",
                "night_max_share": "night load, largest phase share (1/3 = balanced)",
                "catch": "CATCH Power", "model_hint": "model hint", "exclusion_reason": "nb10 exclusion"}


@lru_cache(maxsize=1)
def _details_3p() -> pd.DataFrame:
    p = Fleet.store_dir().joinpath(*DETAILS_3P)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def phase_tables(long: pd.DataFrame, rating_kw: float) -> dict:
    """
    Per-circuit tables (index = AEST time, columns = circuit_id) from `P3Plots.fetch_site_month`'s `long`:
    p, q (% of rating; PV generation positive, per circuit) and V, for the house phases ("m") and PV circuits
    ("pv"); the PV -> house pairing by smallest mean |dV|; and e per house phase (house Q minus the Q of the PV
    circuit(s) paired with it).
    """
    t = {}
    for tag in ("m", "pv"):
        x = long[long.tag == tag].drop_duplicates(["circuit_id", "t_aest"])
        t[tag] = {k: x.pivot(index="t_aest", columns="circuit_id", values=c)
                  for k, c in (("p", "p_est"), ("q", "q_est"), ("v", "V_ok"))}
    house = sorted(t["m"]["p"].columns)
    pvs = sorted(t["pv"]["p"].columns) if len(t["pv"]["p"].columns) else []

    hour = t["pv"]["p"].index.hour
    mid = (hour >= 10) & (hour < 14)
    for c in pvs:                                           # generation positive, circuit by circuit
        med = t["pv"]["p"].loc[mid, c].median()
        if pd.notna(med) and med < 0:
            t["pv"]["p"][c] = -t["pv"]["p"][c]
    for tag in t:
        for k in ("p", "q"):
            t[tag][k] = 100 * t[tag][k] / rating_kw

    pairing, dv = {}, {}
    for c in pvs:
        vp = t["pv"]["v"][c]
        gaps = {h: (vp - t["m"]["v"][h].reindex(vp.index)).abs().mean() for h in house}
        gaps = {h: g for h, g in gaps.items() if pd.notna(g)}
        if gaps:
            pairing[c] = min(gaps, key=gaps.get)
            dv[c] = gaps[pairing[c]]

    qh = t["m"]["q"]
    qp = t["pv"]["q"].reindex(qh.index) if pvs else pd.DataFrame(index=qh.index)
    e = pd.DataFrame({h: qh[h] - sum((qp[c] for c in pvs if pairing.get(c) == h), pd.Series(0.0, index=qh.index))
                      for h in house})
    names = {h: f"L{i + 1}" for i, h in enumerate(house)}
    colors = {h: PHASE_PALETTE[i % 3] for i, h in enumerate(house)}
    return dict(t=t, house=house, pvs=pvs, pairing=pairing, dv=dv, e=e, names=names, colors=colors)


def daily_slopes(w: pd.DataFrame, min_points: int = MIN_DAY_POINTS, min_spread: float = MIN_DAY_SPREAD) -> pd.DataFrame:
    """OLS slope per AEST day for the net meter and the PV circuits (each on its own fit-window intervals)."""
    rows = []
    for day, g in w.groupby("day"):
        r = {"day": day}
        for tag, v, q, win in (("m", "v_avg_m", "q_m_pct", "win_m"), ("pv", "v_avg_pv", "q_pv_pct", "win_pv")):
            x = g[g[win]]
            ok = len(x) >= min_points and (x[v].max() - x[v].min()) >= min_spread
            r[f"slope_{tag}"] = P1Plots.ols(x[v], x[q])[0] if ok else np.nan
            r[f"n_{tag}"] = len(x)
        rows.append(r)
    return pd.DataFrame(rows)


def month_diagnostics(case: pd.Series, w: pd.DataFrame, ph: dict, details: pd.DataFrame | None = None) -> pd.Series:
    """Diagnostics for the site-month, recomputed from the interval data (beta_e, sd(e), PV-off and outside-daylight
    shares as `ami_phase3p_vvar.total_diag_sql`), the PV -> house pairing, and notebook 12's details if given."""
    rating = float(case.rating_kw)
    both = w.win_m & w.win_pv
    e = w.q_m_pct - w.q_pv_pct
    beta_e, _ = P1Plots.ols(w.loc[both, "v_avg_m"], e[both])
    fm = w.win_m
    hour = w.t_aest.dt.hour
    pv_off = fm & w.complete_pv & (w.p_pv.abs() < PV_OFF_FRAC * rating)
    dark = fm & ((hour < DAY_START) | (hour >= DAY_END))
    d = {
        "fitted intervals: net meter / PV / both": f"{int(fm.sum())} / {int(w.win_pv.sum())} / {int(both.sum())}",
        "beta_e (%/V, household Q vs V)": round(beta_e, 3) if np.isfinite(beta_e) else None,
        "net slope − PV slope (%/V)": round(float(case.slope_blind - case.slope_truth), 3),
        "sd(e) (% of rating)": round(float(e[both].std()), 2) if both.sum() > 1 else None,
        "net fitted intervals with PV off": f"{pv_off.sum() / max(fm.sum(), 1):.1%}",
        f"net fitted intervals outside {DAY_START}-{DAY_END} h": f"{dark.sum() / max(fm.sum(), 1):.1%}",
        "PV -> house pairing (mean |dV|)": "; ".join(
            f"{c} -> {ph['names'].get(h, h)} ({ph['dv'][c]:.2f} V)" for c, h in ph["pairing"].items()) or "none",
    }
    if details is not None and len(details):
        r = details[(details.site_id == case.site_id) & (details.year == case.year) & (details.month == case.month)]
        if len(r):
            r = r.iloc[0]
            for col, label in _DETAIL_COLS.items():
                if col in r.index and pd.notna(r[col]):
                    d[label] = round(float(r[col]), 3) if isinstance(r[col], (float, np.floating)) else r[col]
    return pd.Series(d, name="value")


def _colourbar(fig, ax, mappable, label: str):
    """Horizontal colour bar inside the top-right corner of `ax` (keeps the panel layout unchanged)."""
    cax = ax.inset_axes([0.55, 0.95, 0.42, 0.022])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=7)


def draw_explorer_3p(case: pd.Series, w: pd.DataFrame, ph: dict, ds: pd.DataFrame, day, *,
                     colour_by: str = "source", meta: pd.Series | None = None,
                     p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = V_LO, v_hi: float = V_HI):
    """The explorer figure for one site-month and one AEST day. Returns the figure."""
    C, BAND = P3Plots.COLORS, P3Plots.BAND
    rating = float(case.rating_kw)
    day = pd.Timestamp(day).normalize()
    grid = pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left")
    dd = w[w.day == day].drop_duplicates("t_aest").set_index("t_aest").reindex(grid)
    wm, wp = w[w.win_m], w[w.win_pv]
    both = w[w.win_m & w.win_pv]
    b_m, a_m = P1Plots.ols(wm.v_avg_m, wm.q_m_pct)
    b_p, a_p = P1Plots.ols(wp.v_avg_pv, wp.q_pv_pct)
    e_both = both.q_m_pct - both.q_pv_pct
    b_e, a_e = P1Plots.ols(both.v_avg_m, e_both)
    on = w.day == day

    fig = plt.figure(figsize=(17, 15), dpi=95)
    gs = GridSpec(4, 2, width_ratios=[1, 1.5], hspace=0.28, wspace=0.16, figure=fig)
    ax_qv, ax_ev, ax_ds = fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[3, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_q, ax_e, ax_v = (fig.add_subplot(gs[i, 1], sharex=ax_p) for i in (1, 2, 3))
    v_line = np.linspace(v_lo - 4, v_hi + 3, 100)

    # (a) Q vs V, whole month
    ax_qv.axvspan(v_lo, v_hi, color="#9ca3af", alpha=0.10, lw=0, label=f"Fit window [{v_lo}, {v_hi}) V")
    ax_qv.scatter(wp.v_avg_pv, wp.q_pv_pct, s=6, alpha=0.25, color=C["pv"] if colour_by in ("source", "PV off") else "#b8c2cc", lw=0,
                  label=f"PV circuits (total), fitted (n={len(wp):,})")
    lab_m = f"Net meter (total), fitted (n={len(wm):,})"
    if colour_by == "hour of day":
        sc = ax_qv.scatter(wm.v_avg_m, wm.q_m_pct, s=9, c=wm.t_aest.dt.hour + wm.t_aest.dt.minute / 60,
                           cmap="viridis", vmin=0, vmax=24, lw=0, alpha=0.85, label=lab_m)
        _colourbar(fig, ax_qv, sc, "Hour of day (AEST), net-meter points")
    elif colour_by == "PV output":
        sc = ax_qv.scatter(wm.v_avg_m, wm.q_m_pct, s=9, c=wm.p_pv_gen_pct.clip(0, 100), cmap="plasma",
                           vmin=0, vmax=100, lw=0, alpha=0.85, label=lab_m)
        _colourbar(fig, ax_qv, sc, "PV total P (% of rating), net-meter points")
    elif colour_by == "PV off":
        off = wm.complete_pv & (wm.p_pv.abs() < PV_OFF_FRAC * rating)
        ax_qv.scatter(wm.v_avg_m[~off], wm.q_m_pct[~off], s=7, alpha=0.35, color=C["meter"], lw=0,
                      label=f"Net meter, PV on (n={int((~off).sum()):,})")
        ax_qv.scatter(wm.v_avg_m[off], wm.q_m_pct[off], s=14, alpha=0.9, color="#c00000", lw=0,
                      label=f"Net meter, PV off < {PV_OFF_FRAC:.0%} (n={int(off.sum()):,})")
    else:
        ax_qv.scatter(wm.v_avg_m, wm.q_m_pct, s=7, alpha=0.3, color=C["meter"], lw=0, label=lab_m)
    ax_qv.scatter(w.loc[on & w.win_m, "v_avg_m"], w.loc[on & w.win_m, "q_m_pct"], s=26, facecolor="none",
                  edgecolor="black", lw=0.7, label=f"{day.date()} net meter (n={int((on & w.win_m).sum())})")
    ax_qv.scatter(w.loc[on & w.win_pv, "v_avg_pv"], w.loc[on & w.win_pv, "q_pv_pct"], s=26, marker="s",
                  facecolor="none", edgecolor="#6250d6", lw=0.7,
                  label=f"{day.date()} PV circuits (n={int((on & w.win_pv).sum())})")
    ax_qv.plot(v_line, P1Plots.required_q_pct(v_line), color=C["required"], lw=2.2, label="Required curve (AS/NZS 4777.2)")
    for b, a, c, lab in ((b_p, a_p, C["pv"], "PV"), (b_m, a_m, C["meter"], "Net meter")):
        if np.isfinite(b):
            ax_qv.plot(v_line, a + b * v_line, color=c, lw=2, ls="--", label=f"{lab} OLS: {b:.2f} %/V")
    ax_qv.axhline(0, color="black", lw=0.5)
    ax_qv.set(xlabel="Average phase voltage (V)", ylabel="Total Q (% of rating)\n+ supplying, − absorbing",
              xlim=(v_line[0], v_line[-1]),
              title=f"(a) Whole month, coloured by {colour_by}. Pass band [{BAND[0]:.3f}, {BAND[1]:.3f}] %/V")
    ax_qv.legend(fontsize=7, loc="lower left")

    # (b) e vs V
    ax_ev.scatter(both.v_avg_m, e_both, s=6, alpha=0.3, color="#6b7280", lw=0,
                  label=f"Both fit windows (n={len(both):,})")
    sel = on & w.win_m & w.win_pv
    ax_ev.scatter(w.loc[sel, "v_avg_m"], (w.q_m_pct - w.q_pv_pct)[sel], s=24, facecolor="none", edgecolor="black",
                  lw=0.7, label=f"{day.date()}")
    if np.isfinite(b_e):
        ax_ev.plot(v_line, a_e + b_e * v_line, color="#c00000", lw=2, label=f"OLS: beta_e = {b_e:.2f} %/V")
    ax_ev.axhline(0, color="black", lw=0.5)
    ax_ev.axvspan(v_lo, v_hi, color="#9ca3af", alpha=0.10, lw=0)
    ax_ev.set(xlabel="Average phase voltage, net meter (V)", ylabel="e = Q_net − Q_PV\n(% of rating)",
              xlim=(v_line[0], v_line[-1]),
              title=f"(b) Household reactive power. Net − PV slope = {case.slope_blind - case.slope_truth:.2f} %/V")
    ax_ev.legend(fontsize=7, loc="lower left")

    # (c) daily slopes
    ax_ds.axhspan(BAND[0], BAND[1], color=C["required"], alpha=0.10, lw=0, label="Pass band")
    lim = 10.0                                              # daily slopes shown clipped at +-10 %/V
    ax_ds.plot(ds.day + pd.Timedelta(hours=12), ds.slope_m.clip(-lim, lim), "o-", ms=4, lw=0.8, color=C["meter"],
               label="Net meter, daily slope")
    ax_ds.plot(ds.day + pd.Timedelta(hours=12), ds.slope_pv.clip(-lim, lim), "s-", ms=4, lw=0.8, color=C["pv"],
               label="PV circuits, daily slope")
    for b, c in ((b_m, C["meter"]), (b_p, C["pv"])):
        if np.isfinite(b):
            ax_ds.axhline(b, color=c, ls="--", lw=1)
    ax_ds.axvline(day + pd.Timedelta(hours=12), color="black", lw=1.2, alpha=0.6)
    vals = pd.concat([ds.slope_m, ds.slope_pv]).dropna().clip(-lim, lim)
    ax_ds.set_ylim(min(BAND[0] - 1, vals.min() - 0.5 if len(vals) else -5),
                   max(1.0, vals.max() + 0.5 if len(vals) else 1.0))
    ax_ds.axhline(0, color="black", lw=0.5)
    if len(ds):
        ax_ds.set_xlim(ds.day.min(), ds.day.max() + pd.Timedelta(days=1))
    ax_ds.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(ds) // 10)))
    ax_ds.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax_ds.set(xlabel=f"Day of month (dashed: monthly slope; daily slope needs >= {MIN_DAY_POINTS} points over "
                     f">= {MIN_DAY_SPREAD:g} V)", ylabel="Slope (%/V)", title=f"(c) Daily slopes (shown clipped at ±{lim:g} %/V)")
    ax_ds.legend(fontsize=7, loc="lower left", ncol=3)

    # Right: the selected day, per phase and total
    t_ = dd.index
    shade = dd.win_m.astype("boolean").fillna(False).to_numpy(bool)
    tt = ph["t"]
    day_slice = lambda df: df.reindex(t_) if len(df.columns) else df
    hp, hq, hv = day_slice(tt["m"]["p"]), day_slice(tt["m"]["q"]), day_slice(tt["m"]["v"])
    pp, pq = day_slice(tt["pv"]["p"]), day_slice(tt["pv"]["q"])
    ee = ph["e"].reindex(t_)
    for ax in (ax_p, ax_q, ax_e, ax_v):
        ax.fill_between(t_, 0, 1, where=shade, transform=ax.get_xaxis_transform(), color=C["meter"],
                        alpha=0.07, lw=0, step="mid")
    for h in ph["house"]:
        col, nm = ph["colors"][h], ph["names"][h]
        ax_p.plot(t_, hp[h], color=col, lw=0.9, label=f"{nm} net P_est")
        ax_q.plot(t_, hq[h], color=col, lw=0.9, label=f"{nm} net Q_est")
        ax_e.plot(t_, ee[h], color=col, lw=0.9, label=f"{nm} e")
        ax_v.plot(t_, hv[h], color=col, lw=0.8, label=f"{nm} V ({h})")
    for c in ph["pvs"]:
        h = ph["pairing"].get(c)
        col = ph["colors"].get(h, "#6b7280")
        ax_p.plot(t_, pp[c], color=col, lw=0.9, ls="--", label=f"PV {c} -> {ph['names'].get(h, '?')}")
        ax_q.plot(t_, pq[c], color=col, lw=0.9, ls="--", label=f"PV {c}")
    ax_p.plot(t_, dd.p_m_pct, color=C["meter"], lw=1.8, label="Net total")
    ax_p.plot(t_, dd.p_pv_gen_pct, color=C["pv"], lw=1.8, label="PV total")
    ax_p.axhline(100 * p_min_frac, color="black", ls=":", lw=0.9, label=f"Gate {100 * p_min_frac:.0f}%")
    ax_q.plot(t_, dd.q_m_pct, color=C["meter"], lw=1.8, label="Net total")
    ax_q.plot(t_, dd.q_pv_pct, color=C["pv"], lw=1.8, label="PV total")
    ax_q.plot(t_, dd.q_req_pv_pct, color=C["required"], lw=1.6, label="Required at PV V_avg")
    ax_e.plot(t_, dd.q_m_pct - dd.q_pv_pct, color="black", lw=1.6, label="Total e")
    ax_v.plot(t_, dd.v_avg_pv, color=C["pv"], lw=1.6, label="PV average V")
    for ax in (ax_q, ax_e):
        ax.axhline(0, color="black", lw=0.5)
    for v, ls in ((240, ":"), (v_lo, "-"), (v_hi, "-"), (258, ":")):
        ax_v.axhline(v, color="#6b7280", ls=ls, lw=0.7)
    ax_p.set(ylabel="P (% of rating)", title=f"{day.date()} (AEST): per phase (thin; PV dashed) and total (thick). "
                                             f"Shaded: net-meter fit window")
    ax_q.set_ylabel("Q (% of rating)")
    ax_e.set_ylabel("e = Q_net − Q_PV\n(% of rating)")
    ax_v.set_ylabel("Voltage (V)")
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in (ax_p, ax_q, ax_e, ax_v):
        ax.legend(fontsize=6.5, loc="upper left", ncol=4)
    for ax in (ax_p, ax_q, ax_e):
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (ax_qv, ax_ev, ax_ds, ax_p, ax_q, ax_e, ax_v):
        ax.grid(color="#ebebeb", lw=0.5)

    who = "" if meta is None else f" | {meta['manufacturer']} {meta['model']}, {meta['monitoring device']}"
    fig.suptitle(f"Site {case.site_id}, {int(case.year)}-{int(case.month):02d} ({case.get('outcome')}): net meter "
                 f"{case.slope_blind:.2f} %/V ({'pass' if case.pass_blind else 'fail'}) vs PV circuits "
                 f"{case.slope_truth:.2f} %/V ({'pass' if case.pass_truth else 'fail'}); rating {rating:g} kW{who}",
                 fontsize=11, fontweight="bold", y=0.995)
    return fig


def explore_3p(main: pd.DataFrame, ev: pd.DataFrame | None = None, details: pd.DataFrame | None = None, *,
               outcome: str = "FN", how: str = "worst", seed: int = 42, night_hours=NIGHT_HOURS,
               p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = V_LO, v_hi: float = V_HI) -> dict:
    """
    Interactive explorer for notebook 12's oracle table `main` (three-phase inverters): pick the outcome
    (FN/FP/TP/TN) and the order (worst = largest |slope error|, borderline, random), step through site-months and
    days, and colour the month's net-meter points by source / hour of day / PV output / PV off. `ev` (notebook 10
    site evidence) and `details` (notebook 12 details file; loaded from the store if not given) add metadata.
    Returns a dict cache of the loaded site-months (key = (site_id, year, month)).
    """
    details = _details_3p() if details is None else details
    cache: dict = {}
    state = {"busy": False, "cases": pd.DataFrame()}

    w_outcome = widgets.Dropdown(options=["FN", "FP", "TP", "TN"], value=outcome, description="Outcome",
                                 layout=widgets.Layout(width="170px"))
    w_how = widgets.Dropdown(options=["worst", "borderline", "random"], value=how, description="Order",
                             layout=widgets.Layout(width="200px"))
    w_case = widgets.Dropdown(description="Case", layout=widgets.Layout(width="560px"))
    w_case_prev = widgets.Button(description="◀ case", layout=widgets.Layout(width="75px"))
    w_case_next = widgets.Button(description="case ▶", layout=widgets.Layout(width="75px"))
    w_day = widgets.Dropdown(description="Day", layout=widgets.Layout(width="420px"))
    w_day_prev = widgets.Button(description="◀ day", layout=widgets.Layout(width="75px"))
    w_day_next = widgets.Button(description="day ▶", layout=widgets.Layout(width="75px"))
    w_colour = widgets.ToggleButtons(options=COLOUR_MODES, value="source", description="Colour",
                                     style={"button_width": "110px"})
    out_meta, out_fig = widgets.Output(), widgets.Output()

    def load(case):
        key = (int(case.site_id), int(case.year), int(case.month))
        if key not in cache:
            w, long = P3Plots.fetch_site_month(case.site_id, case.year, case.month, float(case.rating_kw),
                                               night_hours=night_hours, p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
            ph = phase_tables(long, float(case.rating_kw))
            cache[key] = dict(w=w, long=long, ph=ph, ds=daily_slopes(w), meta=case_metadata(case, ev))
            cache[key]["diag"] = month_diagnostics(case, w, ph, details)
        return cache[key]

    def current_case():
        return state["cases"].iloc[w_case.value] if len(state["cases"]) and w_case.value is not None else None

    def render(*_):
        if state["busy"]:
            return
        case = current_case()
        with out_fig:
            clear_output(wait=True)
            if case is None or w_day.value is None:
                print("No site-months for this selection.")
                return
            c = load(case)
            draw_explorer_3p(case, c["w"], c["ph"], c["ds"], w_day.value, colour_by=w_colour.value,
                             meta=c["meta"], p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
            plt.show()

    def show_case(*_):
        if state["busy"]:
            return
        case = current_case()
        with out_meta:
            clear_output(wait=True)
        if case is None:
            render()
            return
        with out_fig:
            clear_output(wait=True)
            print(f"Loading site {case.site_id}, {int(case.year)}-{int(case.month):02d} ...")
        c = load(case)
        with out_meta:
            display(widgets.HBox([widgets.HTML(c["meta"].to_frame().to_html()),
                                  widgets.HTML(c["diag"].to_frame("diagnostic").to_html())]))
        w = c["w"]
        per_day = w.groupby("day").agg(n_m=("win_m", "sum"), n_pv=("win_pv", "sum"))
        both = w[w.win_m & w.win_pv]
        start = (both if len(both) else (w[w.win_m] if w.win_m.any() else w)).day.value_counts().idxmax()
        state["busy"] = True
        w_day.options = [(f"{x:%Y-%m-%d} ({x:%a})  fitted: net {int(r.n_m):3d} | PV {int(r.n_pv):3d}", x)
                         for x, r in per_day.iterrows()]
        w_day.value = start
        state["busy"] = False
        render()

    def refresh_cases(*_):
        d = P3Plots.select_cases(main, outcome=w_outcome.value, n=None, how=w_how.value, seed=seed)
        if w_how.value == "random":
            d = d.sample(frac=1, random_state=seed).reset_index(drop=True)
        state["cases"] = d
        state["busy"] = True
        w_case.options = [(f"{i + 1}/{len(d)}  site {r.site_id}  {int(r.year)}-{int(r.month):02d}  net "
                           f"{r.slope_blind:.2f} vs PV {r.slope_truth:.2f} %/V ({r.meter_side})", i)
                          for i, r in d.iterrows()]
        w_case.value = 0 if len(d) else None
        state["busy"] = False
        show_case()

    def step(widget, k):
        vals = [v for _, v in widget.options]
        if widget.value in vals:
            i = vals.index(widget.value) + k
            if 0 <= i < len(vals):
                widget.value = vals[i]

    w_outcome.observe(refresh_cases, names="value")
    w_how.observe(refresh_cases, names="value")
    w_case.observe(show_case, names="value")
    w_day.observe(render, names="value")
    w_colour.observe(render, names="value")
    w_case_prev.on_click(lambda _: step(w_case, -1))
    w_case_next.on_click(lambda _: step(w_case, 1))
    w_day_prev.on_click(lambda _: step(w_day, -1))
    w_day_next.on_click(lambda _: step(w_day, 1))

    display(widgets.VBox([
        widgets.HBox([w_outcome, w_how, w_case_prev, w_case, w_case_next]),
        widgets.HBox([w_day_prev, w_day, w_day_next]),
        w_colour, out_meta, out_fig]))
    refresh_cases()
    return cache


# =========================================================================================
# Single-phase explorer (notebook 11, 3L1P): same layout as explore_3p, for the tested house phase
# =========================================================================================
# For one site-month: (a) Q vs V of the tested phase (`ref_phase`) and the PV circuit, coloured by the chosen mode;
# (b) e = tested-phase Q - PV Q vs V and its slope beta_e (as notebook 11 section 9); (c) daily slopes; right: the
# selected day's P, Q, e and V for the tested phase and the PV circuit, with the other two house phases shown faintly.

DETAILS_1P = ("vvar_phase", "3l1p_vvar", "site_month_details_3l1p.parquet")
_DETAIL_COLS_1P = {"phase_status": "blind phase vs reference (end-to-end)", "pv_q90_over_rating": "PV midday q90 / rating",
                   "affected_year": "missing-circuit year (0 = none)", "model_hint": "model hint",
                   "exclusion_reason": "nb10 exclusion"}


@lru_cache(maxsize=1)
def _details_1p() -> pd.DataFrame:
    p = Fleet.store_dir().joinpath(*DETAILS_1P)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def house_phases_1p(site_id: int, year: int, month: int, rating_kw: float, *, night_hours=NIGHT_HOURS,
                    v_plausible=(150.0, 300.0)) -> dict:
    """Net-meter estimate for every house phase of the site-month (index = AEST time, columns = circuit_id):
    p, q (% of rating) and V (implausible readings set to NaN)."""
    ym = f"{int(year)}-{int(month):02d}"
    con = Fleet.connect("2GB", 2)
    try:
        con.register("_sites", pd.DataFrame({"site_id": [int(site_id)], "rating_kw": [float(rating_kw)]}))
        x = con.execute(f"""
            WITH {Fleet._estimate_sql(ym, "meter_baseline", night_hours)}
            SELECT e.circuit_id, e.t_stamp, e.V, e.p_est, e.q_est FROM est e JOIN _sites s USING (site_id)
        """).df()
    finally:
        con.close()
    t = pd.to_datetime(x.t_stamp)
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    x["t_aest"] = t + pd.Timedelta(hours=10)
    x["V"] = x.V.where(x.V.between(*v_plausible))
    x = x.drop_duplicates(["circuit_id", "t_aest"])
    return {"p": 100 * x.pivot(index="t_aest", columns="circuit_id", values="p_est") / rating_kw,
            "q": 100 * x.pivot(index="t_aest", columns="circuit_id", values="q_est") / rating_kw,
            "v": x.pivot(index="t_aest", columns="circuit_id", values="V")}


def _daily_slopes(d: pd.DataFrame, specs, min_points: int = MIN_DAY_POINTS, min_spread: float = MIN_DAY_SPREAD) -> pd.DataFrame:
    """Daily OLS slopes; `specs` = [(tag, v_col, q_col, window_col), ...]."""
    rows = []
    for day, g in d.groupby("day"):
        r = {"day": day}
        for tag, v, q, win in specs:
            x = g[g[win]]
            ok = len(x) >= min_points and (x[v].max() - x[v].min()) >= min_spread
            r[f"slope_{tag}"] = P1Plots.ols(x[v], x[q])[0] if ok else np.nan
            r[f"n_{tag}"] = len(x)
        rows.append(r)
    return pd.DataFrame(rows)


def month_diagnostics_1p(case: pd.Series, d: pd.DataFrame, house: dict, details: pd.DataFrame | None = None) -> pd.Series:
    """Diagnostics for a 3L1P site-month: beta_e and sd(e) as `ami_phase1p_vvar.beta_e_sql`, PV-off and outside-daylight
    shares as `outside_daylight_sql`, the PV circuit's mean |dV| to each house phase, and notebook 11's details."""
    rating = float(case.rating_kw_oracle)
    both = d.win_m & d.win_pv
    e = d.q_m_pct - d.q_pv_pct
    beta_e, _ = P1Plots.ols(d.loc[both, "V_m"], e[both])
    fm = d.win_m
    hour = d.t_aest.dt.hour
    pv_off = fm & d.p_pv.notna() & (d.p_pv.abs() < PV_OFF_FRAC * rating)
    dark = fm & ((hour < DAY_START) | (hour >= DAY_END))
    vpv = d.set_index("t_aest").V_pv
    tested = int(case.ref_phase)
    gaps = []
    for i, c in enumerate(sorted(house["v"].columns)):
        g = (vpv - house["v"][c].reindex(vpv.index)).abs().mean()
        gaps.append(f"L{i + 1}{'*' if c == tested else ''} {g:.2f} V" if pd.notna(g) else f"L{i + 1} n/a")
    out = {
        "fitted intervals: net meter / PV / both": f"{int(fm.sum())} / {int(d.win_pv.sum())} / {int(both.sum())}",
        "beta_e (%/V, household Q vs V)": round(beta_e, 3) if np.isfinite(beta_e) else None,
        "net slope − PV slope (%/V)": round(float(case.slope_blind - case.slope_truth), 3),
        "sd(e) (% of rating)": round(float(e[both].std()), 2) if both.sum() > 1 else None,
        "net fitted intervals with PV off": f"{pv_off.sum() / max(fm.sum(), 1):.1%}",
        f"net fitted intervals outside {DAY_START}-{DAY_END} h": f"{dark.sum() / max(fm.sum(), 1):.1%}",
        "PV circuit mean |dV| to each house phase (* = tested)": " | ".join(gaps),
    }
    if details is not None and len(details):
        r = details[(details.site_id == case.site_id) & (details.year == case.year) & (details.month == case.month)]
        if len(r):
            r = r.iloc[0]
            for col, label in _DETAIL_COLS_1P.items():
                if col in r.index and pd.notna(r[col]):
                    out[label] = round(float(r[col]), 3) if isinstance(r[col], (float, np.floating)) else r[col]
    return pd.Series(out, name="value")


def draw_explorer_1p(case: pd.Series, d: pd.DataFrame, house: dict, ds: pd.DataFrame, day, *,
                     colour_by: str = "source", meta: pd.Series | None = None,
                     p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = V_LO, v_hi: float = V_HI):
    """The explorer figure for one 3L1P site-month and one AEST day. Returns the figure."""
    C, VV, BAND = P1Plots.COLORS, P1Plots._VV, P1Plots.BAND
    rating = float(case.rating_kw_oracle)
    tested = int(case.ref_phase)
    day = pd.Timestamp(day).normalize()
    grid = pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left")
    dd = d[d.day == day].drop_duplicates("t_aest").set_index("t_aest").reindex(grid)
    wm, wp = d[d.win_m], d[d.win_pv]
    both = d[d.win_m & d.win_pv]
    b_m, a_m = P1Plots.ols(wm.V_m, wm.q_m_pct)
    b_p, a_p = P1Plots.ols(wp.V_pv, wp.q_pv_pct)
    e_both = both.q_m_pct - both.q_pv_pct
    b_e, a_e = P1Plots.ols(both.V_m, e_both)
    on = d.day == day
    phases = sorted(house["v"].columns)
    names = {c: f"L{i + 1}" for i, c in enumerate(phases)}

    fig = plt.figure(figsize=(17, 15), dpi=95)
    gs = GridSpec(4, 2, width_ratios=[1, 1.5], hspace=0.28, wspace=0.16, figure=fig)
    ax_qv, ax_ev, ax_ds = fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[3, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_q, ax_e, ax_v = (fig.add_subplot(gs[i, 1], sharex=ax_p) for i in (1, 2, 3))
    v_line = np.linspace(v_lo - 4, v_hi + 3, 100)

    # (a) Q vs V, whole month
    ax_qv.axvspan(v_lo, v_hi, color="#9ca3af", alpha=0.10, lw=0, label=f"Fit window [{v_lo}, {v_hi}) V")
    ax_qv.scatter(wp.V_pv, wp.q_pv_pct, s=6, alpha=0.25, lw=0, label=f"PV circuit, fitted (n={len(wp):,})",
                  color=C["pv"] if colour_by in ("source", "PV off") else "#b8c2cc")
    lab_m = f"Net meter, tested phase {names.get(tested, tested)}, fitted (n={len(wm):,})"
    if colour_by == "hour of day":
        sc = ax_qv.scatter(wm.V_m, wm.q_m_pct, s=9, c=wm.t_aest.dt.hour + wm.t_aest.dt.minute / 60, cmap="viridis",
                           vmin=0, vmax=24, lw=0, alpha=0.85, label=lab_m)
        _colourbar(fig, ax_qv, sc, "Hour of day (AEST), net-meter points")
    elif colour_by == "PV output":
        sc = ax_qv.scatter(wm.V_m, wm.q_m_pct, s=9, c=wm.p_pv_pct.clip(0, 100), cmap="plasma", vmin=0, vmax=100,
                           lw=0, alpha=0.85, label=lab_m)
        _colourbar(fig, ax_qv, sc, "PV P (% of rating), net-meter points")
    elif colour_by == "PV off":
        off = wm.p_pv.notna() & (wm.p_pv.abs() < PV_OFF_FRAC * rating)
        ax_qv.scatter(wm.V_m[~off], wm.q_m_pct[~off], s=7, alpha=0.35, color=C["meter"], lw=0,
                      label=f"Net meter, PV on (n={int((~off).sum()):,})")
        ax_qv.scatter(wm.V_m[off], wm.q_m_pct[off], s=14, alpha=0.9, color="#c00000", lw=0,
                      label=f"Net meter, PV off < {PV_OFF_FRAC:.0%} (n={int(off.sum()):,})")
    else:
        ax_qv.scatter(wm.V_m, wm.q_m_pct, s=7, alpha=0.3, color=C["meter"], lw=0, label=lab_m)
    ax_qv.scatter(d.loc[on & d.win_m, "V_m"], d.loc[on & d.win_m, "q_m_pct"], s=26, facecolor="none",
                  edgecolor="black", lw=0.7, label=f"{day.date()} net meter (n={int((on & d.win_m).sum())})")
    ax_qv.scatter(d.loc[on & d.win_pv, "V_pv"], d.loc[on & d.win_pv, "q_pv_pct"], s=26, marker="s",
                  facecolor="none", edgecolor="#6250d6", lw=0.7,
                  label=f"{day.date()} PV circuit (n={int((on & d.win_pv).sum())})")
    ax_qv.plot(v_line, P1Plots.required_q_pct(v_line), color=C["required"], lw=2.2, label="Required curve (AS/NZS 4777.2)")
    for b, a, c, lab in ((b_p, a_p, C["pv"], "PV"), (b_m, a_m, C["meter"], "Net meter")):
        if np.isfinite(b):
            ax_qv.plot(v_line, a + b * v_line, color=c, lw=2, ls="--", label=f"{lab} OLS: {b:.2f} %/V")
    ax_qv.axhline(0, color="black", lw=0.5)
    ax_qv.set(xlabel="Voltage (V)", ylabel="Q (% of rating)\n+ supplying, − absorbing", xlim=(v_line[0], v_line[-1]),
              title=f"(a) Whole month, coloured by {colour_by}. Pass band [{BAND[0]:.3f}, {BAND[1]:.3f}] %/V")
    ax_qv.legend(fontsize=7, loc="lower left")

    # (b) e vs V
    ax_ev.scatter(both.V_m, e_both, s=6, alpha=0.3, color="#6b7280", lw=0, label=f"Both fit windows (n={len(both):,})")
    sel = on & d.win_m & d.win_pv
    ax_ev.scatter(d.loc[sel, "V_m"], (d.q_m_pct - d.q_pv_pct)[sel], s=24, facecolor="none", edgecolor="black",
                  lw=0.7, label=f"{day.date()}")
    if np.isfinite(b_e):
        ax_ev.plot(v_line, a_e + b_e * v_line, color="#c00000", lw=2, label=f"OLS: beta_e = {b_e:.2f} %/V")
    ax_ev.axhline(0, color="black", lw=0.5)
    ax_ev.axvspan(v_lo, v_hi, color="#9ca3af", alpha=0.10, lw=0)
    ax_ev.set(xlabel="Voltage, tested phase (V)", ylabel="e = Q_net − Q_PV\n(% of rating)", xlim=(v_line[0], v_line[-1]),
              title=f"(b) Household reactive power on the tested phase. Net − PV slope = "
                    f"{case.slope_blind - case.slope_truth:.2f} %/V")
    ax_ev.legend(fontsize=7, loc="lower left")

    # (c) daily slopes
    lim = 10.0
    ax_ds.axhspan(BAND[0], BAND[1], color=C["required"], alpha=0.10, lw=0, label="Pass band")
    ax_ds.plot(ds.day + pd.Timedelta(hours=12), ds.slope_m.clip(-lim, lim), "o-", ms=4, lw=0.8, color=C["meter"],
               label="Net meter, daily slope")
    ax_ds.plot(ds.day + pd.Timedelta(hours=12), ds.slope_pv.clip(-lim, lim), "s-", ms=4, lw=0.8, color=C["pv"],
               label="PV circuit, daily slope")
    for b, c in ((b_m, C["meter"]), (b_p, C["pv"])):
        if np.isfinite(b):
            ax_ds.axhline(b, color=c, ls="--", lw=1)
    ax_ds.axvline(day + pd.Timedelta(hours=12), color="black", lw=1.2, alpha=0.6)
    vals = pd.concat([ds.slope_m, ds.slope_pv]).dropna().clip(-lim, lim)
    ax_ds.set_ylim(min(BAND[0] - 1, vals.min() - 0.5 if len(vals) else -5), max(1.0, vals.max() + 0.5 if len(vals) else 1.0))
    ax_ds.axhline(0, color="black", lw=0.5)
    if len(ds):
        ax_ds.set_xlim(ds.day.min(), ds.day.max() + pd.Timedelta(days=1))
    ax_ds.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(ds) // 10)))
    ax_ds.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax_ds.set(xlabel=f"Day of month (dashed: monthly slope; daily slope needs >= {MIN_DAY_POINTS} points over "
                     f">= {MIN_DAY_SPREAD:g} V)", ylabel="Slope (%/V)", title=f"(c) Daily slopes (shown clipped at ±{lim:g} %/V)")
    ax_ds.legend(fontsize=7, loc="lower left", ncol=3)

    # Right: the selected day. Tested phase and PV thick; the other two house phases faint.
    t_ = dd.index
    shade = dd.win_m.astype("boolean").fillna(False).to_numpy(bool)
    hp, hq, hv = (house[k].reindex(t_) for k in ("p", "q", "v"))
    for ax in (ax_p, ax_q, ax_e, ax_v):
        ax.fill_between(t_, 0, 1, where=shade, transform=ax.get_xaxis_transform(), color=C["meter"],
                        alpha=0.07, lw=0, step="mid")
    for i, c in enumerate(phases):
        if c == tested:
            continue
        col = PHASE_PALETTE[i % 3]
        ax_p.plot(t_, hp[c], color=col, lw=0.8, alpha=0.6, label=f"{names[c]} P_est (other phase)")
        ax_q.plot(t_, hq[c], color=col, lw=0.8, alpha=0.6, label=f"{names[c]} Q_est (other phase)")
        ax_v.plot(t_, hv[c], color=col, lw=0.8, alpha=0.7, label=f"{names[c]} V ({c})")
    tn = names.get(tested, tested)
    ax_p.plot(t_, dd.p_m_pct, color=C["meter"], lw=1.8, label=f"{tn} P_est (tested)")
    ax_p.plot(t_, dd.p_pv_pct, color=C["pv"], lw=1.8, label="PV circuit P")
    ax_p.axhline(100 * p_min_frac, color="black", ls=":", lw=0.9, label=f"Gate {100 * p_min_frac:.0f}%")
    ax_q.plot(t_, dd.q_m_pct, color=C["meter"], lw=1.8, label=f"{tn} Q_est (tested)")
    ax_q.plot(t_, dd.q_pv_pct, color=C["pv"], lw=1.8, label="PV circuit Q")
    ax_q.plot(t_, dd.q_req_pv_pct, color=C["required"], lw=1.6, label="Required at PV V")
    ax_e.plot(t_, dd.q_m_pct - dd.q_pv_pct, color="black", lw=1.6, label=f"e on {tn}")
    ax_v.plot(t_, dd.V_m, color=C["meter"], lw=1.4, ls="--", label=f"{tn} V (tested, {tested})")
    ax_v.plot(t_, dd.V_pv, color=C["pv"], lw=1.4, label="PV circuit V")
    for ax in (ax_q, ax_e):
        ax.axhline(0, color="black", lw=0.5)
    for v, ls in ((VV["V3"], ":"), (v_lo, "-"), (v_hi, "-"), (VV["V4"], ":")):
        ax_v.axhline(v, color="#6b7280", ls=ls, lw=0.7)
    ax_p.set(ylabel="P (% of rating)", title=f"{day.date()} (AEST): tested phase and PV (thick), other phases (faint). "
                                             f"Shaded: net-meter fit window")
    ax_q.set_ylabel("Q (% of rating)")
    ax_e.set_ylabel("e = Q_net − Q_PV\n(% of rating)")
    ax_v.set_ylabel("Voltage (V)")
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in (ax_p, ax_q, ax_e, ax_v):
        ax.legend(fontsize=6.5, loc="upper left", ncol=4)
    for ax in (ax_p, ax_q, ax_e):
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (ax_qv, ax_ev, ax_ds, ax_p, ax_q, ax_e, ax_v):
        ax.grid(color="#ebebeb", lw=0.5)

    who = "" if meta is None else f" | {meta['manufacturer']} {meta['model']}, {meta['monitoring device']}"
    fig.suptitle(f"Site {case.site_id}, {int(case.year)}-{int(case.month):02d} ({case.get('outcome')}): net meter "
                 f"{case.slope_blind:.2f} %/V ({'pass' if case.pass_blind else 'fail'}) vs PV circuit "
                 f"{case.slope_truth:.2f} %/V ({'pass' if case.pass_truth else 'fail'}); rating {rating:g} kW, "
                 f"tested phase {tested}{who}", fontsize=11, fontweight="bold", y=0.995)
    return fig


def _explorer_widget(main: pd.DataFrame, select, load, draw, *, outcome: str, how: str, seed: int) -> dict:
    """Shared widget: outcome / order / case / day / colour controls, metadata + diagnostics tables, figure.
    `select(main, outcome, how)` -> cases; `load(case)` -> dict(d, meta, diag, ...); `draw(case, c, day, colour)`."""
    cache: dict = {}
    state = {"busy": False, "cases": pd.DataFrame()}
    w_outcome = widgets.Dropdown(options=["FN", "FP", "TP", "TN"], value=outcome, description="Outcome",
                                 layout=widgets.Layout(width="170px"))
    w_how = widgets.Dropdown(options=["worst", "borderline", "random"], value=how, description="Order",
                             layout=widgets.Layout(width="200px"))
    w_case = widgets.Dropdown(description="Case", layout=widgets.Layout(width="600px"))
    w_case_prev = widgets.Button(description="◀ case", layout=widgets.Layout(width="75px"))
    w_case_next = widgets.Button(description="case ▶", layout=widgets.Layout(width="75px"))
    w_day = widgets.Dropdown(description="Day", layout=widgets.Layout(width="420px"))
    w_day_prev = widgets.Button(description="◀ day", layout=widgets.Layout(width="75px"))
    w_day_next = widgets.Button(description="day ▶", layout=widgets.Layout(width="75px"))
    w_colour = widgets.ToggleButtons(options=COLOUR_MODES, value="source", description="Colour",
                                     style={"button_width": "110px"})
    out_meta, out_fig = widgets.Output(), widgets.Output()

    def get(case):
        key = (int(case.site_id), int(case.year), int(case.month))
        if key not in cache:
            cache[key] = load(case)
        return cache[key]

    def current_case():
        return state["cases"].iloc[w_case.value] if len(state["cases"]) and w_case.value is not None else None

    def render(*_):
        if state["busy"]:
            return
        case = current_case()
        with out_fig:
            clear_output(wait=True)
            if case is None or w_day.value is None:
                print("No site-months for this selection.")
                return
            draw(case, get(case), w_day.value, w_colour.value)
            plt.show()

    def show_case(*_):
        if state["busy"]:
            return
        case = current_case()
        with out_meta:
            clear_output(wait=True)
        if case is None:
            render()
            return
        with out_fig:
            clear_output(wait=True)
            print(f"Loading site {case.site_id}, {int(case.year)}-{int(case.month):02d} ...")
        c = get(case)
        with out_meta:
            display(widgets.HBox([widgets.HTML(c["meta"].to_frame().to_html()),
                                  widgets.HTML(c["diag"].to_frame("diagnostic").to_html())]))
        d = c["d"]
        per_day = d.groupby("day").agg(n_m=("win_m", "sum"), n_pv=("win_pv", "sum"))
        both = d[d.win_m & d.win_pv]
        start = (both if len(both) else (d[d.win_m] if d.win_m.any() else d)).day.value_counts().idxmax()
        state["busy"] = True
        w_day.options = [(f"{x:%Y-%m-%d} ({x:%a})  fitted: net {int(r.n_m):3d} | PV {int(r.n_pv):3d}", x)
                         for x, r in per_day.iterrows()]
        w_day.value = start
        state["busy"] = False
        render()

    def refresh_cases(*_):
        d = select(main, w_outcome.value, w_how.value)
        if w_how.value == "random":
            d = d.sample(frac=1, random_state=seed).reset_index(drop=True)
        state["cases"] = d
        state["busy"] = True
        w_case.options = [(f"{i + 1}/{len(d)}  site {r.site_id}  {int(r.year)}-{int(r.month):02d}  net "
                           f"{r.slope_blind:.2f} vs PV {r.slope_truth:.2f} %/V ({r.meter_side})", i)
                          for i, r in d.iterrows()]
        w_case.value = 0 if len(d) else None
        state["busy"] = False
        show_case()

    def step(widget, k):
        vals = [v for _, v in widget.options]
        if widget.value in vals:
            i = vals.index(widget.value) + k
            if 0 <= i < len(vals):
                widget.value = vals[i]

    w_outcome.observe(refresh_cases, names="value")
    w_how.observe(refresh_cases, names="value")
    w_case.observe(show_case, names="value")
    w_day.observe(render, names="value")
    w_colour.observe(render, names="value")
    w_case_prev.on_click(lambda _: step(w_case, -1))
    w_case_next.on_click(lambda _: step(w_case, 1))
    w_day_prev.on_click(lambda _: step(w_day, -1))
    w_day_next.on_click(lambda _: step(w_day, 1))
    display(widgets.VBox([widgets.HBox([w_outcome, w_how, w_case_prev, w_case, w_case_next]),
                          widgets.HBox([w_day_prev, w_day, w_day_next]), w_colour, out_meta, out_fig]))
    refresh_cases()
    return cache


def explore_1p(main: pd.DataFrame, ev: pd.DataFrame | None = None, details: pd.DataFrame | None = None, *,
               outcome: str = "FN", how: str = "worst", seed: int = 42, night_hours=NIGHT_HOURS,
               p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = V_LO, v_hi: float = V_HI) -> dict:
    """
    Interactive explorer for notebook 11's oracle table `main` (3L1P, net meter tested on `ref_phase`): pick the
    outcome and order, step through site-months and days, colour the month's points. `ev` (notebook 10 site
    evidence) and `details` (notebook 11 details file; loaded from the store if not given) add metadata.
    Returns a dict cache of the loaded site-months.
    """
    details = _details_1p() if details is None else details

    def select(m, oc, hw):
        return P1Plots.select_cases(m, outcome=oc, n=None, how=hw, seed=seed)

    def load(case):
        rating = float(case.rating_kw_oracle)
        d = P1Plots.fetch_site_month(case.site_id, case.year, case.month, case.ref_phase, rating,
                                     night_hours=night_hours, p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)
        house = house_phases_1p(case.site_id, case.year, case.month, rating, night_hours=night_hours)
        ds = _daily_slopes(d, [("m", "V_m", "q_m_pct", "win_m"), ("pv", "V_pv", "q_pv_pct", "win_pv")])
        return dict(d=d, house=house, ds=ds, meta=case_metadata(case, ev),
                    diag=month_diagnostics_1p(case, d, house, details))

    def draw(case, c, day, colour):
        draw_explorer_1p(case, c["d"], c["house"], c["ds"], day, colour_by=colour, meta=c["meta"],
                         p_min_frac=p_min_frac, v_lo=v_lo, v_hi=v_hi)

    return _explorer_widget(main, select, load, draw, outcome=outcome, how=how, seed=seed)
