"""Support module for notebooks/02c_explore_voltvar_voltwatt.ipynb.

NOTE ON THE FILE NAME: the notebook this supports is `02c_explore_voltvar_
voltwatt.ipynb`. This module is deliberately NOT named `02c_explore_voltvar_
voltwatt.py`, even though that was the suggested name, because Python module
names cannot start with a digit -- `import 02c_explore_voltvar_voltwatt`
is a syntax error. This file lives next to the notebook and is imported as
`voltvar_voltwatt_lib`.

WHAT THIS PORTS, AND FROM WHERE
--------------------------------------------------------------------------
Curve math, Figure 2.1 minimum reactive-power capability, tolerance handling,
and the Q_impact classification are ported from this project's own already-
validated Solar Analytics pipeline:

  bms_sa_review/shared/ciccada_config.py                       (AS4777 dict)
  bms_sa_review/shared/as4777_curves.py                        (curve math)
  bms_sa_review/data_calc_write/stage2_conformance/
      build_conformance_voltvar.py                             (Q_impact classification, QCAP clamping)
  bms_sa_review/data_query/lib/explore_plots.py                (plot_operational, plot_vvar_month_scatter)

The curve functions (`vw_max_p`, `vvar_required_q`, `q_cap_absorbing`,
`q_conformance_floor_absorbing`, `add_tol_kw`, `q_impact_nearest_edge`) are
copied close to verbatim from `as4777_curves.py`. `classify_voltvar_interval`
re-implements the SQL CTE chain in `build_conformance_voltvar.py`'s
`_insert_sql` (required_q -> tol_band -> clamped -> q_impact -> classified)
as a vectorised pandas/numpy computation instead of an Athena INSERT, for one
site at a time instead of the whole fleet.

WHAT IS DELIBERATELY OUT OF SCOPE
--------------------------------------------------------------------------
- `S_99` / `empirical_limit_basis` and the Method A curtailment-energy
  estimate (`curtailment_voltvar`, `curtailment_eligible`). Both need an
  uncurtailed-PV counterfactual, which the Ausgrid pipeline has not built
  yet (METHODOLOGY_GATES.md gate 7: "Delivery 3 must estimate counterfactual
  PV ... Not estimated in Delivery 2"). Only the Q_impact/capability
  classification against the *required* curve is reproduced here.
- `rating_basis` is fixed to `solar_capacity_kw` (from `site_phase_profile.
  parquet`), passed in by the notebook as an EXPLICIT, LABELLED PROXY for
  S_rated. `DATA_CONTRACT.md`: `s_rated_kva` is null, and neither
  `approved_capacity_kw` nor `solar_capacity_kw` may be silently treated as
  S_rated. Every plot below prints/labels this proxy on the figure itself.
- day/night bucketing, fleet-wide ranking, and table writes: this module
  only classifies and plots one site's already-fetched DataFrame. Nothing
  is written to disk or to a database.

SIGN CONVENTION
--------------------------------------------------------------------------
Every function below uses GENERATOR convention (+Q = supplying/injecting,
-Q = absorbing/consuming), matching `as4777_curves.py` and AS/NZS 4777.2
Fig 3.2 directly -- the same convention this project settled on for
`02c_explore_voltvar_voltwatt.ipynb`. `ausgrid_analysis`'s own derived
column is the opposite (`q_absorbing_var`: +Q = absorbing), so the notebook
must negate it once when building the DataFrames passed in here
(`q_generator = -q_absorbing`). This module does not do that negation
itself -- it assumes whatever `q_kvar`/`Q_kvar` it receives is already in
generator convention, so a bug in the notebook's negation will show up as
values that look "backwards" here rather than being silently double-flipped.
"""

from __future__ import annotations

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# AS/NZS 4777.2:2020 Australia Region A constants.
# Copied verbatim from bms_sa_review/shared/ciccada_config.py (AS4777 dict).
# Generator convention throughout: +Q = supplying, -Q = absorbing.
# ---------------------------------------------------------------------------
from ausgrid_analysis.as4777_curves import (
    Q_CAPABILITY,
    Q_IMPACT_THRESHOLDS,
    TOLERANCE_FRACTION,
    VOLT_VAR,
    VOLT_WATT,
    add_tolerance as _add_tolerance,
    classify_voltvar_interval as _classify_voltvar_scalar,
    q_cap_absorbing,
    q_conformance_floor_absorbing,
    q_impact_nearest_edge,
    vvar_required_q,
    vw_max_p,
)

# Plotting compatibility only. Values come from the tested package constants.
# Q4 stays signed in generator convention.
AS4777 = {
    "VW": {
        "V1": VOLT_WATT.v1, "V2": VOLT_WATT.v2,
        "P1": VOLT_WATT.p1, "P2": VOLT_WATT.p2,
    },
    "VVAR": {
        "V1": VOLT_VAR.v1, "V2": VOLT_VAR.v2,
        "V3": VOLT_VAR.v3, "V4": VOLT_VAR.v4,
        "Q1": VOLT_VAR.q1, "Q4": VOLT_VAR.q4,
    },
    "QCAP": {
        "P_MIN": Q_CAPABILITY.p_min,
        "P_FLAT_MAX": Q_CAPABILITY.p_flat_max,
        "Q_FLAT": Q_CAPABILITY.q_flat,
        "PF_MIN": Q_CAPABILITY.pf_min,
        "P_CIRCLE": Q_CAPABILITY.p_circle,
    },
    "QIMP": dict(zip(("thr1", "thr2", "thr3", "thr4"), Q_IMPACT_THRESHOLDS)),
    "TOL_FRAC": TOLERANCE_FRACTION,
    "INTERVAL_H": 5 / 60,
}

STATUS_ORDER = [
    "conforming", "Q_near_conformant", "Q_significant_shortfall",
    "Q_inactive", "Q_adverse", "Q_major_surplus", "not_assessable",
]
STATUS_COLORS = {
    "conforming": "#1565c0", "Q_near_conformant": "#2e7d32",
    "Q_significant_shortfall": "#ef6c00", "Q_inactive": "#c62828",
    "Q_adverse": "#7f1d1d", "Q_major_surplus": "#6a1b9a",
    "not_assessable": "#9e9e9e",
}
STATUS_LABELS = {
    "conforming": "Conforming (within band)",
    "Q_near_conformant": "Near-conformant (90-110% of required)",
    "Q_significant_shortfall": "Significant shortfall (10-90%)",
    "Q_inactive": "Inactive (no response, within +/-10%)",
    "Q_adverse": "Adverse (wrong direction)",
    "Q_major_surplus": "Major surplus (>110% of required)",
    "not_assessable": "Not assessable",
}


def add_tol_kw(value, capacity, tol_frac=TOLERANCE_FRACTION, sign=+1):
    return _add_tolerance(
        value, capacity, direction=sign, tolerance_fraction=tol_frac
    )


def classify_voltvar_interval(
    v_v, p_kw, q_kvar, s_rated_kw,
    capability_profile="review_corrected",
    tol_frac=TOLERANCE_FRACTION,
):
    """Vector adapter for the tested scalar package implementation."""
    v = np.asarray(v_v, dtype=float)
    p = np.asarray(p_kw, dtype=float)
    q = np.asarray(q_kvar, dtype=float)
    s = np.asarray(s_rated_kw, dtype=float)
    if s.ndim == 0:
        s = np.full(len(v), float(s))
    profile = (
        "figure_2_1_circle"
        if capability_profile == "hossein_m3"
        else capability_profile
    )
    rows = [
        _classify_voltvar_scalar(
            float(vi), float(pi), float(qi), float(si),
            capability_profile=profile,
            tolerance_fraction=tol_frac,
        )
        for vi, pi, qi, si in zip(v, p, q, s, strict=True)
    ]
    return pd.DataFrame({
        "Q_voltvar": [row.q_required for row in rows],
        "Q_cap_absorbing": [row.q_capability_absorbing for row in rows],
        "capability_assessable": [row.capability_assessable for row in rows],
        "Q_min_final": [row.q_min_final for row in rows],
        "Q_max_final": [row.q_max_final for row in rows],
        "Q_impact": [row.q_impact for row in rows],
        "status": [row.status for row in rows],
    })



# ===========================================================================
# 3. Monthly Q-vs-V scatter -- ported from bms_sa_review/data_query/lib/
#    explore_plots.py::plot_vvar_month_scatter, recoloured by the full
#    classify_voltvar_interval() status instead of a simple in/out-of-band flag.
# ===========================================================================
def plot_vvar_month_scatter_ausgrid(scatter_df, 
                                    serial, 
                                    capacity_kw, 
                                    period_label,
                                    capability_profile="review_corrected",
                                    tol=None, 
                                    manufacturer="",
                                    figsize=(12, 8)):
    """Q-vs-V scatter for a full site-month, coloured by Q_impact status.

    scatter_df needs columns V (volts), P_kW, Q_kvar -- GENERATOR convention,
    one row per interval (build with the notebook's `sample` DataFrame).
    capacity_kw is an explicitly labelled proxy for S_rated (solar_capacity_kw),
    not a verified rating -- this is printed on every axis label and title.
    """
    if scatter_df.empty:
        print("No intervals to plot.")
        return None

    S = capacity_kw
    tol = AS4777["TOL_FRAC"] if tol is None else tol
    vv = AS4777["VVAR"]

    d = scatter_df.copy()
    cls = classify_voltvar_interval(d["V"], d["P_kW"], d["Q_kvar"], S, capability_profile, tol)
    d = pd.concat([d.reset_index(drop=True), cls.reset_index(drop=True)], axis=1)
    d["Q_pct"] = d["Q_kvar"] / S * 100.0

    counts = d["status"].value_counts()
    n_total = len(d)

    V_grid = np.linspace(200, 280, 800)
    Q_req_pct = np.array([vvar_required_q(v, S) for v in V_grid]) / S * 100.0
    Q_max_pct = Q_req_pct + tol * 100.0
    Q_min_pct = Q_req_pct - tol * 100.0

    def _draw(ax, x_lo, x_hi, y_lo, y_hi, x_step, y_step, subtitle):
        for status in STATUS_ORDER:
            sub = d[d["status"] == status]
            if sub.empty:
                continue
            n = len(sub)
            ax.scatter(sub["V"], sub["Q_pct"], s=5, alpha=0.35 if status == "conforming" else 0.55,
                       color=STATUS_COLORS[status], zorder=3,
                       label=f"{STATUS_LABELS[status]} ({n:,}, {100*n/n_total:.1f}%)")
        ax.plot(V_grid, Q_req_pct, color="#f59e0b", lw=1.8, zorder=5, label="Required Q (AS4777 curve)")
        ax.fill_between(V_grid, Q_min_pct, Q_max_pct, color="#f59e0b", alpha=0.15, linewidth=0, zorder=2,
                        label=f"+/-{tol*100:.0f}% proxy tolerance (unclamped)")
        ax.axhline(0, color="k", lw=0.5, zorder=1)
        for vx, lbl in [(vv["V1"], f"V1 {vv['V1']:.0f}"), (vv["V2"], f"V2 {vv['V2']:.0f}"),
                        (vv["V3"], f"V3 {vv['V3']:.0f}"), (vv["V4"], f"V4 {vv['V4']:.0f}")]:
            if x_lo <= vx <= x_hi:
                ax.axvline(vx, color="grey", lw=0.6, ls=":", zorder=1)
                ax.text(vx + 0.4, y_hi * 0.95, lbl, fontsize=6, color="grey", va="top", ha="left")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(range(x_lo, x_hi + 1, x_step))
        ax.set_yticks(range(y_lo, y_hi + 1, y_step))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
        ax.set_xlabel("Voltage (V)", fontsize=9)
        ax.set_ylabel("Reactive power (% solar_capacity_kw PROXY, not S_rated)\n+ = supplying, - = absorbing", fontsize=9)
        ax.set_title(
            f"Site {serial}  ·  {manufacturer}  ·  {period_label}\n"
            f"Volt-Var response vs AS/NZS 4777.2:2020 ({capability_profile})\n"
            f"solar_capacity_kw proxy {S:.1f} kW (NOT verified S_rated)",
            fontsize=8.5, fontweight="bold", loc="left")
        ax.legend(fontsize=6.5, loc="lower left", framealpha=0.92, edgecolor="#cccccc", ncol=1)
        ax.grid(color="#ebebeb", lw=0.5)
        ax.set_facecolor("white")

    fig, ax2 = plt.subplots(figsize=figsize, dpi=130)
    _draw(ax2, 230, 260, -60, 40, 2, 10, "operating range zoom")
    plt.tight_layout()
    plt.show()

    print(f"Status breakdown ({n_total:,} intervals total):")
    for status in STATUS_ORDER:
        n = int(counts.get(status, 0))
        print(f"  {status:<26} {n:>7,}  ({100*n/n_total:5.1f}%)")
    return d


# ===========================================================================
# 4. Single-day operational-modes plot -- ported from bms_sa_review/data_query/
#    lib/explore_plots.py::plot_operational (unclamped +/-4% band, matching
#    the source exactly; capability clamping is only applied in the monthly
#    scatter above, not here).
# ===========================================================================
def plot_operational_ausgrid(df_day, serial, capacity_kw, zoom_date, manufacturer="", figsize=(13, 13.5)):
    """5-panel daily plot for Volt-Watt + Volt-Var, ported from explore_plots.plot_operational.

    df_day needs columns: timestamp_local (tz-naive local datetime), voltage,
    P_kW, Q_kvar (GENERATOR convention). capacity_kw is a labelled proxy for
    S_rated, not a verified rating.
    """
    vw, vv, tol = AS4777["VW"], AS4777["VVAR"], AS4777["TOL_FRAC"]
    S = capacity_kw

    t = df_day["timestamp_local"]
    V = df_day["voltage"]
    P = df_day["P_kW"]
    Q = df_day["Q_kvar"]

    def _vw_ceil_pct(v):
        if v < vw["V1"]:
            return 100.0
        if v > vw["V2"]:
            return vw["P2"] * 100.0
        return (1.0 - vw["P2"]) / (vw["V1"] - vw["V2"]) * (v - vw["V2"]) * 100.0 + vw["P2"] * 100.0

    Q_req = V.map(lambda v: vvar_required_q(v, S))
    Q_req_max = Q_req + tol * S
    Q_req_min = Q_req - tol * S

    P_pct = (P / S) * 100.0
    P_ceil_pct = V.map(_vw_ceil_pct) + tol * 100.0
    P_nc_pct = np.where(V.values > vw["V1"], np.maximum(0.0, P_pct.values - P_ceil_pct.values), 0.0)
    Q_pct = (Q / S) * 100.0
    Q_req_pct = (Q_req / S) * 100.0
    Q_req_max_pct = (Q_req_max / S) * 100.0
    Q_req_min_pct = (Q_req_min / S) * 100.0

    nc_vvar_kvar = np.maximum(0, Q - Q_req_max) + np.maximum(0, Q_req_min - Q)
    nc_signed_pct = np.where(
        Q_pct.values > Q_req_max_pct.values, Q_pct.values - Q_req_max_pct.values,
        np.where(Q_pct.values < Q_req_min_pct.values, -(Q_req_min_pct.values - Q_pct.values), 0.0))

    # Same capability-assessable gate as classify_voltvar_interval/Q_CAPABILITY.
    # p_min: below this fraction of capacity, the standard doesn't require any
    # particular reactive response, so these intervals should not count toward
    # a "how much did the site fall short" summary, and are greyed out below.
    capability_assessable = (P.abs() >= Q_CAPABILITY.p_min * S).to_numpy()

    nc_kvarh = nc_vvar_kvar[capability_assessable].sum() * AS4777["INTERVAL_H"]
    nc_kvarh_per_kw = nc_kvarh / S

    vvar_active = V.values > vv["V3"]
    vw_active = V.values > vw["V1"]

    Cv, Cp, Cc, Cq, Cn = "#b45309", "#2e7d32", "#1a1a1a", "#1565c0", "#c62828"
    C_REF, C_VVAR, C_VW, C_GRID = "#f59e0b", "#7c3aed", "#4709b2", "#ebebeb"

    def _mirror_axis(ax_primary, ax_twin, pct_to_unit, unit_ticks, fmt="{:.1f}"):
        lo, hi = ax_primary.get_ylim()
        ax_twin.set_ylim(lo, hi)
        pct_positions = [u / pct_to_unit * 100.0 for u in unit_ticks]
        ax_twin.set_yticks(pct_positions)
        ax_twin.set_yticklabels([fmt.format(u) for u in unit_ticks])

    fig, axes = plt.subplots(5, 1, figsize=figsize, dpi=130, sharex=True,
                             gridspec_kw={"height_ratios": [1.8, 2.2, 1.0, 2.2, 1.0]})
    fig.subplots_adjust(hspace=0.05, left=0.10, right=0.90, top=0.94, bottom=0.05)
    ax_v, ax_p, ax_pnc, ax_q, ax_qnc = axes

    for ax in axes:
        ax.fill_between(t, 0, 1, where=vvar_active, transform=ax.get_xaxis_transform(),
                        color=C_VVAR, alpha=0.07, linewidth=0, zorder=0)
        ax.fill_between(t, 0, 1, where=vw_active, transform=ax.get_xaxis_transform(),
                        color=C_VW, alpha=0.08, linewidth=0, zorder=0)
    _vvar_patch = Patch(color=C_VVAR, alpha=0.30, label=f"V > {vv['V3']:.0f} V — V-Var required")
    _vw_patch = Patch(color=C_VW, alpha=0.30, label=f"V >= {vw['V1']:.0f} V — V-Watt active")

    ax_v.plot(t, V, color=Cv, lw=1.3, zorder=4)
    for vref, ls, al, lbl in [
        (vv["V3"], ":", 0.55, "240 V (V-Var deadband hi)"),
        (vw["V1"], "--", 0.85, "253 V (V-Watt start / V-Var hi ramp)"),
        (vv["V4"], "--", 0.85, "258 V (V-Var max absorb)"),
        (vw["V2"], "--", 0.85, "260 V (V-Watt full curtail)"),
    ]:
        ax_v.axhline(vref, color=Cv, lw=0.8, ls=ls, alpha=al, zorder=3)
        ax_v.text(t.iloc[-1], vref + 0.25, lbl, va="bottom", ha="right", fontsize=6, color=Cv, alpha=min(al + 0.15, 1.0))
    ax_v.set_ylabel("Voltage (V)\n(revenue meter)", fontsize=8.5, color=Cv)
    ax_v.tick_params(axis="y", colors=Cv, labelsize=8)
    ax_v.set_ylim(min(228, V.min() - 2), max(262, V.max() + 2))
    ax_v.grid(color=C_GRID, lw=0.5)
    ax_v.set_facecolor("white")
    ax_v.legend(handles=[_vvar_patch, _vw_patch], fontsize=7, loc="upper left", framealpha=0.92, edgecolor="#cccccc", ncol=1)
    plt.setp(ax_v.get_xticklabels(), visible=False)

    ax_p.plot(t, P_ceil_pct, color=Cc, lw=1.6, zorder=4, label=f"V-Watt ceiling (+{tol*100:.0f}% tol, % proxy)")
    ax_p.plot(t, P_pct, color=Cp, lw=1.3, zorder=5, label="Measured P (% proxy)")
    ax_p.axhline(100, color=Cp, lw=0.6, ls=":", alpha=0.45, zorder=3, label="100% proxy")
    ax_p.axhline(vw["P2"] * 100, color=Cc, lw=0.6, ls=":", alpha=0.45, zorder=3, label=f"V-Watt floor: {vw['P2']*100:.0f}% at >={vw['V2']:.0f} V")
    ax_p.set_ylabel("Active power\n(% proxy)", fontsize=8.5, color=Cp)
    ax_p.tick_params(axis="y", colors=Cp, labelsize=8)
    ax_p.set_ylim(-10, 115)
    ax_p.set_yticks([-10, 0, 20, 40, 60, 80, 100])
    ax_p.grid(color=C_GRID, lw=0.5)
    ax_p.set_facecolor("white")
    ax_p.legend(fontsize=7, loc="upper left", framealpha=0.9)
    plt.setp(ax_p.get_xticklabels(), visible=False)

    ax_pkw = ax_p.twinx()
    _mirror_axis(ax_p, ax_pkw, pct_to_unit=S, unit_ticks=list(np.linspace(0, S, 6)), fmt="{:.1f}")
    ax_pkw.set_ylabel("Active power\n(kW)", fontsize=8.5, color=Cp)
    ax_pkw.tick_params(axis="y", colors=Cp, labelsize=8)

    ax_pnc.bar(t, P_nc_pct, width=pd.Timedelta(minutes=4.5), color=Cn, alpha=0.80, align="center", zorder=4,
              label="V-Watt NC (pp above ceiling)")
    ax_pnc.axhline(0, color="k", lw=0.5, zorder=3)
    ax_pnc.set_ylabel("V-W NC\n(pp)", fontsize=8.5, color=Cn)
    ax_pnc.tick_params(axis="y", colors=Cn, labelsize=8)
    ax_pnc.set_ylim(0, max(P_nc_pct.max() * 1.15, 2.0))
    ax_pnc.grid(color=C_GRID, lw=0.5, axis="y")
    ax_pnc.set_facecolor("white")
    ax_pnc.legend(fontsize=7, loc="upper left", framealpha=0.9)
    plt.setp(ax_pnc.get_xticklabels(), visible=False)

    ax_q.fill_between(t, Q_req_min_pct, Q_req_max_pct, color=C_REF, alpha=0.25, linewidth=0, zorder=1)
    ax_q.plot(t, Q_req_min_pct, color=C_REF, lw=0.8, ls="--", zorder=2)
    ax_q.plot(t, Q_req_max_pct, color=C_REF, lw=0.8, ls="--", zorder=2)
    ax_q.plot(t, Q_req_pct, color=C_REF, lw=1.0, ls="-", alpha=0.6, zorder=2)
    ax_q.plot(t, Q_pct, color=Cq, lw=1.4, zorder=4)
    ax_q.axhline(0, color="k", lw=0.5, zorder=3)
    ax_q.set_ylabel("Reactive power\n(% proxy, + sup / - abs)", fontsize=8.5, color=Cq)
    ax_q.tick_params(axis="y", colors=Cq, labelsize=8)
    _qlim = max(abs(vv["Q1"]), abs(vv["Q4"])) * 100 + 10
    ax_q.set_ylim(-_qlim, _qlim)
    ax_q.grid(color=C_GRID, lw=0.5)
    ax_q.set_facecolor("white")
    ax_q.legend(handles=[
        Patch(color=C_REF, alpha=0.40, label=f"Required Q band (+/-{tol*100:.0f}% proxy, UNCLAMPED)"),
        plt.Line2D([0], [0], color=Cq, lw=1.4, label="Measured Q (% proxy)"),
    ], fontsize=7, loc="upper left", framealpha=0.9)
    plt.setp(ax_q.get_xticklabels(), visible=False)

    ax_qkvar = ax_q.twinx()
    _kvar_ticks = [round(x, 1) for x in np.linspace(-_qlim / 100 * S, _qlim / 100 * S, 7)]
    _mirror_axis(ax_q, ax_qkvar, pct_to_unit=S, unit_ticks=_kvar_ticks, fmt="{:+.1f}")
    ax_qkvar.set_ylabel("Reactive power\n(kvar)", fontsize=8.5, color=Cq)
    ax_qkvar.tick_params(axis="y", colors=Cq, labelsize=8)

    C_NA = "#9e9e9e"  # not-assessable grey, matches STATUS_COLORS['not_assessable']
    qnc_colors = np.where(capability_assessable, Cn, C_NA)
    ax_qnc.bar(t, nc_signed_pct, width=pd.Timedelta(minutes=4.5), color=qnc_colors, alpha=0.80, align="center", zorder=4)
    ax_qnc.axhline(0, color="k", lw=0.5, zorder=3)
    ax_qnc.set_ylabel("V-Var NC\n(pp, signed)", fontsize=8.5, color=Cn)
    ax_qnc.tick_params(axis="y", colors=Cn, labelsize=8)
    ax_qnc.grid(color=C_GRID, lw=0.5, axis="y")
    ax_qnc.set_facecolor("white")
    ax_qnc.legend(handles=[
        plt.Line2D([0], [0], color=Cn, lw=4, alpha=0.80, label="V-Var NC (pp outside UNCLAMPED band, signed)"),
        plt.Line2D([0], [0], color=C_NA, lw=4, alpha=0.80,
                   label=f"Not assessable (|P| < {Q_CAPABILITY.p_min*100:.0f}% proxy)"),
    ], fontsize=7, loc="upper left", framealpha=0.9)
    ax_qnc.text(0.99, 0.95,
               f"Daily V-Var NC (unclamped band, assessable only):  {nc_kvarh:.3f} kvarh"
               f"   ({nc_kvarh_per_kw:.4f} kvarh / kW proxy)",
               transform=ax_qnc.transAxes, ha="right", va="top", fontsize=7.5, color=Cn,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=Cn, alpha=0.85, lw=0.7))
    ax_qnc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_qnc.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_qnc.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax_qnc.tick_params(axis="x", which="major", labelsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.suptitle(
        f"Site {serial}  ·  {manufacturer}  ·  {zoom_date}\n"
        f"Operational modes: Volt-Watt & Volt-Var  ·  solar_capacity_kw proxy {S:.1f} kW "
        f"(NOT verified S_rated)  ·  net-meter, revenue-meter voltage",
        fontsize=9.5, fontweight="bold", y=0.975)
    plt.show()

    n_vw_nc = int((P_nc_pct > 0).sum())
    n_vvar_nc = int((nc_vvar_kvar[capability_assessable] > 0).sum())
    print(f"\nVolt-Watt:  {n_vw_nc} intervals above the unclamped ceiling")
    print(f"Volt-Var:   {n_vvar_nc} intervals outside the unclamped band  |  "
          f"{nc_kvarh:.3f} kvarh  ({nc_kvarh_per_kw:.4f} kvarh/kW proxy)")
