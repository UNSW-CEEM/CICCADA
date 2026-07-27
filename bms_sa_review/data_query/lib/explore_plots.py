"""
Single-day diagnostic plots for fleet exploration
======================================================================

  plot_operational(...)  5-panel: voltage, Volt-Watt P vs ceiling + NC,
                         Volt-VAr Q vs required band + NC. For the two
                         autonomous power-quality response modes.

  plot_protective(...)   2-panel: voltage with OV/UV thresholds, and active
                         power. For sustained-operation and anti-islanding,
                         whose verdict is "did P drop below 4% in time".

Every function takes a prepared day DataFrame and explicit params. No globals.
The day DataFrame needs: t_stamp_aest (tz-aware AEST), voltage, P_kW, Q_kvar.
Use `to_aest()` below to build t_stamp_aest from a UTC t_stamp.
"""

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

from shared.ciccada_config import AS4777, FIXED_OFFSET
from shared.as4777_curves import vvar_required_q, vw_max_p

# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------
def to_aest(series):
    """UTC datetime series -> AEST (UTC+10, fixed, no DST)."""
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    return series.dt.tz_convert(FIXED_OFFSET)


def _strip_tz(series):
    """Drop the tz label without shifting values (matplotlib fill_between)."""
    return series.dt.tz_localize(None)

def plot_operational(df_day, site_id, ac_capacity_kw, zoom_date,
                     as4777, manufacturer="", figsize=(13, 13.5)):
    """
    5-panel unified daily plot for OPERATIONAL modes (Volt-Watt + Volt-VAr).
 
    Panel 0: Voltage — regime zones shaded, key setpoint lines labelled.
    Panel 1: Active power vs V-Watt ceiling. Measured + ceiling BOTH on % S_rated
             (left). kW axis (right) is a pure mirror of the same values.
    Panel 2: Volt-Watt non-conformance (percentage points above ceiling).
    Panel 3: Reactive power vs required V-VAr band. Measured Q AND the required
             band BOTH on % S_rated (left) — so they are directly comparable.
             kvar axis (right) is a pure mirror. This is the fix: previously the
             measured line was effectively read against the kvar scale while the
             band lived on the % scale, making a small under-absorption look like
             the inverter was idle.
    Panel 4: Volt-VAr non-conformance (signed pp outside band).
 
    Both twin axes are locked to `_mirror_axis`, which copies the primary axis
    limits and only relabels ticks in the physical unit — guaranteeing the two
    y-scales can never drift apart again.
    """
    vw  = as4777["VW"]
    vv  = as4777["VVAR"]
    tol = as4777["TOL_FRAC"]
    S   = ac_capacity_kw
 
    t = _strip_tz(df_day["t_stamp_aest"])
    V = df_day["voltage"]
    P = df_day["P_kW"]
    Q = df_day["Q_kvar"]
 
    # ---- curves (keystone) --------------------------------------------------
    def _vw_ceil_pct(v):
        if v < vw["V1"]: return 100.0
        if v > vw["V2"]: return vw["P2"] * 100.0
        return (1.0 - vw["P2"]) / (vw["V1"] - vw["V2"]) * (v - vw["V2"]) * 100.0 \
               + vw["P2"] * 100.0
 
    # V-VAr required Q in kvar, from the single-source-of-truth curve
    Q_req     = V.map(lambda v: vvar_required_q(v, S))
    Q_req_max = Q_req + tol * S
    Q_req_min = Q_req - tol * S
 
    # ---- everything in % S_rated (the common unit) --------------------------
    P_pct         = (P / S) * 100.0
    P_ceil_pct    = V.map(_vw_ceil_pct) + tol * 100.0
    P_nc_pct = np.where(
        V.values > vw["V1"],
        np.maximum(0.0, P_pct.values - P_ceil_pct.values),
        0.0,
    ) 
    Q_pct         = (Q         / S) * 100.0
    Q_req_pct     = (Q_req     / S) * 100.0
    Q_req_max_pct = (Q_req_max / S) * 100.0
    Q_req_min_pct = (Q_req_min / S) * 100.0
 
    # V-VAr NC: signed pp for bars, kvar magnitude for energy totals
    nc_vvar_kvar  = np.maximum(0, Q - Q_req_max) + np.maximum(0, Q_req_min - Q)
    nc_signed_pct = np.where(
        Q_pct.values > Q_req_max_pct.values,
         Q_pct.values - Q_req_max_pct.values,
        np.where(
            Q_pct.values < Q_req_min_pct.values,
            -(Q_req_min_pct.values - Q_pct.values),
            0.0))
 
    INTERVAL_H      = as4777.get("INTERVAL_H", 5 / 60)
    nc_kvarh        = nc_vvar_kvar.sum() * INTERVAL_H
    nc_kvarh_per_kw = nc_kvarh / S
 
    vvar_active = V.values >  vv["V3"]
    vw_active = V.values > vw["V1"] 
    
    # ---- colours ------------------------------------------------------------
    Cv, Cp, Cc, Cq, Cn = "#b45309", "#2e7d32", "#1a1a1a", "#1565c0", "#c62828"
    C_REF, C_VVAR, C_VW, C_GRID = "#f59e0b", "#7c3aed", "#4709b2", "#ebebeb"
 
    # ---- helper: make a twin axis a PURE MIRROR of the primary --------------
    # Copies the primary's [lo, hi] and only converts tick LABELS to the physical
    # unit (kW or kvar). Because the transform is identical, the mirrored series
    # sits at exactly the same height as on the primary axis — the two scales
    # can never disagree.
    def _mirror_axis(ax_primary, ax_twin, pct_to_unit, unit_ticks, fmt="{:.1f}"):
        lo, hi = ax_primary.get_ylim()
        ax_twin.set_ylim(lo, hi)                      # same limits, same transform
        # place ticks at the % positions that correspond to round unit values
        pct_positions = [u / pct_to_unit * 100.0 for u in unit_ticks]
        ax_twin.set_yticks(pct_positions)
        ax_twin.set_yticklabels([fmt.format(u) for u in unit_ticks])
 
    # ---- layout -------------------------------------------------------------
    fig, axes = plt.subplots(
        5, 1, figsize=figsize, dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [1.8, 2.2, 1.0, 2.2, 1.0]})
    fig.subplots_adjust(hspace=0.05, left=0.10, right=0.90, top=0.95, bottom=0.05)
    ax_v, ax_p, ax_pnc, ax_q, ax_qnc = axes
 
    for ax in axes:
        ax.fill_between(t, 0, 1, where=vvar_active,
                        transform=ax.get_xaxis_transform(),
                        color=C_VVAR, alpha=0.07, linewidth=0, zorder=0)
        ax.fill_between(t, 0, 1, where=vw_active,
                        transform=ax.get_xaxis_transform(),
                        color=C_VW, alpha=0.08, linewidth=0, zorder=0)
    _vvar_patch = Patch(color=C_VVAR, alpha=0.30,
                        label=f"V > {vv['V3']:.0f} V — V-VAr required")
    _vw_patch   = Patch(color=C_VW,   alpha=0.30,
                        label=f"V >= {vw['V1']:.0f} V — V-Watt active")
 
    # ═══ PANEL 0 Voltage ═══════════════════════════════════════════════════
    ax_v.plot(t, V, color=Cv, lw=1.3, zorder=4)
    for vref, ls, al, lbl in [
        (vv["V3"], ":",  0.55, "240 V (V-VAr deadband hi)"),
        (vw["V1"], "--", 0.85, "253 V (V-Watt start / V-VAr hi ramp)"),
        (vv["V4"], "--", 0.85, "258 V (V-VAr max absorb)"),
        (vw["V2"], "--", 0.85, "260 V (V-Watt full curtail)"),
    ]:
        ax_v.axhline(vref, color=Cv, lw=0.8, ls=ls, alpha=al, zorder=3)
        ax_v.text(t.iloc[-1], vref + 0.25, lbl, va="bottom", ha="right",
                  fontsize=6, color=Cv, alpha=min(al + 0.15, 1.0))
    ax_v.set_ylabel("Voltage (V)", fontsize=8.5, color=Cv)
    ax_v.tick_params(axis="y", colors=Cv, labelsize=8)
    ax_v.set_ylim(min(228, V.min() - 2), max(262, V.max() + 2))
    ax_v.grid(color=C_GRID, lw=0.5); ax_v.set_facecolor("white")
    ax_v.legend(handles=[_vvar_patch, _vw_patch], fontsize=7,
                loc="upper left", framealpha=0.92, edgecolor="#cccccc", ncol=1)
    plt.setp(ax_v.get_xticklabels(), visible=False)
 
    # ═══ PANEL 1 Active power — both series on % S_rated ═══════════════════
    ax_p.plot(t, P_ceil_pct, color=Cc, lw=1.6, zorder=4,
              label=f"V-Watt ceiling (+{tol*100:.0f}% tol, % S_rated)")
    ax_p.plot(t, P_pct, color=Cp, lw=1.3, zorder=5,
              label="Measured P (% S_rated)")
    ax_p.axhline(100, color=Cp, lw=0.6, ls=":", alpha=0.45, zorder=3,
                 label="100% S_rated (nameplate)")
    ax_p.axhline(vw["P2"] * 100, color=Cc, lw=0.6, ls=":", alpha=0.45, zorder=3,
                 label=f"V-Watt floor: {vw['P2']*100:.0f}% at >={vw['V2']:.0f} V")
    ax_p.set_ylabel("Active power\n(% S_rated)", fontsize=8.5, color=Cp)
    ax_p.tick_params(axis="y", colors=Cp, labelsize=8)
    ax_p.set_ylim(-2, 115); ax_p.set_yticks([0, 20, 40, 60, 80, 100])
    ax_p.grid(color=C_GRID, lw=0.5); ax_p.set_facecolor("white")
    ax_p.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    plt.setp(ax_p.get_xticklabels(), visible=False)
 
    # kW mirror (right) — identical transform, labels only
    ax_pkw = ax_p.twinx()
    _mirror_axis(ax_p, ax_pkw, pct_to_unit=S,
                 unit_ticks=list(np.linspace(0, S, 6)), fmt="{:.1f}")
    ax_pkw.set_ylabel("Active power\n(kW)", fontsize=8.5, color=Cp)
    ax_pkw.tick_params(axis="y", colors=Cp, labelsize=8)
 
    # ═══ PANEL 2 Volt-Watt NC ══════════════════════════════════════════════
    ax_pnc.bar(t, P_nc_pct, width=pd.Timedelta(minutes=4.5),
               color=Cn, alpha=0.80, align="center", zorder=4,
               label="V-Watt NC (pp above ceiling)")
    ax_pnc.axhline(0, color="k", lw=0.5, zorder=3)
    ax_pnc.set_ylabel("V-W NC\n(pp)", fontsize=8.5, color=Cn)
    ax_pnc.tick_params(axis="y", colors=Cn, labelsize=8)
    ax_pnc.set_ylim(0, max(P_nc_pct.max() * 1.15, 2.0))
    ax_pnc.grid(color=C_GRID, lw=0.5, axis="y"); ax_pnc.set_facecolor("white")
    ax_pnc.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    plt.setp(ax_pnc.get_xticklabels(), visible=False)
 
    # ═══ PANEL 3 Reactive power — measured AND band both on % S_rated ══════
    # THE FIX: everything plotted here is in % S_rated. The blue measured line
    # and the orange required band share one scale, so their vertical distance
    # IS the conformance gap. The kvar axis on the right is a pure mirror.
    ax_q.fill_between(t, Q_req_min_pct, Q_req_max_pct,
                      color=C_REF, alpha=0.25, linewidth=0, zorder=1)
    ax_q.plot(t, Q_req_min_pct, color=C_REF, lw=0.8, ls="--", zorder=2)
    ax_q.plot(t, Q_req_max_pct, color=C_REF, lw=0.8, ls="--", zorder=2)
    ax_q.plot(t, Q_req_pct,     color=C_REF, lw=1.0, ls="-",  alpha=0.6, zorder=2)
    ax_q.plot(t, Q_pct,         color=Cq,    lw=1.4, zorder=4)
    ax_q.axhline(0, color="k", lw=0.5, zorder=3)
    ax_q.set_ylabel("Reactive power\n(% S_rated, + sup / - abs)",
                    fontsize=8.5, color=Cq)
    ax_q.tick_params(axis="y", colors=Cq, labelsize=8)
    _qlim = max(vv["Q1"], vv["Q4"]) * 100 + 10
    ax_q.set_ylim(-_qlim, _qlim)
    ax_q.grid(color=C_GRID, lw=0.5); ax_q.set_facecolor("white")
    ax_q.legend(handles=[
        Patch(color=C_REF, alpha=0.40,
              label=f"Required Q band (±{tol*100:.0f}% S_rated)"),
        plt.Line2D([0], [0], color=Cq, lw=1.4, label="Measured Q (% S_rated)"),
    ], fontsize=7.5, loc="upper left", framealpha=0.9)
    plt.setp(ax_q.get_xticklabels(), visible=False)
 
    # kvar mirror (right) — identical transform, labels only
    ax_qkvar = ax_q.twinx()
    _kvar_ticks = [round(x, 1) for x in
                   np.linspace(-_qlim / 100 * S, _qlim / 100 * S, 7)]
    _mirror_axis(ax_q, ax_qkvar, pct_to_unit=S, unit_ticks=_kvar_ticks, fmt="{:+.1f}")
    ax_qkvar.set_ylabel("Reactive power\n(kvar)", fontsize=8.5, color=Cq)
    ax_qkvar.tick_params(axis="y", colors=Cq, labelsize=8)
 
    # ═══ PANEL 4 Volt-VAr NC (signed pp) ══════════════════════════════════
    ax_qnc.bar(t, nc_signed_pct, width=pd.Timedelta(minutes=4.5),
               color=Cn, alpha=0.80, align="center", zorder=4)
    ax_qnc.axhline(0, color="k", lw=0.5, zorder=3)
    ax_qnc.set_ylabel("V-VAr NC\n(pp, signed)", fontsize=8.5, color=Cn)
    ax_qnc.tick_params(axis="y", colors=Cn, labelsize=8)
    ax_qnc.grid(color=C_GRID, lw=0.5, axis="y"); ax_qnc.set_facecolor("white")
    ax_qnc.legend(handles=[
        plt.Line2D([0], [0], color=Cn, lw=4, alpha=0.80,
                   label="V-VAr NC (pp outside band, signed)"),
    ], fontsize=7.5, loc="upper left", framealpha=0.9)
    ax_qnc.text(0.99, 0.95,
                f"Daily V-VAr NC:  {nc_kvarh:.3f} kvarh"
                f"   ({nc_kvarh_per_kw:.4f} kvarh / kW nameplate)",
                transform=ax_qnc.transAxes, ha="right", va="top",
                fontsize=7.5, color=Cn,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=Cn, alpha=0.85, lw=0.7))
    ax_qnc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_qnc.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_qnc.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax_qnc.tick_params(axis="x", which="major", labelsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")
 
    fig.suptitle(
        f"Site {site_id}  ·  {manufacturer}  ·  {zoom_date}\n"
        f"Operational modes: Volt-Watt & Volt-VAr  ·  Nameplate {S:.0f} kW AC",
        fontsize=10, fontweight="bold", y=0.975)
    plt.show()
 
    n_vw_nc   = int((P_nc_pct > 0).sum())
    n_vvar_nc = int((nc_vvar_kvar > 0).sum())
    print(f"\nVolt-Watt:  {n_vw_nc} non-conforming intervals")
    print(f"Volt-VAr:   {n_vvar_nc} non-conforming intervals  |  "
          f"{nc_kvarh:.3f} kvarh  ({nc_kvarh_per_kw:.4f} kvarh/kW)")

def plot_protective(df_day, site_id, ac_capacity_kw, zoom_date,
                    manufacturer="", figsize=(12, 7),
                    sustop_v_ceil=258, ai_ov1=265, ai_ov2=275,
                    ai_uv1=180, ai_uv2=70):
    """
    2-panel daily plot for PROTECTIVE functions (Sustained-Op + Anti-Islanding).

    Panel 1 Voltage:
        Zoomed to observed range. OV thresholds (258, 265, 275 V) shown as
        reference lines with left-spine labels and zone shading.
        UV thresholds (180, 70 V) shown as a text annotation only — they sit far
        below normal operating range and would compress the y-axis uselessly.

    Panel 2 Active power P only:
        The conformance criterion is whether P drops below 4% of nameplate
        within the required time after the voltage threshold is exceeded.
        Reactive power is not a compliance criterion for these modes.
    """
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Patch

    t   = _strip_tz(df_day["t_stamp_aest"])
    V   = df_day["voltage"]
    P   = df_day["P_kW"]
    S   = ac_capacity_kw

    so  = sustop_v_ceil    # 258 V
    ov1 = ai_ov1           # 265 V
    ov2 = ai_ov2           # 275 V
    uv1 = ai_uv1
    uv2 = ai_uv2

    C_VOLT = "#b45309"; C_P = "#2e7d32"; C_GRID = "#ebebeb"
    C_SO   = "#f59e0b"
    C_OV   = "#c62828"
    C_UV   = "#1565c0"

    fig = plt.figure(figsize=figsize, dpi=140)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.08)
    gs   = fig.add_gridspec(2, 1, height_ratios=[1.6, 2.0], hspace=0.06)
    ax_v = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax_v)

    # ── OV zone shading ──────────────────────────────────────────────────
    for ax in (ax_v, ax_p):
        ax.fill_between(t, 0, 1,
                        where=(V.values >= so) & (V.values < ov1),
                        transform=ax.get_xaxis_transform(),
                        color=C_SO, alpha=0.18, linewidth=0, zorder=0)
        ax.fill_between(t, 0, 1,
                        where=(V.values >= ov1) & (V.values < ov2),
                        transform=ax.get_xaxis_transform(),
                        color=C_OV, alpha=0.18, linewidth=0, zorder=0)
        ax.fill_between(t, 0, 1,
                        where=V.values >= ov2,
                        transform=ax.get_xaxis_transform(),
                        color=C_OV, alpha=0.35, linewidth=0, zorder=0)

    # ═══ PANEL 1 — Voltage ═══════════════════════════════════════════════
    ax_v.plot(t, V, color=C_VOLT, lw=1.4, zorder=4)
    ax_v.fill_between(t, so, V, where=V.values >= so,
                      color=C_VOLT, alpha=0.18, linewidth=0, zorder=2)

    v_lo = max(V.min() - 3, 230)
    v_hi = max(V.max() + 5, ov2 + 3)
    ax_v.set_ylim(v_lo, v_hi)

    OV_REFS = [
        (so,  C_SO,      "-",  1.2, 0.90, "bold",   f"{so:.0f} V  Sust-Op limit"),
        (ov1, C_OV,      "--", 1.0, 0.85, "bold",   f"{ov1:.0f} V  OV1 (trip ≤2 s)"),
        (ov2, "#7f1d1d", ":",  0.8, 0.65, "normal", f"{ov2:.0f} V  OV2 (trip ≤0.2 s)"),
    ]
    for vref, col, ls, lw_, al, fw, _ in OV_REFS:
        if v_lo <= vref <= v_hi:
            ax_v.axhline(vref, color=col, lw=lw_, ls=ls, alpha=al, zorder=3)

    fig.canvas.draw()
    blend = mtransforms.blended_transform_factory(ax_v.transAxes, ax_v.transData)
    for vref, col, _, _, al, fw, lbl in OV_REFS:
        if v_lo <= vref <= v_hi:
            ax_v.annotate("", xy=(0, vref), xycoords=blend,
                xytext=(-0.055, vref), textcoords=blend,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.9,
                                alpha=al, shrinkA=0, shrinkB=0),
                zorder=7, clip_on=False)
            ax_v.text(-0.06, vref, lbl, transform=blend,
                      ha="right", va="center", fontsize=7.5,
                      color=col, fontweight=fw, alpha=al, clip_on=False)

    # UV annotation — zorder=6 ensures it sits above the voltage line (zorder=4)
    ax_v.text(0.99, 0.04,
              f"Under-voltage thresholds (below observable range):\n"
              f"  UV1: V ≤ {uv1:.0f} V — trip ≤11 s\n"
              f"  UV2: V ≤ {uv2:.0f} V — trip ≤2 s",
              transform=ax_v.transAxes, ha="right", va="bottom",
              fontsize=6.5, color=C_UV, zorder=6,
              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        ec=C_UV, alpha=0.90, lw=0.7))

    visible_ticks = [v for v in [230, 240, 250, 258, 260, 265, 270, 275, 280]
                     if v_lo <= v <= v_hi]
    ax_v.set_yticks(visible_ticks)
    ax_v.set_yticklabels([str(v) for v in visible_ticks])
    ax_v.set_ylabel("Voltage (V)", fontsize=8.5, color=C_VOLT)
    ax_v.tick_params(axis="y", colors=C_VOLT, labelsize=8)
    ax_v.grid(True, color=C_GRID, lw=0.5, zorder=0)
    ax_v.set_facecolor("white")
    plt.setp(ax_v.get_xticklabels(), visible=False)
    ax_v.legend(handles=[
        Patch(color=C_SO, alpha=0.50,
              label=f"Sust-Op zone ≥{so:.0f} V  (cease in ≤15 min)"),
        Patch(color=C_OV, alpha=0.55,
              label=f"Anti-island OV1 ≥{ov1:.0f} V  (cease in ≤2 s)"),
    ], fontsize=7.5, loc="upper left", framealpha=0.92, edgecolor="#cccccc")

    # ═══ PANEL 2 — Active power ══════════════════════════════════════════
    # Conformance criterion: did P drop below 4% nameplate after the
    # voltage threshold was exceeded? Reactive power is not assessed here.
    ax_p.plot(t, P, color=C_P, lw=1.6, zorder=4,
              label="Active power P (kW)")
    ax_p.axhline(S,      color=C_P, lw=0.7, ls=":",  alpha=0.45, zorder=3,
                 label=f"Nameplate {S:.0f} kW")
    ax_p.axhline(S*0.04, color=C_P, lw=0.8, ls="--", alpha=0.65, zorder=3,
                 label=f"4% floor ({S*0.04:.2f} kW) — conformance threshold")
    ax_p.axhline(0, color="k", lw=0.5, zorder=3)
    ax_p.set_ylabel("Active power (kW)", fontsize=8.5, color=C_P)
    ax_p.tick_params(axis="y", colors=C_P, labelsize=8)
    ax_p.set_ylim(bottom=-0.3)
    ax_p.legend(fontsize=7.5, loc="upper left", framealpha=0.92, edgecolor="#cccccc")
    ax_p.grid(True, color=C_GRID, lw=0.5, zorder=0)
    ax_p.set_facecolor("white")

    ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_p.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_p.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax_p.tick_params(axis="x", which="major", labelsize=8, pad=3)
    ax_p.tick_params(axis="x", which="minor", length=2)
    fig.autofmt_xdate(rotation=0, ha="center")

    ax_v.set_title(
        f"Site {site_id}  ·  {manufacturer}  ·  {zoom_date}\n"
        f"Protective functions: Sustained-Op & Anti-Islanding  ·  Nameplate {S:.0f} kW",
        fontsize=10, fontweight="bold", pad=6, loc="left")
    plt.show()

# ---------------------------------------------------------------------------
# Monthly Volt-VAr scatter: Q vs V for one site-month   [was 01 cell 64 / 3.3c]
# ---------------------------------------------------------------------------
def plot_vvar_month_scatter(scatter_df, site_id, ac_capacity_kw, period_label,
                            manufacturer="", tol=None):
    """
    Q-vs-V scatter for a full site-month, every 5-min interval as a dot, with
    the AS/NZS 4777.2 required-Q curve and +/-4% tolerance band overlaid.

    Two panels: a 200-280 V / +/-100% overview, and a 230-260 V / -60..+40%
    operating-range zoom. Points are coloured by per-interval conformance
    (inside vs outside the tolerance band).

    The required-Q curve comes from `as4777_curves.vvar_required_q` (the same
    code that scored conformance_voltvar_v2) — NOT a local re-implementation.

    Parameters
    ----------
    scatter_df : DataFrame with columns V, Q_kvar, P_kW (one row per interval).
                 Build it with site_selection.pull_month_scatter().
    period_label : e.g. "2024-03" for the title.
    tol : tolerance fraction; defaults to AS4777["TOL_FRAC"].
    """
    if scatter_df.empty:
        print("No intervals to plot.")
        return None

    S   = ac_capacity_kw
    tol = AS4777["TOL_FRAC"] if tol is None else tol
    vv  = AS4777["VVAR"]

    # per-interval required Q + conformance flag (curve from the keystone)
    d = scatter_df.copy()
    d["Q_req"]     = d["V"].map(lambda v: vvar_required_q(v, S))
    d["Q_req_max"] = d["Q_req"] + tol * S
    d["Q_req_min"] = d["Q_req"] - tol * S
    d["Q_pct"]     = d["Q_kvar"] / S * 100.0
    d["nc"] = (d["Q_kvar"] > d["Q_req_max"]) | (d["Q_kvar"] < d["Q_req_min"])

    n_total = len(d)
    n_nc    = int(d["nc"].sum())
    pct_nc  = 100.0 * n_nc / n_total if n_total else 0.0
    ok_mask = ~d["nc"]

    # reference curve on a fine grid
    V_grid    = np.linspace(200, 280, 800)
    Q_req_pct = np.array([vvar_required_q(v, S) for v in V_grid]) / S * 100.0
    Q_max_pct = Q_req_pct + tol * 100.0
    Q_min_pct = Q_req_pct - tol * 100.0

    C_OK, C_NC, C_REF = "#1565c0", "#c62828", "#f59e0b"

    def _draw(ax, x_lo, x_hi, y_lo, y_hi, x_step, y_step, subtitle):
        ax.scatter(d.loc[ok_mask, "V"], d.loc[ok_mask, "Q_pct"],
                   s=4, alpha=0.20, color=C_OK, zorder=3,
                   label=f"Conforming ({int(ok_mask.sum()):,})")
        ax.scatter(d.loc[~ok_mask, "V"], d.loc[~ok_mask, "Q_pct"],
                   s=6, alpha=0.45, color=C_NC, zorder=4,
                   label=f"Non-conforming ({n_nc:,}, {pct_nc:.1f}%)")
        ax.plot(V_grid, Q_req_pct, color=C_REF, lw=1.8, zorder=5,
                label="Required Q (AS4777 curve)")
        ax.fill_between(V_grid, Q_min_pct, Q_max_pct, color=C_REF, alpha=0.20,
                        linewidth=0, zorder=2,
                        label=f"+/-{tol*100:.0f}% S_rated tolerance")
        ax.axhline(0, color="k", lw=0.5, zorder=1)
        for vx, lbl in [(vv["V1"], f"V1 {vv['V1']:.0f}"), (vv["V2"], f"V2 {vv['V2']:.0f}"),
                        (vv["V3"], f"V3 {vv['V3']:.0f}"), (vv["V4"], f"V4 {vv['V4']:.0f}")]:
            if x_lo <= vx <= x_hi:
                ax.axvline(vx, color="grey", lw=0.6, ls=":", zorder=1)
                ax.text(vx + 0.4, y_hi * 0.95, lbl, fontsize=6, color="grey",
                        va="top", ha="left")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(range(x_lo, x_hi + 1, x_step))
        ax.set_yticks(range(y_lo, y_hi + 1, y_step))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
        ax.set_xlabel("Voltage (V)", fontsize=9)
        ax.set_ylabel("Reactive power (% S_rated)\n+ = supplying,  - = absorbing", fontsize=9)
        ax.set_title(
            f"Site {site_id}  ·  {manufacturer}  ·  {period_label}  ·  {subtitle}\n"
            f"Volt-VAr response vs AS/NZS 4777.2:2020  ·  Nameplate {S:.0f} kW AC",
            fontsize=9, fontweight="bold", loc="left")
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.92, edgecolor="#cccccc")
        ax.grid(color="#ebebeb", lw=0.5)
        ax.set_facecolor("white")

    fig1, ax1 = plt.subplots(figsize=(9, 5.5), dpi=130)
    _draw(ax1, 200, 280, -100, 100, 5, 20, "overview")
    plt.tight_layout(); plt.show()

    fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=130)
    _draw(ax2, 230, 260, -60, 40, 2, 10, "operating range zoom")
    plt.tight_layout(); plt.show()

    print(f"Non-conforming: {n_nc:,} / {n_total:,} intervals ({pct_nc:.1f}%)")
    return d
