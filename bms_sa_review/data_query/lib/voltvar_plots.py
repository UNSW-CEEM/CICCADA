"""Plots for Volt-VAr symptom and attribution evidence."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Patch
import seaborn as sns
import pytz

FIXED_OFFSET = pytz.FixedOffset(600)  # AEST = UTC+10


PURPLE = "#6d28d9"
RED = "#b91c1c"
GRID = "#e5e7eb"


def plot_evidence_tiers(tiers):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
    bars = ax.barh(tiers.evidence_tier, tiers.n_intervals, color=PURPLE)
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set(xlabel="Intervals", title="Volt-VAr attribution evidence funnel")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_group_energy(frame, group_col, title):
    d = frame.sort_values("attributed_kwh")
    fig, ax = plt.subplots(figsize=(9, max(3, .35*len(d))), dpi=130)
    bars = ax.barh(d[group_col].astype(str), d.attributed_kwh, color=PURPLE)
    ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
    ax.set(xlabel="Attributed energy (kWh; counterfactual covered)", title=title)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_concentration(ranked, shares):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.plot(ranked.site_share_pct, ranked.cumulative_energy_share_pct,
            color=PURPLE, lw=2)
    ax.plot([0,100],[0,100], ls="--", color="grey")
    for x, y in shares.items():
        ax.scatter([x],[y], color=RED)
        ax.annotate(f"Top {x}%: {y:.1f}%", (x,y), xytext=(5,5),
                    textcoords="offset points", fontsize=8)
    ax.set(xlim=(0,100), ylim=(0,100),
           xlabel="Affected sites ranked by attributed energy (%)",
           ylabel="Cumulative attributed energy (%)")
    ax.grid(color=GRID)
    plt.tight_layout()
    return fig


def plot_site_intervals(frame, site_id, year):
    if frame.empty:
        print("No intervals returned")
        return None
    d = frame.copy()
    d["time"] = pd.to_datetime(d.t_stamp) + pd.Timedelta(hours=10)
    fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True,dpi=130)
    axes[0].plot(d.time,d.V,color="#b45309",label="Average site voltage")
    axes[0].axhline(240,ls="--",color="grey")
    axes[0].axhline(253,ls="--",color="grey")
    axes[0].set_ylabel("V")
    axes[0].legend()
    axes[1].plot(d.time,d.P_kW,label="Measured P")
    axes[1].plot(d.time,d.uncurtailed_P,ls="--",label="Counterfactual P")
    axes[1].plot(d.time,d.pmax_measured_q_kw,ls=":",label="P headroom from measured Q")
    axes[1].plot(d.time,d.Q_kvar.abs(),label="|Q|",alpha=.8)
    symptom = d.apparent_limit_symptom.fillna(False)
    axes[1].scatter(d.loc[symptom,"time"],d.loc[symptom,"P_kW"],s=15,
                    color=RED,label="Apparent-limit symptom")
    axes[1].set_ylabel("kW / kvar")
    axes[1].legend(ncol=2,fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for ax in axes: ax.grid(color=GRID)
    fig.suptitle(f"Site {site_id}, {year}: Volt-VAr attribution inputs")
    plt.tight_layout()
    return fig


def plot_pq_circle(frame, site_id):
    if frame.empty:
        return None
    limit = frame.empirical_limit.median()
    theta = np.linspace(-np.pi/2,np.pi/2,300)
    fig, ax = plt.subplots(figsize=(6,6),dpi=130)
    ax.plot(limit*np.cos(theta),limit*np.sin(theta),color="black",label="Empirical limit")
    sc=ax.scatter(frame.P_kW,frame.Q_kvar,c=frame.V,s=8,alpha=.5,cmap="plasma")
    ax.set(xlabel="P (kW)",ylabel="Q (kvar; negative = absorbing)",
           title=f"Site {site_id}: operating points")
    ax.set_aspect("equal")
    ax.grid(color=GRID)
    fig.colorbar(sc,ax=ax,label="Average site voltage (V)")
    ax.legend()
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════
# Legacy-style plot functions (adapted from the pre-evidence-tier codebase)
# ═════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# Single-day curtailment plot  (was plot_varcurt_day)
# ─────────────────────────────────────────────────────────────

def plot_varcurt_day(day_df, site_id, date_str, ac_capacity_kw, s_limit,
                     params, restrict_to_peak_hours=False):
    """
    Full-day plot: voltage panel + power panel showing S, P, |Q|,
    P_potential, P_max_given_Q, and the Volt-VAr curtailment area
    (shaded red).

    Parameters
    ----------
    day_df : DataFrame
        Output of fetch_day_data(). Must contain: t_stamp, V, P_kW,
        Q_kvar, P_potential_kW.
    site_id : int
    date_str : str   ('YYYY-MM-DD', for titles only)
    ac_capacity_kw : float
    s_limit : float  (empirical apparent-power limit)
    params : VoltVarParams
    restrict_to_peak_hours : bool
    """
    from bms_sa_review.shared.ciccada_config import AS4777

    if day_df.empty:
        print(f"No data for site {site_id} on {date_str}.")
        return

    df = day_df.copy()
    df["t"] = (pd.to_datetime(df["t_stamp"])
               .dt.tz_localize("UTC").dt.tz_convert(FIXED_OFFSET)
               .dt.tz_localize(None))
    df["S_apparent"]    = np.sqrt(df["P_kW"]**2 + df["Q_kvar"]**2)
    df["P_max_given_Q"] = np.sqrt(np.clip(s_limit**2 - df["Q_kvar"]**2, 0, None))

    V_LOW      = params.v_low
    V_HIGH     = params.v_high
    PEAK_START = params.peak_hour_start
    PEAK_END   = params.peak_hour_end

    eligible   = (df["V"] > V_LOW) & (df["V"] < V_HIGH)
    peak_solar = (df["t"].dt.hour >= PEAK_START) & (df["t"].dt.hour < PEAK_END)

    curtail_signal = (
        eligible & (df["Q_kvar"] < 0) &
        df["P_potential_kW"].notna() &
        (df["P_potential_kW"] > df["P_max_given_Q"] + 0.02)
    )
    curtailed = curtail_signal & peak_solar if restrict_to_peak_hours else curtail_signal

    df["varcurt_kW"] = np.where(
        curtailed,
        np.maximum(0, df["P_potential_kW"] - df["P_max_given_Q"]),
        0,
    )
    df["varcurt_kW_allday"] = np.where(
        curtail_signal,
        np.maximum(0, df["P_potential_kW"] - df["P_max_given_Q"]),
        0,
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7.5), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [1.3, 2]},
    )
    pal = sns.color_palette("deep")
    C_V, C_P, C_Q = "#b45309", pal[2], pal[0]
    C_S, C_POT, C_CURT = "#1a1a1a", "#e8702a", "#c62828"

    for ax in axes:
        ax.fill_between(
            df["t"], 0, 1, where=eligible.values,
            transform=ax.get_xaxis_transform(),
            color="#7c3aed", alpha=0.06, linewidth=0, zorder=0,
        )

    # Top: voltage
    ax = axes[0]
    v_line,     = ax.plot(df["t"], df["V"], color=C_V, lw=1.4, label="Measured voltage")
    v_low_line  = ax.axhline(V_LOW,  color=C_V, lw=0.7, ls=":",  alpha=0.6,
                             label=f"V_LOW = {V_LOW:.1f} V")
    v_high_line = ax.axhline(V_HIGH, color=C_V, lw=0.9, ls="--", alpha=0.8,
                             label=f"V_HIGH = {V_HIGH:.1f} V")
    eligible_patch = Patch(
        facecolor="#7c3aed", alpha=0.15, edgecolor="none",
        label=f"Volt-VAr eligible ({V_LOW:.1f}~{V_HIGH:.1f} V)",
    )
    ax.set_ylabel("Voltage (V)", color=C_V)
    ax.tick_params(axis="y", colors=C_V)
    ax.legend(handles=[v_line, v_low_line, v_high_line, eligible_patch],
              fontsize=7.5, loc="upper left", framealpha=0.92)

    # Bottom: power
    ax = axes[1]
    curt_label = ("Volt-VAr curtailment (peak-hour only)"
                  if restrict_to_peak_hours else "Volt-VAr curtailment (full day)")
    ax.fill_between(
        df["t"], df["P_max_given_Q"], df["P_potential_kW"],
        where=curtailed.values, color=C_CURT, alpha=0.30, zorder=2,
        label=curt_label,
    )
    if df["P_potential_kW"].notna().any():
        ax.plot(df["t"], df["P_potential_kW"], color=C_POT, lw=1.6, ls="--",
                alpha=0.9, zorder=4, label="P potential (uncurtailed, clear-sky)")
    ax.plot(df["t"], df["P_max_given_Q"], color="#888888", lw=1.2, ls=":",
            zorder=3, label="P max given measured Q  (√(S²−Q²))")
    ax.plot(df["t"], df["S_apparent"], color=C_S, lw=2.0, zorder=5,
            label="Apparent power S = √(P²+Q²)")
    ax.plot(df["t"], df["P_kW"], color=C_P, lw=1.6, zorder=4,
            label="Active power P (measured)")
    ax.plot(df["t"], df["Q_kvar"].abs(), color=C_Q, lw=1.4, zorder=4,
            label="Reactive power |Q| (absorbing)")
    ax.axhline(s_limit, color=C_S, lw=0.8, ls=":", alpha=0.5,
               label=f"S_limit = {s_limit:.1f} kVA")
    ax.set_ylabel("Power (kW / kvar)")
    ax.legend(fontsize=7.3, loc="upper left", framealpha=0.92, ncol=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate(rotation=0, ha="center")

    total_kwh        = (df["varcurt_kW"]        * AS4777["INTERVAL_H"]).sum()
    total_kwh_allday = (df["varcurt_kW_allday"] * AS4777["INTERVAL_H"]).sum()

    window_note = (f"{total_kwh:.2f} kWh in peak window"
                   if restrict_to_peak_hours else f"{total_kwh:.2f} kWh all day")
    fig.suptitle(
        f"Site {site_id} | Volt-VAr curtailment, {date_str}  |  {window_note}\n"
        f"Nameplate AC capacity = {ac_capacity_kw:.1f} kW",
        fontsize=10.5, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    plt.show()

    peak_kwh = (
        total_kwh if restrict_to_peak_hours
        else (df["varcurt_kW_allday"] * AS4777["INTERVAL_H"])[peak_solar].sum()
    )
    print(f"Curtailment within peak-hour window "
          f"({PEAK_START}:00-{PEAK_END}:00): {peak_kwh:.2f} kWh")
    print(f"Curtailment across the full day:                       "
          f"{total_kwh_allday:.2f} kWh")
    return fig


# ─────────────────────────────────────────────────────────────
# Apparent-power circle plot  (was plot_apparent_power_circle)
# ─────────────────────────────────────────────────────────────

def plot_apparent_power_circle(intervals_df, site_id, s_limit, params):
    """
    P–Q scatter with apparent-power circle for one site's eligible intervals.

    Parameters
    ----------
    intervals_df : DataFrame
        Output of fetch_method_b_intervals(). Must contain: P_kW, Q_kvar,
        V, empirical_limit, rating_capacity, uncurtailed_P.
    site_id : int
    s_limit : float
    params : VoltVarParams
    """
    if intervals_df.empty:
        print("No intervals to plot.")
        return None

    varcurt = intervals_df.copy()
    cap = varcurt["rating_capacity"].median()
    tol = params.tolerance_fraction * cap

    S_apparent = np.sqrt(varcurt["P_kW"]**2 + varcurt["Q_kvar"]**2)
    on_circle  = S_apparent >= (s_limit - tol)

    low_v    = varcurt["V"] < 240
    normal_v = ~low_v

    fig = plt.figure(figsize=(7.5, 8.6), dpi=120)
    ax  = fig.add_axes([0.12, 0.12, 0.75, 0.80])
    cax = fig.add_axes([0.75, 0.35, 0.020, 0.30])

    theta = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.plot(
        s_limit * np.cos(theta), s_limit * np.sin(theta),
        color="k", lw=1.4, zorder=5, label=f"S_limit = {s_limit:.1f} kVA",
    )

    ax.scatter(
        varcurt.loc[low_v & ~on_circle, "P_kW"],
        varcurt.loc[low_v & ~on_circle, "Q_kvar"],
        color="lightgrey", s=12, alpha=0.7, zorder=2, label="V < 240 V",
    )
    ax.scatter(
        varcurt.loc[low_v & on_circle, "P_kW"],
        varcurt.loc[low_v & on_circle, "Q_kvar"],
        color="lightgrey", s=38, alpha=0.9, zorder=4,
        edgecolors="grey", linewidths=1.0,
    )

    ax.scatter(
        varcurt.loc[normal_v & ~on_circle, "P_kW"],
        varcurt.loc[normal_v & ~on_circle, "Q_kvar"],
        c=varcurt.loc[normal_v & ~on_circle, "V"],
        cmap="plasma", s=12, alpha=0.45, zorder=3, label="Within headroom",
    )

    sc = ax.scatter(
        varcurt.loc[normal_v & on_circle, "P_kW"],
        varcurt.loc[normal_v & on_circle, "Q_kvar"],
        c=varcurt.loc[normal_v & on_circle, "V"],
        cmap="plasma", s=38, alpha=0.95, zorder=6,
        edgecolors="#c62828", linewidths=1.3,
        label=f"At apparent-power limit (n={on_circle.sum()})",
    )

    if "uncurtailed_P" in varcurt.columns:
        ax.scatter(
            varcurt["uncurtailed_P"], np.zeros(len(varcurt)),
            marker="|", color="green", s=40, alpha=0.25,
            label="P potential (clear-sky)",
        )

    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Active power P (kW)")
    ax.set_ylabel("Reactive power Q (kvar)   [− = absorbing]")
    ax.set_title(
        f"Site {site_id}: operating points vs apparent-power circle\n"
        f"clear-sky | {params.v_low:.0f}~{params.v_high:.0f} V  |  "
        f"{on_circle.sum()}/{len(varcurt)} points "
        f"({on_circle.mean() * 100:.0f}%) at the limit",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.set_aspect("equal")

    cbar = fig.colorbar(sc, cax=cax, orientation="vertical")
    vmin = varcurt.loc[normal_v, "V"].min()
    vmax = varcurt.loc[normal_v, "V"].max()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
    cbar.set_label("Voltage (V)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# Concentration / Lorenz curve  (was plot_concentration_curve)
# ─────────────────────────────────────────────────────────────

def plot_concentration_curve(ranked, concentration, total_kwh,
                             title="Volt-VAr curtailment concentration across sites"):
    """
    Cumulative contribution plot.

    Parameters
    ----------
    ranked : DataFrame
        Output of vm.concentration(). Must contain:
        site_share_pct, cumulative_energy_share_pct.
    concentration : dict
        {percentile: cumulative_share_pct}.
    total_kwh : float
    """
    if ranked.empty:
        print("No sites with positive estimated curtailment.")
        return None

    n_affected = len(ranked)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=140)

    ax.plot(
        ranked["site_share_pct"],
        ranked["cumulative_energy_share_pct"],
        lw=2.4,
    )
    ax.plot([0, 100], [0, 100], ls="--", lw=1, alpha=0.5)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of affected sites, ranked by estimated curtailed energy (%)")
    ax.set_ylabel("Cumulative share of estimated curtailed energy (%)")
    ax.set_title(title, weight="bold")
    ax.grid(True, alpha=0.35)

    for x, y in concentration.items():
        ax.axvline(x, ls=":", lw=0.9, alpha=0.7)
        ax.axhline(y, ls=":", lw=0.9, alpha=0.7)
        ax.text(x + 1, y + 2, f"Top {x}% = {y:.1f}%", fontsize=9, va="bottom")

    ax.text(
        0.98, 0.05,
        f"Affected sites: {n_affected:,}\n"
        f"Total estimated curtailment: {total_kwh:,.0f} kWh",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="lightgrey", alpha=0.95),
    )

    plt.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# Year-by-year summary comparison
# ─────────────────────────────────────────────────────────────

def plot_yearly_comparison(method_a_yearly, method_b_yearly):
    """
    2x2 panel comparing Method A symptoms and Method B attribution by year.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=130)
    years_a = method_a_yearly["year"].astype(str)
    years_b = method_b_yearly["year"].astype(str)

    w = 0.35
    # A. Sites
    ax = axes[0, 0]
    x_a = np.arange(len(years_a))
    ax.bar(x_a - w/2, method_a_yearly["eligible_sites"], w,
           label="Eligible", color="#93c5fd")
    ax.bar(x_a + w/2, method_a_yearly["symptom_sites"], w,
           label="Symptom", color=PURPLE)
    ax.set_xticks(x_a); ax.set_xticklabels(years_a)
    ax.set_ylabel("Sites"); ax.set_title("A. Method A: sites", weight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

    # B. Tier 4 sites
    ax = axes[0, 1]
    x_b = np.arange(len(years_b))
    ax.bar(x_b - w/2, method_b_yearly["eligible_sites"], w,
           label="Eligible", color="#93c5fd")
    t4 = method_b_yearly.get("tier4_affected_sites", 0)
    ax.bar(x_b + w/2, t4, w, label="Tier 4", color=PURPLE)
    ax.set_xticks(x_b); ax.set_xticklabels(years_b)
    ax.set_ylabel("Sites"); ax.set_title("B. Method B: Tier 4 sites", weight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

    # C. Attributed kWh
    ax = axes[1, 0]
    bars = ax.bar(years_b, method_b_yearly["attributed_measured_q_kwh"], color=PURPLE)
    ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
    ax.set_ylabel("kWh"); ax.set_title("C. Attributed energy", weight="bold")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

    # D. % of covered potential
    ax = axes[1, 1]
    pct = method_b_yearly["attributed_pct_of_covered_potential"]
    bars = ax.bar(years_b, pct, color=RED)
    ax.bar_label(bars, fmt="{:.4f}%", padding=3, fontsize=8)
    ax.set_ylabel("%"); ax.set_title("D. Attributed / covered potential", weight="bold")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

    fig.suptitle("Volt-VAr curtailment: year-by-year summary",
                 fontsize=14, weight="bold", y=1.01)
    plt.tight_layout()
    return fig
