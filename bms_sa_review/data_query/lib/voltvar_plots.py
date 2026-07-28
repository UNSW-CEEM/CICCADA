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

# ═════════════════════════════════════════════════════════════════════════
# Fleet summary volt-var method A vs method B comparison
# ═════════════════════════════════════════════════════════════════════════

def plot_fleet_summary(method_a_enriched, method_a_yearly,
                       method_b_enriched, method_b_yearly, params,
                       eligible_context=None, all_context=None):
    """
    One concise 2x2 figure contrasting Method A (screening proxy) and
    Method B (counterfactual attribution) at fleet scale.

      A. Energy cascade — potential generation -> Method A -> Method B (log x)
      B. Method A vs Method B attributed energy, by year (log y)
      C. Method B evidence-tier funnel (tier 1 -> tier 4 intervals)
      D. Curtailment intensity — share of potential generation (%)

    eligible_context / all_context are optional; if omitted, the denominator
    bars in panels A and D are skipped.
    """
    def _kwh(x):
        if x >= 1e6:  return f"{x/1e6:.1f}M"
        if x >= 1e3:  return f"{x/1e3:.1f}k"
        return f"{x:.0f}"

    # ── Totals ──────────────────────────────────────────────────
    a_proxy = method_a_enriched.headroom_displacement_kwh.sum()
    b_attr  = method_b_enriched.attributed_measured_q_kwh.sum()
    cov_pot = method_b_enriched.covered_potential_kwh.sum()
    all_pot = all_context.all_potential_kWh.sum() if all_context is not None else None
    elig_pot = eligible_context.eligible_potential_kWh.sum() if eligible_context is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=130)

    # ── A. Energy cascade (log x) ───────────────────────────────
    ax = axes[0, 0]
    labels, values, colors = [], [], []
    if all_pot is not None:
        labels.append("All potential\ngeneration");     values.append(all_pot);  colors.append("#9ca3af")
    if elig_pot is not None:
        labels.append("Eligible potential\ngeneration"); values.append(elig_pot); colors.append("#6b7280")
    labels.append("Method A\nproxy");        values.append(a_proxy); colors.append(PURPLE)
    labels.append("Method B\nattribution");  values.append(b_attr);  colors.append(RED)

    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    for yi, v in zip(y, values):
        ax.text(v * 1.15, yi, _kwh(v) + " kWh", va="center", fontsize=8.5)
    ax.set_xlim(right=max(values) * 4)
    ax.set(xlabel="Energy (kWh, log scale)",
           title="A. Energy cascade: potential → curtailment")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

    # ── B. Method A vs B by year (log y grouped bars) ───────────
    ax = axes[0, 1]
    yrs = method_a_yearly["year"].astype(str).tolist()
    x = np.arange(len(yrs)); w = 0.38
    a_by = method_a_yearly["headroom_displacement_proxy_kwh"].values
    b_by = (method_b_yearly.set_index("year")
            .reindex(method_a_yearly["year"])
            ["attributed_measured_q_kwh"].values)
    ba = ax.bar(x - w/2, a_by, w, color=PURPLE, label="Method A proxy")
    bb = ax.bar(x + w/2, b_by, w, color=RED,    label="Method B attribution")
    ax.bar_label(ba, fmt="{:,.0f}", padding=2, fontsize=7.5)
    ax.bar_label(bb, fmt="{:,.0f}", padding=2, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(yrs)
    ax.set(ylabel="Curtailed energy (kWh, log)",
           title="B. Method A vs B, by year")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

    # ── C. Evidence tier funnel ─────────────────────────────────
    ax = axes[1, 0]
    tier_cols = ["tier1_absorbing_intervals", "tier2_apparent_limit_intervals",
                 "tier3_counterfactual_above_headroom_intervals",
                 "tier4_attributed_intervals"]
    tier_names = ["T1  absorbing (Q<0)", "T2  at apparent limit",
                  "T3  counterfactual > headroom", "T4  attributed"]
    tier_vals = [method_b_yearly[c].sum() for c in tier_cols]
    shades = ["#c4b5fd", "#a78bfa", "#8b5cf6", PURPLE]
    yy = np.arange(len(tier_names))[::-1]
    tb = ax.barh(yy, tier_vals, color=shades)
    ax.set_yticks(yy); ax.set_yticklabels(tier_names, fontsize=9)
    for yi, v in zip(yy, tier_vals):
        ax.text(v, yi, f" {v:,.0f}", va="center", fontsize=8.5)
    ax.set_xlim(right=max(tier_vals) * 1.18)
    ax.set(xlabel="Intervals",
           title="C. Method B evidence-tier funnel")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

    # ── D. Curtailment intensity (% of potential) ───────────────
    ax = axes[1, 1]
    ilabels, ivals, icolors = [], [], []
    if elig_pot:
        ilabels.append("Method A /\neligible potential"); ivals.append(100*a_proxy/elig_pot); icolors.append(PURPLE)
    if all_pot:
        ilabels.append("Method A /\nall potential");      ivals.append(100*a_proxy/all_pot);  icolors.append("#a78bfa")
    if cov_pot:
        ilabels.append("Method B /\ncovered potential");  ivals.append(100*b_attr/cov_pot);   icolors.append(RED)
    xb = np.arange(len(ilabels))
    ib = ax.bar(xb, ivals, color=icolors, width=0.6)
    ax.bar_label(ib, fmt="{:.4f}%", padding=3, fontsize=8.5)
    ax.set_xticks(xb); ax.set_xticklabels(ilabels, fontsize=8.5)
    ax.set(ylabel="Share of potential generation (%)",
           title="D. Curtailment intensity")
    ax.set_ylim(top=max(ivals) * 1.25 if ivals else 1)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

    fig.suptitle(
        f"Fleet-wide Volt-VAr curtailment  |  {list(params.years)}  |  "
        f"Method A proxy {_kwh(a_proxy)} kWh  vs  Method B attribution {_kwh(b_attr)} kWh "
        f"({a_proxy/b_attr:.1f}×)",
        fontsize=13, weight="bold", y=1.01)
    plt.tight_layout()
    return fig

# ═════════════════════════════════════════════════════════════════════════
# Method A vs B comparison plots + combined legacy Method A detail figure.
# ═════════════════════════════════════════════════════════════════════════

def _fmt_num(x, decimals=1):
    if pd.isna(x): return "n/a"
    if abs(x) >= 1_000_000: return f"{x/1_000_000:.{decimals}f}M"
    if abs(x) >= 1_000:     return f"{x/1_000:.{decimals}f}k"
    if abs(x) >= 10:        return f"{x:,.0f}"
    if abs(x) >= 1:         return f"{x:,.1f}"
    if abs(x) >= 0.01:      return f"{x:,.3f}"
    return f"{x:,.5f}"


def _fmt_pct(x, decimals=4):
    if pd.isna(x): return "n/a"
    if abs(x) >= 10: return f"{x:.1f}%"
    if abs(x) >= 1:  return f"{x:.2f}%"
    return f"{x:.{decimals}f}%"


# ─────────────────────────────────────────────────────────────
# Side-by-side DNSP breakdown: Method A proxy vs Method B attribution
# ─────────────────────────────────────────────────────────────

def plot_group_energy_compare(a_by, b_by, group_col,
                              suptitle="Volt-VAr curtailment by DNSP"):
    """
    Two panels sharing DNSP order: Method A proxy energy (left, purple) and
    Method B attributed energy (right, red).

    a_by : output of vm.group_breakdown_a (has 'proxy_kwh')
    b_by : output of vm.group_breakdown   (has 'attributed_kwh')
    """
    merged = (a_by[[group_col, "proxy_kwh"]]
              .merge(b_by[[group_col, "attributed_kwh"]], on=group_col, how="outer")
              .fillna(0)
              .sort_values("attributed_kwh"))
    order = merged[group_col].astype(str).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.5, .42 * len(merged))),
                             dpi=130, sharey=True)

    b1 = axes[0].barh(order, merged["proxy_kwh"], color=PURPLE)
    axes[0].bar_label(b1, fmt="{:,.0f}", padding=3, fontsize=8)
    axes[0].set(xlabel="Method A proxy energy (kWh)",
                title="A. Method A: Apparent-limit proxy")
    axes[0].grid(axis="x", color=GRID); axes[0].set_axisbelow(True)

    b2 = axes[1].barh(order, merged["attributed_kwh"], color=RED)
    axes[1].bar_label(b2, fmt="{:,.0f}", padding=3, fontsize=8)
    axes[1].set(xlabel="Method B attributed energy (kWh)",
                title="B. Method B: Counterfactual attribution")
    axes[1].grid(axis="x", color=GRID); axes[1].set_axisbelow(True)

    fig.suptitle(suptitle, fontsize=13, weight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Side-by-side concentration / Lorenz: Method A vs Method B
# ─────────────────────────────────────────────────────────────

def plot_concentration_compare(a_ranked, a_shares, a_total,
                               b_ranked, b_shares, b_total):
    """
    Two Lorenz panels: Method A proxy (left, purple) and Method B attribution
    (right, red). Each *_ranked is a vm.concentration() output with
    site_share_pct / cumulative_energy_share_pct; *_shares its dict; *_total
    the summed kWh for the annotation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=130)

    for ax, ranked, shares, total, colour, name in [
        (axes[0], a_ranked, a_shares, a_total, PURPLE, "Method A (proxy)"),
        (axes[1], b_ranked, b_shares, b_total, RED,    "Method B (attribution)"),
    ]:
        if ranked is None or ranked.empty:
            ax.text(0.5, 0.5, "No positive sites", ha="center", va="center")
            ax.set_title(name, weight="bold"); continue
        ax.plot(ranked["site_share_pct"], ranked["cumulative_energy_share_pct"],
                color=colour, lw=2.4)
        ax.plot([0, 100], [0, 100], ls="--", lw=1, color="grey", alpha=0.6)
        for x, y in shares.items():
            ax.axvline(x, ls=":", lw=0.8, alpha=0.6)
            ax.axhline(y, ls=":", lw=0.8, alpha=0.6)
            ax.text(x + 1, y + 2, f"Top {x}% = {y:.1f}%", fontsize=8)
        ax.set(xlim=(0, 100), ylim=(0, 100),
               xlabel="Share of affected sites (%)",
               ylabel="Cumulative share of curtailed energy (%)",
               title=name)
        ax.grid(True, color=GRID); ax.set_axisbelow(True)
        ax.text(0.98, 0.05,
                f"Affected sites: {len(ranked):,}\nTotal: {total:,.0f} kWh",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="lightgrey", alpha=0.95))

    fig.suptitle("Curtailment concentration across sites. Method A vs Method B",
                 fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Combined Method A fleet detail (legacy Figure 1 + Figure 2 in one block)
# ─────────────────────────────────────────────────────────────

def _hist_panel(ax, df, value_col, title, xlabel, bins, denominator_col=None):
    d = df.copy()
    mask = d[denominator_col].fillna(0) > 0 if (denominator_col in d.columns) else pd.Series(True, index=d.index)
    pos = d[mask & (d[value_col].fillna(0) > 0)]
    if pos.empty:
        ax.set_title(title, fontsize=10.5, weight="bold")
        ax.text(0.5, 0.5, "No positive values", ha="center", va="center")
        ax.set_xlabel(xlabel); return
    sns.histplot(data=pos, x=value_col, hue="year", bins=bins, element="step",
                 stat="percent", common_norm=False, fill=True, alpha=0.22,
                 linewidth=1.6, ax=ax)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10.5, weight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel("Share of positive site-years, %")
    ax.grid(axis="y", alpha=0.25)


def plot_method_a_detail(summary_by_year, overall_summary, site_year_distribution):
    """
    Legacy Method A fleet detail, combined into ONE figure:
      rows 1-2 : denominator funnels + year-by-year rates (was Figure 1)
      rows 3-5 : positive-value distribution histograms (was Figure 2)

    All Method A (symptom scan). Build the inputs with
    vm.build_method_a_context(method_a_enriched, eligible_context,
                              all_context, config.interval_h).
    """
    os0 = overall_summary.iloc[0]
    splot = pd.concat([summary_by_year.assign(year=summary_by_year["year"].astype(str)),
                       overall_summary], ignore_index=True)

    fig = plt.figure(figsize=(15, 21), dpi=120)
    gs = fig.add_gridspec(5, 2, hspace=0.42, wspace=0.22)

    # A. timestamp funnel
    ax = fig.add_subplot(gs[0, 0])
    tv = pd.DataFrame({"c": ["All\nsite-intervals", "Eligible\nsite-intervals",
                             "Flagged\nsite-intervals"],
                       "v": [os0["all_intervals"], os0["eligible_intervals"],
                             os0["flagged_intervals"]]})
    ax.barh(tv["c"][::-1], tv["v"][::-1], color=sns.color_palette("Blues")[3])
    ax.set_xscale("log"); ax.set_xlabel("5-min site-intervals (log)")
    ax.set_title("A. Timestamp funnel: all → eligible → flagged", weight="bold")
    for i, v in enumerate(tv["v"][::-1]): ax.text(v * 1.1, i, f"{v:,.0f}", va="center", fontsize=8.5)
    ax.text(0.98, 0.05,
            f"Flagged/eligible = {_fmt_pct(os0['pct_eligible_intervals_flagged'])}\n"
            f"Flagged/all = {_fmt_pct(os0['pct_all_intervals_flagged'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="lightgrey"))
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

    # B. energy funnel
    ax = fig.add_subplot(gs[0, 1])
    ev = pd.DataFrame({"c": ["All potential\ngeneration", "Eligible potential\ngeneration",
                             "Est. curtailed\nenergy"],
                       "v": [os0["all_potential_kWh"], os0["eligible_potential_kWh"],
                             os0["est_curtailed_kWh"]]})
    ax.barh(ev["c"][::-1], ev["v"][::-1], color=sns.color_palette("Reds")[3])
    ax.set_xscale("log"); ax.set_xlabel("Energy, kWh (log)")
    ax.set_title("B. Energy funnel: all → eligible → curtailed", weight="bold")
    for i, v in enumerate(ev["v"][::-1]): ax.text(v * 1.1, i, f"{_fmt_num(v)} kWh", va="center", fontsize=8.5)
    ax.text(0.98, 0.05,
            f"Curtailed/eligible = {_fmt_pct(os0['pct_eligible_potential_curtailed'])}\n"
            f"Curtailed/all = {_fmt_pct(os0['pct_all_potential_curtailed'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="lightgrey"))
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

    # C. year-by-year timestamp rates
    ax = fig.add_subplot(gs[1, 0])
    rdf = splot[["year", "pct_eligible_intervals_flagged", "pct_all_intervals_flagged"]].melt(
        id_vars="year", var_name="metric", value_name="percent")
    rdf["metric"] = rdf["metric"].map({"pct_eligible_intervals_flagged": "Flagged / eligible",
                                       "pct_all_intervals_flagged": "Flagged / all"})
    sns.pointplot(data=rdf, x="percent", y="year", hue="metric", dodge=0.35,
                  markers="o", linestyles="", ax=ax)
    ax.set_xscale("log"); ax.set_xlabel("Share of timestamps flagged (%, log)")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_pct(v)))
    ax.set_ylabel(""); ax.set_title("C. Extent of detected symptoms", weight="bold")
    ax.legend(title="", loc="lower center", fontsize=8)
    yorder = list(splot["year"])
    for _, r in rdf.iterrows():
        if pd.notna(r["percent"]) and r["percent"] > 0:
            ax.text(r["percent"] * 1.15, yorder.index(r["year"]),
                    _fmt_pct(r["percent"]), va="center", fontsize=7.5)
    ax.grid(True, color=GRID); ax.set_axisbelow(True)

    # D. year-by-year energy rates
    ax = fig.add_subplot(gs[1, 1])
    edf = splot[["year", "pct_eligible_potential_curtailed", "pct_all_potential_curtailed"]].melt(
        id_vars="year", var_name="metric", value_name="percent")
    edf["metric"] = edf["metric"].map({"pct_eligible_potential_curtailed": "Curtailed / eligible",
                                       "pct_all_potential_curtailed": "Curtailed / all"})
    sns.pointplot(data=edf, x="percent", y="year", hue="metric", dodge=0.35,
                  markers="o", linestyles="", ax=ax)
    ax.set_xscale("log"); ax.set_xlabel("Share of potential generation curtailed (%, log)")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_pct(v)))
    ax.set_ylabel(""); ax.set_title("D. Estimated energy impact", weight="bold")
    ax.legend(title="", loc="lower center", fontsize=8)
    yorder = list(splot["year"])
    for _, r in edf.iterrows():
        if pd.notna(r["percent"]) and r["percent"] > 0:
            ax.text(r["percent"] * 1.15, yorder.index(r["year"]),
                    _fmt_pct(r["percent"]), va="center", fontsize=7.5)
    ax.grid(True, color=GRID); ax.set_axisbelow(True)

    # E-J. distributions
    dist = site_year_distribution.copy()
    dist["year"] = dist["year"].astype(str)
    metrics = [
        ("pct_eligible_timestamps_flagged", "E. Extent vs eligible cases",
         "Flagged / eligible site-intervals, %", np.logspace(-3, 2, 35), "n_eligible_intervals"),
        ("pct_all_timestamps_flagged", "F. Extent vs all timestamps",
         "Flagged / all site-intervals, %", np.logspace(-5, 1, 35), "n_all_intervals"),
        ("est_curtailed_kWh", "G. Absolute curtailed energy",
         "Est. curtailed energy per site-year, kWh", np.logspace(-2, 4, 35), "n_all_intervals"),
        ("avg_est_curtailed_kW_when_flagged", "H. Avg curtailed power when flagged",
         "Avg curtailed power during flagged intervals, kW", np.logspace(-3, 2, 35), "n_flagged_intervals"),
        ("pct_eligible_potential_generation_curtailed", "I. Energy impact vs eligible gen",
         "Curtailed / eligible potential, %", np.logspace(-4, 2, 35), "eligible_potential_kWh"),
        ("pct_all_potential_generation_curtailed", "J. Energy impact vs all gen",
         "Curtailed / all potential, %", np.logspace(-6, 1, 35), "all_potential_kWh"),
    ]
    positions = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]
    for (r, c), (col, title, xlabel, bins, den) in zip(positions, metrics):
        ax = fig.add_subplot(gs[r, c])
        _hist_panel(ax, dist, col, title, xlabel, bins, den)

    return fig