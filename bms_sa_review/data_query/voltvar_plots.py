"""
Visualisation functions for Volt-VAr curtailment analysis.

All functions receive DataFrames (already fetched) and explicit parameters.

Standard constants (AS4777, FIXED_OFFSET) are imported from ciccada_config.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns

from ciccada_config import AS4777, FIXED_OFFSET

# ═════════════════════════════════════════════════════════════
# Single-day curtailment plot
# ═════════════════════════════════════════════════════════════

def plot_varcurt_day(day_df, 
                     site_id, 
                     date_str, 
                     ac_capacity_kw, 
                     s_limit,
                     params, 
                     restrict_to_peak_hours=False):
    """
    Full-day plot: voltage panel + power panel showing S, P, |Q|, P_potential,
    P_max_given_Q, and the Volt-VAr curtailment area (shaded red).

    Parameters
    ----------
    day_df : pd.DataFrame
        Output of voltvar_queries.fetch_day_data(). Must contain columns:
        t_stamp, V, P_kW, Q_kvar, P_potential_kW.
    site_id : int
    date_str : str
        'YYYY-MM-DD', used for titles only.
    ac_capacity_kw : float
    s_limit : float
    params : dict
        The PARAMS dict (V_LOW, V_HIGH, PEAK_HOUR_START, PEAK_HOUR_END).
    restrict_to_peak_hours : bool
        If True, red shading and kWh total only count peak-hour curtailment.
    """
    if day_df.empty:
        print(f"No data for site {site_id} on {date_str}.")
        return

    # Derived columns
    df = day_df.copy()
    df["t"] = (pd.to_datetime(df["t_stamp"])
               .dt.tz_localize("UTC").dt.tz_convert(FIXED_OFFSET)
               .dt.tz_localize(None))
    df["S_apparent"]    = np.sqrt(df["P_kW"]**2 + df["Q_kvar"]**2)
    df["P_max_given_Q"] = np.sqrt(np.clip(s_limit**2 - df["Q_kvar"]**2, 0, None))

    V_LOW      = params["V_LOW"]
    V_HIGH     = params["V_HIGH"]
    PEAK_START = params["PEAK_HOUR_START"]
    PEAK_END   = params["PEAK_HOUR_END"]

    eligible   = (df["V"] > V_LOW) & (df["V"] < V_HIGH)
    peak_solar = (df["t"].dt.hour >= PEAK_START) & (df["t"].dt.hour < PEAK_END)

    # Curtailment criteria
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

    # Plot
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7.5), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [1.3, 2]},
    )
    pal = sns.color_palette("deep")
    C_V, C_P, C_Q = "#b45309", pal[2], pal[0]
    C_S, C_POT, C_CURT = "#1a1a1a", "#e8702a", "#c62828"

    # Violet eligible band on both panels
    for ax in axes:
        ax.fill_between(
            df["t"], 0, 1, where=eligible.values,
            transform=ax.get_xaxis_transform(),
            color="#7c3aed", alpha=0.06, linewidth=0, zorder=0,
        )

    # Top panel: voltage
    ax = axes[0]
    v_line,     = ax.plot(df["t"], df["V"], color=C_V, lw=1.4, label="Measured voltage")
    v_low_line  = ax.axhline(V_LOW,  color=C_V, lw=0.7, ls=":",  alpha=0.6,
                             label=f"V_LOW = {V_LOW:.1f} V")
    v_high_line = ax.axhline(V_HIGH, color=C_V, lw=0.9, ls="--", alpha=0.8,
                             label=f"V_HIGH = {V_HIGH:.1f} V")
    eligible_patch = Patch(
        facecolor="#7c3aed", alpha=0.15, edgecolor="none",
        label=f"Volt-VAr eligible voltage range ({V_LOW:.1f}~{V_HIGH:.1f} V)",
    )
    ax.set_ylabel("Voltage (V)", color=C_V)
    ax.tick_params(axis="y", colors=C_V)
    ax.legend(
        handles=[v_line, v_low_line, v_high_line, eligible_patch],
        fontsize=7.5, loc="upper left", framealpha=0.92,
    )

    # Bottom panel: power
    ax = axes[1]
    curt_label = ("Volt-VAr curtailment (peak-hour only)"
                  if restrict_to_peak_hours else
                  "Volt-VAr curtailment (full day)")
    ax.fill_between(
        df["t"], df["P_max_given_Q"], df["P_potential_kW"],
        where=curtailed.values, color=C_CURT, alpha=0.30, zorder=2,
        label=curt_label,
    )
    if df["P_potential_kW"].notna().any():
        ax.plot(df["t"], df["P_potential_kW"], color=C_POT, lw=1.6, ls="--",
                alpha=0.9, zorder=4, label="P potential (uncurtailed, clear-sky)")
    ax.plot(df["t"], df["P_max_given_Q"], color="#888888", lw=1.2, ls=":",
            zorder=3, label="P max given measured Q  (√(S_limit² − Q²))")
    ax.plot(df["t"], df["S_apparent"], color=C_S, lw=2.0, zorder=5,
            label="Apparent power S = √(P²+Q²)")
    ax.plot(df["t"], df["P_kW"], color=C_P, lw=1.6, zorder=4,
            label="Active power P (measured)")
    ax.plot(df["t"], df["Q_kvar"].abs(), color=C_Q, lw=1.4, zorder=4,
            label="Reactive power |Q| (absorbing)")
    ax.axhline(s_limit, color=C_S, lw=0.8, ls=":", alpha=0.5,
               label=f"S_limit (S_99) = {s_limit:.1f} kVA")
    ax.set_ylabel("Power (kW / kvar)")
    ax.legend(fontsize=7.3, loc="upper left", framealpha=0.92, ncol=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate(rotation=0, ha="center")

    # Title and totals
    total_kwh        = (df["varcurt_kW"]        * AS4777["INTERVAL_H"]).sum()
    total_kwh_allday = (df["varcurt_kW_allday"] * AS4777["INTERVAL_H"]).sum()

    window_note = (f"{total_kwh:.2f} kWh in peak window"
                   if restrict_to_peak_hours else
                   f"{total_kwh:.2f} kWh all day")
    fig.suptitle(
        f"Site {site_id} | Volt-VAr curtailment, {date_str}  |  {window_note}\n"
        f"Nameplate AC capacity = {ac_capacity_kw:.1f} kW",
        fontsize=10.5, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # Peak-hour total (always printed for comparison)
    peak_kwh = (
        total_kwh if restrict_to_peak_hours
        else (df["varcurt_kW_allday"] * AS4777["INTERVAL_H"])[peak_solar].sum()
    )
    print(f"Curtailment within peak-hour window "
          f"({PEAK_START}:00-{PEAK_END}:00): {peak_kwh:.2f} kWh")
    print(f"Curtailment across the full day:                       "
          f"{total_kwh_allday:.2f} kWh")


# ═════════════════════════════════════════════════════════════
# Apparent-power circle plot
# ═════════════════════════════════════════════════════════════

def plot_apparent_power_circle(varcurt_df, site_id, s_limit, params):
    """
    P–Q scatter with apparent-power circle for one site's flagged intervals.

    Parameters
    ----------
    varcurt_df : pd.DataFrame
        Method B output for one site. Must contain columns:
        P_meas_kW, Q_meas_kvar, V_max, s_limit, ac_capacity_kw, P_potential_kW.
    site_id : int
    s_limit : float
    params : dict
        The PARAMS dict (S_TOL_FRAC, V_LOW, V_HIGH).
    """
    if varcurt_df.empty:
        print("No intervals to plot.")
        return

    varcurt = varcurt_df.copy()
    cap = varcurt["ac_capacity_kw"].median()
    tol = params["S_TOL_FRAC"] * cap

    S_apparent = np.sqrt(varcurt["P_meas_kW"]**2 + varcurt["Q_meas_kvar"]**2)
    on_circle  = S_apparent >= (s_limit - tol)

    low_v    = varcurt["V_max"] < 240
    normal_v = ~low_v

    # Figure layout
    fig = plt.figure(figsize=(7.5, 8.6), dpi=120)
    ax  = fig.add_axes([0.12, 0.12, 0.75, 0.80])
    cax = fig.add_axes([0.75, 0.35, 0.020, 0.30])

    # Apparent-power circle
    theta = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.plot(
        s_limit * np.cos(theta), s_limit * np.sin(theta),
        color="k", lw=1.4, zorder=5, label=f"S_limit = {s_limit:.1f} kVA",
    )

    # Low-voltage points (< 240 V) in grey
    ax.scatter(
        varcurt.loc[low_v & ~on_circle, "P_meas_kW"],
        varcurt.loc[low_v & ~on_circle, "Q_meas_kvar"],
        color="lightgrey", s=12, alpha=0.7, zorder=2, label="V < 240 V",
    )
    ax.scatter(
        varcurt.loc[low_v & on_circle, "P_meas_kW"],
        varcurt.loc[low_v & on_circle, "Q_meas_kvar"],
        color="lightgrey", s=38, alpha=0.9, zorder=4,
        edgecolors="grey", linewidths=1.0,
    )

    # Coloured points not at limit
    ax.scatter(
        varcurt.loc[normal_v & ~on_circle, "P_meas_kW"],
        varcurt.loc[normal_v & ~on_circle, "Q_meas_kvar"],
        c=varcurt.loc[normal_v & ~on_circle, "V_max"],
        cmap="plasma", s=12, alpha=0.45, zorder=3, label="Within headroom",
    )

    # Coloured points at limit
    sc = ax.scatter(
        varcurt.loc[normal_v & on_circle, "P_meas_kW"],
        varcurt.loc[normal_v & on_circle, "Q_meas_kvar"],
        c=varcurt.loc[normal_v & on_circle, "V_max"],
        cmap="plasma", s=38, alpha=0.95, zorder=6,
        edgecolors="#c62828", linewidths=1.3,
        label=f"At apparent-power limit (n={on_circle.sum()})",
    )

    # Potential power markers
    ax.scatter(
        varcurt["P_potential_kW"], np.zeros(len(varcurt)),
        marker="|", color="green", s=40, alpha=0.25,
        label="P potential (clear-sky)",
    )

    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Active power P (kW)")
    ax.set_ylabel("Reactive power Q (kvar)   [− = absorbing]")
    ax.set_title(
        f"Site {site_id}: operating points vs apparent-power circle\n"
        f"clear-sky | {params['V_LOW']:.0f}~{params['V_HIGH']:.0f} V  |  "
        f"{on_circle.sum()}/{len(varcurt)} points "
        f"({on_circle.mean() * 100:.0f}%) at the limit",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.set_aspect("equal")

    # Compact vertical colorbar
    cbar = fig.colorbar(sc, cax=cax, orientation="vertical")
    vmin = varcurt.loc[normal_v, "V_max"].min()
    vmax = varcurt.loc[normal_v, "V_max"].max()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
    cbar.set_label("Voltage (V)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.show()



# ═════════════════════════════════════════════════════════════
# Concentration / Lorenz curve
# ═════════════════════════════════════════════════════════════

def plot_concentration_curve(curt_rank, 
                             concentration, 
                             total_kwh,
                             title="Volt-VAr curtailment concentration across sites"):
    """
    Cumulative contribution plot showing how curtailment is distributed
    across affected sites.

    Parameters
    ----------
    curt_rank : pd.DataFrame
        Output of voltvar_metrics.compute_concentration().
        Must contain: share_of_affected_sites_pct, cum_share_pct.
    concentration : dict
        {percentile: cumulative_share_pct} from compute_concentration().
    total_kwh : float
        Total estimated curtailment across positive sites.
    title : str
    """
    if curt_rank.empty:
        print("No sites with positive estimated curtailment.")
        return

    n_affected = len(curt_rank)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=140)

    ax.plot(
        curt_rank["share_of_affected_sites_pct"],
        curt_rank["cum_share_pct"],
        lw=2.4,
    )
    ax.plot([0, 100], [0, 100], ls="--", lw=1, alpha=0.5)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of affected sites, ranked by estimated curtailed energy (%)")
    ax.set_ylabel("Cumulative share of estimated curtailed energy (%)")
    ax.set_title(title, weight="bold")
    ax.grid(True, alpha=0.35)

    # Annotate thresholds
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