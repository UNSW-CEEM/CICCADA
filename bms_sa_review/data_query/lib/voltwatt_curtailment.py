"""
Volt-Watt curtailment: single-site day plot
======================================================================

Mirrors voltvar_queries.fetch_day_data / voltvar_plots.plot_varcurt_day, but
for Volt-Watt: instead of the apparent-power circle (P/Q/S), this uses the
AS/NZS 4777.2 Volt-Watt P-ceiling and the GHI counterfactual (uncurtailed_P)
to show where generation was curtailed by the Volt-Watt response, and where
the site was outright non-conformant (measured P above the ceiling).

    fetch_voltwatt_day_data(...)     -> one AEST day of site-level telemetry
                                        + the GHI counterfactual.
    plot_voltwatt_curtailment_day(.) -> voltage panel + power panel, with
                                        curtailment (orange) and
                                        non-conformance (red) shaded.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from matplotlib.patches import Patch

from shared.as4777_curves import vw_max_p, add_tol_kw
from shared.pipeline_options import capacity_column, voltage_aggregate_sql

FIXED_OFFSET = pytz.FixedOffset(600)  # AEST = UTC+10


# ═════════════════════════════════════════════════════════════════════════
# Full-day telemetry fetch for the single-site day plot
# ═════════════════════════════════════════════════════════════════════════

def fetch_voltwatt_day_data(aq, site_id, date_str,
                            database,
                            all_uncurtailedpv_table,
                            metadata_table="meta_up23c",
                            rating_basis="ac_capacity_kw",
                            voltage_aggregation="avg"):
    """
    One full AEST day of telemetry + GHI counterfactual for a single site.

    Queries raw ts (circuit-level), aggregates to site level (matching the
    same rating_basis / voltage_aggregation used to build
    conformance_voltwattghi_v2), LEFT JOINs the uncurtailed_P counterfactual
    from all_uncurtailedpv. The Volt-Watt ceiling itself is computed
    afterwards in `plot_voltwatt_curtailment_day`, in Python.

    Parameters
    ----------
    aq : the notebook's query function
    site_id : int
    date_str : str  ('YYYY-MM-DD')
    database : str            (e.g. SAI)
    all_uncurtailedpv_table : str  (e.g. TABLES['all_uncurtailedpv'])
    metadata_table : str      (default 'meta_up23c')

    Returns: t_stamp, V, P_kW, rating_capacity, P_potential_kW
    """
    rating_col = capacity_column(rating_basis)
    voltage_sql = voltage_aggregate_sql(voltage_aggregation, "t.voltage")

    return aq(f"""
        WITH site_meta AS (
            SELECT DISTINCT circuit_id, circuit_polarity, {rating_col} AS rating_capacity
            FROM {metadata_table}
            WHERE is_pv = True AND site_id = {int(site_id)}
        ),
        meas AS (
            SELECT
                t.t_stamp,
                {voltage_sql}                                AS V,
                sum(t.power * sm.circuit_polarity) / 1000.0   AS P_kW,
                max(sm.rating_capacity)                       AS rating_capacity
            FROM ts t
            JOIN site_meta sm ON t.circuit_id = sm.circuit_id
            WHERE t.is_pv = True
              AND date(t.t_stamp + interval '10' hour) = DATE '{date_str}'
            GROUP BY t.t_stamp
        )
        SELECT
            m.t_stamp, m.V, m.P_kW, m.rating_capacity,
            u.uncurtailed_P AS P_potential_kW
        FROM meas m
        LEFT JOIN (
            SELECT site_id, t_stamp, avg(uncurtailed_P) AS uncurtailed_P
            FROM {all_uncurtailedpv_table}
            WHERE site_id = {int(site_id)}
            GROUP BY site_id, t_stamp
        ) u ON u.site_id = {int(site_id)} AND u.t_stamp = m.t_stamp
        ORDER BY m.t_stamp
    """, database=database)


# ═════════════════════════════════════════════════════════════════════════
# Plot
# ═════════════════════════════════════════════════════════════════════════

def plot_voltwatt_curtailment_day(day_df, site_id, date_str, as4777,
                                  ac_capacity_kw=None, redact_site=False):
    """
    Full-day plot: voltage panel + power panel showing measured P, the
    Volt-Watt ceiling (+ tolerance), the GHI counterfactual (P_potential_kW),
    Volt-Watt-attributed curtailment (orange, conformant but sun-limited),
    and outright non-conformance (red, measured P above the ceiling).

    Parameters
    ----------
    day_df : DataFrame
        Output of fetch_voltwatt_day_data(). Must contain: t_stamp, V,
        P_kW, rating_capacity, P_potential_kW.
    site_id : int
    date_str : str   ('YYYY-MM-DD', for titles only)
    as4777 : dict    (AS4777 config, e.g. from ciccada_config)
    ac_capacity_kw : float, optional (defaults to day_df["rating_capacity"])
    """
    if day_df.empty:
        print(f"No data for site {site_id} on {date_str}.")
        return

    df = day_df.copy()
    df["t"] = (pd.to_datetime(df["t_stamp"])
               .dt.tz_localize("UTC").dt.tz_convert(FIXED_OFFSET)
               .dt.tz_localize(None))

    S = ac_capacity_kw if ac_capacity_kw is not None else df["rating_capacity"].median()
    vw = as4777["VW"]
    tol = as4777["TOL_FRAC"]

    df["P_ceiling_kW"] = df["V"].apply(lambda v: add_tol_kw(vw_max_p(v, S), S, tol))

    V1 = vw["V1"]
    eligible = df["V"] > V1

    nonconformant = eligible & (df["P_kW"] > df["P_ceiling_kW"])
    curtailed = (
        eligible
        & ~nonconformant
        & df["P_potential_kW"].notna()
        & (df["P_potential_kW"] > df["P_ceiling_kW"])
    )

    df["curtailed_kW"] = np.where(
        curtailed, np.maximum(0, df["P_potential_kW"] - df["P_kW"]), 0,
    )
    df["excess_kW"] = np.where(
        nonconformant, np.maximum(0, df["P_kW"] - df["P_ceiling_kW"]), 0,
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7.5), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [1.3, 2]},
    )
    pal = sns.color_palette("deep")
    C_V, C_P, C_CEIL = "#b45309", pal[2], "#1a1a1a"
    C_POT, C_CURT, C_NC = "#e8702a", "#e8702a", "#c62828"

    for ax in axes:
        ax.fill_between(
            df["t"], 0, 1, where=eligible.values,
            transform=ax.get_xaxis_transform(),
            color="#4709b2", alpha=0.06, linewidth=0, zorder=0,
        )

    # Top: voltage
    ax = axes[0]
    v_line,  = ax.plot(df["t"], df["V"], color=C_V, lw=1.4, label="Measured voltage")
    v1_line  = ax.axhline(V1, color=C_V, lw=0.9, ls="--", alpha=0.8,
                          label=f"V1 = {V1:.1f} V (V-Watt start)")
    eligible_patch = Patch(
        facecolor="#4709b2", alpha=0.15, edgecolor="none",
        label=f"V-Watt active (V > {V1:.1f} V)",
    )
    ax.set_ylabel("Voltage (V)", color=C_V)
    ax.tick_params(axis="y", colors=C_V)
    ax.legend(handles=[v_line, v1_line, eligible_patch],
              fontsize=7.5, loc="upper left", framealpha=0.92)

    # Bottom: power
    ax = axes[1]
    ax.fill_between(
        df["t"], df["P_kW"], df["P_potential_kW"],
        where=curtailed.values, interpolate=True,
        color=C_CURT, alpha=0.30, zorder=2,
        label="V-Watt curtailment (uncurtailed_P − measured P)",
    )
    ax.fill_between(
        df["t"], df["P_ceiling_kW"], df["P_kW"],
        where=nonconformant.values, interpolate=True,
        color=C_NC, alpha=0.35, zorder=2,
        label="Non-conformant (measured P above ceiling)",
    )
    if df["P_potential_kW"].notna().any():
        ax.plot(df["t"], df["P_potential_kW"], color=C_POT, lw=1.6, ls="--",
                alpha=0.9, zorder=4, label="P potential (uncurtailed, clear-sky)")
    ax.plot(df["t"], df["P_ceiling_kW"], color=C_CEIL, lw=1.6, zorder=5,
            label="V-Watt ceiling (+4% tol)")
    ax.plot(df["t"], df["P_kW"], color=C_P, lw=1.6, zorder=4,
            label="Active power P (measured)")
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=7.3, loc="upper left", framealpha=0.92, ncol=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate(rotation=0, ha="center")

    interval_h = as4777["INTERVAL_H"]
    curtailed_kwh = (df["curtailed_kW"] * interval_h).sum()
    excess_kwh    = (df["excess_kW"] * interval_h).sum()

    site_label = "Site [redacted]" if redact_site else f"Site {site_id}"
    fig.suptitle(
        f"{site_label} | Volt-Watt curtailment, {date_str}  |  "
        f"{curtailed_kwh:.2f} kWh curtailed, {excess_kwh:.2f} kWh non-conformant excess\n"
        f"Nameplate AC capacity = {S:.1f} kW",
        fontsize=10.5, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    plt.show()

    print(f"Curtailed (sun-confirmed, conformant): {curtailed_kwh:.2f} kWh "
          f"across {int(curtailed.sum())} intervals")
    print(f"Non-conformant excess (measured > ceiling): {excess_kwh:.2f} kWh "
          f"across {int(nonconformant.sum())} intervals")
    return fig
