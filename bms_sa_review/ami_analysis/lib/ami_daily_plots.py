"""
Daily plots for one site: P, Q and V of every house phase (ac_load_net) and PV circuit
(pv_site_net), plus the voltage gaps between each PV circuit and each house phase (the
voltage-matching check used in notebook 10).

Data: ami_raw_phaseseparate (polarity-corrected P/Q). Times are shown in AEST (UTC+10).
House phases have no A/B/C label in the data, so they are named L1-L3 in circuit-id order.

Usage:
    from bms_sa_review.ami_analysis.lib import ami_daily_plots as Daily
    Daily.show_site_day(674258435, month="2025-01", blind_month=blind_month, truth_month=truth_month)
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet
from bms_sa_review.ami_analysis.lib import ami_phase_vvar as Phase

V_PLAUSIBLE = (150, 300)      # voltages outside this range are treated as bad readings (set to NaN)
DAY_HOURS = (10, 14)          # default midday window (AEST), as in notebook 10


def best_day(site_id: int, month: str, day_hours: tuple[int, int] = DAY_HOURS) -> pd.Timestamp:
    """
    The AEST day in `month` to plot: among days that have house-phase data, the one with the most
    midday PV output (from the site's PV circuit(s)). If no day has both, falls back to a day with
    house data (most midday house activity), then to the best PV-only day, and says so.
    """
    d = Fleet.month_dir("ami_raw_phaseseparate", month)
    h = "(EXTRACT(hour FROM t_stamp) + 10) % 24"
    con = Fleet.connect("2GB", 2)
    try:
        row = con.execute(f"""
            SELECT CAST(t_stamp + INTERVAL 10 HOUR AS DATE) AS d,
                   sum(abs(P_kw_signed)) FILTER (WHERE circuit_type = 'pv_site_net') AS e_pv,
                   sum(abs(P_kw_signed)) FILTER (WHERE circuit_type = 'ac_load_net') AS e_load,
                   count(*) FILTER (WHERE circuit_type = 'ac_load_net') AS n_load
            FROM read_parquet('{d.as_posix()}/*.parquet')
            WHERE site_id = {int(site_id)} AND {h} >= {day_hours[0]} AND {h} < {day_hours[1]}
            GROUP BY 1
            ORDER BY (n_load > 0) DESC, e_pv DESC NULLS LAST, e_load DESC NULLS LAST
            LIMIT 1
        """).fetchone()
    finally:
        con.close()
    if row is None:
        raise ValueError(f"No midday data for site {site_id} in {month}")
    day, e_pv, _, n_load = row
    if not n_load:
        print(f"Note: no day in {month} has house-phase data; showing the best PV day ({day}).")
    elif e_pv is None:
        print(f"Note: no PV-circuit data in {month}; showing the house day with most midday activity ({day}).")
    return pd.Timestamp(day)


def load_site_day(site_id: int, date) -> pd.DataFrame:
    """Every circuit of one site for one AEST day (00:00-24:00 AEST)."""
    day = pd.Timestamp(date).normalize()
    t0, t1 = day - pd.Timedelta(hours=10), day + pd.Timedelta(hours=14)      # AEST day in UTC
    months = sorted({t0.strftime("%Y-%m"), (t1 - pd.Timedelta(seconds=1)).strftime("%Y-%m")})
    parts = []
    con = Fleet.connect("2GB", 2)
    try:
        for m in months:
            d = Fleet.month_dir("ami_raw_phaseseparate", m)
            if d.exists():
                parts.append(con.execute(f"""
                    SELECT circuit_id, circuit_type, t_stamp, V, P_kw_signed AS P_kw, Q_kvar_signed AS Q_kvar
                    FROM read_parquet('{d.as_posix()}/*.parquet')
                    WHERE site_id = {int(site_id)}
                      AND t_stamp >= TIMESTAMP '{t0}' AND t_stamp < TIMESTAMP '{t1}'
                """).df())
    finally:
        con.close()
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        raise ValueError(f"No data for site {site_id} on {day.date()}")
    t = pd.to_datetime(df.t_stamp)
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df["t_aest"] = t + pd.Timedelta(hours=10)
    bad = ~df.V.between(*V_PLAUSIBLE) & df.V.notna()
    if bad.any():
        print(f"Note: {bad.sum()} voltage readings outside {V_PLAUSIBLE} V set to NaN "
              f"(max {df.V[bad].max():,.0f} V, circuits {sorted(df.circuit_id[bad].unique())})")
        df.loc[bad, "V"] = np.nan
    return df.sort_values(["circuit_type", "circuit_id", "t_aest"]).reset_index(drop=True)


def phase_labels(df: pd.DataFrame) -> dict:
    """Name house phases L1-L3 and PV circuits PV1-PV3, in circuit-id order."""
    load = sorted(df.loc[df.circuit_type == "ac_load_net", "circuit_id"].unique())
    pv = sorted(df.loc[df.circuit_type == "pv_site_net", "circuit_id"].unique())
    return {**{c: f"L{i + 1} ({c})" for i, c in enumerate(load)},
            **{c: f"PV{i + 1} ({c})" for i, c in enumerate(pv)}}


def voltage_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    """|V_PV - V_phase| (mean and median) for every PV circuit x house phase on this day, ranked by the mean,
    with the ratio second-best / best (the same test as notebook 10's DV_RATIO_MIN)."""
    lab = phase_labels(df)
    v = df.pivot_table(index="t_aest", columns="circuit_id", values="V")
    pv = sorted(df.loc[df.circuit_type == "pv_site_net", "circuit_id"].unique())
    load = sorted(df.loc[df.circuit_type == "ac_load_net", "circuit_id"].unique())
    if not pv or not load:
        print(f"  (no voltage comparison: {len(pv)} PV circuit(s) and {len(load)} house phase(s) on this day)")
        return pd.DataFrame()
    rows = []
    for p in pv:
        gaps = {l: (v[p] - v[l]).abs().mean() for l in load}
        meds = {l: (v[p] - v[l]).abs().median() for l in load}
        order = sorted(gaps, key=lambda k: (np.isnan(gaps[k]), gaps[k]))
        best = gaps[order[0]]
        second = gaps[order[1]] if len(order) > 1 else np.nan
        for l in load:
            rows.append({"PV circuit": lab[p], "house phase": lab[l],
                         "mean |dV| (V)": round(gaps[l], 3),
                         "median |dV| (V)": round(meds[l], 3),
                         "n common": int((v[p].notna() & v[l].notna()).sum()),
                         "rank": order.index(l) + 1,
                         "ratio 2nd/best": round(second / best, 1) if (l == order[0] and best > 0) else np.nan})
    return pd.DataFrame(rows).sort_values(["PV circuit", "rank"]).reset_index(drop=True)


def plot_site_day(df: pd.DataFrame, site_id: int, day, *, day_hours: tuple[int, int] = DAY_HOURS,
                  v_band: tuple[float, float] = (242, 253)):
    """Four panels: P, Q, V per circuit, and V_PV - V_phase. Returns the figure."""
    lab = phase_labels(df)
    load = sorted(df.loc[df.circuit_type == "ac_load_net", "circuit_id"].unique())
    pv = sorted(df.loc[df.circuit_type == "pv_site_net", "circuit_id"].unique())
    colors = dict(zip(load, ["#2a78d6", "#eb6834", "#1baf7a", "#888780"]))
    pv_styles = dict(zip(pv, [("k", "--"), ("#6250d6", "--"), ("#e87ba4", "--")]))
    day = pd.Timestamp(day).normalize()

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), dpi=110, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})
    for c in load:
        g = df[df.circuit_id == c]
        for ax, col in zip(axes[:3], ["P_kw", "Q_kvar", "V"]):
            ax.plot(g.t_aest, g[col], color=colors[c], lw=1, label=f"{lab[c]} net")
    for c in pv:
        g = df[df.circuit_id == c]
        col, ls = pv_styles[c]
        for ax, y in zip(axes[:3], ["P_kw", "Q_kvar", "V"]):
            ax.plot(g.t_aest, g[y], color=col, ls=ls, lw=1.1, label=f"{lab[c]}")

    # Voltage gaps: each PV circuit minus each house phase
    v = df.pivot_table(index="t_aest", columns="circuit_id", values="V")
    for p in pv:
        for l in load:
            axes[3].plot(v.index, v[p] - v[l], color=colors[l], lw=1,
                         ls="-" if len(pv) == 1 else ["-", "--", ":"][pv.index(p) % 3],
                         label=f"{lab[p]} − {lab[l]}")
    axes[3].axhline(0, color="k", lw=0.6)

    for ax in axes:
        ax.axvspan(day + pd.Timedelta(hours=day_hours[0]), day + pd.Timedelta(hours=day_hours[1]),
                   color="#f2f2f2", zorder=0)
        ax.grid(color="#e6e6e6", lw=0.5)
    axes[2].axhspan(*v_band, color="#1baf7a", alpha=0.08, zorder=0)
    axes[0].set_ylabel("P (kW)\n+ = as stored")
    axes[1].set_ylabel("Q (kvar)")
    axes[2].set_ylabel("V (V)")
    axes[3].set_ylabel("V_PV − V_phase (V)")
    axes[0].legend(fontsize=7, ncol=3, loc="upper left")
    axes[3].legend(fontsize=7, ncol=3, loc="upper left")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[3].set_xlabel("Time (AEST)")
    axes[0].set_title(f"Site {site_id} - {day.date()}  (grey band: midday window {day_hours[0]}-{day_hours[1]} h; "
                      f"green band: {v_band[0]}-{v_band[1]} V)", fontsize=10, loc="left")
    fig.tight_layout()
    plt.show()
    return fig


def show_site_day(site_id: int, date=None, month: str | None = None, *,
                  day_hours: tuple[int, int] = DAY_HOURS,
                  blind_month: pd.DataFrame | None = None,
                  truth_month: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Tables + plot for one site-day. Give `date` ('YYYY-MM-DD'), or `month` ('YYYY-MM') to pick its
    best PV day. Pass notebook 10's `blind_month` / `truth_month` to also show its results for that
    site-month. Returns the interval data.
    """
    if date is None and month is None:
        raise ValueError("Give either date or month")
    day = pd.Timestamp(date) if date is not None else best_day(site_id, month, day_hours)
    df = load_site_day(site_id, day)
    lab = phase_labels(df)
    types = df.drop_duplicates("circuit_id").circuit_type
    print(f"Site {site_id}, {day.date()}: {df.circuit_id.nunique()} circuits "
          f"({(types == 'ac_load_net').sum()} house phases, {(types == 'pv_site_net').sum()} PV)")

    y, m = day.year, day.month
    if blind_month is not None:
        b = blind_month[(blind_month.site_id == site_id) & (blind_month.year == y) & (blind_month.month == m)]
        if len(b):
            b = b.assign(blind_phase=b.blind_load_circuit_id.map(lambda c: lab.get(c, c) if pd.notna(c) else None))
            print("Blind configuration (this month):")
            display(b[["cohort", "blind_config", "blind_phase", "top_share", "min_share", "top_over_rating"]].round(2))
    if truth_month is not None:
        t = truth_month[(truth_month.site_id == site_id) & (truth_month.year == y) & (truth_month.month == m)]
        if len(t):
            t = t.assign(pv=t.pv_circuit_id.map(lambda c: lab.get(c, c)),
                         ref_phase=t.load_circuit_id.map(lambda c: lab.get(c, c)))
            print("Reference pairing (this month, every 15 min):")
            display(t[["pv", "ref_phase", "best_dv", "second_dv", "dv_ratio", "confident"]].round(3))

    print("Voltage gaps on this day (every 5 min):")
    display(voltage_delta_table(df))
    plot_site_day(df, site_id, day, day_hours=day_hours)
    return df


# ---------------------------------------------------------------------------
# Section 5 of notebook 10, visualised: the blind "export signature" for one site and month.
# P_est = night baseline - net reading, per house phase (same SQL as notebook 10:
# polarity correction + monthly night-time median baseline).
#   (a) one day: P_est per house phase, with the PV circuit's own output for comparison
#   (b) the whole month: distribution of midday P_est per phase; the dot is the 90th percentile
#       (the "signal" the classifier uses)
#   (c) the signal shares with the thresholds, and the resulting blind configuration
# ---------------------------------------------------------------------------
NIGHT_HOURS = (0, 4)          # default night-baseline window (AEST), as in notebooks 06/07/10


def site_rating(site_id: int) -> tuple[float, str]:
    """Inverter rating (kW) and cohort label (e.g. '3L1P') of a site, from the site metadata."""
    meta = pd.read_parquet(Fleet.Config.store_path("ami_site_metadata_residential"),
                           columns=["site_id", "ac_capacity_kw", "n_load_phases", "n_pv_phases"])
    row = meta.loc[meta.site_id == site_id]
    if row.empty:
        raise ValueError(f"Site {site_id} not in ami_site_metadata_residential")
    r = row.iloc[0]
    return float(r.ac_capacity_kw), f"{int(r.n_load_phases)}L{int(r.n_pv_phases)}P"


def load_site_month_est(site_id: int, month: str, rating_kw: float, *,
                        night_hours: tuple[int, int] = NIGHT_HOURS,
                        day_hours: tuple[int, int] = DAY_HOURS) -> pd.DataFrame:
    """Net-meter estimate (P_est, Q_est) for every house phase of one site for one month."""
    con = Fleet.connect("2GB", 2)
    try:
        con.register("_sites", pd.DataFrame({"site_id": [site_id], "rating_kw": [rating_kw]}))
        df = con.execute(f"""
            WITH {Fleet._estimate_sql(month, "meter_baseline", night_hours)}
            SELECT e.circuit_id, e.t_stamp, e.V, e.p_est, e.q_est, b.base_p_kw, b.base_q_kvar
            FROM est e LEFT JOIN baseline b USING (circuit_id)
        """).df()
    finally:
        con.close()
    if df.empty:
        raise ValueError(f"No net-meter data for site {site_id} in {month}")
    t = pd.to_datetime(df.t_stamp)
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df["t_aest"] = t + pd.Timedelta(hours=10)
    df["midday"] = df.t_aest.dt.hour.between(day_hours[0], day_hours[1] - 1)
    return df.sort_values(["circuit_id", "t_aest"]).reset_index(drop=True)


def show_export_signature(site_id: int, month: str, date=None, *,
                          night_hours: tuple[int, int] = NIGHT_HOURS,
                          day_hours: tuple[int, int] = DAY_HOURS,
                          share_1p: float = 0.6, min_share_3p: float = 0.1,
                          min_signal_frac: float = 0.1) -> pd.DataFrame:
    """
    Table + three panels showing how notebook 10's blind classifier sees one site-month.
    Defaults match notebook 10's settings; pass the notebook's values to be sure they agree.
    Returns the month's net-meter estimate.
    """
    rating, cohort = site_rating(site_id)
    day = pd.Timestamp(date) if date is not None else best_day(site_id, month, day_hours)

    est = load_site_month_est(site_id, month, rating, night_hours=night_hours, day_hours=day_hours)
    pv_day = load_site_day(site_id, day)
    pv_day = pv_day[pv_day.circuit_type == "pv_site_net"]
    load = sorted(est.circuit_id.unique())
    lab = {c: f"L{i + 1} ({c})" for i, c in enumerate(load)}
    colors = dict(zip(load, ["#2a78d6", "#eb6834", "#1baf7a", "#888780"]))

    # Monthly signal per phase: 90th percentile of midday P_est (as Phase.phase_signature_sql)
    mid = est[est.midday & est.p_est.notna()]
    sig = mid.groupby("circuit_id").p_est.quantile(0.9).reindex(load)
    signal = sig.clip(lower=0)
    share = signal / signal.sum() if signal.sum() > 0 else signal * np.nan

    # Classification with the given settings (same function as notebook 10)
    c = Phase.classify_configuration(
        sig.rename("p_mid_q90").rename_axis("circuit_id").reset_index()
           .assign(site_id=site_id, year=int(month[:4]), month=int(month[5:])),
        pd.DataFrame({"site_id": [site_id], "rating_kw": [rating]}),
        share_1p=share_1p, min_share_3p=min_share_3p, min_signal_frac=min_signal_frac).iloc[0]

    print(f"Site {site_id} ({cohort}), rating {rating:g} kW, month {month}, example day {day.date()}")
    tab = pd.DataFrame({"house phase": [lab[x] for x in load],
                        "night baseline P (kW)": est.groupby("circuit_id").base_p_kw.first().reindex(load).round(2).values,
                        "midday P_est q90 (kW)": sig.round(2).values,
                        "signal (>=0)": signal.round(2).values,
                        "share": share.round(3).values,
                        "signal / rating": (signal / rating).round(2).values})
    display(tab)
    print(f"Blind configuration: {c.blind_config}"
          + (f" on {lab.get(c.blind_load_circuit_id, c.blind_load_circuit_id)}" if c.blind_config == "1P" else "")
          + f"  | top share {c.top_share:.2f}, smallest share {c.min_share:.2f}, "
            f"total signal {c.total:.2f} kW ({c.total / rating:.0%} of rating), top / rating {c.top_over_rating:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), dpi=110, gridspec_kw={"width_ratios": [2.2, 1.2, 1]})

    # (a) one day
    d0 = pd.Timestamp(day).normalize()
    dd = est[(est.t_aest >= d0) & (est.t_aest < d0 + pd.Timedelta(days=1))]
    for x in load:
        g = dd[dd.circuit_id == x]
        axes[0].plot(g.t_aest, g.p_est, color=colors[x], lw=1.1, label=f"{lab[x]} P_est")
    for i, (pc, g) in enumerate(pv_day.groupby("circuit_id")):
        axes[0].plot(g.t_aest, g.P_kw.abs(), color="k", ls="--", lw=1, label=f"PV circuit {pc} (|P|)" if i < 3 else None)
    axes[0].axvspan(d0 + pd.Timedelta(hours=day_hours[0]), d0 + pd.Timedelta(hours=day_hours[1]),
                    color="#f2f2f2", zorder=0)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].axhline(rating, color="#a32d2d", lw=0.8, ls=":", label=f"rating {rating:g} kW")
    axes[0].axhline(rating / 3, color="#a32d2d", lw=0.8, ls="-.", label="rating / 3")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].set(xlabel="Time (AEST)", ylabel="kW",
                title=f"(a) {d0.date()}: P_est = night baseline − net, per house phase")
    axes[0].legend(fontsize=7, loc="upper left")

    # (b) the whole month, midday only
    data = [mid.loc[mid.circuit_id == x, "p_est"].values for x in load]
    bp = axes[1].boxplot(data, showfliers=False, patch_artist=True)
    axes[1].set_xticks(range(1, len(load) + 1), [lab[x].split()[0] for x in load])
    for patch, x in zip(bp["boxes"], load):
        patch.set_facecolor(colors[x]); patch.set_alpha(0.35)
    axes[1].scatter(range(1, len(load) + 1), sig.values, color="k", zorder=3, label="90th percentile = signal")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].axhline(rating, color="#a32d2d", lw=0.8, ls=":")
    axes[1].axhline(rating / 3, color="#a32d2d", lw=0.8, ls="-.")
    axes[1].set(ylabel="Midday P_est (kW)", title=f"(b) {month}: midday {day_hours[0]}-{day_hours[1]} h, all days")
    axes[1].legend(fontsize=7, loc="upper left")

    # (c) shares and thresholds
    axes[2].bar([lab[x].split()[0] for x in load], share.values, color=[colors[x] for x in load])
    axes[2].axhline(share_1p, color="k", ls="--", lw=0.9, label=f"SHARE_1P = {share_1p:g}")
    axes[2].axhline(min_share_3p, color="k", ls=":", lw=0.9, label=f"MIN_SHARE_3P = {min_share_3p:g}")
    axes[2].axhline(1 / 3, color="#888780", ls="-.", lw=0.8, label="1/3 (balanced)")
    axes[2].set(ylim=(0, 1.05), ylabel="Share of midday export signal", title=f"(c) blind: {c.blind_config}")
    axes[2].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    plt.show()
    return est