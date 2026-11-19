"""
Notebook 10: look at PV circuits whose voltage-matched house phase changes between months.

For one site and PV circuit, two independent kinds of evidence, month by month (all months on disk):
  voltage  mean |V_pv - V_house| for each house phase (the notebook 10 matching; smallest = matched phase)
  power    correlation, at midday (10-14 h AEST), between the PV circuit's output |P_pv| and each house phase's
           export (-P_house): the phase that carries the inverter's current should track the PV output closely.
           This does not use voltage, so it can confirm or contradict the voltage match.
Plus two example days, one in the last month before the switch and one in the first month after, showing the
voltages and powers of the PV circuit and the three house phases.

Data: ami_raw_phaseseparate (polarity-corrected P). Times in AEST (UTC + 10). House phases are L1-L3 in circuit-id order.

Usage (notebook 10, after the truth_month cell):
    from bms_sa_review.ami_analysis.lib import phase_switch_plots as Switch
    changed = Switch.changed_circuits(truth_month)
    Switch.show_switch(changed.site_id.iloc[0], changed.pv_circuit_id.iloc[0], truth_month)
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from bms_sa_review.ami_analysis.lib import ami_daily_plots as Daily
from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet

V_PLAUSIBLE = (150, 300)
MIDDAY = (10, 14)
POWER_CORR_MIN = 0.5  # "power match" only if the best midday correlation reaches this
PHASE_PALETTE = ["#e69f00", "#56b4e9", "#cc79a7"]


def changed_circuits(truth_month: pd.DataFrame) -> pd.DataFrame:
    """PV circuits whose confident monthly match points to more than one house phase, with the months on each."""
    c = truth_month[truth_month.confident]
    g = c.groupby(["site_id", "pv_circuit_id"])
    out = g.agg(
        n_confident=("load_circuit_id", "size"), n_phases=("load_circuit_id", "nunique")
    ).reset_index()
    out = out[out.n_phases > 1]
    if "cohort" in c:
        out = out.merge(
            c[["site_id", "cohort"]].drop_duplicates(), on="site_id", how="left"
        )
    return out.sort_values("n_confident", ascending=False).reset_index(drop=True)


def monthly_evidence(site_id: int, pv_circuit_id: int) -> pd.DataFrame:
    """One row per month and house phase: mean |dV| to the PV circuit (all timestamps) and the midday correlation
    between |P_pv| and the house phase's export."""
    src = (Fleet.store_dir() / "ami_raw_phaseseparate").as_posix() + "/*/*.parquet"
    h = "(EXTRACT(hour FROM t_stamp) + 10) % 24"
    con = Fleet.connect("2GB", 2)
    try:
        df = con.execute(f"""
            WITH r AS (SELECT circuit_id, circuit_type, t_stamp,
                              CASE WHEN V BETWEEN {V_PLAUSIBLE[0]} AND {V_PLAUSIBLE[1]} THEN V END AS V,
                              P_kw_signed AS P, {h} AS hr
                       FROM read_parquet('{src}') WHERE site_id = {int(site_id)}),
                 pv AS (SELECT t_stamp, V AS v_pv, abs(P) AS p_pv, hr FROM r WHERE circuit_id = {int(pv_circuit_id)}),
                 hs AS (SELECT circuit_id, t_stamp, V AS v_h, -P AS exp_h FROM r WHERE circuit_type = 'ac_load_net')
            SELECT strftime(pv.t_stamp + INTERVAL 10 HOUR, '%Y-%m') AS ym, hs.circuit_id AS load_circuit_id,
                   avg(abs(pv.v_pv - hs.v_h)) AS mean_abs_dv,
                   corr(pv.p_pv, hs.exp_h) FILTER (WHERE pv.hr >= {MIDDAY[0]} AND pv.hr < {MIDDAY[1]}) AS power_corr,
                   count(*) AS n
            FROM pv JOIN hs USING (t_stamp)
            GROUP BY ALL HAVING count(*) >= 288 ORDER BY 1, 2
        """).df()
    finally:
        con.close()
    return df


def _switch_months(truth_month: pd.DataFrame | None, site_id: int, pv_circuit_id: int):
    """(month before, month after, old phase, new phase) from the confident monthly matches, or None."""
    if truth_month is None:
        return None
    c = truth_month[
        truth_month.confident
        & (truth_month.site_id == site_id)
        & (truth_month.pv_circuit_id == pv_circuit_id)
    ].sort_values(["year", "month"])
    if c.load_circuit_id.nunique() < 2:
        return None
    ym = c.year.astype(str) + "-" + c.month.map("{:02d}".format)
    lc = c.load_circuit_id.to_numpy()
    i = int(np.argmax(lc[1:] != lc[:-1])) + 1  # first change
    return ym.iloc[i - 1], ym.iloc[i], int(lc[i - 1]), int(lc[i])


def show_switch(
    site_id: int,
    pv_circuit_id: int,
    truth_month: pd.DataFrame | None = None,
    *,
    days: tuple | None = None,
):
    """
    Figure for one PV circuit: (a) monthly mean |dV| to each house phase, (b) monthly midday power correlation with
    each house phase, (c, d) one day before and one day after the first switch (voltage and power). `days` =
    ("YYYY-MM-DD", "YYYY-MM-DD") overrides the two example days. Returns the monthly evidence table.
    """
    ev = monthly_evidence(site_id, pv_circuit_id)
    if ev.empty:
        raise ValueError(f"No data for site {site_id}, PV circuit {pv_circuit_id}")
    phases = sorted(ev.load_circuit_id.unique())
    name = {c: f"L{i + 1} ({c})" for i, c in enumerate(phases)}
    col = {c: PHASE_PALETTE[i % 3] for i, c in enumerate(phases)}
    sw = _switch_months(truth_month, site_id, pv_circuit_id)

    meta = Fleet.Config.store_path("ami_circuit_metadata")
    cm = pd.read_parquet(meta)
    cm = cm[cm.site_id == site_id][
        ["circuit_id", "circuit_type", "device_type", "min_time", "max_time"]
    ]
    print(f"Site {site_id}, PV circuit {pv_circuit_id}")
    display(cm.sort_values("circuit_id").reset_index(drop=True))
    if sw:
        print(
            f"Confident voltage match: {name.get(sw[2], sw[2])} until {sw[0]}, {name.get(sw[3], sw[3])} from {sw[1]}"
        )

    wide_dv = ev.pivot(index="ym", columns="load_circuit_id", values="mean_abs_dv")
    wide_pc = ev.pivot(index="ym", columns="load_circuit_id", values="power_corr")
    summary = pd.concat(
        {
            "mean |dV| (V)": wide_dv.rename(columns=name),
            "midday power corr": wide_pc.rename(columns=name),
        },
        axis=1,
    )
    summary["voltage match"] = wide_dv.idxmin(axis=1).map(name)
    summary["phases with data"] = wide_dv.notna().sum(axis=1)
    best_pc = wide_pc.max(axis=1)
    summary["power match"] = (
        wide_pc.idxmax(axis=1).map(name).where(best_pc >= POWER_CORR_MIN, "none")
    )
    display(summary.round(3))

    if days is None and sw:
        days = (Daily.best_day(site_id, sw[0]), Daily.best_day(site_id, sw[1]))
    n_rows = 2 + (2 if days else 0)
    fig, axes = plt.subplots(
        n_rows, 2 if days else 1, figsize=(16, 3.4 * n_rows), dpi=100, squeeze=False
    )
    x = pd.to_datetime(wide_dv.index + "-01")

    ax = plt.subplot2grid((n_rows, 2), (0, 0), colspan=2, fig=fig)
    for c in phases:
        ax.plot(x, wide_dv[c], "o-", color=col[c], ms=4, label=name[c])
    ax.set_yscale("log")
    ax.set(
        ylabel="mean |V_pv − V_house| (V, log)",
        title="(a) Voltage: distance from the PV circuit to each house phase "
        "(lowest = voltage match)",
    )
    ax2 = plt.subplot2grid((n_rows, 2), (1, 0), colspan=2, fig=fig)
    for c in phases:
        ax2.plot(x, wide_pc[c], "o-", color=col[c], ms=4, label=name[c])
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set(
        ylabel="corr(|P_pv|, house export)",
        ylim=(-1.05, 1.05),
        title=f"(b) Power: midday ({MIDDAY[0]}-{MIDDAY[1]} h) correlation of PV output with each house phase's "
        f"export (highest = phase carrying the inverter current)",
    )
    for a in (ax, ax2):
        if sw:
            a.axvline(
                pd.Timestamp(sw[1] + "-01") - pd.Timedelta(days=15),
                color="black",
                ls="--",
                lw=1,
            )
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.grid(color="#ebebeb", lw=0.5)
        a.legend(fontsize=8, loc="best")
    for a in axes[:2].ravel():
        a.remove()

    if days:
        for j, dday in enumerate(days):
            dday = pd.Timestamp(dday).normalize()
            df = Daily.load_site_day(site_id, dday)
            av, ap = axes[2, j], axes[3, j]
            for c in phases:
                g = df[df.circuit_id == c]
                av.plot(g.t_aest, g.V, color=col[c], lw=0.9, label=name[c])
                ap.plot(
                    g.t_aest,
                    -g.P_kw,
                    color=col[c],
                    lw=0.9,
                    label=f"{name[c]} export (−P)",
                )
            g = df[df.circuit_id == pv_circuit_id]
            av.plot(
                g.t_aest,
                g.V,
                color="#1f4e79",
                lw=1.4,
                ls="--",
                label=f"PV {pv_circuit_id}",
            )
            ap.plot(
                g.t_aest,
                g.P_kw.abs(),
                color="#1f4e79",
                lw=1.4,
                ls="--",
                label=f"PV {pv_circuit_id} |P|",
            )
            ap.axhline(0, color="black", lw=0.5)
            tag = "before" if j == 0 else "after"
            av.set(
                ylabel="V",
                title=f"({'c' if j == 0 else 'd'}) {dday.date()} ({tag} the switch): voltages",
            )
            ap.set(
                ylabel="kW", title=f"{dday.date()}: PV output and house-phase export"
            )
            for a in (av, ap):
                a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                a.grid(color="#ebebeb", lw=0.5)
                a.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    plt.show()
    return summary
