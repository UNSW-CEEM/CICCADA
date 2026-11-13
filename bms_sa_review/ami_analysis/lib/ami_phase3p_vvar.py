"""
Notebook 12: the Volt-VAr slope test for three-phase houses with a THREE-PHASE inverter (cohort 3L3P, reference
configuration 3P).

AS/NZS 4777.2:2020 Clause 3.3.2.1 (printed p. 20): "For three-phase inverters, the inverter shall use the average of
the three single-phase voltages as the reference voltage to determine the corresponding volt response action."
So the test uses the TOTAL of the three phases:

  net meter  per timestamp: P_tot = sum_i (base_P_i - P_net_i), Q_tot = sum_i (base_Q_i - Q_net_i), each house phase
             with its own monthly 00-04 h night baseline (`Fleet._estimate_sql`, unchanged); V_avg = mean of the
             three phase voltages (V_max as a sensitivity).
  reference  the same sums over the three PV circuits (their own P/Q, no baseline), against their mean voltage.

A timestamp is used only if all three circuits report P, Q and a plausible voltage (150-300 V). Gate: P_tot >= 0.20 x
rating on the net meter (signed), |P_tot| >= 0.20 x rating on the PV circuits. Window 242 <= V < 253 V, q = 100 Q_tot /
rating, OLS, assessable if n >= 20 and dV >= 5 V, band [-4.000, -2.667] %/V: all as for single-phase.

Clause 2.11 (printed p. 13) requires balanced output current from three-phase inverters, but allows unbalanced
currents in voltage balance mode (Clause 3.3.2.4). Per-phase imbalance is therefore flagged, never excluded.

Fleet, Phase and the notebook-11 modules are not changed; the per-phase fits reuse `Fleet.run_months`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet
from bms_sa_review.ami_analysis.lib import ami_phase_vvar as Phase
from bms_sa_review.ami_analysis.lib import ami_conformance_plot as ConformancePlot

KEYS = ["site_id", "year", "month"]
V_PLAUSIBLE = (150.0, 300.0)
V_MODES = ("avg", "max")
BLIND_ONLY_REASONS = ("blind_disagrees", "blind_inconsistent", "too_few_decided_months")


# =========================================================================================
# Sites and populations
# =========================================================================================
def populations(ev: pd.DataFrame, circuit_meta: pd.DataFrame) -> pd.DataFrame:
    """
    One row per reference-3P site (3L3P, ref_config 3P) with flags:
      in_B  all reference-3P sites (end-to-end grading)
      in_A  clean (use_in_12); in_A3 = in_A and n_decided >= 3
      in_C  excluded in notebook 10 only for blind reasons (no phase-reference check applies to 3P)
      catch any PV circuit is a CATCH Power device (active power uncorrected in the Phase-5 build)
    """
    p = ev[(ev.cohort == "3L3P") & (ev.ref_config == "3P")].copy()
    pv = circuit_meta[circuit_meta.circuit_type == "pv_site_net"]
    catch = pv.groupby("site_id").device_type.apply(lambda s: (s == "CATCH Power").any())
    p["catch"] = p.site_id.map(catch).fillna(False).astype(bool)
    p["in_B"] = True
    p["in_A"] = p.use_in_12.astype(bool)
    p["in_A3"] = p.in_A & (p.n_decided >= 3)
    p["in_C"] = p.exclusion_reason.isin(BLIND_ONLY_REASONS)
    return p.reset_index(drop=True)


# =========================================================================================
# SQL: per-timestamp totals over the three circuits
# =========================================================================================
def _ts_cte(name: str, month: str, source: str, night_hours, v_plausible) -> str:
    """CTE `name`: one row per (site_id, t_stamp) with circuit counts, P/Q totals and mean/max plausible voltage."""
    lo, hi = v_plausible
    ok_v = f"V BETWEEN {lo} AND {hi}"
    return f"""{name} AS (
        WITH {Fleet._estimate_sql(month, source, night_hours)}
        SELECT site_id, t_stamp, count(*) AS n_c, count(p_est) AS n_p, count(q_est) AS n_q,
               count(CASE WHEN {ok_v} THEN 1 END) AS n_v,
               sum(p_est) AS p_tot, sum(q_est) AS q_tot,
               avg(CASE WHEN {ok_v} THEN V END) AS v_avg, max(CASE WHEN {ok_v} THEN V END) AS v_max
        FROM est GROUP BY site_id, t_stamp)"""


def _complete(alias: str = "", n: int = 3) -> str:
    a = f"{alias}." if alias else ""
    return f"({a}n_c = {n} AND {a}n_p = {n} AND {a}n_q = {n} AND {a}n_v = {n})"


def total_fit_sql(month: str, source: str = "meter_baseline", *, night_hours=(0, 4),
                  p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = 242, v_hi: float = 253,
                  v_plausible=V_PLAUSIBLE, n_phases: int = 3, sign_min_frac: float = 0.05) -> str:
    """
    One row per (site in `_sites`, v_mode in avg/max) for `month`: timestamp counts, and the OLS fit of
    100 Q_tot / rating on V_avg (or V_max) inside the window. Sites without data still get rows.

    Also `n_signed` / `n_negative`: circuits with a midday (10-14 h AEST) median P_est above sign_min_frac x rating / 3
    in magnitude, and how many of those are negative. On the PV circuits generation is positive, so a site-month with
    SOME (not all) negative PV circuits has a reversed circuit and its summed P and Q are wrong (a reference error).
    """
    year, mon = (int(x) for x in month.split("-"))
    gate = "p_tot" if source == "meter_baseline" else "abs(p_tot)"
    comp = _complete(n=n_phases)
    win = lambda col, mode: (f"SELECT site_id, '{mode}' AS v_mode, {col} AS vx, 100.0 * q_tot / rating_kw AS y FROM x "
                             f"WHERE {comp} AND {col} >= {v_lo} AND {col} < {v_hi} AND {gate} >= {p_min_frac} * rating_kw")
    h = "(EXTRACT(hour FROM e.t_stamp) + 10) % 24"
    sign_cte = f"""sgn AS (
        WITH {Fleet._estimate_sql(month, source, night_hours)}
        SELECT site_id, count(*) AS n_signed, sum(CASE WHEN mp < 0 THEN 1 ELSE 0 END) AS n_negative
        FROM (SELECT e.site_id, e.circuit_id, median(e.p_est) AS mp
              FROM est e JOIN _sites s USING (site_id)
              WHERE {h} >= 10 AND {h} < 14 AND abs(e.p_est) > {sign_min_frac} * s.rating_kw / {n_phases}
              GROUP BY e.site_id, e.circuit_id)
        GROUP BY site_id)"""
    return f"""
        WITH {_ts_cte('x0', month, source, night_hours, v_plausible)},
        {sign_cte},
        x AS (SELECT x0.*, s.rating_kw FROM x0 JOIN _sites s USING (site_id)),
        cnt AS (SELECT site_id, count(*) AS n_ts, sum(CASE WHEN {comp} THEN 1 ELSE 0 END) AS n_complete
                FROM x GROUP BY site_id),
        win AS ({win('v_avg', 'avg')} UNION ALL {win('v_max', 'max')}),
        fit AS (SELECT site_id, v_mode, regr_count(y, vx) AS n_window, regr_slope(y, vx) AS slope,
                       regr_intercept(y, vx) AS intercept, regr_r2(y, vx) AS r_squared,
                       regr_sxx(y, vx) AS sxx, regr_syy(y, vx) AS syy, regr_sxy(y, vx) AS sxy,
                       min(vx) AS v_min_obs, max(vx) AS v_max_obs
                FROM win GROUP BY site_id, v_mode)
        SELECT s.site_id, m.v_mode, {year} AS year, {mon} AS month, '{source}' AS source, s.rating_kw,
               coalesce(c.n_ts, 0) AS n_ts, coalesce(c.n_complete, 0) AS n_complete,
               coalesce(g.n_signed, 0) AS n_signed, coalesce(g.n_negative, 0) AS n_negative,
               coalesce(f.n_window, 0) AS n_window, f.v_min_obs, f.v_max_obs, f.slope, f.intercept, f.r_squared,
               CASE WHEN f.n_window > 2 AND f.sxx > 0
                    THEN sqrt(greatest(f.syy - f.sxy * f.sxy / f.sxx, 0) / (f.n_window - 2) / f.sxx) END AS slope_stderr
        FROM _sites s CROSS JOIN (VALUES ('avg'), ('max')) AS m(v_mode)
        LEFT JOIN cnt c ON c.site_id = s.site_id
        LEFT JOIN sgn g ON g.site_id = s.site_id
        LEFT JOIN fit f ON f.site_id = s.site_id AND f.v_mode = m.v_mode
    """


def run_total_months(months: list[str], out_dir: Path, sites: pd.DataFrame, *, source: str,
                     overwrite: bool = False, memory_limit: str = "8GB", threads: int = 4, **sql_kwargs) -> pd.DataFrame:
    """
    `total_fit_sql` month by month for `sites` (site_id, rating_kw), written to out_dir/dt_month=YYYY-MM/part.parquet
    (temp file + rename, so a crash never leaves half a month). Months on disk are skipped unless `overwrite`.
    """
    out_dir = Path(out_dir)
    table = "ami_meter_residential" if source == "meter_baseline" else "ami_raw_phaseseparate"
    log = []
    for month in months:
        final = out_dir / f"dt_month={month}" / "part.parquet"
        tmp = final.with_name("part.parquet.tmp")
        if final.exists() and not overwrite:
            log.append({"month": month, "status": "skipped (already done)"})
            continue
        if not Fleet.month_dir(table, month).exists():
            log.append({"month": month, "status": "no partition"})
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        con = Fleet.connect(memory_limit, threads)
        try:
            con.register("_sites", sites[["site_id", "rating_kw"]])
            con.execute(f"COPY ({total_fit_sql(month, source, **sql_kwargs)}) TO '{tmp.as_posix()}' (FORMAT parquet)")
            con.close()
            os.replace(tmp, final)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 -- log and continue with the next month
            con.close()
            tmp.unlink(missing_ok=True)
            status = f"error: {type(exc).__name__}: {exc}"
        log.append({"month": month, "status": status, "seconds": round(time.time() - t0, 1)})
        print(f"total {source} {month}: {status} ({time.time() - t0:.1f} s)")
    return pd.DataFrame(log)


def load_total(out_dir: Path) -> pd.DataFrame:
    files = (Path(out_dir) / "dt_month=*" / "part.parquet").as_posix()
    return duckdb.sql(f"SELECT * EXCLUDE (dt_month) FROM read_parquet('{files}', hive_partitioning=1)").df()


def add_fit_status(d: pd.DataFrame, *, min_n_window: int, min_v_spread: float, tolerance_pct: float,
                   v_lo: float, v_hi: float) -> pd.DataFrame:
    """
    `v_spread`, `fit_status` (first match wins): no_data (no timestamps) -> incomplete (no timestamp with all three
    circuits) -> pv_polarity_mixed (reference only: some but not all PV circuits have a negative midday P, i.e. one
    circuit is reversed, so the summed Q is wrong) -> too_few_points -> narrow_voltage_range -> constant_q (reference
    only: slope and its stderr are 0) -> ok; and `passed` (band test; NA when not ok).
    """
    d = d.copy()
    d["v_spread"] = d.v_max_obs - d.v_min_obs
    ref = d.source == "ground_truth"
    const_q = ref & (d.slope == 0) & (d.slope_stderr == 0)
    mixed = ref & (d.n_negative > 0) & (d.n_negative < d.n_signed)
    d["fit_status"] = np.select(
        [d.n_ts == 0, d.n_complete == 0, mixed, d.n_window < min_n_window, d.v_spread < min_v_spread, const_q],
        ["no_data", "incomplete", "pv_polarity_mixed", "too_few_points", "narrow_voltage_range", "constant_q"],
        default="ok")
    ok = d.fit_status == "ok"
    bad_v = ok & ~(d.v_min_obs.between(v_lo, v_hi) & d.v_max_obs.between(v_lo, v_hi))
    assert not bad_v.any(), f"{bad_v.sum()} fits use voltages outside [{v_lo}, {v_hi})"
    pf = ConformancePlot.add_slope_pass_fail(d[ok], tolerance_pct)
    d["passed"] = pd.Series(pd.NA, index=d.index, dtype="boolean")
    d.loc[ok, "passed"] = pf.passed.astype("boolean")
    return d


# =========================================================================================
# Oracle table and end-to-end route
# =========================================================================================
def oracle_table(meter: pd.DataFrame, truth: pd.DataFrame, pop: pd.DataFrame, blind_month: pd.DataFrame,
                 v_mode: str = "avg") -> pd.DataFrame:
    """
    One row per (graded site, month with a reference row): the net-meter total fit (`slope_blind`, `pass_blind`) and
    the PV-circuit total fit (`slope_truth`, `pass_truth`) for `v_mode`, population flags, the blind result that
    month and notebook 10's `month_ok`. No phase choice exists for 3P: the oracle IS the test on the total.
    """
    cols = ["slope", "intercept", "r_squared", "n_window", "n_complete", "fit_status", "passed"]
    t = truth[truth.v_mode == v_mode][KEYS + cols + ["rating_kw"]].rename(
        columns={c: f"{c}_truth" for c in cols} | {"slope": "slope_truth", "passed": "pass_truth"})
    m = meter[meter.v_mode == v_mode][KEYS + cols].rename(
        columns={c: f"{c}_meter" for c in cols} | {"slope": "slope_blind", "passed": "pass_blind"})
    u = (t.merge(m, on=KEYS, how="left")
         .merge(pop[["site_id", "in_A", "in_A3", "in_C", "catch", "model_hint", "exclusion_reason", "n_decided"]],
                on="site_id")
         .merge(blind_month[KEYS + ["blind_config", "month_ok", "top_share", "min_share"]]
                .rename(columns={"top_share": "blind_top_share", "min_share": "blind_min_share"}), on=KEYS, how="left"))
    u["fit_status_meter"] = u.fit_status_meter.fillna("no_data")
    u["month_ok"] = u.month_ok.astype("boolean").fillna(False).astype(bool)
    u["blind_config"] = u.blind_config.fillna("no_signature")
    u["both_ok"] = (u.fit_status_meter == "ok") & (u.fit_status_truth == "ok")
    u["slope_err"] = u.slope_blind - u.slope_truth
    return u


def route_end_to_end(orc: pd.DataFrame) -> pd.DataFrame:
    """
    Every graded site-month with an assessable reference, routed by that month's blind configuration:
      3P -> total test; 1P -> wrong route (error); anything else -> not assessable.
    `outcome`: not_assessable_config | error_routed_1P | not_assessable_vvar | agree | disagree.
    """
    r = orc[orc.fit_status_truth == "ok"].copy()
    r["route"] = np.where(r.blind_config.isin(["1P", "3P"]), r.blind_config, "undecided")
    ok = r.fit_status_meter == "ok"
    agree = (r.pass_blind == r.pass_truth).fillna(False).astype(bool)
    r["outcome"] = np.select(
        [r.route == "undecided", r.route == "1P", (r.route == "3P") & ~ok, agree],
        ["not_assessable_config", "error_routed_1P", "not_assessable_vvar", "agree"], default="disagree")
    return r


def step_table(r: pd.DataFrame) -> pd.DataFrame:
    n = len(r)
    rows = [
        ("reference assessable (PV total fit ok)", n),
        ("1. configuration: undecided (not assessable)", (r.route == "undecided").sum()),
        ("1. configuration: routed 1P (wrong route)", (r.route == "1P").sum()),
        ("1. configuration: routed 3P", (r.route == "3P").sum()),
        ("2. Volt-VAr: net-meter total not assessable", (r.outcome == "not_assessable_vvar").sum()),
        ("2. Volt-VAr: verdict agrees with reference", (r.outcome == "agree").sum()),
        ("2. Volt-VAr: verdict disagrees", (r.outcome == "disagree").sum()),
    ]
    t = pd.DataFrame(rows, columns=["step", "site_months"]).set_index("step")
    t["% of reference assessable"] = (100 * t.site_months / max(n, 1)).round(1)
    return t


def end_to_end_summary(r: pd.DataFrame) -> pd.DataFrame:
    """end_to_end = agree / (agree + disagree + routed 1P); given_3P_route = agree / (agree + disagree)."""
    def acc(mask):
        g = r[mask]
        return {"site_months": len(g), "sites": g.site_id.nunique(),
                "accuracy": (g.outcome == "agree").mean() if len(g) else np.nan}
    out = pd.DataFrame({
        "end_to_end (routed 1P = error)": acc(r.outcome.isin(["agree", "disagree", "error_routed_1P"])),
        "given a 3P route": acc(r.outcome.isin(["agree", "disagree"])),
    }).T
    out["coverage_%_of_reference"] = (100 * out.site_months.astype(float) / max(len(r), 1)).round(1)
    return out


# =========================================================================================
# Per-phase checks (Clause 2.11) and household load imbalance
# =========================================================================================
def phase_shares(phase_fits: pd.DataFrame, total: pd.DataFrame, *, min_n_window: int, min_v_spread: float,
                 min_total_slope: float = -1.0) -> pd.DataFrame:
    """
    Per site-month (3 per-phase fits ok AND the total fit ok with slope <= min_total_slope, so the share is
    meaningful): each phase's slope / total slope. Per-phase fits normalise by the FULL rating, so a balanced
    inverter gives about 1/3 per phase. Returns min/max share and the sum of the three per-phase slopes.
    """
    f = Fleet.add_fit_status(phase_fits, min_n_window=min_n_window, min_v_spread=min_v_spread)
    f = f[f.fit_status == "ok"]
    g = f.groupby(KEYS).agg(n_phase_ok=("slope", "size"), sum_phase_slopes=("slope", "sum"),
                            min_phase_slope=("slope", "max"), max_phase_slope=("slope", "min")).reset_index()
    g = g[g.n_phase_ok == 3]
    t = total[(total.v_mode == "avg") & (total.fit_status == "ok") & (total.slope <= min_total_slope)]
    g = g.merge(t[KEYS + ["slope"]].rename(columns={"slope": "total_slope"}), on=KEYS)
    # slopes are negative: the least negative phase slope gives the smallest share
    g["min_share"] = g.min_phase_slope / g.total_slope
    g["max_share"] = g.max_phase_slope / g.total_slope
    g["sum_ratio"] = g.sum_phase_slopes / g.total_slope
    return g


def load_imbalance(phase_meter: pd.DataFrame) -> pd.DataFrame:
    """
    Per site-month: household load imbalance across the three house phases from their night baselines,
    `night_max_share` = largest phase baseline P / sum of the three (1/3 = balanced, 1 = all load on one phase).
    NA unless there are three baselines, none negative, with a positive sum (a negative night baseline, i.e. net
    export at night, would make the share meaningless).
    """
    b = phase_meter.dropna(subset=["circuit_id", "base_p_kw"])
    g = b.groupby(KEYS).agg(n=("base_p_kw", "size"), tot=("base_p_kw", "sum"), mx=("base_p_kw", "max"),
                            mn=("base_p_kw", "min")).reset_index()
    valid = (g.n == 3) & (g.mn >= 0) & (g.tot > 0)
    g["night_max_share"] = np.where(valid, g.mx / g.tot.where(valid), np.nan)
    return g[KEYS + ["night_max_share"]]


# =========================================================================================
# Diagnostics on the total: phantom export and household reactive power (beta_e)
# =========================================================================================
def total_diag_sql(month: str, *, night_hours=(0, 4), p_min_frac: float = Fleet.P_MIN_FRAC, v_lo: float = 242,
                   v_hi: float = 253, v_plausible=V_PLAUSIBLE, day_start: int = 7, day_end: int = 18,
                   pv_off_frac: float = 0.05) -> str:
    """
    Per site in `_sites` for `month`, over the net-meter total fit's intervals (V_avg window):
      n_window_fit, n_outside_daylight, n_pv_off (PV total |P| < pv_off_frac x rating), n_pv_missing (PV total not
      complete); and on the intervals in BOTH fit windows: e = 100 (Q_tot_meter - Q_tot_pv) / rating,
      beta_e = OLS slope of e on the net-meter V_avg, sd(e), n_shared.
    """
    year, mon = (int(x) for x in month.split("-"))
    h = "(EXTRACT(hour FROM m.t_stamp) + 10) % 24"
    t_comp, t_gate = _complete("t"), f"abs(t.p_tot) >= {p_min_frac} * m.rating_kw"
    shared = (f"{t_comp} AND t.v_avg >= {v_lo} AND t.v_avg < {v_hi} AND {t_gate} AND t.q_tot IS NOT NULL")
    return f"""
        WITH {_ts_cte('m0', month, 'meter_baseline', night_hours, v_plausible)},
             {_ts_cte('t0', month, 'ground_truth', night_hours, v_plausible)},
        m AS (SELECT m0.*, s.rating_kw FROM m0 JOIN _sites s USING (site_id)
              WHERE {_complete('m0')} AND m0.v_avg >= {v_lo} AND m0.v_avg < {v_hi}
                AND m0.p_tot >= {p_min_frac} * s.rating_kw),
        j AS (SELECT m.site_id, m.t_stamp, m.v_avg AS v_b, m.rating_kw,
                     100.0 * (m.q_tot - t.q_tot) / m.rating_kw AS e,
                     CASE WHEN {h} < {day_start} OR {h} >= {day_end} THEN 1 ELSE 0 END AS dark,
                     CASE WHEN {t_comp} AND abs(t.p_tot) < {pv_off_frac} * m.rating_kw THEN 1 ELSE 0 END AS pv_off,
                     CASE WHEN t.site_id IS NULL OR NOT {t_comp} THEN 1 ELSE 0 END AS pv_missing,
                     coalesce({shared}, false) AS shared
              FROM m LEFT JOIN t0 t ON t.site_id = m.site_id AND t.t_stamp = m.t_stamp)
        SELECT site_id, {year} AS year, {mon} AS month, count(*) AS n_window_fit,
               sum(dark) AS n_outside_daylight, sum(pv_off) AS n_pv_off, sum(pv_missing) AS n_pv_missing,
               count(*) FILTER (WHERE shared) AS n_shared,
               regr_slope(e, v_b) FILTER (WHERE shared) AS beta_e,
               stddev_samp(e) FILTER (WHERE shared) AS e_sd
        FROM j GROUP BY site_id
    """


def run_total_diag(units: pd.DataFrame, months: list[str], cache_dir: Path, tag: str, *, memory_limit: str,
                   threads: int, overwrite: bool = False, **sql_kwargs) -> pd.DataFrame:
    """total_diag_sql for `units` (KEYS + rating_kw), month by month, cached per month."""
    cache_dir = Path(cache_dir) / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for month in months:
        y, mo = int(month[:4]), int(month[5:])
        u = units[(units.year == y) & (units.month == mo)].drop_duplicates("site_id")
        path = cache_dir / f"{month}.parquet"
        if u.empty:
            continue
        if path.exists() and not overwrite:
            parts.append(pd.read_parquet(path))
            continue
        con = Fleet.connect(memory_limit, threads)
        try:
            con.register("_sites", u[["site_id", "rating_kw"]])
            df = con.execute(total_diag_sql(month, **sql_kwargs)).df()
        finally:
            con.close()
        df.to_parquet(path, index=False)
        parts.append(df)
        print(f"total diagnostics {month}: {len(df):,} site-months")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =========================================================================================
# Comparison with the other cohorts
# =========================================================================================
def summer_group(d: pd.DataFrame) -> pd.Series:
    return np.where(d.month.isin([1, 2]), "Jan-Feb", "Mar-Dec")


def load_3l1p_main(dir_3l1p: Path) -> pd.DataFrame:
    """Notebook 11's oracle site-months (clean, month_ok, both assessable) with pass_* and slope_* (blind / truth)."""
    d = pd.read_parquet(Path(dir_3l1p) / "site_month_details_3l1p.parquet")
    r = pd.read_parquet(Path(dir_3l1p) / "site_month_results_3l1p.parquet")
    pick = lambda src, tag: r[(r.source == src) & (r.fit_status == "ok")][KEYS + ["passed", "slope"]].rename(
        columns={"passed": f"pass_{tag}", "slope": f"slope_{tag}"})
    return (d[d.in_A & d.month_ok][KEYS].merge(pick("meter_oracle", "blind"), on=KEYS)
            .merge(pick("ground_truth", "truth"), on=KEYS))


def load_1l1p_main(fleet_results: pd.DataFrame, tolerance_pct: float) -> pd.DataFrame:
    """Notebook 07's single-phase site-months with both fits ok, with pass_* and slope_* (blind / truth)."""
    out = []
    for src, tag in [("meter_baseline", "blind"), ("ground_truth", "truth")]:
        f = fleet_results[(fleet_results.source == src) & (fleet_results.fit_status == "ok")]
        out.append(ConformancePlot.add_slope_pass_fail(f, tolerance_pct)[KEYS + ["passed", "slope"]].rename(
            columns={"passed": f"pass_{tag}", "slope": f"slope_{tag}"}))
    return out[0].merge(out[1], on=KEYS)


# =========================================================================================
# Output
# =========================================================================================
def to_result_columns(fits: pd.DataFrame, *, source: str, cohort: str = "3L3P") -> pd.DataFrame:
    """Phase.RESULT_COLUMNS rows; one inverter per site, so unit_id = site_id."""
    out = fits.copy()
    out["cohort"], out["source"], out["unit_id"] = cohort, source, out.site_id
    return out[Phase.RESULT_COLUMNS].reset_index(drop=True)
