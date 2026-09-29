"""
Fleet-wide, month-by-month Volt-VAr slope fits -- notebook 07.

Method (validated in 05_exploratory_data_analysis, Section 12), per site-month:
  1. Polarity-corrected net meter reading (ami_meter_residential x circuit_polarity).
  2. Night-time baseline, per circuit and per MONTH (median P/Q over AEST night hours):
         P_est = base_P - P_net        Q_est = base_Q - Q_net
  3. Signed net-export gate: keep intervals with P_est >= QCAP.P_MIN * rating.
     (abs(P) would also admit heavy household import, when the PV is producing ~nothing.)
  4. OLS of Q_est (% of rating) on V over [VVAR.V3, VW.V1), via DuckDB regr_* aggregates.

Memory: everything is aggregated inside DuckDB; only one row per site-month reaches Python.
Resume: each month is written to its own Parquet file; a rerun skips months already on disk.

Months are the store's `dt_month` partitions (UTC calendar months), same as every other
notebook's `year`/`month` scope.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import side effect: puts `bms_sa_review` on sys.path for the `shared.*` import below.
from bms_sa_review.synthetic_ami_creation.config import ami_config as Config
from bms_sa_review.ami_analysis.lib import ami_polarity as Polarity
from shared.ciccada_config import AS4777  # noqa: E402

# --- AS/NZS 4777.2 constants (same sources as ami_conformance_plot.py) -------------------
V_LO = AS4777["VVAR"]["V3"]          # 240 V: start of the absorbing ramp
V_HI = AS4777["VW"]["V1"]            # 253 V: Volt-Watt starts, window ends (exclusive)
P_MIN_FRAC = AS4777["QCAP"]["P_MIN"] # 0.20: minimum output for a capability assessment
REQUIRED_SLOPE = -AS4777["VVAR"]["Q4"] / (AS4777["VVAR"]["V4"] - AS4777["VVAR"]["V3"]) * 100

SOURCES = ("meter_baseline", "ground_truth")


# =========================================================================================
# Paths and DuckDB connection
# =========================================================================================
def store_dir() -> Path:
    """Local (non-OneDrive) data store root, e.g. %LOCALAPPDATA%/ciccada/ami_store."""
    return Config.store_path("ami_meter_residential").parent


def month_dir(table: str, month: str) -> Path:
    """Folder of one month's partition, e.g. ami_meter_residential/dt_month=2025-01."""
    return store_dir() / table / f"dt_month={month}"


def connect(memory_limit: str = "8GB", threads: int = 4) -> duckdb.DuckDBPyConnection:
    """DuckDB connection that spills to local disk OUTSIDE OneDrive and keeps memory capped."""
    tmp = store_dir().parent / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    return con


# =========================================================================================
# Site eligibility
# =========================================================================================
def build_eligibility(site_meta: pd.DataFrame, rating_column: str = "ac_capacity_kw") -> pd.DataFrame:
    """
    One row per residential site with `rating_kw` and an `eligibility` reason (first match wins):
    multi_phase -> flex_export -> no_rating -> export_limited -> eligible.
    `export_limited`: export limit below the gate threshold, so net export can never pass it.
    """
    d = site_meta[["site_id", rating_column, "export_limit_kw", "flex_export_detected",
                   "n_load_phases", "n_pv_phases"]].rename(columns={rating_column: "rating_kw"})
    conditions = [
        (d.n_load_phases != 1) | (d.n_pv_phases != 1),
        d.flex_export_detected.fillna(False).astype(bool),
        d.rating_kw.isna() | (d.rating_kw <= 0),
        d.export_limit_kw < P_MIN_FRAC * d.rating_kw,
    ]
    d["eligibility"] = np.select(conditions, ["multi_phase", "flex_export", "no_rating", "export_limited"],
                                 default="eligible")
    return d


# =========================================================================================
# SQL: one month -> one row per eligible site
# =========================================================================================
def _meter_estimate_sql(month: str, night_hours: tuple[int, int]) -> str:
    """Polarity-corrected net reading -> per-circuit monthly night baseline -> P_est / Q_est."""
    meter = (f"(SELECT site_id, circuit_id, t_stamp, V, P_kw, Q_kvar "
             f"FROM read_parquet('{month_dir('ami_meter_residential', month).as_posix()}/*.parquet') "
             f"WHERE site_id IN (SELECT site_id FROM _sites))")
    polarity = (f"(SELECT circuit_id, circuit_polarity "
                f"FROM read_parquet('{Config.store_path('ami_circuit_metadata').as_posix()}'))")
    return f"""
        pc AS ({Polarity.build_polarity_corrected_sql(meter, polarity)}),
        baseline AS (   -- this month's night-time median, per circuit
            SELECT circuit_id,
                   median(P_kw) AS base_p_kw, median(Q_kvar) AS base_q_kvar, count(*) AS n_night
            FROM pc
            WHERE (EXTRACT(hour FROM t_stamp) + 10) % 24 >= {night_hours[0]}
              AND (EXTRACT(hour FROM t_stamp) + 10) % 24 <  {night_hours[1]}
            GROUP BY circuit_id
        ),
        est AS (
            SELECT pc.site_id, pc.circuit_id, pc.V,
                   b.base_p_kw - pc.P_kw   AS p_est,
                   b.base_q_kvar - pc.Q_kvar AS q_est
            FROM pc JOIN baseline b USING (circuit_id)
        )"""


def _truth_estimate_sql(month: str) -> str:
    """Ground truth: the physically separate PV circuit's own P/Q -- no baseline needed."""
    return f"""
        pc AS (
            SELECT site_id, circuit_id, t_stamp, V, P_kw_signed AS P_kw, Q_kvar_signed AS Q_kvar
            FROM read_parquet('{month_dir('ami_raw_phaseseparate', month).as_posix()}/*.parquet')
            WHERE circuit_type = 'pv_site_net' AND site_id IN (SELECT site_id FROM _sites)
        ),
        baseline AS (
            SELECT circuit_id, NULL::DOUBLE AS base_p_kw, NULL::DOUBLE AS base_q_kvar,
                   NULL::BIGINT AS n_night
            FROM pc GROUP BY circuit_id
        ),
        est AS (SELECT site_id, circuit_id, V, P_kw AS p_est, Q_kvar AS q_est FROM pc)"""


def month_fit_sql(month: str, source: str = "meter_baseline", *,
                  night_hours: tuple[int, int] = (0, 4), p_min_frac: float = P_MIN_FRAC,
                  v_lo: float = V_LO, v_hi: float = V_HI) -> str:
    """
    One row per site in the registered `_sites` table (site_id, rating_kw) for `month`:
    circuit, interval counts, baseline, and the OLS fit of Q% on V inside the assessable
    window. Sites with no data that month still get a row (circuit_id NULL).
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    year, mon = (int(x) for x in month.split("-"))
    estimate = _meter_estimate_sql(month, night_hours) if source == "meter_baseline" else _truth_estimate_sql(month)
    return f"""
        WITH {estimate},
        circuits AS (SELECT site_id, circuit_id, count(*) AS n_intervals FROM pc GROUP BY ALL),
        win AS (        -- assessable window: voltage range + signed net-export gate
            SELECT e.site_id, e.circuit_id, e.V AS x, 100.0 * e.q_est / s.rating_kw AS y
            FROM est e JOIN _sites s USING (site_id)
            WHERE e.V >= {v_lo} AND e.V < {v_hi}
              AND e.p_est >= {p_min_frac} * s.rating_kw
              AND e.q_est IS NOT NULL
        ),
        fit AS (
            SELECT site_id, circuit_id,
                   regr_count(y, x) AS n_window,
                   regr_slope(y, x) AS slope, regr_intercept(y, x) AS intercept,
                   regr_r2(y, x) AS r_squared,
                   regr_sxx(y, x) AS sxx, regr_syy(y, x) AS syy, regr_sxy(y, x) AS sxy,
                   min(x) AS v_min_obs, max(x) AS v_max_obs
            FROM win GROUP BY ALL
        )
        SELECT s.site_id, c.circuit_id, {year} AS year, {mon} AS month, '{source}' AS source,
               s.rating_kw, c.n_intervals, b.n_night, b.base_p_kw, b.base_q_kvar,
               coalesce(f.n_window, 0) AS n_window, f.v_min_obs, f.v_max_obs,
               f.slope, f.intercept, f.r_squared,
               CASE WHEN f.n_window > 2 AND f.sxx > 0      -- OLS standard error of the slope
                    THEN sqrt(greatest(f.syy - f.sxy * f.sxy / f.sxx, 0) / (f.n_window - 2) / f.sxx)
               END AS slope_stderr
        FROM _sites s
        LEFT JOIN circuits c USING (site_id)
        LEFT JOIN baseline b ON b.circuit_id = c.circuit_id
        LEFT JOIN fit f ON f.site_id = c.site_id AND f.circuit_id = c.circuit_id
    """


# =========================================================================================
# Resumable month-by-month runner
# =========================================================================================
def run_months(months: list[str], out_dir: Path, sites: pd.DataFrame, *,
               source: str = "meter_baseline", overwrite: bool = False,
               memory_limit: str = "8GB", threads: int = 4, **fit_kwargs) -> pd.DataFrame:
    """
    Fit every month in `months` for `sites` (site_id, rating_kw) and write each to
    out_dir/dt_month=YYYY-MM/part.parquet. Months already on disk are skipped (resume),
    unless `overwrite`. Each month uses a fresh connection and writes via a temp file +
    rename, so a crash never leaves a half-written file behind. A failed month is logged
    and the run moves on. Returns (and appends to out_dir/run_log.csv) one log row per month.
    """
    out_dir = Path(out_dir)
    table = "ami_meter_residential" if source == "meter_baseline" else "ami_raw_phaseseparate"
    log = []
    for month in months:
        final = out_dir / f"dt_month={month}" / "part.parquet"
        tmp = final.with_name("part.parquet.tmp")
        entry = {"month": month, "source": source, "started": pd.Timestamp.now()}

        if final.exists() and not overwrite:
            log.append({**entry, "status": "skipped (already done)"})
            print(f"{month}: already done, skipped")
            continue
        if not month_dir(table, month).exists():
            log.append({**entry, "status": "no partition"})
            print(f"{month}: no {table} partition, skipped")
            continue

        final.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        con = connect(memory_limit, threads)
        try:
            con.register("_sites", sites[["site_id", "rating_kw"]])
            sql = month_fit_sql(month, source, **fit_kwargs)
            con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT parquet)")
            n_rows, n_fit = con.execute(
                f"SELECT count(*), count(slope) FROM read_parquet('{tmp.as_posix()}')").fetchone()
            con.close()
            os.replace(tmp, final)  # atomic: the month only "exists" once fully written
            entry.update(status="ok", n_sites=n_rows, n_fitted=n_fit)
        except Exception as exc:  # noqa: BLE001 -- log it, keep going with the next month
            con.close()
            tmp.unlink(missing_ok=True)
            entry.update(status=f"error: {type(exc).__name__}: {exc}")
        entry["seconds"] = round(time.time() - t0, 1)
        log.append(entry)
        print(f"{month}: {entry['status']} ({entry['seconds']} s)")

        pd.DataFrame([entry]).to_csv(out_dir / "run_log.csv", mode="a", index=False,
                                     header=not (out_dir / "run_log.csv").exists())
    return pd.DataFrame(log)


def load_results(out_dir: Path) -> pd.DataFrame:
    """All months written by run_months, as one DataFrame (one row per site-month)."""
    files = (Path(out_dir) / "dt_month=*" / "part.parquet").as_posix()
    return duckdb.sql(f"SELECT * EXCLUDE (dt_month) FROM read_parquet('{files}', hive_partitioning=1)").df()


# =========================================================================================
# Post-processing (cheap, pandas): thresholds can change without re-running the months
# =========================================================================================
def add_fit_status(results: pd.DataFrame, *, min_n_window: int, min_v_spread: float) -> pd.DataFrame:
    """
    Adds `v_spread` and `fit_status` (first match wins):
    no_data -> no_baseline -> too_few_points -> narrow_voltage_range -> ok.
    """
    d = results.copy()
    d["v_spread"] = d.v_max_obs - d.v_min_obs
    conditions = [
        d.circuit_id.isna(),
        (d.source == "meter_baseline") & d.base_q_kvar.isna(),
        d.n_window < min_n_window,
        d.v_spread < min_v_spread,
    ]
    d["fit_status"] = np.select(conditions, ["no_data", "no_baseline", "too_few_points", "narrow_voltage_range"],
                                default="ok")
    return d


def compare_to_truth(blind: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Site-months where BOTH fits are `ok`, side by side, with slope error (blind - truth)."""
    keys = ["site_id", "year", "month"]
    cols = keys + ["slope", "intercept", "r_squared", "n_window"]
    both = (blind.loc[blind.fit_status == "ok", cols]
            .merge(truth.loc[truth.fit_status == "ok", cols], on=keys, suffixes=("_blind", "_truth")))
    both["slope_err"] = both.slope_blind - both.slope_truth
    return both


# =========================================================================================
# Plots
# =========================================================================================
def plot_monthly_slopes(results: pd.DataFrame, title: str = "", ax=None):
    """Box plot of `ok` site slopes per month, against the required slope."""
    ok = results.loc[results.fit_status == "ok"]
    months = sorted(ok[["year", "month"]].drop_duplicates().itertuples(index=False))
    data = [ok.loc[(ok.year == y) & (ok.month == m), "slope"].values for y, m in months]
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5), dpi=130)
    ax.boxplot(data, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#2a78d6", "alpha": 0.35}, medianprops={"color": "black"})
    ax.set_xticks(range(1, len(months) + 1), [f"{y}-{m:02d}\n(n={len(v):,})" for (y, m), v in zip(months, data)],
                  fontsize=7.5)
    ax.axhline(REQUIRED_SLOPE, color="#f59e0b", linewidth=2, linestyle="--",
               label=f"Required slope = {REQUIRED_SLOPE:.2f} %/V")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Fitted slope (% of rating per V)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#ebebeb")
    return ax


def plot_blind_vs_truth(compare: pd.DataFrame, lim: tuple[float, float] = (-8, 2), ax=None):
    """Blind slope vs ground-truth slope per site-month; points on the diagonal = agreement."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=130)
    ax.scatter(compare.slope_truth, compare.slope_blind, s=6, alpha=0.3, color="#2a78d6")
    ax.plot(lim, lim, color="black", linewidth=0.8, label="1:1")
    ax.axhline(REQUIRED_SLOPE, color="#f59e0b", linewidth=1, linestyle="--", label="Required slope")
    ax.axvline(REQUIRED_SLOPE, color="#f59e0b", linewidth=1, linestyle="--")
    ax.set(xlim=lim, ylim=lim, xlabel="Ground-truth slope (%/V)", ylabel="Blind slope (%/V)")
    err = compare.slope_err.abs()
    ax.set_title(f"n={len(compare):,} site-months; median |error| {err.median():.2f} %/V",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(color="#ebebeb")
    return ax
