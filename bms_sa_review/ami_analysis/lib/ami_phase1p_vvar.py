"""
Notebook 11: the single-phase Volt-VAr slope test on three-phase houses with ONE single-phase inverter
(cohort 3L1P, reference configuration 1P).

The test is unchanged from notebooks 06/07 (`ami_fleet_vvar`): monthly 00-04 h AEST night baseline,
P_est/Q_est = base - net, signed gate P_est >= 0.20 x rating, 242 <= V < 253 V, q = 100 Q_est / rating,
OLS of q on V, assessable if n >= 20 and dV >= 5 V, conformant if the slope is within +-20 % of
-3.333 %/V, i.e. in [-4.000, -2.667] %/V.

What is new here is only WHICH net-meter circuit is tested:
  * oracle      : the house phase the PV circuit is on (voltage matching, notebook 10),
  * blind route : the phase the blind classifier picked, in months it decided "1P".
`Fleet.month_fit_sql` already fits every circuit of a site, so one run gives all three house phases.

Inputs come from notebook 10 (`vvar_phase/phase_config/`). Nothing in `ami_fleet_vvar` or
`ami_phase_vvar` is changed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet
from bms_sa_review.ami_analysis.lib import ami_phase_vvar as Phase
from bms_sa_review.ami_analysis.lib import ami_conformance_plot as ConformancePlot
from bms_sa_review.ami_analysis.lib import ami_vvar_validation as Valid

KEYS = ["site_id", "year", "month"]

# Sites excluded in notebook 10 ONLY because of the blind method (all reference checks before them passed)
BLIND_ONLY_REASONS = ("blind_disagrees", "blind_inconsistent", "phase_disagrees", "too_few_decided_months")

# Single-phase (1L1P) validation, quoted from the saved outputs of notebook 06 (cell 21, blind vs ground
# truth, constant-Q months not assessable) and notebook 09 (section 9.4, "OLS (as is)", by year).
SINGLE_PHASE_REFERENCE = pd.DataFrame([
    {"group": "all years", "n": 74805, "TP": 14137, "FP": 1288, "FN": 2823, "TN": 56557,
     "accuracy": 0.945, "precision": 0.916, "recall": 0.834, "f1": 0.873, "source": "06, cell 21"},
    {"group": "2024", "FN": 1381, "FP": 624, "accuracy": 0.953, "f1": 0.886, "source": "09, section 9.4"},
    {"group": "2025", "FN": 1442, "FP": 664, "accuracy": 0.934, "f1": 0.858, "source": "09, section 9.4"},
])


# =========================================================================================
# Sites and populations
# =========================================================================================
def fit_sites(ev: pd.DataFrame) -> pd.DataFrame:
    """Sites to fit: every 3L1P site whose reference configuration is a single-phase inverter (1P)."""
    s = ev[(ev.cohort == "3L1P") & (ev.ref_config == "1P")]
    return s[["site_id", "rating_kw"]].drop_duplicates("site_id").reset_index(drop=True)


def check_site_list(out_dir: Path, sites: pd.DataFrame, overwrite: bool) -> None:
    """
    `Fleet.run_months` skips months already on disk. Guard against re-using a cache that was built for a
    different site list (e.g. a test sample): stop unless the list is the same or `overwrite` is set.
    """
    out_dir = Path(out_dir)
    path = out_dir / "sites.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        old = pd.read_parquet(path)
        same = set(old.site_id) == set(sites.site_id)
        if not same:
            raise RuntimeError(f"{out_dir} was built for {len(old):,} other sites; set OVERWRITE = True "
                               "(or delete the folder) to rebuild it for the current site list.")
    sites[["site_id", "rating_kw"]].to_parquet(path, index=False)


def populations(ev: pd.DataFrame, *, ref_dv_ratio_min: float) -> pd.DataFrame:
    """
    One row per graded site (3L1P, reference 1P) with population flags:
      in_B  all reference-graded 1P sites (end-to-end grading)
      in_A  clean sites from notebook 10 (use_in_11): the main oracle result
      in_A3 in_A and n_decided >= 3 (stricter subset)
      in_C  excluded in notebook 10 only for blind reasons AND with a strong voltage reference
            (all PV circuits paired, consensus ok, median dv_ratio >= ref_dv_ratio_min): the
            selection-bias check. The phase-reference check is applied here because the notebook-10
            check order stops at the first failure, so e.g. too_few_decided_months sites were never tested.
    """
    p = ev[(ev.cohort == "3L1P") & (ev.ref_config == "1P")].copy()
    p["ref_strong"] = (p.ref_ok_all.astype("boolean").fillna(False).astype(bool)
                       & (p.dv_ratio_min >= ref_dv_ratio_min).fillna(False))
    p["in_B"] = True
    p["in_A"] = p.use_in_11.astype(bool)
    p["in_A3"] = p.in_A & (p.n_decided >= 3)
    p["blind_only_excluded"] = p.exclusion_reason.isin(BLIND_ONLY_REASONS)
    p["in_C"] = p.blind_only_excluded & p.ref_strong
    p["ref_phase"] = p.ref_phase.astype("Int64")
    p["blind_phase"] = p.blind_phase.astype("Int64")
    return p.reset_index(drop=True)


def reference_phase_month(truth_pairing: pd.DataFrame) -> pd.DataFrame:
    """Per site-month: the house phase the PV circuit is CONFIDENTLY voltage-matched to that month."""
    c = truth_pairing[truth_pairing.confident]
    c = c.sort_values("dv_ratio", ascending=False).drop_duplicates(KEYS)
    return c[KEYS + ["load_circuit_id", "dv_ratio"]].rename(
        columns={"load_circuit_id": "ref_phase_month", "dv_ratio": "dv_ratio_month"})


# =========================================================================================
# Fits
# =========================================================================================
def load_fits(out_dir: Path, sites: pd.DataFrame, *, min_n_window: int, min_v_spread: float,
              tolerance_pct: float, v_lo: float, v_hi: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (meter, truth) fits written by `Fleet.run_months` under out_dir/{meter,truth}, restricted to `sites`,
    with `fit_status` (as notebook 07, including `constant_q` for reference PV circuits that report a
    constant Q) and `passed` (NA when not assessable). `meter` has one row per house-phase circuit.
    """
    out = []
    for name in ("meter", "truth"):
        r = Fleet.load_results(Path(out_dir) / name)
        r = r[r.site_id.isin(sites.site_id)]
        r = Fleet.add_fit_status(r, min_n_window=min_n_window, min_v_spread=min_v_spread)
        if name == "truth":
            cq = (r.fit_status == "ok") & (r.slope == 0) & (r.slope_stderr == 0)
            r.loc[cq, "fit_status"] = "constant_q"
        ok = r.fit_status == "ok"
        # Voltage outliers (e.g. ~214,869 V readings) cannot enter a fit: the window is [v_lo, v_hi)
        bad_v = ok & ~(r.v_min_obs.between(v_lo, v_hi) & r.v_max_obs.between(v_lo, v_hi))
        assert not bad_v.any(), f"{name}: {bad_v.sum()} fits use voltages outside [{v_lo}, {v_hi})"
        pf = ConformancePlot.add_slope_pass_fail(r[ok], tolerance_pct)
        r["passed"] = pd.Series(pd.NA, index=r.index, dtype="boolean")
        r.loc[ok, "passed"] = pf.passed.astype("boolean")
        r["circuit_id"] = r.circuit_id.astype("Int64")
        out.append(r.reset_index(drop=True))
    meter, truth = out
    n_dup = truth.dropna(subset=["circuit_id"]).duplicated(KEYS).sum()
    assert n_dup == 0, f"{n_dup} site-months with more than one PV circuit in the reference"
    return meter, truth


def _pick(meter: pd.DataFrame, units: pd.DataFrame, circuit_col: str, tag: str) -> pd.DataFrame:
    """The meter fit of circuit `circuit_col` for each row of `units` (KEYS + circuit_col)."""
    cols = ["circuit_id", "slope", "intercept", "r_squared", "n_window", "fit_status", "passed",
            "base_p_kw", "base_q_kvar", "rating_kw"]
    m = meter[KEYS + cols].rename(columns={"circuit_id": circuit_col})
    out = units.merge(m, on=KEYS + [circuit_col], how="left")
    out["fit_status"] = out.fit_status.fillna("no_circuit_data")
    return out.rename(columns={c: f"{c}_{tag}" for c in cols if c != "circuit_id"})


def oracle_table(meter: pd.DataFrame, truth: pd.DataFrame, pop: pd.DataFrame,
                 blind_month: pd.DataFrame, ref_month: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (graded site, month with a reference PV fit row): the meter fit on the site's reference
    phase (`ref_phase`), the reference PV fit, and the month filters:
      month_ok      notebook 10's filter (clean site, blind result agrees that month, confident match)
      ref_month_ok  reference-only filter: the PV circuit is confidently matched to ref_phase that month
                    (no blind information used); for the selection-bias comparison.
    """
    t = truth[KEYS + ["circuit_id", "slope", "fit_status", "passed", "n_window"]].rename(
        columns={"circuit_id": "pv_circuit_id", "slope": "slope_truth", "fit_status": "fit_status_truth",
                 "passed": "pass_truth", "n_window": "n_window_truth"})
    u = t.merge(pop[["site_id", "ref_phase", "in_A", "in_A3", "in_C", "model_hint", "exclusion_reason"]],
                on="site_id")
    u = _pick(meter, u, "ref_phase", "oracle")
    u = u.merge(blind_month[KEYS + ["month_ok", "blind_config", "n_circuits"]], on=KEYS, how="left")
    u["month_ok"] = u.month_ok.astype("boolean").fillna(False).astype(bool)
    u = u.merge(ref_month[KEYS + ["ref_phase_month"]], on=KEYS, how="left")
    u["ref_month_ok"] = (u.ref_phase_month == u.ref_phase).fillna(False).astype(bool)
    u["both_ok"] = (u.fit_status_oracle == "ok") & (u.fit_status_truth == "ok")
    return u.rename(columns={"slope_oracle": "slope_blind", "passed_oracle": "pass_blind"})


# =========================================================================================
# Metrics
# =========================================================================================
def metrics(df: pd.DataFrame, by=None, pred: str = "pass_blind", actual: str = "pass_truth") -> pd.DataFrame:
    """Valid.classification_metrics (positive = conformant) overall or per group, plus pass rates and sites."""
    def one(g):
        p, a = g[pred].astype(bool), g[actual].astype(bool)
        return {"sites": g.site_id.nunique(), **Valid.classification_metrics(p, a),
                "pass_rate_tested_%": 100 * p.mean(), "pass_rate_reference_%": 100 * a.mean()}
    d = df.dropna(subset=[pred, actual])
    if by is None:
        return pd.DataFrame([one(d)], index=["all"])
    return pd.DataFrame({k: one(g) for k, g in d.groupby(by)}).T


def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42, level: float = 0.95,
                 pred: str = "pass_blind", actual: str = "pass_truth") -> dict:
    """
    Site-cluster bootstrap CI of accuracy and F1: resample SITES with replacement (site-months of one site
    are not independent), pool their confusion counts, recompute. Returns point estimates and CIs.
    """
    d = df.dropna(subset=[pred, actual]).copy()
    p, a = d[pred].astype(bool), d[actual].astype(bool)
    d["TP"], d["FP"], d["FN"], d["TN"] = (p & a), (p & ~a), (~p & a), (~p & ~a)
    c = d.groupby("site_id")[["TP", "FP", "FN", "TN"]].sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    s = c[idx].sum(axis=1)                              # (n_boot, 4) pooled counts
    tp, fp, fn, tn = s.T
    acc = (tp + tn) / s.sum(axis=1)
    f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.where(fp + fn > 0, 0.0, np.nan))
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    TP, FP, FN, TN = c.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_point = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else np.nan
    if np.isnan(f1).all():
        f1 = np.array([np.nan, np.nan])
    return {"sites": len(c), "n": int(c.sum()), "accuracy": (TP + TN) / c.sum(),
            "accuracy_lo": np.nanquantile(acc, q[0]), "accuracy_hi": np.nanquantile(acc, q[1]),
            "f1": f1_point, "f1_lo": np.quantile(f1, q[0]) if np.isnan(f1).all() else np.nanquantile(f1, q[0]),
            "f1_hi": np.quantile(f1, q[1]) if np.isnan(f1).all() else np.nanquantile(f1, q[1])}


# =========================================================================================
# End-to-end blind route
# =========================================================================================
def route_end_to_end(meter: pd.DataFrame, truth: pd.DataFrame, pop: pd.DataFrame, blind_month: pd.DataFrame,
                     ref_month: pd.DataFrame, phase_choice: str = "month") -> pd.DataFrame:
    """
    Every graded site-month with an ASSESSABLE reference (PV fit ok), routed by the blind method only:
      1. configuration: that month's blind_config (1P -> single-phase test; 3P -> wrong route; anything
         else or no net-meter signature -> not assessable);
      2. phase: that month's blind phase (`phase_choice="month"`) or the site's majority blind phase
         (`"site"`), graded against the confidently matched phase that month, else the site consensus
         phase when the site's reference is strong, else unknown;
      3. Volt-VAr: the single-phase test on the routed circuit, graded against the PV fit.
    `outcome`: not_assessable_config | error_routed_3P | not_assessable_vvar | agree | disagree.
    Also carries the oracle fit (reference phase) for the same months.
    """
    t = truth[truth.fit_status == "ok"][KEYS + ["slope", "passed"]].rename(
        columns={"slope": "slope_truth", "passed": "pass_truth"})
    u = t.merge(pop[["site_id", "ref_phase", "ref_strong", "blind_phase", "in_A", "in_C"]], on="site_id")
    u = u.merge(blind_month[KEYS + ["blind_config", "blind_load_circuit_id", "month_ok"]], on=KEYS, how="left")
    u["blind_config"] = u.blind_config.fillna("no_signature")
    u = u.merge(ref_month[KEYS + ["ref_phase_month"]], on=KEYS, how="left")

    u["route"] = np.where(u.blind_config.isin(["1P", "3P"]), u.blind_config, "undecided")
    phase = (u.blind_load_circuit_id if phase_choice == "month" else u.blind_phase).astype("Float64")
    u["route_circuit"] = phase.where(u.route == "1P").astype("Int64")
    grade = u.ref_phase_month.astype("Int64").fillna(u.ref_phase.where(u.ref_strong).astype("Int64"))
    u["grade_phase"] = grade
    u["phase_status"] = np.select(
        [u.route != "1P", grade.isna(), (u.route_circuit == grade).fillna(False).astype(bool)],
        ["-", "no_reference", "correct"], default="wrong")

    u = _pick(meter, u, "route_circuit", "route")
    u = _pick(meter, u, "ref_phase", "oracle")
    ok = u.fit_status_route == "ok"
    agree = (u.passed_route == u.pass_truth).fillna(False).astype(bool)
    u["outcome"] = np.select(
        [u.route == "undecided", u.route == "3P", (u.route == "1P") & ~ok, agree],
        ["not_assessable_config", "error_routed_3P", "not_assessable_vvar", "agree"], default="disagree")
    return u


def step_table(r: pd.DataFrame) -> pd.DataFrame:
    """Where reference-assessable site-months are lost or wrong, step by step."""
    n = len(r)
    one = r[r.route == "1P"]
    rows = [
        ("reference assessable (PV fit ok)", n),
        ("1. configuration: undecided (not assessable)", (r.route == "undecided").sum()),
        ("1. configuration: routed 3P (wrong route)", (r.route == "3P").sum()),
        ("1. configuration: routed 1P", len(one)),
        ("2. phase: correct", (one.phase_status == "correct").sum()),
        ("2. phase: wrong", (one.phase_status == "wrong").sum()),
        ("2. phase: no reference phase to grade", (one.phase_status == "no_reference").sum()),
        ("3. Volt-VAr: routed circuit not assessable", (r.outcome == "not_assessable_vvar").sum()),
        ("3. Volt-VAr: verdict agrees with reference", (r.outcome == "agree").sum()),
        ("3. Volt-VAr: verdict disagrees", (r.outcome == "disagree").sum()),
    ]
    t = pd.DataFrame(rows, columns=["step", "site_months"]).set_index("step")
    t["% of reference assessable"] = (100 * t.site_months / n).round(1)
    return t


def end_to_end_summary(r: pd.DataFrame) -> pd.DataFrame:
    """
    Accuracy at each step, all on site-months with an assessable reference:
      end_to_end      agree / (agree + disagree + routed 3P)   -- routed-3P months count as errors
      given_1P_route  agree / (agree + disagree)
      by phase status (correct / wrong)
      oracle_same     the oracle (reference phase) on the same routed-and-assessable months, for comparison
    """
    def acc(mask):
        g = r[mask]
        return {"site_months": len(g), "sites": g.site_id.nunique(),
                "accuracy": (g.outcome == "agree").mean() if len(g) else np.nan}
    graded = r.outcome.isin(["agree", "disagree", "error_routed_3P"])
    routed = r.outcome.isin(["agree", "disagree"])
    both = routed & (r.fit_status_oracle == "ok")
    oracle = r[both]
    rows = {
        "end_to_end (routed 3P = error)": acc(graded),
        "given a 1P route": acc(routed),
        "given a 1P route, phase correct": acc(routed & (r.phase_status == "correct")),
        "given a 1P route, phase wrong": acc(routed & (r.phase_status == "wrong")),
        "blind route, months where the oracle is also assessable": acc(both),
        "oracle, same months": {"site_months": len(oracle), "sites": oracle.site_id.nunique(),
                                "accuracy": (oracle.passed_oracle == oracle.pass_truth).mean() if len(oracle) else np.nan},
    }
    out = pd.DataFrame(rows).T
    out["coverage_%_of_reference"] = (100 * out.site_months.astype(float) / len(r)).round(1)
    return out


# =========================================================================================
# Low PV output months
# =========================================================================================
def pv_output_ratio(pv_output_dir: Path, pop: pd.DataFrame) -> pd.DataFrame:
    """Per site-month: PV circuit midday q90 / rating (notebook 10's `pv_output` cache)."""
    pv = pd.concat([pd.read_parquet(f) for f in sorted(Path(pv_output_dir).glob("*.parquet"))], ignore_index=True)
    pv = pv.merge(pop[["site_id", "rating_kw"]], on="site_id")
    g = pv.groupby(KEYS).agg(pv_mid_q90=("pv_mid_q90", "sum"), rating_kw=("rating_kw", "first")).reset_index()
    g["pv_q90_over_rating"] = g.pv_mid_q90 / g.rating_kw
    return g[KEYS + ["pv_q90_over_rating"]]


# =========================================================================================
# Missing paired-phase circuit
# =========================================================================================
def affected_sites(blind_month: pd.DataFrame, min_incomplete: int = 6) -> pd.DataFrame:
    """Clean 1P sites with >= min_incomplete 'incomplete' months (fewer than 3 house circuits) in a year."""
    c = blind_month[(blind_month.exclusion_reason == "clean") & (blind_month.expected == "1P")]
    inc = (c.blind_config == "incomplete").groupby([c.site_id, c.year]).sum().unstack(fill_value=0)
    inc.columns = [f"incomplete_{y}" for y in inc.columns]
    inc["affected_year"] = np.select([inc.get("incomplete_2024", 0) >= min_incomplete,
                                      inc.get("incomplete_2025", 0) >= min_incomplete], [2024, 2025], 0)
    return inc.reset_index()


def presence_sql(table: str, month: str, glob: str = "*.parquet") -> str:
    """Rows per circuit in one month of a store table, for the circuits in a registered `_circ` table."""
    src = f"read_parquet('{Fleet.month_dir(table, month).as_posix()}/{glob}')"
    return (f"SELECT circuit_id, '{table}' AS tbl, '{month}' AS ym, count(*) AS n_rows FROM {src} "
            f"WHERE circuit_id IN (SELECT circuit_id FROM _circ) GROUP BY ALL")


def athena_presence_sql(circuit_ids, start: str = "2024-01-01", end: str = "2026-01-01") -> str:
    """
    Proposed (not run) Athena check: rows per circuit and month in the Solar Analytics `ts` table. Check
    the table's partition columns first and add a partition filter if there is one (cost).
    """
    ids = ", ".join(str(int(c)) for c in sorted(set(circuit_ids)))
    return f"""SELECT circuit_id, date_trunc('month', t_stamp) AS month_utc, count(*) AS n_rows
FROM ts
WHERE circuit_id IN ({ids})
  AND t_stamp >= TIMESTAMP '{start}' AND t_stamp < TIMESTAMP '{end}'
GROUP BY 1, 2
ORDER BY 1, 2"""


# =========================================================================================
# Other-phase effect: household reactive power on the tested phase (notebook 09's beta_e)
# =========================================================================================
def beta_e_sql(month: str, *, night_hours=(0, 4), p_min_frac: float = Fleet.P_MIN_FRAC,
               v_lo: float = 242, v_hi: float = 253) -> str:
    """
    One row per unit in the registered `_units` table (site_id, meter_circuit_id, rating_kw) for `month`, on
    the intervals inside BOTH fit windows (meter circuit and PV circuit, each with its own gate and voltage,
    exactly as notebook 09): e = 100 (Q_est - Q_pv) / rating = the household reactive load on that circuit
    relative to its night median; beta_e = OLS slope of e on the meter voltage; sd and mean of e.
    Requires `_sites` (site_id, rating_kw) as well, for the estimate CTEs.
    """
    year, mon = (int(x) for x in month.split("-"))
    m = Fleet._estimate_sql(month, "meter_baseline", night_hours)
    t = Fleet._estimate_sql(month, "ground_truth", night_hours)
    g = p_min_frac
    return f"""
        WITH m AS (WITH {m} SELECT site_id, circuit_id, t_stamp, V, p_est, q_est FROM est),
             t AS (WITH {t} SELECT site_id, t_stamp, V, p_est, q_est FROM est),
             s AS (
                SELECT u.site_id, m.V AS v_b,
                       100.0 * (m.q_est - t.q_est) / u.rating_kw AS e
                FROM _units u
                JOIN m ON m.site_id = u.site_id AND m.circuit_id = u.meter_circuit_id
                JOIN t ON t.site_id = u.site_id AND t.t_stamp = m.t_stamp
                WHERE m.V >= {v_lo} AND m.V < {v_hi} AND m.p_est >= {g} * u.rating_kw AND m.q_est IS NOT NULL
                  AND t.V >= {v_lo} AND t.V < {v_hi} AND abs(t.p_est) >= {g} * u.rating_kw
                  AND t.q_est IS NOT NULL
             )
        SELECT site_id, {year} AS year, {mon} AS month, count(*) AS n_shared,
               regr_slope(e, v_b) AS beta_e, stddev_samp(e) AS e_sd, avg(e) AS e_mean
        FROM s GROUP BY ALL
    """


def run_beta_e(units: pd.DataFrame, months: list[str], cache_dir: Path, tag: str, *,
               memory_limit: str, threads: int, overwrite: bool = False, **sql_kwargs) -> pd.DataFrame:
    """beta_e_sql for `units` (KEYS + meter_circuit_id + rating_kw), month by month, cached per month."""
    cache_dir = Path(cache_dir) / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for month in months:
        y, mo = int(month[:4]), int(month[5:])
        u = units[(units.year == y) & (units.month == mo)]
        path = cache_dir / f"{month}.parquet"
        if u.empty:
            continue
        if path.exists() and not overwrite:
            parts.append(pd.read_parquet(path))
            continue
        con = Fleet.connect(memory_limit, threads)
        try:
            con.register("_units", u[["site_id", "meter_circuit_id", "rating_kw"]].astype(
                {"site_id": "int64", "meter_circuit_id": "int64", "rating_kw": "float64"}))
            con.register("_sites", u[["site_id", "rating_kw"]].drop_duplicates("site_id"))
            df = con.execute(beta_e_sql(month, **sql_kwargs)).df()
        finally:
            con.close()
        df.to_parquet(path, index=False)
        parts.append(df)
        print(f"beta_e {tag} {month}: {len(df):,} site-months")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def distribution_table(groups: dict, cols: list[str]) -> pd.DataFrame:
    """Median, IQR and median |x| of `cols` for each named frame in `groups`."""
    rows = {}
    for name, d in groups.items():
        r = {"site_months": len(d)}
        for c in cols:
            x = d[c].dropna()
            r.update({f"{c}_median": x.median(), f"{c}_p25": x.quantile(.25), f"{c}_p75": x.quantile(.75),
                      f"{c}_median_abs": x.abs().median()})
        rows[name] = r
    return pd.DataFrame(rows).T


# =========================================================================================
# Output
# =========================================================================================
def to_result_columns(fits: pd.DataFrame, *, source: str, cohort: str = "3L1P") -> pd.DataFrame:
    """
    Rows in Phase.RESULT_COLUMNS format. `fits`: per-site-month fit columns (slope, intercept, r_squared,
    slope_stderr, n_window, v_min_obs, v_max_obs, v_spread, fit_status, passed, rating_kw).
    One single-phase inverter per site, so unit_id = site_id.
    """
    out = fits.copy()
    out["cohort"], out["source"], out["unit_id"] = cohort, source, out.site_id
    return out[Phase.RESULT_COLUMNS].reset_index(drop=True)


# =========================================================================================
# "Phantom export" check: fitted net-meter intervals with no PV output, or outside daylight
# =========================================================================================
def outside_daylight_sql(month: str, *, night_hours=(0, 4), p_min_frac: float = Fleet.P_MIN_FRAC,
                         v_lo: float = 242, v_hi: float = 253, day_start: int = 7, day_end: int = 18,
                         pv_off_frac: float = 0.05) -> str:
    """
    "Phantom export" check. One row per unit in the registered `_units` table (site_id, meter_circuit_id,
    rating_kw) for `month`, over the intervals in the net-meter fit window (`n_window_fit`, exactly the fit's
    intervals):
      n_outside_daylight  AEST hour < day_start or >= day_end
      n_pv_off            the PV circuit's |P| < pv_off_frac x rating at the same timestamp: the net meter
                          shows "export" above the gate while the PV produces almost nothing. This happens when
                          a regular night load raises the 00-04 h baseline and base - net then exceeds the gate.
      n_pv_missing        no PV-circuit reading at that timestamp
    Requires `_sites` (site_id, rating_kw) for the estimate CTEs.
    """
    year, mon = (int(x) for x in month.split("-"))
    h = "(EXTRACT(hour FROM m.t_stamp) + 10) % 24"
    m = Fleet._estimate_sql(month, "meter_baseline", night_hours)
    t = Fleet._estimate_sql(month, "ground_truth", night_hours)
    return f"""
        WITH m AS (WITH {m}
                   SELECT e.site_id, e.circuit_id, e.t_stamp FROM est e JOIN _sites s USING (site_id)
                   WHERE {Fleet._window_sql("meter_baseline", p_min_frac, v_lo, v_hi)}),
             t AS (WITH {t} SELECT site_id, t_stamp, p_est FROM est)
        SELECT m.site_id, {year} AS year, {mon} AS month, count(*) AS n_window_fit,
               sum(CASE WHEN {h} < {day_start} OR {h} >= {day_end} THEN 1 ELSE 0 END) AS n_outside_daylight,
               sum(CASE WHEN abs(t.p_est) < {pv_off_frac} * u.rating_kw THEN 1 ELSE 0 END) AS n_pv_off,
               sum(CASE WHEN t.p_est IS NULL THEN 1 ELSE 0 END) AS n_pv_missing
        FROM m
        JOIN _units u ON u.site_id = m.site_id AND u.meter_circuit_id = m.circuit_id
        LEFT JOIN t ON t.site_id = m.site_id AND t.t_stamp = m.t_stamp
        GROUP BY ALL
    """


def run_outside_daylight(units: pd.DataFrame, months: list[str], cache_dir: Path, tag: str, *,
                         memory_limit: str, threads: int, overwrite: bool = False, **sql_kwargs) -> pd.DataFrame:
    """outside_daylight_sql for `units` (KEYS + meter_circuit_id + rating_kw), month by month, cached per month."""
    cache_dir = Path(cache_dir) / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for month in months:
        y, mo = int(month[:4]), int(month[5:])
        u = units[(units.year == y) & (units.month == mo)]
        path = cache_dir / f"{month}.parquet"
        if u.empty:
            continue
        if path.exists() and not overwrite:
            parts.append(pd.read_parquet(path))
            continue
        con = Fleet.connect(memory_limit, threads)
        try:
            con.register("_units", u[["site_id", "meter_circuit_id", "rating_kw"]].astype(
                {"site_id": "int64", "meter_circuit_id": "int64", "rating_kw": "float64"}))
            con.register("_sites", u[["site_id", "rating_kw"]].drop_duplicates("site_id"))
            df = con.execute(outside_daylight_sql(month, **sql_kwargs)).df()
        finally:
            con.close()
        df.to_parquet(path, index=False)
        parts.append(df)
        print(f"outside daylight {tag} {month}: {len(df):,} site-months")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def outside_daylight_table(d: pd.DataFrame, share_col: str = "share_pv_off", by: str = "outcome",
                           thresholds=(0.10, 0.25)) -> pd.DataFrame:
    """Per group: site-months, median of `share_col`, and % of site-months with `share_col` above each threshold."""
    g = d.groupby(by)
    out = pd.DataFrame({"site_months": g.size(), "sites": g.site_id.nunique(),
                        f"median_{share_col}": g[share_col].median()})
    for t in thresholds:
        out[f"% site-months > {t:.0%}"] = 100 * g[share_col].apply(lambda s: (s > t).mean())
    return out
