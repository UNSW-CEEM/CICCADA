"""
Notebook 13: fleet Volt-VAr conformance (AS/NZS 4777.2:2020) from the net meter only, for three cohorts:

  1L1P  single-phase house, single-phase inverter      (notebook 07: blind fit, all eligible sites)
  3L1P  three-phase house, single-phase inverter       (notebook 11: blind route on the blind phase)
  3L3P  three-phase house, three-phase inverter        (notebook 12: total of the three phases vs V_avg)

Main population: for 3L1P/3L3P the clean sites (`use_in_11` / `use_in_12`) and their `month_ok` months, CATCH Power
excluded for 3L3P (what notebooks 11/12 validated). Sensitivity "all decided": every month the blind method routes.

A site-month is conformant if its slope is in [-4.000, -2.667] %/V. Site level (as notebook 07): conformant if it
passes in ALL its assessable months; sensitivity: in at least half of them.

Bias-adjusted rate (sensitivity only): each cohort's observed site-month rate corrected with that cohort's validated
sensitivity (recall) and specificity, p_adj = (p_obs + spec - 1) / (sens + spec - 1) (Rogan and Gladen, 1978,
American Journal of Epidemiology 107(1):71-76; verify the reference before citing). It assumes the validation error
rates carry over to the whole cohort. Because this dataset also has the PV-circuit reference, the adjustment is
checked out of sample: error rates from one year, applied to the other year, against that year's reference rate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bms_sa_review.ami_analysis.lib import ami_conformance_plot as ConformancePlot
from bms_sa_review.ami_analysis.lib import ami_phase1p_vvar as P1

KEYS = ["site_id", "year", "month"]
COHORTS = ("1L1P", "3L1P", "3L3P")
VARIANTS = ("main", "all_decided", "n_decided_3", "catch_included")


# =========================================================================================
# Cohort site-months (net-meter verdict, with the reference verdict where it exists)
# =========================================================================================
def _with_truth(blind: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    t = truth[truth.fit_status == "ok"][KEYS + ["passed", "slope"]].rename(
        columns={"passed": "pass_truth", "slope": "slope_truth"})
    return blind.merge(t, on=KEYS, how="left")


def cohort_1l1p(fleet_dir: Path, tolerance_pct: float) -> pd.DataFrame:
    """Notebook 07: blind fits `ok` at eligible single-phase sites (all variants are the same population)."""
    fleet_dir = Path(fleet_dir)
    r = pd.read_parquet(fleet_dir / "site_month_results.parquet")
    elig = pd.read_parquet(fleet_dir / "site_eligibility.parquet")
    r = r[r.site_id.isin(elig.loc[elig.eligibility == "eligible", "site_id"])]
    pf = lambda src: ConformancePlot.add_slope_pass_fail(r[(r.source == src) & (r.fit_status == "ok")], tolerance_pct)
    b = pf("meter_baseline")[KEYS + ["slope", "passed"]]
    t = pf("ground_truth").assign(fit_status="ok")
    return _with_truth(b, t).assign(cohort="1L1P")


def cohort_3l1p(dir_3l1p: Path, variant: str = "main") -> pd.DataFrame:
    """Notebook 11: blind-route fits `ok`. main = clean & month_ok; all_decided = every 1P-routed month;
    n_decided_3 = A3 & month_ok."""
    d = pd.read_parquet(Path(dir_3l1p) / "site_month_details_3l1p.parquet")
    r = pd.read_parquet(Path(dir_3l1p) / "site_month_results_3l1p.parquet")
    b = r[(r.source == "meter_blind_route") & (r.fit_status == "ok")][KEYS + ["slope", "passed"]]
    if variant in ("main", "catch_included"):
        b = b.merge(d[d.in_A & d.month_ok][KEYS], on=KEYS)
    elif variant == "n_decided_3":
        b = b.merge(d[d.in_A3 & d.month_ok][KEYS], on=KEYS)
    elif variant != "all_decided":
        raise ValueError(variant)
    return _with_truth(b, r[r.source == "ground_truth"]).assign(cohort="3L1P")


def cohort_3l3p(dir_3l3p: Path, variant: str = "main") -> pd.DataFrame:
    """Notebook 12: net-meter total fits `ok` (V_avg). main = clean & month_ok & no CATCH Power; all_decided = every
    month the blind method calls 3P (no CATCH); n_decided_3 = A3 & month_ok (no CATCH); catch_included = main + CATCH."""
    d = pd.read_parquet(Path(dir_3l3p) / "site_month_details_3l3p.parquet")
    r = pd.read_parquet(Path(dir_3l3p) / "site_month_results_3l3p.parquet")
    b = r[(r.source == "meter_total") & (r.fit_status == "ok")][KEYS + ["slope", "passed"]]
    sel = {"main": d.in_A & d.month_ok & ~d.catch,
           "all_decided": (d.blind_config == "3P") & ~d.catch,
           "n_decided_3": d.in_A3 & d.month_ok & ~d.catch,
           "catch_included": d.in_A & d.month_ok}[variant]
    b = b.merge(d[sel][KEYS], on=KEYS)
    return _with_truth(b, r[r.source == "ground_truth_total"]).assign(cohort="3L3P")


def load_fleet(fleet_dir: Path, dir_3l1p: Path, dir_3l3p: Path, *, tolerance_pct: float,
               variant: str = "main") -> pd.DataFrame:
    """All three cohorts as one site-month table: cohort, KEYS, slope, passed, slope_truth, pass_truth."""
    parts = [cohort_1l1p(fleet_dir, tolerance_pct), cohort_3l1p(dir_3l1p, variant), cohort_3l3p(dir_3l3p, variant)]
    f = pd.concat(parts, ignore_index=True)
    f["passed"] = f.passed.astype(bool)
    f["pass_truth"] = f.pass_truth.astype("boolean")
    dup = f.groupby("site_id").cohort.nunique()
    assert (dup == 1).all(), f"{(dup > 1).sum()} sites appear in more than one cohort"
    assert not f.duplicated(KEYS).any(), "duplicated site-months"
    return f


def add_metadata(f: pd.DataFrame, site_meta: pd.DataFrame, mandatory: pd.Timestamp) -> pd.DataFrame:
    """OEM, DNSP, install date/year and era (installed before / after the 2020 standard became mandatory)."""
    m = site_meta[["site_id", "manufacturer", "dnsp_name", "pv_install_date", "ac_capacity_kw"]].rename(
        columns={"manufacturer": "oem", "dnsp_name": "dnsp", "pv_install_date": "install_date",
                 "ac_capacity_kw": "rating_kw"})
    out = f.merge(m, on="site_id", how="left")
    out[["oem", "dnsp"]] = out[["oem", "dnsp"]].fillna("Unknown")
    out["install_date"] = pd.to_datetime(out.install_date)
    out["install_year"] = out.install_date.dt.year
    out["install_era"] = np.where(out.install_date >= mandatory, "after 2020 standard mandatory",
                                  np.where(out.install_date.isna(), "unknown", "before"))
    return out


# =========================================================================================
# Rates
# =========================================================================================
def site_level(f: pd.DataFrame, share_main: float = 1.0, share_sens: float = 0.5) -> pd.DataFrame:
    """One row per site: assessable months, share passed, conformant (>= share_main) and conformant_sens."""
    meta = [c for c in ["cohort", "oem", "dnsp", "install_year", "install_era", "rating_kw"] if c in f.columns]
    s = (f.groupby("site_id").agg(n_months=("passed", "size"), share_pass=("passed", "mean"),
                                  **{c: (c, "first") for c in meta}).reset_index())
    s["conformant"] = s.share_pass >= share_main
    s["conformant_sens"] = s.share_pass >= share_sens
    return s


def cohort_rates(f: pd.DataFrame, s: pd.DataFrame, by=("cohort",)) -> pd.DataFrame:
    """Site-month and site-level conformance per group (pooled row 'all' added)."""
    by = list(by)

    def one(fm, ss):
        return pd.Series({"sites": ss.site_id.nunique(), "site_months": len(fm),
                          "site_month_conformant_%": 100 * fm.passed.mean(),
                          "site_conformant_%": 100 * ss.conformant.mean(),
                          "site_conformant_%_(>=50%_of_months)": 100 * ss.conformant_sens.mean(),
                          "median_months_per_site": ss.n_months.median()})
    rows = {k if isinstance(k, tuple) else (k,): one(g, s[s.site_id.isin(g.site_id)]) for k, g in f.groupby(by)}
    rows[("all",) * len(by)] = one(f, s)
    out = pd.DataFrame(rows).T
    out.index.names = by
    return out


def rate_table(f: pd.DataFrame, s: pd.DataFrame, col: str, min_sites: int = 10) -> pd.DataFrame:
    """Conformance by `col` (site-month and site level), groups with < min_sites sites hidden; cohort mix shown."""
    g = f.groupby(col)
    out = pd.DataFrame({"sites": g.site_id.nunique(), "site_months": g.size(),
                        "site_month_conformant_%": 100 * g.passed.mean()})
    out["site_conformant_%"] = 100 * s.groupby(col).conformant.mean()
    mix = s.groupby([col, "cohort"]).size().unstack(fill_value=0)
    out = out.join(mix.add_prefix("sites_"), how="left")
    return out[out.sites >= min_sites].sort_values("sites", ascending=False)


# =========================================================================================
# Validation error rates and the bias-adjusted rate
# =========================================================================================
def validation_sets(fleet_dir: Path, dir_3l1p: Path, dir_3l3p: Path, tolerance_pct: float) -> dict:
    """
    The site-months each cohort was validated on (pass_blind / pass_truth), as reported in notebooks 06/11/12:
    1L1P: notebook 07 both fits ok (eligible sites); 3L1P: clean, month_ok, oracle = blind phase on those months;
    3L3P: clean, month_ok, no CATCH Power.
    """
    v1 = cohort_1l1p(fleet_dir, tolerance_pct).dropna(subset=["pass_truth"])
    v3 = cohort_3l1p(dir_3l1p, "main").dropna(subset=["pass_truth"])
    v33 = cohort_3l3p(dir_3l3p, "main").dropna(subset=["pass_truth"])
    return {name: v.rename(columns={"passed": "pass_blind"})[KEYS + ["pass_blind", "pass_truth"]]
            for name, v in [("1L1P", v1), ("3L1P", v3), ("3L3P", v33)]}


def rogan_gladen(p_obs, sens, spec):
    """Prevalence corrected for test error; clipped to [0, 1]."""
    return np.clip((np.asarray(p_obs) + spec - 1) / (sens + spec - 1), 0, 1)


def _site_counts_conf(v: pd.DataFrame) -> np.ndarray:
    p, a = v.pass_blind.astype(bool), v.pass_truth.astype(bool)
    c = pd.DataFrame({"site_id": v.site_id, "TP": p & a, "FP": p & ~a, "FN": ~p & a, "TN": ~p & ~a})
    return c.groupby("site_id")[["TP", "FP", "FN", "TN"]].sum().to_numpy(float)


def adjusted_rates(f: pd.DataFrame, validation: dict, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Per cohort: observed site-month rate (all fleet site-months), validated sensitivity / specificity, and the
    bias-adjusted rate with a bootstrap interval (fleet sites and validation sites resampled independently).
    """
    rng = np.random.default_rng(seed)
    rows = {}
    for cohort, g in f.groupby("cohort"):
        fc = g.groupby("site_id").passed.agg(["sum", "size"]).to_numpy(float)
        vc = _site_counts_conf(validation[cohort])
        TP, FP, FN, TN = vc.sum(axis=0)
        sens, spec = TP / (TP + FN), TN / (TN + FP)
        p_obs = g.passed.mean()
        fi = rng.integers(0, len(fc), size=(n_boot, len(fc)))
        vi = rng.integers(0, len(vc), size=(n_boot, len(vc)))
        fs = fc[fi].sum(axis=1)
        vs = vc[vi].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            b_sens = vs[:, 0] / (vs[:, 0] + vs[:, 2])
            b_spec = vs[:, 3] / (vs[:, 3] + vs[:, 1])
            b_adj = rogan_gladen(fs[:, 0] / fs[:, 1], b_sens, b_spec)
        rows[cohort] = {
            "sites": g.site_id.nunique(), "site_months": len(g),
            "observed_%": 100 * p_obs, "sensitivity": sens, "specificity": spec,
            "adjusted_%": 100 * float(rogan_gladen(p_obs, sens, spec)),
            "adjusted_lo_%": 100 * np.nanquantile(b_adj, 0.025), "adjusted_hi_%": 100 * np.nanquantile(b_adj, 0.975),
        }
    return pd.DataFrame(rows).T


def adjusted_cross_year(f: pd.DataFrame, validation: dict) -> pd.DataFrame:
    """
    Out-of-sample check of the adjustment. On the validation site-months themselves the adjusted rate equals the
    reference rate by construction, so instead: sensitivity / specificity from one year's validation set, applied to
    the OTHER year's fleet site-months that have an assessable reference, compared with that year's reference rate.
    """
    rows = {}
    for cohort, g in f.groupby("cohort"):
        v = validation[cohort]
        for train, test in [(2024, 2025), (2025, 2024)]:
            vc = _site_counts_conf(v[v.year == train]).sum(axis=0)
            TP, FP, FN, TN = vc
            sens, spec = TP / (TP + FN), TN / (TN + FP)
            t = g[(g.year == test)].dropna(subset=["pass_truth"])
            p_obs = t.passed.mean()
            rows[(cohort, f"{train} -> {test}")] = {
                "site_months": len(t), "sensitivity_train": sens, "specificity_train": spec,
                "net_meter_%": 100 * p_obs, "adjusted_%": 100 * float(rogan_gladen(p_obs, sens, spec)),
                "reference_%": 100 * t.pass_truth.astype(bool).mean()}
    out = pd.DataFrame(rows).T
    out["error_net_meter_pp"] = out["net_meter_%"] - out["reference_%"]
    out["error_adjusted_pp"] = out["adjusted_%"] - out["reference_%"]
    return out


def validation_summary(validation: dict) -> pd.DataFrame:
    """P1.metrics per cohort on its validation set (the numbers reported in notebooks 06/11/12)."""
    return pd.DataFrame({k: P1.metrics(v).iloc[0] for k, v in validation.items()}).T
