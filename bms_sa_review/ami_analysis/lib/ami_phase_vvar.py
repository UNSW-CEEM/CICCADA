"""
Three-phase houses: phase configuration and phase pairing (notebook 10), and the
shared result format used by notebooks 11-13.

Cohorts (from `ami_site_metadata_residential`):
  3L1P  three-phase house (3 `ac_load_net` circuits), one PV circuit
  3L3P  three-phase house, three PV circuits

The net meter is one `ac_load_net` circuit per house phase, so a three-phase site
has three net-meter circuits. Two questions decide how the Volt-VAr test applies:

  1. Which configuration is it (single-phase inverter on one phase, or a
     three-phase inverter)? Blind: from the per-phase export signature of the net
     meter only. Reference: `n_pv_phases` in the site metadata.
  2. For a single-phase inverter, which house phase is it on? Blind: the phase with
     the export signature. Reference: the load circuit whose voltage matches the PV
     circuit's voltage (same phase -> near-identical voltage).

AS/NZS 4777.2:2020 Clause 3.3.2.1: a three-phase inverter uses the average of the
three single-phase voltages as the reference voltage for the volt response modes.

This module does not change `ami_fleet_vvar`; it reuses its estimate/window SQL so
the night baseline, sign handling and gate stay identical to notebooks 06/07.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bms_sa_review.ami_analysis.lib import ami_fleet_vvar as Fleet

COHORTS = {"3L1P": (3, 1), "3L3P": (3, 3)}

# Shared per-unit-month result format for notebooks 11-13 (07's single-phase results map onto it
# with cohort = "1L1P" and unit_id = site_id).
RESULT_COLUMNS = [
    "cohort", "site_id", "unit_id", "year", "month", "source",
    "n_window", "v_min_obs", "v_max_obs", "v_spread", "slope", "intercept", "r_squared",
    "slope_stderr", "fit_status", "passed", "rating_kw",
]


def _aest_hour(col: str = "t_stamp") -> str:
    return f"(EXTRACT(hour FROM {col}) + 10) % 24"


# =========================================================================================
# Eligibility
# =========================================================================================
def build_phase_eligibility(site_meta: pd.DataFrame, cohorts: dict = COHORTS,
                            rating_column: str = "ac_capacity_kw") -> pd.DataFrame:
    """
    One row per residential site with `cohort` (None if the phase mix is not in `cohorts`) and
    `eligibility` (first match wins): other_phase_mix -> flex_export -> no_rating ->
    export_limited -> eligible. Same rules as `Fleet.build_eligibility`, minus the
    single-phase requirement.
    """
    d = site_meta[["site_id", rating_column, "export_limit_kw", "flex_export_detected",
                   "n_load_phases", "n_pv_phases"]].rename(columns={rating_column: "rating_kw"}).copy()
    d["cohort"] = None
    for name, (n_load, n_pv) in cohorts.items():
        d.loc[(d.n_load_phases == n_load) & (d.n_pv_phases == n_pv), "cohort"] = name
    conditions = [
        d.cohort.isna(),
        d.flex_export_detected.fillna(False).astype(bool),
        d.rating_kw.isna() | (d.rating_kw <= 0),
        d.export_limit_kw < Fleet.P_MIN_FRAC * d.rating_kw,
    ]
    d["eligibility"] = np.select(conditions, ["other_phase_mix", "flex_export", "no_rating", "export_limited"],
                                 default="eligible")
    return d


# =========================================================================================
# Reference phase pairing: voltage matching of PV circuits to load circuits
# =========================================================================================
V_PLAUSIBLE = (150.0, 300.0)   # voltage readings outside this range are treated as bad data


def voltage_match_sql(month: str, sample_every_min: int = 15,
                      v_plausible: tuple[float, float] = V_PLAUSIBLE) -> str:
    """
    One row per (site_id, pv_circuit_id, load_circuit_id) for `month`: number of common
    timestamps, mean and median |V_pv - V_load|, from `ami_raw_phaseseparate`. Uses every
    `sample_every_min`-th minute only (voltage matching needs few points) and drops voltage
    readings outside `v_plausible` (single bad readings of ~10^5 V otherwise dominate the mean).
    Requires a registered `_sites` table (site_id).
    """
    src = f"read_parquet('{Fleet.month_dir('ami_raw_phaseseparate', month).as_posix()}/*.parquet')"
    keep = (f"EXTRACT(minute FROM t_stamp) % {sample_every_min} = 0 AND site_id IN (SELECT site_id FROM _sites) "
            f"AND V BETWEEN {v_plausible[0]} AND {v_plausible[1]}")
    return f"""
        WITH r AS (SELECT site_id, circuit_id, circuit_type, t_stamp, V FROM {src}
                   WHERE {keep}),
        pv AS (SELECT site_id, circuit_id AS pv_circuit_id, t_stamp, V AS v_pv FROM r WHERE circuit_type = 'pv_site_net'),
        ld AS (SELECT site_id, circuit_id AS load_circuit_id, t_stamp, V AS v_load FROM r WHERE circuit_type = 'ac_load_net')
        SELECT site_id, pv_circuit_id, load_circuit_id, count(*) AS n_common,
               avg(abs(v_pv - v_load)) AS mean_abs_dv,
               median(abs(v_pv - v_load)) AS median_abs_dv
        FROM pv JOIN ld USING (site_id, t_stamp)
        GROUP BY ALL
    """


def assign_phase_truth(vm: pd.DataFrame, *, ratio_min: float = 2.0, max_best_dv: float = 1.0,
                       min_common: int = 100) -> pd.DataFrame:
    """
    From `voltage_match_sql` rows (one month or several, with year/month columns), the best
    load circuit per (site, month, PV circuit): `best_dv`, `second_dv`, `dv_ratio` and
    `confident` (= ratio >= ratio_min, best <= max_best_dv, enough common timestamps).
    """
    keys = ["site_id", "year", "month", "pv_circuit_id"]
    v = vm[vm.n_common >= min_common].sort_values(keys + ["mean_abs_dv"]).copy()
    v["rank"] = v.groupby(keys).cumcount()
    best = v[v["rank"] == 0].drop(columns="rank").rename(columns={"mean_abs_dv": "best_dv"})
    sec = v[v["rank"] == 1][keys + ["mean_abs_dv"]].rename(columns={"mean_abs_dv": "second_dv"})
    out = best.merge(sec, on=keys, how="left")
    out["dv_ratio"] = out.second_dv / out.best_dv.where(out.best_dv > 0)
    out["confident"] = (out.dv_ratio >= ratio_min) & (out.best_dv <= max_best_dv)
    return out


def consensus_truth(truth_month: pd.DataFrame) -> pd.DataFrame:
    """
    Site-level reference pairing: for each (site, PV circuit), the load circuit chosen in most
    confident months, with `n_months_confident` and `agree_share` (share of confident months
    that chose it).
    """
    c = truth_month[truth_month.confident]
    counts = c.groupby(["site_id", "pv_circuit_id", "load_circuit_id"]).size().rename("n").reset_index()
    tot = counts.groupby(["site_id", "pv_circuit_id"]).n.transform("sum")
    counts["n_months_confident"] = tot
    counts["agree_share"] = counts.n / tot
    best = counts.sort_values("n", ascending=False).groupby(["site_id", "pv_circuit_id"], as_index=False).first()
    return best.rename(columns={"n": "n_months_agree"})


# =========================================================================================
# Reference configuration: what the PV circuit(s) themselves say (not blind)
# =========================================================================================
# `n_pv_phases` in the site metadata counts PV MEASURING circuits (CTs), not inverter phases:
# a three-phase inverter monitored with one CT is labelled as one PV phase. The reference
# configuration therefore uses the PV circuits' own output relative to the inverter rating.

# Inverter-model hints (case-insensitive regex -> phase type). Supporting evidence only, never
# used to override the PV-circuit rule. Heuristic, extend/verify against datasheets.
MODEL_HINTS = {
    r"\bsymo\b|tripower|\bstp\s?\d|-3-|\b3p|three": "3P",
    r"\bprimo\b|\bgalvo\b|sunny boy|\bsb\s?\d|\b1p": "1P",
}


def pv_output_sql(month: str, day_hours: tuple[int, int] = (10, 14)) -> str:
    """
    One row per (site_id, PV circuit) for `month`: midday 90th percentile of |P| of the PV
    circuit itself (kW), from `ami_raw_phaseseparate`. Requires a registered `_sites` table.
    """
    year, mon = (int(x) for x in month.split("-"))
    h = _aest_hour("t_stamp")
    src = f"read_parquet('{Fleet.month_dir('ami_raw_phaseseparate', month).as_posix()}/*.parquet')"
    return f"""
        SELECT site_id, circuit_id AS pv_circuit_id, {year} AS year, {mon} AS month,
               count(*) AS n_mid, quantile_cont(abs(P_kw_signed), 0.9) AS pv_mid_q90
        FROM {src}
        WHERE circuit_type = 'pv_site_net' AND site_id IN (SELECT site_id FROM _sites)
          AND {h} >= {day_hours[0]} AND {h} < {day_hours[1]} AND P_kw_signed IS NOT NULL
        GROUP BY ALL
    """


def model_hint(model, hints: dict = MODEL_HINTS) -> str | None:
    """'1P' / '3P' if the model string matches one of `hints`, else None."""
    import re
    if not isinstance(model, str):
        return None
    for pattern, label in hints.items():
        if re.search(pattern, model, flags=re.IGNORECASE):
            return label
    return None


def reference_configuration(pv_out: pd.DataFrame, ratings: pd.DataFrame, *, full_min: float = 0.55,
                            partial_max: float = 0.45, balance_min_share: float = 0.2,
                            min_months: int = 1, min_pv_over_rating: float = 0.05) -> pd.DataFrame:
    """
    Site-level reference configuration from the PV circuits' own midday output.

    Per site-month: pv_over_rating = sum over PV circuits of pv_mid_q90 / rating, and each PV
    circuit's share of that sum. Months with pv_over_rating < min_pv_over_rating (no PV output)
    are ignored. Per site: medians over the remaining months.

      1 PV circuit : pv_over_rating >= full_min     -> 1P          (single-phase inverter, fully monitored)
                     pv_over_rating <= partial_max  -> 3P_partial  (three-phase inverter, CT on one phase)
                     in between                     -> uncertain
      3 PV circuits: every share >= balance_min_share -> 3P        (three-phase inverter, fully monitored)
                     otherwise                        -> multi     (unbalanced: e.g. several inverters)
      other counts                                    -> uncertain
    """
    keys = ["site_id", "year", "month"]
    m = pv_out.merge(ratings[["site_id", "rating_kw"]], on="site_id")
    month_tot = m.groupby(keys).agg(pv_sum=("pv_mid_q90", "sum"), n_pv=("pv_circuit_id", "nunique"),
                                    rating_kw=("rating_kw", "first")).reset_index()
    month_tot["pv_over_rating"] = month_tot.pv_sum / month_tot.rating_kw
    m = m.merge(month_tot[keys + ["pv_sum"]], on=keys)
    m["pv_share"] = m.pv_mid_q90 / m.pv_sum.where(m.pv_sum > 0)
    min_share = m.groupby(keys).pv_share.min().rename("min_pv_share").reset_index()
    month_tot = month_tot.merge(min_share, on=keys)
    month_tot = month_tot[month_tot.pv_over_rating >= min_pv_over_rating]

    site = (month_tot.groupby("site_id")
            .agg(n_months_pv=("month", "size"), n_pv_circuits=("n_pv", "max"),
                 pv_over_rating=("pv_over_rating", "median"), min_pv_share=("min_pv_share", "median"))
            .reset_index())
    site = site[site.n_months_pv >= min_months]
    one, three = site.n_pv_circuits == 1, site.n_pv_circuits == 3
    site["ref_config"] = np.select(
        [one & (site.pv_over_rating >= full_min),
         one & (site.pv_over_rating <= partial_max),
         three & (site.min_pv_share >= balance_min_share),
         three],
        ["1P", "3P_partial", "3P", "multi"], default="uncertain")
    return site


# =========================================================================================
# Blind configuration: per-phase export signature of the net meter
# =========================================================================================
def phase_signature_sql(month: str, *, night_hours: tuple[int, int] = (0, 4),
                        day_hours: tuple[int, int] = (10, 14)) -> str:
    """
    One row per (site_id, load circuit) for `month`, from the net meter only: midday
    statistics of P_est = base_P - P_net (kW; positive = export relative to the night
    baseline). Uses the same polarity correction and monthly night baseline as
    `Fleet.month_fit_sql`. Requires a registered `_sites` table (site_id, rating_kw).
    """
    year, mon = (int(x) for x in month.split("-"))
    h = _aest_hour("e.t_stamp")
    return f"""
        WITH {Fleet._estimate_sql(month, "meter_baseline", night_hours)}
        SELECT e.site_id, e.circuit_id, {year} AS year, {mon} AS month,
               count(*)                                   AS n_day,
               median(e.p_est)                            AS p_mid_median,
               quantile_cont(e.p_est, 0.9)                AS p_mid_q90,
               avg(e.V)                                   AS v_mid_mean
        FROM est e
        WHERE {h} >= {day_hours[0]} AND {h} < {day_hours[1]} AND e.p_est IS NOT NULL
        GROUP BY ALL
    """


def classify_configuration(sig: pd.DataFrame, ratings: pd.DataFrame, *, share_1p: float = 0.6,
                           min_share_3p: float = 0.2, min_signal_frac: float = 0.1,
                           n_phase_circuits: int = 3, stat: str = "p_mid_q90") -> pd.DataFrame:
    """
    Blind configuration per site-month from `phase_signature_sql` rows.

    signal_i = max(stat_i, 0) per load circuit; share_i = signal_i / sum(signal).
      incomplete fewer (or more) than n_phase_circuits load circuits that month: the shares are
                 not comparable (e.g. one circuit alone always has a share of 1)
      no_signal  total signal < min_signal_frac x rating
      1P         top share >= share_1p                (blind phase = top circuit)
      3P         three circuits and every share >= min_share_3p
      unclear    anything else
    Also returns `top_over_rating` = top signal / rating (about 1 for a single-phase inverter,
    about 1/3 for a balanced three-phase inverter) as a diagnostic.
    """
    s = sig.merge(ratings[["site_id", "rating_kw"]], on="site_id")
    s["signal"] = s[stat].clip(lower=0)
    keys = ["site_id", "year", "month"]
    g = s.groupby(keys)
    s["total"] = g.signal.transform("sum")
    s["share"] = s.signal / s.total.where(s.total > 0)
    s["n_circuits"] = g.circuit_id.transform("count")
    s = s.sort_values(keys + ["share"], ascending=[True, True, True, False])
    top = s.groupby(keys, as_index=False).first()
    min_share = s.groupby(keys).share.min().rename("min_share").reset_index()
    out = top.merge(min_share, on=keys).rename(columns={"circuit_id": "top_circuit_id", "share": "top_share",
                                                        "signal": "top_signal"})
    out["top_over_rating"] = out.top_signal / out.rating_kw
    out["blind_config"] = np.select(
        [out.n_circuits != n_phase_circuits,
         out.total < min_signal_frac * out.rating_kw,
         out.top_share >= share_1p,
         out.min_share >= min_share_3p],
        ["incomplete", "no_signal", "1P", "3P"], default="unclear")
    out["blind_load_circuit_id"] = out.top_circuit_id.where(out.blind_config == "1P")
    return out[keys + ["rating_kw", "n_circuits", "total", "top_circuit_id", "top_share", "min_share",
                       "top_over_rating", "blind_config", "blind_load_circuit_id"]]


def site_vote(site_month: pd.DataFrame) -> pd.DataFrame:
    """
    Site-level blind configuration = most frequent monthly configuration among months with a
    decision (1P/3P); for 1P, the most frequent blind phase circuit. `vote_share` = share of
    decided months agreeing.
    """
    dec = site_month[site_month.blind_config.isin(["1P", "3P"])]
    cfg = dec.groupby(["site_id", "blind_config"]).size().rename("n").reset_index()
    cfg["vote_share"] = cfg.n / cfg.groupby("site_id").n.transform("sum")
    cfg = cfg.sort_values("n", ascending=False).groupby("site_id", as_index=False).first()
    ph = (dec[dec.blind_config == "1P"].groupby(["site_id", "blind_load_circuit_id"]).size()
          .rename("n_phase").reset_index().sort_values("n_phase", ascending=False)
          .groupby("site_id", as_index=False).first())
    out = cfg.merge(ph, on="site_id", how="left")
    out.loc[out.blind_config != "1P", "blind_load_circuit_id"] = np.nan
    return out.rename(columns={"n": "n_months_decided_for"})
