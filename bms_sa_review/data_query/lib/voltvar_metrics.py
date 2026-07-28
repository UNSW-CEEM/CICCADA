"""Metrics for the Volt-VAr evidence-tier analysis."""

import numpy as np
import pandas as pd


def method_a_summary(site_year, interval_h):
    d = site_year.copy()
    d["headroom_displacement_kwh"] = d["headroom_displacement_kw_sum"] * interval_h
    d["affected"] = d["symptom_count"] > 0
    rows = []
    for year, g in d.groupby("year"):
        rows.append({
            "year": int(year), "eligible_sites": g.site_id.nunique(),
            "symptom_sites": g.loc[g.affected, "site_id"].nunique(),
            "eligible_intervals": int(g.eligible_count.sum()),
            "symptom_intervals": int(g.symptom_count.sum()),
            "symptom_site_pct": 100*g.affected.mean(),
            "symptom_interval_pct": 100*g.symptom_count.sum()/g.eligible_count.sum(),
            "headroom_displacement_proxy_kwh": g.headroom_displacement_kwh.sum(),
        })
    return d, pd.DataFrame(rows)


def method_b_summary(site_year, interval_h):
    d = site_year.copy()
    d["attributed_measured_q_kwh"] = d["attributed_measured_q_kw_sum"] * interval_h
    d["required_q_scenario_kwh"] = d["required_q_scenario_kw_sum"] * interval_h
    d["covered_potential_kwh"] = d["covered_potential_kw_sum"] * interval_h
    d["covered_measured_kwh"] = d["covered_measured_kw_sum"] * interval_h
    d["tier4_affected"] = d["tier4_attributed_count"] > 0
    rows = []
    for year, g in d.groupby("year"):
        potential = g.covered_potential_kwh.sum()
        attributed = g.attributed_measured_q_kwh.sum()
        rows.append({
            "year": int(year), "eligible_sites": g.site_id.nunique(),
            "counterfactual_covered_sites": g.loc[g.counterfactual_covered_count>0,"site_id"].nunique(),
            "tier1_absorbing_intervals": int(g.tier1_absorbing_count.sum()),
            "tier2_apparent_limit_intervals": int(g.tier2_symptom_count.sum()),
            "tier3_counterfactual_above_headroom_intervals": int(g.tier3_count.sum()),
            "tier4_attributed_intervals": int(g.tier4_attributed_count.sum()),
            "tier4_affected_sites": g.loc[g.tier4_affected,"site_id"].nunique(),
            "attributed_measured_q_kwh": attributed,
            "required_q_scenario_kwh": g.required_q_scenario_kwh.sum(),
            "counterfactual_covered_potential_kwh": potential,
            "attributed_pct_of_covered_potential": 100*attributed/potential if potential else np.nan,
        })
    return d, pd.DataFrame(rows)


def evidence_tier_table(site_year):
    labels = [
        ("Tier 1: absorbing Q", "tier1_absorbing_count"),
        ("Tier 2: apparent-limit symptom", "tier2_symptom_count"),
        ("Tier 3: counterfactual above measured-Q headroom", "tier3_count"),
        ("Tier 4: symptom + attributable displacement", "tier4_attributed_count"),
    ]
    return pd.DataFrame([{
        "evidence_tier": label,
        "n_intervals": int(site_year[col].sum()),
        "n_sites": int(site_year.loc[site_year[col]>0, "site_id"].nunique()),
    } for label, col in labels])


def group_breakdown(site_year_enriched, meta, by, min_sites=20):
    site = (site_year_enriched.groupby("site_id", as_index=False)
        .agg(counterfactual_covered_count=("counterfactual_covered_count","sum"),
             tier4_attributed_count=("tier4_attributed_count","sum"),
             attributed_measured_q_kwh=("attributed_measured_q_kwh","sum"),
             covered_potential_kwh=("covered_potential_kwh","sum")))
    site["affected"] = site.tier4_attributed_count > 0
    d = site.merge(meta, on="site_id", how="left", validate="one_to_one")
    d[by] = d[by].fillna("Unknown")
    out = (d.groupby(by, dropna=False)
        .agg(n_sites=("site_id","nunique"),
             affected_sites=("affected","sum"),
             attributed_kwh=("attributed_measured_q_kwh","sum"),
             covered_potential_kwh=("covered_potential_kwh","sum"))
        .reset_index())
    out["affected_site_pct"] = 100*out.affected_sites/out.n_sites
    out["attributed_pct_of_covered_potential"] = (
        100*out.attributed_kwh/out.covered_potential_kwh.replace(0,np.nan))
    return out[out.n_sites>=min_sites].sort_values("attributed_kwh", ascending=False)


def concentration(site_year_enriched, metric="attributed_measured_q_kwh",
                  percentiles=(1,5,10,20,50)):
    site = site_year_enriched.groupby("site_id", as_index=False)[metric].sum()
    site = site[site[metric]>0].sort_values(metric, ascending=False).reset_index(drop=True)
    if site.empty:
        return site, {}
    site["site_share_pct"] = 100*(np.arange(len(site))+1)/len(site)
    site["cumulative_energy_share_pct"] = 100*site[metric].cumsum()/site[metric].sum()
    shares = {p: float(np.interp(p, site.site_share_pct,
                                 site.cumulative_energy_share_pct)) for p in percentiles}
    return site, shares


def sensitivity_table(results):
    """results is an iterable of (label, method_b_site_year)."""
    rows = []
    for label, d in results:
        rows.append({
            "scenario": label,
            "covered_sites": d.loc[d.counterfactual_covered_count>0,"site_id"].nunique(),
            "tier4_sites": d.loc[d.tier4_attributed_count>0,"site_id"].nunique(),
            "tier4_intervals": int(d.tier4_attributed_count.sum()),
            "attributed_kw_sum": d.attributed_measured_q_kw_sum.sum(),
        })
    return pd.DataFrame(rows)

# ═════════════════════════════════════════════════════════════
# Fleet summary — text presentation
# ═════════════════════════════════════════════════════════════
def print_fleet_summary(method_a_enriched, method_a_yearly,
                        method_b_enriched, method_b_yearly, params,
                        eligible_context=None, all_context=None):
    """
    Fleet summary contextualising BOTH methods.

    Method A (apparent-limit symptom scan) = loose upper-bound proxy.
    Method B (counterfactual attribution)  = tight, sun-limited estimate.

    If eligible_context / all_context are supplied (from
    fetch_eligible_context / fetch_all_timestamp_context), the Method A block
    also reports the fleet denominators and %-of-potential-generation figures,
    matching the legacy fleet summary. If omitted, those lines are skipped.
    """
    W = 88
    interval_h = getattr(params, "interval_h", None)

    # ── Method A core totals ────────────────────────────────────
    n_elig_sites  = method_a_enriched.site_id.nunique()
    n_symp_sites  = int(method_a_enriched.loc[method_a_enriched.affected, "site_id"].nunique())
    tot_elig_int  = int(method_a_enriched.eligible_count.sum())
    tot_symp_int  = int(method_a_enriched.symptom_count.sum())
    tot_proxy_kwh = method_a_enriched.headroom_displacement_kwh.sum()

    print("=" * W)
    print(f"VOLT-VAR CURTAILMENT — FLEET SUMMARY ({list(params.years)})")
    print("=" * W)
    print()
    print(f"Detection window: {params.v_low:.0f}-{params.v_high:.0f} V,  "
          f"{params.peak_hour_start}:00-{params.peak_hour_end}:00 AEST,  "
          f"GHI filter {'ON' if params.apply_ghi_filter else 'OFF'},  "
          f"flex {params.flex_selection}")
    print()

    # ── Optional fleet denominators ─────────────────────────────
    if all_context is not None:
        n_all_sites   = all_context.site_id.nunique()
        tot_all_int   = int(all_context.n_all_intervals.sum())
        tot_all_pot   = all_context.all_potential_kWh.sum()
        print("Denominator — all observed timestamps:")
        print(f"  Comparable PV sites:                     {n_all_sites:>14,}")
        print(f"  All observed site-intervals:             {tot_all_int:>14,}")
        print(f"  All potential generation:                {tot_all_pot:>14,.0f} kWh")
        print()

    if eligible_context is not None:
        n_el_sites  = eligible_context.site_id.nunique()
        tot_el_int  = int(eligible_context.n_eligible_intervals.sum())
        tot_el_pot  = eligible_context.eligible_potential_kWh.sum()
        print("Denominator — eligible cases (peak-solar, V-band):")
        print(f"  Eligible sites:                          {n_el_sites:>14,}")
        print(f"  Eligible site-intervals:                 {tot_el_int:>14,}")
        print(f"  Eligible potential generation:           {tot_el_pot:>14,.0f} kWh")
        print()

    # ── Method A numerator ──────────────────────────────────────
    print("-" * W)
    print("METHOD A — apparent-limit symptom scan  (loose upper-bound proxy)")
    print("-" * W)
    print(f"  Eligible sites:                          {n_elig_sites:>14,}")
    print(f"  Sites showing symptom:                   {n_symp_sites:>14,}"
          + (f"  ({100*n_symp_sites/n_elig_sites:.1f}% of eligible)" if n_elig_sites else ""))
    print(f"  Eligible intervals (Method A cohort):    {tot_elig_int:>14,}")
    print(f"  Symptom (flagged) intervals:             {tot_symp_int:>14,}"
          + (f"  ({100*tot_symp_int/tot_elig_int:.4f}%)" if tot_elig_int else ""))
    print(f"  Est. energy curtailed (Method A proxy):  {tot_proxy_kwh:>14,.1f} kWh")
    if eligible_context is not None and tot_el_pot > 0:
        print(f"  Share of eligible potential generation:  {100*tot_proxy_kwh/tot_el_pot:>14.4f}%")
    if all_context is not None and tot_all_pot > 0:
        print(f"  Share of all potential generation:       {100*tot_proxy_kwh/tot_all_pot:>14.4f}%")
    print()

    # ── Method B numerator ──────────────────────────────────────
    n_cov_sites = int(method_b_enriched.loc[
        method_b_enriched.counterfactual_covered_count > 0, "site_id"].nunique())
    n_t4_sites  = int(method_b_enriched.loc[method_b_enriched.tier4_affected, "site_id"].nunique())
    tot_t4_int  = int(method_b_enriched.tier4_attributed_count.sum())
    tot_attr    = method_b_enriched.attributed_measured_q_kwh.sum()
    tot_cov_pot = method_b_enriched.covered_potential_kwh.sum()

    print("-" * W)
    print("METHOD B — counterfactual attribution  (sun-limited, defensible estimate)")
    print("-" * W)
    print(f"  Counterfactual-covered sites:            {n_cov_sites:>14,}")
    print(f"  Tier 4 affected sites:                   {n_t4_sites:>14,}")
    print(f"  Tier 4 attributed intervals:             {tot_t4_int:>14,}")
    print(f"  Attributed curtailment (measured Q):     {tot_attr:>14,.1f} kWh")
    print(f"  Counterfactual-covered potential:        {tot_cov_pot:>14,.0f} kWh")
    if tot_cov_pot > 0:
        print(f"  Attributed / covered potential:          {100*tot_attr/tot_cov_pot:>14.4f}%")
    print()

    # ── Method A vs Method B contrast ───────────────────────────
    print("-" * W)
    print("METHOD A vs METHOD B")
    print("-" * W)
    if tot_attr > 0:
        print(f"  Method A proxy is {tot_proxy_kwh/tot_attr:>.1f}x the Method B attribution.")
    print(f"  Method A over-counts because it assumes every interval where Q ate")
    print(f"  circle-room was sun-limited; Method B only counts intervals where the")
    print(f"  clear-sky counterfactual confirms real generation was lost.")
    print()

    # ── Year-by-year tables ─────────────────────────────────────
    print("=" * W)
    print("YEAR-BY-YEAR — METHOD A")
    print("=" * W)
    a = method_a_yearly.copy()
    if "headroom_displacement_proxy_kwh" in a:
        a["headroom_displacement_proxy_kwh"] = a["headroom_displacement_proxy_kwh"].round(1)
    for c in ["symptom_site_pct", "symptom_interval_pct"]:
        if c in a: a[c] = a[c].round(2)
    print(a.to_string(index=False))
    print()

    print("=" * W)
    print("YEAR-BY-YEAR — METHOD B")
    print("=" * W)
    b = method_b_yearly.copy()
    for c in ["attributed_measured_q_kwh", "required_q_scenario_kwh",
              "counterfactual_covered_potential_kwh"]:
        if c in b: b[c] = b[c].round(1)
    if "attributed_pct_of_covered_potential" in b:
        b["attributed_pct_of_covered_potential"] = b["attributed_pct_of_covered_potential"].round(4)
    print(b.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
# Method A support for DNSP breakdown + legacy fleet-detail assembly.
# ═════════════════════════════════════════════════════════════════════════

def group_breakdown_a(method_a_enriched, meta, by, min_sites=20):
    """
    Method A version of group_breakdown: aggregate the symptom-scan proxy by a
    metadata column (e.g. dnsp_name). Uses headroom_displacement_kwh (the
    Method A proxy), NOT the counterfactual attribution.

    Returns one row per group: n_sites, affected_sites, proxy_kwh,
    affected_site_pct.
    """
    site = (method_a_enriched.groupby("site_id", as_index=False)
        .agg(symptom_count=("symptom_count", "sum"),
             proxy_kwh=("headroom_displacement_kwh", "sum")))
    site["affected"] = site.symptom_count > 0
    d = site.merge(meta, on="site_id", how="left", validate="one_to_one")
    d[by] = d[by].fillna("Unknown")
    out = (d.groupby(by, dropna=False)
        .agg(n_sites=("site_id", "nunique"),
             affected_sites=("affected", "sum"),
             proxy_kwh=("proxy_kwh", "sum"))
        .reset_index())
    out["affected_site_pct"] = 100 * out.affected_sites / out.n_sites
    return out[out.n_sites >= min_sites].sort_values("proxy_kwh", ascending=False)


def build_method_a_context(method_a_enriched, eligible_context, all_context,
                           interval_h):
    """
    Assemble the frames the legacy Method A fleet-detail figures need, from the
    evidence-tier outputs.

    Returns
    -------
    summary_by_year : per-year denominators + numerator + pct columns
    overall_summary : single 'All years' roll-up row (same columns)
    site_year_distribution : per (site_id, year) row with the derived pct /
        energy columns and the denominator columns the histograms filter on
    """
    # ── site-year distribution (merge numerator + both denominators) ──────
    a = method_a_enriched[["site_id", "year", "symptom_count",
                           "headroom_displacement_kwh"]].copy()
    e = eligible_context[["site_id", "year", "n_eligible_intervals",
                          "eligible_potential_kWh"]].copy()
    al = all_context[["site_id", "year", "n_all_intervals",
                      "all_potential_kWh"]].copy()

    d = a.merge(e, on=["site_id", "year"], how="outer") \
         .merge(al, on=["site_id", "year"], how="outer")

    for c in ["symptom_count", "headroom_displacement_kwh",
              "n_eligible_intervals", "eligible_potential_kWh",
              "n_all_intervals", "all_potential_kWh"]:
        d[c] = d[c].fillna(0)

    d["n_flagged_intervals"] = d["symptom_count"]
    d["est_curtailed_kWh"] = d["headroom_displacement_kwh"]

    def _safe(n, den):
        return np.where(den > 0, n / den * 100, np.nan)

    d["pct_eligible_timestamps_flagged"] = _safe(d.symptom_count, d.n_eligible_intervals)
    d["pct_all_timestamps_flagged"]      = _safe(d.symptom_count, d.n_all_intervals)
    d["pct_eligible_potential_generation_curtailed"] = _safe(d.est_curtailed_kWh, d.eligible_potential_kWh)
    d["pct_all_potential_generation_curtailed"]      = _safe(d.est_curtailed_kWh, d.all_potential_kWh)
    d["avg_est_curtailed_kW_when_flagged"] = np.where(
        d.symptom_count > 0,
        d.est_curtailed_kWh / (d.symptom_count * interval_h), np.nan)

    # ── per-year summary ─────────────────────────────────────────────────
    rows = []
    for yr, g in d.groupby("year"):
        flagged = g.symptom_count.sum()
        ei = g.n_eligible_intervals.sum()
        ai = g.n_all_intervals.sum()
        ep = g.eligible_potential_kWh.sum()
        ap = g.all_potential_kWh.sum()
        est = g.est_curtailed_kWh.sum()
        rows.append({
            "year": int(yr),
            "all_sites": int((g.n_all_intervals > 0).sum()),
            "all_intervals": int(ai),
            "all_potential_kWh": ap,
            "eligible_sites": int((g.n_eligible_intervals > 0).sum()),
            "eligible_intervals": int(ei),
            "eligible_potential_kWh": ep,
            "affected_sites": int((g.symptom_count > 0).sum()),
            "flagged_intervals": int(flagged),
            "est_curtailed_kWh": est,
            "pct_eligible_intervals_flagged": 100 * flagged / ei if ei else np.nan,
            "pct_all_intervals_flagged": 100 * flagged / ai if ai else np.nan,
            "pct_eligible_potential_curtailed": 100 * est / ep if ep else np.nan,
            "pct_all_potential_curtailed": 100 * est / ap if ap else np.nan,
        })
    summary_by_year = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    # ── overall 'All years' roll-up ──────────────────────────────────────
    flagged = d.symptom_count.sum(); ei = d.n_eligible_intervals.sum()
    ai = d.n_all_intervals.sum(); ep = d.eligible_potential_kWh.sum()
    ap = d.all_potential_kWh.sum(); est = d.est_curtailed_kWh.sum()
    overall_summary = pd.DataFrame([{
        "year": "All years",
        "all_sites": d.loc[d.n_all_intervals > 0, "site_id"].nunique(),
        "all_intervals": int(ai), "all_potential_kWh": ap,
        "eligible_sites": d.loc[d.n_eligible_intervals > 0, "site_id"].nunique(),
        "eligible_intervals": int(ei), "eligible_potential_kWh": ep,
        "affected_sites": d.loc[d.symptom_count > 0, "site_id"].nunique(),
        "flagged_intervals": int(flagged), "est_curtailed_kWh": est,
        "pct_eligible_intervals_flagged": 100 * flagged / ei if ei else np.nan,
        "pct_all_intervals_flagged": 100 * flagged / ai if ai else np.nan,
        "pct_eligible_potential_curtailed": 100 * est / ep if ep else np.nan,
        "pct_all_potential_curtailed": 100 * est / ap if ap else np.nan,
    }])

    return summary_by_year, overall_summary, d