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
                        method_b_enriched, method_b_yearly, params):
    """
    Print a formatted fleet summary combining Method A and Method B results.

    Parameters
    ----------
    method_a_enriched : DataFrame
        Output[0] of method_a_summary (site-year rows with 'affected' flag).
    method_a_yearly : DataFrame
        Output[1] of method_a_summary (one row per year).
    method_b_enriched : DataFrame
        Output[0] of method_b_summary (site-year rows with kWh columns).
    method_b_yearly : DataFrame
        Output[1] of method_b_summary (one row per year).
    params : VoltVarParams
        Detection parameters (used for header labels only).
    """
    years = list(params.years)

    # ── Method A totals ─────────────────────────────────────────
    n_eligible  = method_a_enriched.site_id.nunique()
    n_symptom   = int(method_a_enriched.loc[method_a_enriched.affected, "site_id"].nunique())
    total_elig  = int(method_a_enriched.eligible_count.sum())
    total_symp  = int(method_a_enriched.symptom_count.sum())
    total_proxy = method_a_enriched.headroom_displacement_kwh.sum()

    # ── Method B totals ─────────────────────────────────────────
    n_covered   = int(method_b_enriched.loc[
        method_b_enriched.counterfactual_covered_count > 0, "site_id"
    ].nunique())
    n_t4        = int(method_b_enriched.loc[method_b_enriched.tier4_affected, "site_id"].nunique())
    total_t4    = int(method_b_enriched.tier4_attributed_count.sum())
    total_attr  = method_b_enriched.attributed_measured_q_kwh.sum()
    total_pot   = method_b_enriched.covered_potential_kwh.sum()

    W = 88
    print("=" * W)
    print(f"VOLT-VAR CURTAILMENT — FLEET SUMMARY ({years})")
    print("=" * W)
    print()
    print(f"Detection window: {params.v_low:.0f}–{params.v_high:.0f} V,  "
          f"{params.peak_hour_start}:00–{params.peak_hour_end}:00 AEST,  "
          f"GHI filter {'ON' if params.apply_ghi_filter else 'OFF'},  "
          f"flex {params.flex_selection}")
    print()

    print("Method A — apparent-limit symptom scan:")
    print(f"  Eligible sites:                          {n_eligible:>10,}")
    print(f"  Sites with ≥1 symptom interval:          {n_symptom:>10,}  "
          f"({100 * n_symptom / n_eligible:.1f}%)" if n_eligible else "")
    print(f"  Total eligible intervals:                {total_elig:>10,}")
    print(f"  Total symptom intervals:                 {total_symp:>10,}  "
          f"({100 * total_symp / total_elig:.4f}%)" if total_elig else "")
    print(f"  Headroom displacement proxy:             {total_proxy:>10,.1f} kWh")
    print()

    print("Method B — counterfactual attribution:")
    print(f"  Counterfactual-covered sites:             {n_covered:>10,}")
    print(f"  Tier 4 affected sites:                    {n_t4:>10,}")
    print(f"  Tier 4 attributed intervals:              {total_t4:>10,}")
    print(f"  Attributed curtailment (measured Q):      {total_attr:>10,.1f} kWh")
    print(f"  Covered potential generation:             {total_pot:>10,.0f} kWh")
    if total_pot > 0:
        print(f"  Attributed / covered potential:           "
              f"{100 * total_attr / total_pot:>10.4f}%")
    print()

    print("=" * W)
    print("YEAR-BY-YEAR — METHOD A")
    print("=" * W)
    display_a = method_a_yearly.copy()
    for c in ["symptom_site_pct", "symptom_interval_pct"]:
        if c in display_a.columns:
            display_a[c] = display_a[c].round(2)
    if "headroom_displacement_proxy_kwh" in display_a.columns:
        display_a["headroom_displacement_proxy_kwh"] = (
            display_a["headroom_displacement_proxy_kwh"].round(1)
        )
    print(display_a.to_string(index=False))
    print()

    print("=" * W)
    print("YEAR-BY-YEAR — METHOD B")
    print("=" * W)
    display_b = method_b_yearly.copy()
    for c in ["attributed_measured_q_kwh", "required_q_scenario_kwh",
              "counterfactual_covered_potential_kwh"]:
        if c in display_b.columns:
            display_b[c] = display_b[c].round(1)
    if "attributed_pct_of_covered_potential" in display_b.columns:
        display_b["attributed_pct_of_covered_potential"] = (
            display_b["attributed_pct_of_covered_potential"].round(4)
        )
    print(display_b.to_string(index=False))
    print()

    # ── Top 10 site-years ───────────────────────────────────────
    top = (method_b_enriched
           .sort_values("attributed_measured_q_kwh", ascending=False)
           .head(10)
           [["site_id", "year", "tier4_attributed_count",
             "attributed_measured_q_kwh", "covered_potential_kwh"]]
           .copy())
    print("=" * W)
    print("TOP 10 SITE-YEARS BY ATTRIBUTED CURTAILMENT")
    print("=" * W)
    print(top.to_string(index=False))
