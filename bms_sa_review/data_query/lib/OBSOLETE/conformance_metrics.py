"""
conformance_metrics.py  —  DataFrame-level metrics for the conformance analysis
================================================================================

Everything that turns raw query output into a VERDICT or a RATE. Kept separate
from `conformance_queries` because these are methodological decisions, not data
retrieval.

All functions take DataFrames in and return DataFrames out. No SQL, no globals.

R5 — RESOLVED
-------------
Reduced non-conformance is:

    reduced = Q_adverse + Q_inactive + Q_significant_shortfall

i.e. inverters that are actually failing: responding in the wrong direction, not
responding at all, or responding at only 10-90% of what the curve requires.

Q_near_conformant (Q_impact 0.9-1.1) is EXCLUDED. Those inverters deliver
90-110% of the required reactive power. They are not failing.

Milestone 3 summed Q_near_conformant and excluded the shortfall band. That
followed from the swapped column names (R4) and is NOT reproduced here.
`milestone3_reduced()` can reproduce it if you need to explain the discrepancy
against the published figure, but it is not a project result.

NOTE ON THE STORED COLUMNS
--------------------------
`vvar_reduced()` derives the total from the five bucket columns rather than
reading `nonconformance_voltvar_red_sum` / `_count`. Deliberate: any table built
before the R5 fix landed has the WRONG value stored in those two columns.
Deriving from the buckets is correct against both the old and the rebuilt table,
and costs nothing. Once conformance_voltvar_v2 is rebuilt with the fixed builder,
the stored column will agree with the derived one — `check_stored_red()` below
verifies that.
"""

import numpy as np
import pandas as pd

from bms_sa_review.data_query.OBSOLETEciccada_config import AS4777

# The five Q_impact bands, as named in conformance_voltvar_v2.
BANDS = ["adverse", "inactive", "shortfall", "near_conformant", "surplus"]

# R5: the bands that constitute an actual failure.
REDUCED_BANDS = ["adverse", "inactive", "shortfall"]


# ═════════════════════════════════════════════════════════════
# 1. Site-level conformance verdict
# ═════════════════════════════════════════════════════════════

def site_conformance(df, thresh=AS4777["SITE_CONF_THRESH"], label=""):
    """
    Apply the 10%-rule to per-site counts from
    `conformance_queries.fetch_site_conformance()`.

    A site is CONFORMANT if fewer than `thresh` of its evaluated intervals were
    non-conformant. The threshold is a project convention, not a standard — the
    standard says nothing about how often an inverter may fail before the site
    is deemed non-compliant.

    Expects: site_id, nonconf_count, total_count.
    Adds:    nonconf_frac, conformant.
    """
    d = df[df["total_count"] > 0].copy()
    d["nonconf_frac"] = d["nonconf_count"] / d["total_count"]
    d["conformant"]   = d["nonconf_frac"] <= thresh

    rate = d["conformant"].mean()
    name = label or "(unlabelled)"
    print(f"{name:32s} sites={len(d):6,}  fleet conformant={rate*100:5.1f}%  "
          f"(<= {thresh*100:.0f}% nonconf intervals)")
    return d


# ═════════════════════════════════════════════════════════════
# 2. Volt-VAr reduced non-conformance  (R5 — resolved)
# ═════════════════════════════════════════════════════════════

def vvar_reduced(vvar_comp, thresh=AS4777["SITE_CONF_THRESH"]):
    """
    Volt-VAr per-site conformance verdict, project definition:

        reduced = adverse + inactive + significant_shortfall

    Derived from the bucket columns, not from the stored
    `nonconformance_voltvar_red_*` (see module docstring).

    Expects: adverse, inactive, shortfall, near_conformant, surplus, total_count.
    Adds:    nonconf_count, nonconf_frac, conformant.
    """
    d = vvar_comp[vvar_comp["total_count"] > 0].copy()

    d["nonconf_count"] = d[REDUCED_BANDS].sum(axis=1)
    d["nonconf_frac"]  = d["nonconf_count"] / d["total_count"]
    d["conformant"]    = d["nonconf_frac"] <= thresh

    print("Volt-VAr (reduced = adverse + inactive + significant_shortfall)")
    print(f"  {len(d):,} sites  fleet conformant = {d['conformant'].mean()*100:.1f}%  "
          f"(<= {thresh*100:.0f}% nonconf intervals)")
    return d


def milestone3_reduced(vvar_comp, thresh=AS4777["SITE_CONF_THRESH"]):
    """
    RECONCILIATION ONLY — never quote this as a result.

    Reproduces Milestone 3's definition:

        reduced = adverse + inactive + near_conformant

    which counts the 0.9-1.1 band (essentially conformant) and ignores the
    0.1-0.9 band (the actual failures). Artefact of the swapped names (R4).
    """
    d = vvar_comp[vvar_comp["total_count"] > 0].copy()
    d["nonconf_count"] = d[["adverse", "inactive", "near_conformant"]].sum(axis=1)
    d["nonconf_frac"]  = d["nonconf_count"] / d["total_count"]
    d["conformant"]    = d["nonconf_frac"] <= thresh

    print("[MILESTONE 3 REPRODUCTION — not a project result]")
    print(f"  {len(d):,} sites  fleet conformant = {d['conformant'].mean()*100:.1f}%")
    return d


def reconcile_vvar(vvar_comp, thresh=AS4777["SITE_CONF_THRESH"]):
    """
    Print both definitions side by side and quantify the gap.

    Run ONCE to get the methods-section sentence: "Correcting the reduced
    non-conformance definition changes the fleet Volt-VAr conformance rate from
    X% to Y%." You will need this, because your Volt-VAr figure will not match
    the published one and a reader will ask why.
    """
    ours = vvar_reduced(vvar_comp, thresh)
    print()
    m3 = milestone3_reduced(vvar_comp, thresh)
    print()

    r_ours = ours["conformant"].mean() * 100
    r_m3   = m3["conformant"].mean() * 100

    print(f"Milestone 3 definition : {r_m3:5.1f}% conformant")
    print(f"Corrected definition   : {r_ours:5.1f}% conformant")
    print(f"Difference             : {r_ours - r_m3:+5.1f} percentage points")
    print()
    print("The corrected definition counts inverters responding at only 10-90% of")
    print("required Q as failures, and stops counting near-conformant ones.")

    return pd.DataFrame([
        {"definition": "milestone3 (adverse+inactive+near_conformant)",
         "pct_conformant": round(r_m3, 1)},
        {"definition": "corrected (adverse+inactive+shortfall)",
         "pct_conformant": round(r_ours, 1)},
    ])


def vvar_band_shares(vvar_comp):
    """
    Each Q_impact band as a share of evaluated intervals, fleet-wide.

    Requires no interpretive choice, so no definitional dispute can contaminate
    it. The cleanest Volt-VAr figure for the paper.
    """
    totals = vvar_comp[BANDS].sum()
    total_evaluated = vvar_comp["total_count"].sum()

    out = pd.DataFrame({
        "interval_count":   totals,
        "pct_of_evaluated": (totals / total_evaluated * 100).round(2),
    })
    out["counts_as_failure"] = [b in REDUCED_BANDS for b in BANDS]
    out.loc["TOTAL_EVALUATED"] = [total_evaluated, 100.0, None]
    return out


def check_stored_red(vvar_comp_with_stored):
    """
    Verify the STORED nonconformance_voltvar_red_count agrees with the derived
    one. Run this after rebuilding conformance_voltvar_v2 with the fixed builder.

    Before the rebuild: expect a mismatch (the stored column summed
    near_conformant). After: expect zero mismatched sites.

    Requires the query to have also selected nonconformance_voltvar_red_count.
    """
    if "nonconformance_voltvar_red_count" not in vvar_comp_with_stored.columns:
        print("Stored red column not in the DataFrame — nothing to check.")
        return None

    d = vvar_comp_with_stored.copy()
    d["derived_red"] = d[REDUCED_BANDS].sum(axis=1)
    d["mismatch"]    = d["derived_red"] != d["nonconformance_voltvar_red_count"]

    n_bad = int(d["mismatch"].sum())
    if n_bad == 0:
        print("Stored red column agrees with the derived value. Table is rebuilt.")
    else:
        print(f"{n_bad:,} of {len(d):,} sites: stored red != derived red.")
        print("Expected if conformance_voltvar_v2 predates the R5 fix.")
        print("The derived value is the correct one — vvar_reduced() uses it.")
    return d


# ═════════════════════════════════════════════════════════════
# 3. Curtailment energy
# ═════════════════════════════════════════════════════════════

def curtailment_energy(df, label="", has_gen=False):
    """
    Convert the kW sums from `fetch_curtailment_energy()` into kWh, and report
    curtailment as a share of eligible generation.

    Denominator is (actual generation + curtailed energy) — what the fleet WOULD
    have generated in those conditions. Actual generation alone understates it.

    Expects: site_id, curt_kw_sum, curt_intervals [, gen_kw_sum].
    """
    d = df.copy()
    d["curt_kwh"] = d["curt_kw_sum"] * AS4777["INTERVAL_H"]

    total_curt = d["curt_kwh"].sum()
    msg = f"{label:28s} fleet curtailed = {total_curt:,.0f} kWh across {len(d):,} sites"

    if has_gen and "gen_kw_sum" in d.columns:
        d["gen_kwh"] = d["gen_kw_sum"] * AS4777["INTERVAL_H"]
        denom = d["gen_kwh"].sum() + total_curt
        if denom > 0:
            d.attrs["pct_of_eligible"] = total_curt / denom * 100
            msg += f"  | {total_curt / denom * 100:.2f}% of eligible generation"

    print(msg)
    return d.sort_values("curt_kwh", ascending=False)


# ═════════════════════════════════════════════════════════════
# 4. Volt-Watt report metrics
# ═════════════════════════════════════════════════════════════

def vw_report_metrics(vw_report, meta, thresh=AS4777["SITE_CONF_THRESH"]):
    """
    Derive the milestone-report metrics from `fetch_vw_report()`:

        nc_mwh        cumulative non-conformance energy (MWh)
        norm_nc_wh_kw normalised NC: Wh excess per kW nameplate per eligible
                      interval -> makes a 3 kW and a 20 kW system comparable
        nonconf_site  the 10%-rule verdict
        any_nonconf   at least one NC interval (a much looser bar; the gap
                      between the two is informative)
    """
    d = vw_report.merge(meta[["site_id", "ac_capacity_kw"]], on="site_id", how="left")
    d = d[d["total_count"] > 0].copy()

    d["nc_mwh"] = d["nc_sum_kw"] * AS4777["INTERVAL_H"] / 1000.0

    denom = d["total_count"] * d["ac_capacity_kw"]
    d["norm_nc_wh_kw"] = np.where(
        denom > 0,
        d["nc_sum_kw"] * AS4777["INTERVAL_H"] * 1000.0 / denom,
        np.nan,
    )

    d["nonconf_frac"] = d["nc_count"] / d["total_count"]
    d["nonconf_site"] = d["nonconf_frac"] >= thresh
    d["conform_site"] = ~d["nonconf_site"]
    d["any_nonconf"]  = d["nc_count"] > 0

    n_total   = len(d)
    n_nonconf = int(d["nonconf_site"].sum())
    print(f"GHI-eligible sites:            {n_total:,}")
    print(f"Non-conformant (>={thresh*100:.0f}% rule): {n_nonconf:,}  "
          f"({n_nonconf / n_total * 100:.1f}%)")
    print(f"Conformant     (< {thresh*100:.0f}% rule): {n_total - n_nonconf:,}  "
          f"({(n_total - n_nonconf) / n_total * 100:.1f}%)")
    print(f"\nFor comparison — 'any NC' rate: {d['any_nonconf'].mean()*100:.1f}%")
    print("The gap shows how many sites fail occasionally but stay under the threshold.")

    return d


# ═════════════════════════════════════════════════════════════
# 5. Breakdown by state / DNSP / OEM
# ═════════════════════════════════════════════════════════════

def breakdown(site_df, meta, by, min_sites=1):
    """
    Group per-site Volt-Watt metrics by state, DNSP, OEM, or a list of those.
    Expects site_df to have been through `vw_report_metrics()`.
    """
    d = site_df.merge(meta, on="site_id", how="left", suffixes=("", "_meta"))
    cols = [by] if isinstance(by, str) else list(by)
    d = d.dropna(subset=cols)

    g = (
        d.groupby(by, dropna=True)
         .agg(
             n_sites            = ("site_id",       "nunique"),
             pct_nonconf_10pct  = ("nonconf_site",  lambda s: s.mean() * 100),
             pct_any_nonconf    = ("any_nonconf",   lambda s: s.mean() * 100),
             total_nc_instances = ("nc_count",      "sum"),
             total_nc_mwh       = ("nc_mwh",        "sum"),
             mean_norm_nc_wh_kw = ("norm_nc_wh_kw", "mean"),
         )
         .reset_index()
    )

    g = g[g["n_sites"] >= min_sites].sort_values("pct_nonconf_10pct", ascending=False)

    for col in ("total_nc_instances", "total_nc_mwh", "mean_norm_nc_wh_kw",
                "pct_any_nonconf", "pct_nonconf_10pct"):
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0).astype(float)

    return g.round({
        "pct_nonconf_10pct": 1, "pct_any_nonconf": 1,
        "total_nc_mwh": 2, "mean_norm_nc_wh_kw": 3,
    })


# ═════════════════════════════════════════════════════════════
# 6. Fleet summary
# ═════════════════════════════════════════════════════════════

def fleet_summary(results):
    """
    Assemble the headline table.

    `results` : dict of {mechanism_label: site_df_with_conformant_column}

    Volt-VAr and Volt-Watt come from rebuilt _v2 tables; sustained-op and
    anti-islanding come from LEGACY tables that still carry R1/R2/R3. Their rates
    are not strictly comparable — footnote this in any write-up.
    """
    rows = []
    for label, d in results.items():
        rows.append({
            "mechanism":      label,
            "n_sites":        len(d),
            "pct_conformant": d["conformant"].mean() * 100,
        })
    return pd.DataFrame(rows).round(1)
