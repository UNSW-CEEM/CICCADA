"""
voltvar_diagnostics.py  —  Pre-flight checks and diagnostics
=============================================================

Everything in `03_voltvar_curtailment_detection.ipynb` that CHECKS or DIAGNOSES
rather than analyses, lifted out so the notebook stays an orchestrator.

Covers:
  * check_schema()            — do the tables have the columns we need?     [was cell 10]
  * check_ghi_coverage()      — which years does structured_data cover?     [was cell 7]
  * check_sign_convention()   — is Q negative (absorbing) at high voltage?  [was cell 17]
  * fleet_funnel()            — how many sites survive each filter?         [was cell 34]
  * flex_export_prevalence()  — how many flex-export sites, and where?      [R18 evidence]
  * flex_export_impact()      — what does excluding them do to the results? [R18 evidence]

BUG FIXED IN PHASE 4A
---------------------
`fleet_funnel()` took `PARAMS["GHI_YEARS"][0]` and silently reported a
single-year funnel while the headline Method A numbers were multi-year. It now
takes an explicit `year` argument, so the caller cannot accidentally compare a
one-year funnel against a two-year result.
"""

import pandas as pd

from bms_sa_review.shared.ciccada_config import AS4777, TABLES

SD  = TABLES["structured_data"]
UNC = TABLES["all_uncurtailedpv"]


# ═════════════════════════════════════════════════════════════
# 1. Schema checks
# ═════════════════════════════════════════════════════════════

def check_schema(aq_func, database):
    """
    Confirm the tables have the columns the analysis depends on.

    Returns
    -------
    dict with keys: has_ghi (bool), sd_columns (list), meta_ok (bool)

    `has_ghi` feeds straight into the queries' `has_ghi` argument.
    """
    out = {"has_ghi": False, "sd_columns": [], "meta_ok": False}

    try:
        _sd = aq_func(f"SELECT * FROM {SD} LIMIT 1", database=database)
        out["sd_columns"] = list(_sd.columns)
        out["has_ghi"] = "ghi" in _sd.columns and "ghi_cs" in _sd.columns
        print(f"{SD} columns:")
        print(f"  {out['sd_columns']}")
        print(f"\n  ghi + ghi_cs present? {out['has_ghi']}")
    except Exception as e:
        print(f"[!] {SD} not accessible: {e}")
        print("    Will fall back to the measured-power clear-sky proxy.")

    _meta = aq_func("SELECT * FROM meta_up23c LIMIT 1", database=database)
    has_s99  = "s_99" in _meta.columns
    has_cap  = "ac_capacity_kw" in _meta.columns
    has_flex = "flex_export_detected" in _meta.columns
    out["meta_ok"] = has_s99 and has_cap and has_flex

    print(f"\nmeta_up23c has s_99?                 {has_s99}")
    print(f"meta_up23c has ac_capacity_kw?       {has_cap}")
    print(f"meta_up23c has flex_export_detected? {has_flex}   (needed for R18)")

    return out


def check_ghi_coverage(aq_func, database):
    """
    Which years does the counterfactual actually cover?

    Method A's GHI clear-sky filter and Method B both depend on these tables.
    If a year in PARAMS["GHI_YEARS"] is missing here, that year silently
    produces nothing.
    """
    sd_years = aq_func(f"""
        SELECT year, count(*) AS n_rows, count(DISTINCT site_id) AS n_sites
        FROM {SD}
        GROUP BY year
        ORDER BY year
    """, database=database)

    unc_years = aq_func(f"""
        SELECT year, count(*) AS n_rows, count(DISTINCT site_id) AS n_sites
        FROM {UNC}
        GROUP BY year
        ORDER BY year
    """, database=database)

    print(f"{SD} coverage by year:")
    print(sd_years.to_string(index=False))
    print(f"\n{UNC} coverage by year:")
    print(unc_years.to_string(index=False))
    print("\nIf a year in PARAMS['GHI_YEARS'] is absent above, it will contribute nothing.")

    return sd_years, unc_years


# ═════════════════════════════════════════════════════════════
# 2. Sign-convention check
# ═════════════════════════════════════════════════════════════

def check_sign_convention(year, aq_func, database):
    """
    In the 245-253 V band the standard REQUIRES inverters to absorb reactive
    power, i.e. Q < 0 under the generator convention. If the fleet average comes
    back positive, either the sign convention is flipped somewhere, or the fleet
    genuinely isn't responding — and those two look identical in the data, so
    this must be checked before anything else is believed.

    Expected: roughly -0.2 kvar.
    """
    sign_check = aq_func(f"""
        SELECT
            round(avg(t.energy_reactive * m.circuit_polarity / 1000.0 * 12), 3)
                AS avg_Q_kvar_high_V,
            count(*) AS n
        FROM ts t
        JOIN (SELECT DISTINCT circuit_id, circuit_polarity
              FROM meta_up23c WHERE is_pv = True) m
          ON t.circuit_id = m.circuit_id
        WHERE t.year = {year} AND t.is_pv = True
          AND t.voltage > 245 AND t.voltage < 253
    """, database=database)

    q_avg = sign_check["avg_Q_kvar_high_V"].iloc[0]
    n     = int(sign_check["n"].iloc[0])

    print(f"Average Q in the 245-253 V band ({year}): {q_avg} kvar  (n = {n:,})")
    if q_avg < 0:
        print("  OK — negative means absorbing, as the standard requires.")
    else:
        print("  [!] POSITIVE. Either the sign is flipped (negate energy_reactive")
        print("      everywhere), or the fleet genuinely isn't absorbing. Investigate")
        print("      before trusting any curtailment number.")

    return sign_check


# ═════════════════════════════════════════════════════════════
# 3. Fleet funnel
# ═════════════════════════════════════════════════════════════

def fleet_funnel(year, params, aq_func, database):
    """
    Trace how many sites survive each successive filter, for ONE year.

    PHASE 4A: `year` is now an explicit argument. Previously this silently used
    PARAMS["GHI_YEARS"][0] and produced a single-year funnel that got compared
    against multi-year headline numbers.

    Runs 4 separate Athena queries. Cheap-ish, but not free.
    """
    P = params
    _limit_col = "s_99" if P["USE_S99"] else "ac_capacity_kw"
    flex_where = "AND flex_export_detected = False" if P.get("EXCLUDE_FLEX", True) else ""

    print(f"Fleet funnel for {year} — running 4 Athena queries...")
    print(f"  (limit basis = {_limit_col}, "
          f"exclude_flex = {P.get('EXCLUDE_FLEX', True)})\n")

    base_meta = f"""
        is_pv = True
        {flex_where}
        AND ac_capacity_kw > 0
        AND ac_capacity_kw <= {P['MAX_AC_CAPACITY_KW']}
    """

    # Step 0 — every PV site in the size cohort
    step0 = aq_func(f"""
        SELECT count(DISTINCT site_id) AS n
        FROM meta_up23c
        WHERE {base_meta}
    """, database=database)

    # Step 1 — + a valid apparent-power limit
    step1 = aq_func(f"""
        SELECT count(DISTINCT site_id) AS n
        FROM meta_up23c
        WHERE {base_meta}
          AND {_limit_col} > 0
    """, database=database)

    # Step 2 — + at least one interval in the Volt-VAr band this year
    step2 = aq_func(f"""
        SELECT count(DISTINCT sm.site_id) AS n
        FROM ts t
        JOIN (SELECT DISTINCT site_id, circuit_id FROM meta_up23c
              WHERE {base_meta} AND {_limit_col} > 0) sm
          ON t.circuit_id = sm.circuit_id
        WHERE t.year = {year} AND t.is_pv = True
          AND t.voltage > {P['V_LOW']} AND t.voltage < {P['V_HIGH']}
    """, database=database)

    # Step 3 — + also inside the peak-solar window
    step3 = aq_func(f"""
        SELECT count(DISTINCT sm.site_id) AS n
        FROM ts t
        JOIN (SELECT DISTINCT site_id, circuit_id FROM meta_up23c
              WHERE {base_meta} AND {_limit_col} > 0) sm
          ON t.circuit_id = sm.circuit_id
        WHERE t.year = {year} AND t.is_pv = True
          AND t.voltage > {P['V_LOW']} AND t.voltage < {P['V_HIGH']}
          AND hour(t.t_stamp + interval '10' hour)
              BETWEEN {P['PEAK_HOUR_START']} AND {P['PEAK_HOUR_END']}
    """, database=database)

    funnel = pd.DataFrame([
        {"step": "0. PV sites in size cohort",              "n_sites": int(step0["n"].iloc[0])},
        {"step": f"1. + valid {_limit_col}",                "n_sites": int(step1["n"].iloc[0])},
        {"step": f"2. + any interval {P['V_LOW']:.0f}-{P['V_HIGH']:.0f} V",
                                                            "n_sites": int(step2["n"].iloc[0])},
        {"step": f"3. + in peak window {P['PEAK_HOUR_START']}:00-{P['PEAK_HOUR_END']}:00",
                                                            "n_sites": int(step3["n"].iloc[0])},
    ])
    funnel["retained_pct"] = (
        funnel["n_sites"] / funnel["n_sites"].iloc[0] * 100
    ).round(1)

    print(funnel.to_string(index=False))
    return funnel


# ═════════════════════════════════════════════════════════════
# 4. R18 evidence — flexible-export sites
# ═════════════════════════════════════════════════════════════

def flex_export_prevalence(aq_func, database):
    """
    How many flex-export sites are there, and are they concentrated in the DNSPs
    that actually run flexible-export programs (any DNSP especially)? If the flag
    correlates with known programs, it's trustworthy.
    """
    overall = aq_func("""
        SELECT flex_export_detected, count(DISTINCT site_id) AS n_sites
        FROM meta_up23c
        WHERE is_pv = True
        GROUP BY flex_export_detected
    """, database=database)

    by_dnsp = aq_func("""
        SELECT state, dnsp_name, count(DISTINCT site_id) AS n_flex_sites
        FROM meta_up23c
        WHERE is_pv = True AND flex_export_detected = True
        GROUP BY state, dnsp_name
        ORDER BY n_flex_sites DESC
    """, database=database)

    print("Flex-export prevalence:")
    print(overall.to_string(index=False))
    print("\nFlagged sites by state / DNSP:")
    print(by_dnsp.to_string(index=False))

    return overall, by_dnsp


def flex_export_impact(candidates, aq_func, database):
    """
    What does R18 actually do to the headline numbers?

    Call this with the Method A `candidates` DataFrame produced WITHOUT the flex
    filter (params EXCLUDE_FLEX=False), and it reports how many affected sites
    and how much curtailed energy the filter removes. This is the sensitivity
    sentence for the paper.

    Parameters
    ----------
    candidates : pd.DataFrame
        Method A output. Needs columns: site_id, est_curtailed_kWh.
    """
    if candidates.empty:
        print("No Method A candidates — nothing to compare.")
        return None

    ids = ",".join(str(s) for s in candidates["site_id"].unique())

    flex_sites = aq_func(f"""
        SELECT DISTINCT site_id
        FROM meta_up23c
        WHERE is_pv = True
          AND flex_export_detected = True
          AND site_id IN ({ids})
    """, database=database)

    flex_set = set(flex_sites["site_id"].values)
    keep     = ~candidates["site_id"].isin(flex_set)

    sites_all,  sites_kept = candidates["site_id"].nunique(), candidates.loc[keep, "site_id"].nunique()
    kwh_all,    kwh_kept   = candidates["est_curtailed_kWh"].sum(), candidates.loc[keep, "est_curtailed_kWh"].sum()

    print(f"{'Metric':<34}{'Incl. flex':>14}{'Excl. flex':>14}{'Delta':>12}")
    print("-" * 74)
    print(f"{'Affected sites':<34}{sites_all:>14,}{sites_kept:>14,}"
          f"{sites_kept - sites_all:>+12,}")
    print(f"{'Est. curtailed kWh':<34}{kwh_all:>14,.0f}{kwh_kept:>14,.0f}"
          f"{kwh_kept - kwh_all:>+12,.0f}")
    if kwh_all > 0:
        print(f"{'Change in curtailment':<34}{'':>14}{'':>14}"
              f"{(kwh_kept / kwh_all - 1) * 100:>+11.1f}%")

    return pd.DataFrame([{
        "sites_incl_flex": sites_all,  "sites_excl_flex": sites_kept,
        "kwh_incl_flex":   kwh_all,    "kwh_excl_flex":   kwh_kept,
        "n_flex_in_affected": len(flex_set),
    }])
