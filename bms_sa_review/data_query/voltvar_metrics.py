"""
Pandas aggregation and metric computation for Volt-VAr curtailment analysis.

"""

import numpy as np
import pandas as pd

from ciccada_config import AS4777


# Helper: safe percentage
def _safe_pct(numerator, denominator):
    """Element-wise percentage, returning NaN where denominator is zero."""
    return np.where(denominator > 0, numerator / denominator * 100, np.nan)


# ═════════════════════════════════════════════════════════════
# Year-level summary
# ═════════════════════════════════════════════════════════════

def compute_summary_by_year(all_context_by_site_year,
                            eligible_context_by_site_year,
                            all_years_method_a):
    """
    Merge denominators and numerators into a year-level summary table.

    Parameters
    ----------
    all_context_by_site_year : pd.DataFrame
        Output of fetch_all_timestamp_context_for_year, concatenated.
    eligible_context_by_site_year : pd.DataFrame
        Output of fetch_eligible_context_for_year, concatenated.
    all_years_method_a : pd.DataFrame
        Concatenated per-year Method A results (_all_years).

    Returns
    -------
    pd.DataFrame
        One row per year with counts, kWh, and percentage metrics.
    """
    interval_h = AS4777["INTERVAL_H"]

    # Numerator by year
    curtailment_by_year = (
        all_years_method_a
        .groupby("year", as_index=False)
        .agg(
            affected_sites=("site_id", "nunique"),
            flagged_intervals=("n_flagged_intervals", "sum"),
            est_curtailed_kWh=("est_curtailed_kWh", "sum"),
        )
    )

    # Denominator: eligible
    eligible_by_year = (
        eligible_context_by_site_year
        .groupby("year", as_index=False)
        .agg(
            eligible_sites=("site_id", "nunique"),
            eligible_intervals=("n_eligible_intervals", "sum"),
            eligible_potential_kWh=("eligible_potential_kWh", "sum"),
        )
    )

    # Denominator: all
    all_by_year = (
        all_context_by_site_year
        .groupby("year", as_index=False)
        .agg(
            all_sites=("site_id", "nunique"),
            all_intervals=("n_all_intervals", "sum"),
            all_potential_kWh=("all_potential_kWh", "sum"),
        )
    )

    # Merge
    df = (
        all_by_year
        .merge(eligible_by_year, on="year", how="left")
        .merge(curtailment_by_year, on="year", how="left")
        .fillna({
            "eligible_sites": 0, "eligible_intervals": 0,
            "eligible_potential_kWh": 0, "affected_sites": 0,
            "flagged_intervals": 0, "est_curtailed_kWh": 0,
        })
    )

    # Cast integer columns
    int_cols = [
        "all_sites", "all_intervals", "eligible_sites",
        "eligible_intervals", "affected_sites", "flagged_intervals",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    # Percentages
    df["pct_eligible_sites_affected"]   = _safe_pct(df["affected_sites"],   df["eligible_sites"])
    df["pct_eligible_intervals_flagged"] = _safe_pct(df["flagged_intervals"], df["eligible_intervals"])
    df["pct_all_intervals_flagged"]      = _safe_pct(df["flagged_intervals"], df["all_intervals"])
    df["pct_eligible_potential_curtailed"] = _safe_pct(df["est_curtailed_kWh"], df["eligible_potential_kWh"])
    df["pct_all_potential_curtailed"]      = _safe_pct(df["est_curtailed_kWh"], df["all_potential_kWh"])

    df["avg_est_curtailed_kW_when_flagged"] = np.where(
        df["flagged_intervals"] > 0,
        df["est_curtailed_kWh"] / (df["flagged_intervals"] * interval_h),
        np.nan,
    )

    return df

# ═════════════════════════════════════════════════════════════
# Site-year distribution table
# ═════════════════════════════════════════════════════════════

def compute_site_year_distribution(all_context_by_site_year,
                                   eligible_context_by_site_year,
                                   all_years_method_a):
    """
    Build a site × year table with all denominators and numerators merged.

    Returns
    -------
    pd.DataFrame
        One row per (site_id, year) with interval counts, hours, kWh,
        and percentage metrics.
    """
    interval_h = AS4777["INTERVAL_H"]

    curtailment_by_site_year = (
        all_years_method_a
        .groupby(["site_id", "year"], as_index=False)
        .agg(
            n_flagged_intervals=("n_flagged_intervals", "sum"),
            avg_V=("avg_V", "mean"),
            avg_P_kW=("avg_P_kW", "mean"),
            avg_Q_kvar=("avg_Q_kvar", "mean"),
            avg_s_limit=("avg_s_limit", "mean"),
            est_curtailed_kWh=("est_curtailed_kWh", "sum"),
        )
    )

    df = (
        all_context_by_site_year
        .merge(eligible_context_by_site_year, on=["site_id", "year"], how="left")
        .merge(curtailment_by_site_year, on=["site_id", "year"], how="left")
    )

    # Fill missing numerators / eligible counts with zero
    for col in ["n_eligible_intervals", "eligible_potential_kWh",
                "n_flagged_intervals", "est_curtailed_kWh"]:
        df[col] = df[col].fillna(0)

    # Integer casts
    for col in ["n_all_intervals", "n_eligible_intervals", "n_flagged_intervals"]:
        df[col] = df[col].astype(int)

    # Flags
    df["is_eligible_site_year"] = df["n_eligible_intervals"] > 0
    df["is_affected_site_year"] = df["n_flagged_intervals"] > 0

    # Hours
    df["all_hours"]      = df["n_all_intervals"]      * interval_h
    df["eligible_hours"] = df["n_eligible_intervals"]  * interval_h
    df["flagged_hours"]  = df["n_flagged_intervals"]   * interval_h

    # Percentages
    df["pct_eligible_timestamps_flagged"] = _safe_pct(
        df["n_flagged_intervals"], df["n_eligible_intervals"]
    )
    df["pct_all_timestamps_flagged"] = _safe_pct(
        df["n_flagged_intervals"], df["n_all_intervals"]
    )
    df["pct_eligible_potential_generation_curtailed"] = _safe_pct(
        df["est_curtailed_kWh"], df["eligible_potential_kWh"]
    )
    df["pct_all_potential_generation_curtailed"] = _safe_pct(
        df["est_curtailed_kWh"], df["all_potential_kWh"]
    )
    df["avg_est_curtailed_kW_when_flagged"] = np.where(
        df["n_flagged_intervals"] > 0,
        df["est_curtailed_kWh"] / (df["n_flagged_intervals"] * interval_h),
        np.nan,
    )

    return df

# ═════════════════════════════════════════════════════════════
# Overall multi-year fleet summary (returns a dict)
# ═════════════════════════════════════════════════════════════

def compute_fleet_summary(candidates, all_context_by_site_year,
                          eligible_context_by_site_year):
    """
    Compute overall multi-year fleet summary metrics.

    Returns
    -------
    dict
        Keys: n_sites_affected, total_flagged, total_est_kwh, etc.
        Returns None if candidates is empty.
    """
    if len(candidates) == 0:
        return None

    interval_h = AS4777["INTERVAL_H"]

    n_sites_affected     = candidates["site_id"].nunique()
    total_flagged        = int(candidates["n_flagged_intervals"].sum())
    total_est_kwh        = candidates["est_curtailed_kWh"].sum()
    total_flagged_hours  = total_flagged * interval_h

    n_all_sites              = all_context_by_site_year["site_id"].nunique()
    total_all_intervals      = int(all_context_by_site_year["n_all_intervals"].sum())
    total_all_potential_kwh  = all_context_by_site_year["all_potential_kWh"].sum()

    n_eligible_sites             = eligible_context_by_site_year["site_id"].nunique()
    total_eligible_intervals     = int(eligible_context_by_site_year["n_eligible_intervals"].sum())
    total_eligible_potential_kwh = eligible_context_by_site_year["eligible_potential_kWh"].sum()

    def _pct(num, den):
        return num / den * 100 if den > 0 else np.nan

    return {
        "n_all_sites":                    n_all_sites,
        "total_all_intervals":            total_all_intervals,
        "total_all_potential_kwh":        total_all_potential_kwh,
        "n_eligible_sites":              n_eligible_sites,
        "total_eligible_intervals":      total_eligible_intervals,
        "total_eligible_potential_kwh":   total_eligible_potential_kwh,
        "n_sites_affected":              n_sites_affected,
        "total_flagged":                 total_flagged,
        "total_flagged_hours":           total_flagged_hours,
        "total_est_kwh":                 total_est_kwh,
        "pct_sites_affected":            _pct(n_sites_affected, n_eligible_sites),
        "pct_eligible_intervals_flagged": _pct(total_flagged, total_eligible_intervals),
        "pct_all_intervals_flagged":      _pct(total_flagged, total_all_intervals),
        "pct_eligible_potential_curtailed": _pct(total_est_kwh, total_eligible_potential_kwh),
        "pct_all_potential_curtailed":     _pct(total_est_kwh, total_all_potential_kwh),
        "avg_curtailed_kw_when_flagged": (
            total_est_kwh / total_flagged_hours if total_flagged_hours > 0 else np.nan
        ),
        "median_intervals_per_affected_site": candidates["n_flagged_intervals"].median(),
    }

