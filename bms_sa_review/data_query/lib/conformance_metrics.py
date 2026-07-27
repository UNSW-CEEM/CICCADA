"""DataFrame metrics with explicit populations and denominators."""

import numpy as np
import pandas as pd


VVAR_FAILURE_BANDS = ("adverse", "inactive", "shortfall")


def validate_metadata(meta):
    if meta["site_id"].duplicated().any():
        raise AssertionError("Metadata is not one row per site")
    conflict_cols = [c for c in meta if c.startswith("n_") and c.endswith("_values")]
    return pd.DataFrame({
        "field": conflict_cols,
        "sites_with_multiple_values": [int((meta[c] > 1).sum()) for c in conflict_cols],
    })


def aggregate_sites(site_year, numerator, denominator, threshold=0.10,
                    min_intervals=1):
    """Aggregate site-year rows and apply the project threshold consistently."""
    sum_cols = site_year.select_dtypes(include="number").columns.difference(["site_id", "year"])
    d = site_year.groupby("site_id", as_index=False)[list(sum_cols)].sum(min_count=1)
    d = d[d[denominator].fillna(0) >= min_intervals].copy()
    d["nonconf_count"] = d[numerator].fillna(0)
    d["denominator_count"] = d[denominator]
    d["nonconf_frac"] = d["nonconf_count"] / d["denominator_count"]
    d["conformant"] = d["nonconf_frac"] < threshold
    d["nonconformant"] = d["nonconf_frac"] >= threshold
    d["any_nonconformance"] = d["nonconf_count"] > 0
    return d


def prepare_vw_conformance(vw_site_year, config, min_intervals=None):
    return aggregate_sites(
        vw_site_year, "nonconf_count", "exposed_count",
        config.site_nonconf_threshold,
        config.min_site_intervals if min_intervals is None else min_intervals,
    )


def prepare_vw_response(vw_response_site_year, config, min_intervals=None):
    return aggregate_sites(
        vw_response_site_year, "nonconf_count", "response_supported_count",
        config.site_nonconf_threshold,
        config.min_site_intervals if min_intervals is None else min_intervals,
    )


def prepare_vvar_conformance(vvar_site_year, config, min_intervals=None):
    d = vvar_site_year.copy()
    d["reduced_nonconf_count"] = d[list(VVAR_FAILURE_BANDS)].sum(axis=1)
    return aggregate_sites(
        d, "reduced_nonconf_count", "capability_assessable_count",
        config.site_nonconf_threshold,
        config.min_site_intervals if min_intervals is None else min_intervals,
    )


def fleet_result(
    site_df,
    label,
    question,
    denominator_label,
    *,
    table_site_count=None,
):
    """
    Summarise a site-level conformance result.

    table_site_count:
        Number of distinct sites represented in the source result table,
        including sites that had no intervals in the selected assessment
        denominator.

    n_assessed_sites:
        Sites retained after applying the assessment denominator and any
        minimum-interval requirement.
    """
    denominator_intervals = site_df["denominator_count"].sum()
    nonconforming_intervals = site_df["nonconf_count"].sum()

    return pd.DataFrame([{
        "metric": label,
        "question": question,
        "denominator": denominator_label,

        "n_sites_contained_in_table": (
            int(table_site_count)
            if table_site_count is not None
            else int(len(site_df))
        ),

        "n_assessed_sites": int(len(site_df)),
        "n_intervals": int(denominator_intervals),
        "n_nonconforming_intervals": int(nonconforming_intervals),

        "interval_nonconf_pct": (
            100 * nonconforming_intervals / denominator_intervals
            if denominator_intervals > 0 else np.nan
        ),

        "fleet_conformant_pct": (
            100 * site_df["conformant"].mean()
            if len(site_df) else np.nan
        ),

        "fleet_any_nonconf_pct": (
            100 * site_df["any_nonconformance"].mean()
            if len(site_df) else np.nan
        ),
    }])


def minimum_interval_sensitivity(site_year, numerator, denominator, config,
                                 minimums=(1, 5, 10, 30, 100, 300)):
    rows = []
    for minimum in minimums:
        d = aggregate_sites(site_year, numerator, denominator,
                            config.site_nonconf_threshold, minimum)
        rows.append({
            "minimum_intervals": minimum,
            "n_sites": len(d),
            "fleet_conformant_pct": d["conformant"].mean() * 100 if len(d) else np.nan,
            "median_nonconf_frac": d["nonconf_frac"].median() if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def coverage_summary(vw_response_site_year):
    x = vw_response_site_year.copy()
    exposed = x["exposed_count"].sum()
    return pd.DataFrame([{
        "exposed_intervals": int(exposed),
        "response_supported_intervals": int(x["response_supported_count"].sum()),
        "missing_counterfactual_intervals": int(x["missing_counterfactual_count"].sum()),
        "response_supported_pct_of_exposed": 100 * x["response_supported_count"].sum() / exposed if exposed else np.nan,
        "missing_counterfactual_pct_of_exposed": 100 * x["missing_counterfactual_count"].sum() / exposed if exposed else np.nan,
    }])


def energy_summary(energy_site_year, interval_h, label):
    d = energy_site_year.copy()
    for src, dst in [
        ("curtailed_kw_sum", "curtailed_kwh"),
        ("measured_kw_sum", "measured_kwh"),
        ("potential_kw_sum", "potential_kwh"),
        ("curtailed_interval_measured_kw_sum", "curtailed_interval_measured_kwh"),
        ("curtailed_interval_potential_kw_sum", "curtailed_interval_potential_kwh"),
    ]:
        if src in d:
            d[dst] = d[src].fillna(0) * interval_h
    rows = []
    for year, g in d.groupby("year"):
        potential = g.get("potential_kwh", pd.Series(dtype=float)).sum()
        measured = g.get("measured_kwh", pd.Series(dtype=float)).sum()
        curtailed = g["curtailed_kwh"].sum()
        event_potential = g.get(
            "curtailed_interval_potential_kwh", pd.Series(dtype=float)
        ).sum()
        rows.append({
            "metric": label, "year": int(year),
            "counterfactual_covered_sites": g.loc[
                g["counterfactual_covered_count"] > 0, "site_id"
            ].nunique(),
            "counterfactual_voltage_exposed_sites": g.loc[
                g["counterfactual_exposed_count"] > 0, "site_id"
            ].nunique(),
            "response_opportunity_sites": g.loc[
                g["response_opportunity_count"] > 0, "site_id"
            ].nunique(),
            "curtailed_sites": g.loc[g["curtailed_count"] > 0, "site_id"].nunique(),
            "counterfactual_covered_intervals": int(g["counterfactual_covered_count"].sum()),
            "counterfactual_voltage_exposed_intervals": int(g["counterfactual_exposed_count"].sum()),
            "response_opportunity_intervals": int(g["response_opportunity_count"].sum()),
            "curtailed_intervals": int(g["curtailed_count"].sum()),
            "curtailed_kwh": curtailed,
            "total_measured_generation_kwh": measured,
            "total_counterfactual_generation_kwh": potential,
            "curtailed_interval_counterfactual_kwh": event_potential,
            "curtailed_pct_within_curtailed_intervals": 100 * curtailed / event_potential
                if event_potential else np.nan,
            "curtailed_pct_of_total_counterfactual_generation": 100 * curtailed / potential
                if potential else np.nan,
            "curtailed_pct_of_measured_plus_vw_loss": 100 * curtailed / (measured + curtailed)
                if measured + curtailed else np.nan,
        })
    return d, pd.DataFrame(rows)

def legacy_exposed_energy_summary(
    energy_site_year,
    interval_h,
    label="Volt-Watt loss over exposed measured generation plus loss",
):
    """
    Reproduce the legacy exposed-generation percentage while distinguishing
    table population from sites with positive attributed curtailment.
    """
    d = energy_site_year.copy()

    d["exposed_measured_kwh"] = (
        d["exposed_measured_kw_sum"].fillna(0) * interval_h
    )

    d["curtailed_kwh"] = (
        d["curtailed_kw_sum"].fillna(0) * interval_h
    )

    rows = []

    for year, g in d.groupby("year"):
        measured = g["exposed_measured_kwh"].sum()
        curtailed = g["curtailed_kwh"].sum()
        denominator = measured + curtailed

        rows.append({
            "metric": label,
            "year": int(year),

            "sites_contained_in_table":
                int(g["site_id"].nunique()),

            "sites_with_positive_curtailment":
                int(g.loc[g["curtailed_count"] > 0, "site_id"].nunique()),

            "voltage_exposed_intervals":
                int(g["exposed_count"].sum()),

            "curtailed_intervals":
                int(g["curtailed_count"].sum()),

            "measured_voltage_exposed_kwh":
                measured,

            "identified_voltwatt_loss_kwh":
                curtailed,

            "measured_plus_identified_loss_kwh":
                denominator,

            "identified_loss_pct_of_measured_plus_loss": (
                100 * curtailed / denominator
                if denominator > 0 else np.nan
            ),
        })

    return d, pd.DataFrame(rows)

def add_vw_normalized_nonconformance(site_df, meta, interval_h,
                                     capacity_col="ac_capacity_kw"):
    """Add Wh per kW-nameplate per exposed interval to a VW site table."""
    if meta["site_id"].duplicated().any():
        raise AssertionError("Metadata merge would fan out: duplicate site_id")
    d = site_df.merge(
        meta[["site_id", capacity_col]], on="site_id", how="left",
        validate="one_to_one",
    )
    d["nonconformance_wh"] = d["nonconf_kw_sum"].fillna(0) * interval_h * 1000
    d["capacity_interval_exposure_kw"] = (
        d[capacity_col] * d["denominator_count"]
    )
    d["normalized_nonconformance_wh_per_kw_interval"] = np.where(
        d["capacity_interval_exposure_kw"] > 0,
        d["nonconformance_wh"] / d["capacity_interval_exposure_kw"],
        np.nan,
    )
    return d

def vw_nonconformance_magnitude(
    site_df,
    meta,
    interval_h,
    capacity_col="ac_capacity_kw",
):
    """
    Add Volt-Watt exceedance-energy metrics.

    nonconforming_kwh:
        Time-integrated active-power exceedance above the permitted
        voltage-dependent Volt-Watt ceiling.

    normalized_nonconformance_kwh_per_kw_interval:
        Nonconforming energy divided by site nameplate and the number of
        assessed voltage-exposed intervals.
    """
    if meta["site_id"].duplicated().any():
        raise AssertionError(
            "Metadata merge would fan out because site_id is duplicated"
        )

    d = site_df.merge(
        meta[["site_id", capacity_col]],
        on="site_id",
        how="left",
        validate="one_to_one",
    )

    d["nonconforming_kwh"] = (
        d["nonconf_kw_sum"].fillna(0) * interval_h
    )

    d["capacity_interval_exposure_kw"] = (
        d[capacity_col] * d["denominator_count"]
    )

    d["normalized_nonconformance_kwh_per_kw_interval"] = np.where(
        d["capacity_interval_exposure_kw"] > 0,
        d["nonconforming_kwh"]
        / d["capacity_interval_exposure_kw"],
        np.nan,
    )

    return d

def vw_nonconformance_magnitude_breakdown(
    site_df,
    meta,
    by,
    interval_h,
    capacity_col="ac_capacity_kw",
    min_sites=20,
):
    """
    Summarise Volt-Watt nonconforming timesteps, nonconforming energy,
    and capacity-normalised nonconforming energy by a metadata field.
    """
    d = vw_nonconformance_magnitude(
        site_df,
        meta,
        interval_h,
        capacity_col=capacity_col,
    )

    d = d.merge(
        meta[["site_id", by]],
        on="site_id",
        how="left",
        validate="one_to_one",
    )

    d[by] = d[by].fillna("Unknown")

    out = (
        d.groupby(by, dropna=False)
        .agg(
            n_sites=("site_id", "nunique"),

            assessed_intervals=(
                "denominator_count",
                "sum",
            ),

            nonconforming_intervals=(
                "nonconf_count",
                "sum",
            ),

            nonconforming_kwh=(
                "nonconforming_kwh",
                "sum",
            ),

            capacity_interval_exposure_kw=(
                "capacity_interval_exposure_kw",
                "sum",
            ),

            mean_site_normalized_nonconformance=(
                "normalized_nonconformance_kwh_per_kw_interval",
                "mean",
            ),

            median_site_normalized_nonconformance=(
                "normalized_nonconformance_kwh_per_kw_interval",
                "median",
            ),
        )
        .reset_index()
    )

    out["interval_nonconf_pct"] = (
        100
        * out["nonconforming_intervals"]
        / out["assessed_intervals"].replace(0, np.nan)
    )

    out["fleet_weighted_nonconforming_kwh_per_kw_interval"] = (
        out["nonconforming_kwh"]
        / out["capacity_interval_exposure_kw"].replace(0, np.nan)
    )

    return (
        out[out["n_sites"] >= min_sites]
        .sort_values(
            "fleet_weighted_nonconforming_kwh_per_kw_interval",
            ascending=False,
        )
    )

def vw_normalized_group_breakdown(site_df, meta, by, interval_h,
                                  capacity_col="ac_capacity_kw", min_sites=20):
    """Group VW site verdicts and normalized excess magnitude."""
    d = add_vw_normalized_nonconformance(
        site_df, meta, interval_h, capacity_col=capacity_col
    )
    d[by] = d[by].fillna("Unknown") if by in d else "Unknown"
    if by not in d.columns or (d[by] == "Unknown").all():
        # The first merge intentionally selected only capacity; attach grouping metadata now.
        d = d.drop(columns=[by], errors="ignore").merge(
            meta[["site_id", by]], on="site_id", how="left", validate="one_to_one"
        )
        d[by] = d[by].fillna("Unknown")
    out = (d.groupby(by, dropna=False)
        .agg(n_sites=("site_id", "nunique"),
             fleet_conformant_pct=("conformant", lambda x: 100*x.mean()),
             fleet_any_nonconf_pct=("any_nonconformance", lambda x: 100*x.mean()),
             total_nonconformance_wh=("nonconformance_wh", "sum"),
             total_capacity_interval_exposure_kw=("capacity_interval_exposure_kw", "sum"),
             mean_site_normalized_nc=("normalized_nonconformance_wh_per_kw_interval", "mean"),
             median_site_normalized_nc=("normalized_nonconformance_wh_per_kw_interval", "median"))
        .reset_index())
    out["fleet_weighted_normalized_nc_wh_per_kw_interval"] = (
        out["total_nonconformance_wh"]
        / out["total_capacity_interval_exposure_kw"].replace(0, np.nan)
    )
    return out[out.n_sites >= min_sites].sort_values(
        "fleet_weighted_normalized_nc_wh_per_kw_interval", ascending=False
    )


def vvar_curtailment_summary(vvar_site_year, interval_h):
    """Summarise the Stage 2 empirical-limit symptom and its CF coverage."""
    d = vvar_site_year.copy()
    d["curtailed_kwh"] = d["curtailed_kw_sum"].fillna(0) * interval_h
    rows = []
    for year, g in d.groupby("year"):
        symptom = g["symptom_count"].sum()
        missing = g["symptom_missing_counterfactual_count"].sum()
        rows.append({
            "year": int(year),
            "symptom_sites": g.loc[g.symptom_count > 0, "site_id"].nunique(),
            "symptom_intervals": int(symptom),
            "symptom_intervals_missing_counterfactual": int(missing),
            "counterfactual_coverage_pct": 100 * (symptom-missing) / symptom if symptom else np.nan,
            "curtailed_intervals": int(g.curtailed_count.sum()),
            "curtailed_kwh": g.curtailed_kwh.sum(),
        })
    return d, pd.DataFrame(rows)


def energy_group_breakdown(energy_site_year, meta, by,
                           energy_col="curtailed_kwh", min_sites=20):
    if meta["site_id"].duplicated().any():
        raise AssertionError("Metadata merge would fan out: duplicate site_id")
    site = energy_site_year.groupby("site_id", as_index=False)[energy_col].sum()
    d = site.merge(meta, on="site_id", how="left", validate="one_to_one")
    d[by] = d[by].fillna("Unknown")
    out = (d.groupby(by, dropna=False)
        .agg(n_sites=("site_id", "nunique"),
             sites_with_curtailment=(energy_col, lambda x: int((x > 0).sum())),
             curtailed_kwh=(energy_col, "sum"),
             median_site_kwh=(energy_col, "median"))
        .reset_index())
    return out[out.n_sites >= min_sites].sort_values("curtailed_kwh", ascending=False)


def group_breakdown(site_df, meta, by, min_sites=20):
    if meta["site_id"].duplicated().any():
        raise AssertionError("Metadata merge would fan out: duplicate site_id")
    d = site_df.merge(meta, on="site_id", how="left", validate="one_to_one")
    keys = [by] if isinstance(by, str) else list(by)
    for key in keys:
        d[key] = d[key].fillna("Unknown")
    out = (d.groupby(keys, dropna=False)
        .agg(n_sites=("site_id", "nunique"),
             denominator_intervals=("denominator_count", "sum"),
             nonconforming_intervals=("nonconf_count", "sum"),
             fleet_conformant_pct=("conformant", lambda x: x.mean() * 100),
             fleet_any_nonconf_pct=("any_nonconformance", lambda x: x.mean() * 100))
        .reset_index())
    out["interval_nonconf_pct"] = 100 * out["nonconforming_intervals"] / out["denominator_intervals"]
    return out[out["n_sites"] >= min_sites].sort_values("fleet_conformant_pct")


def monthly_rates(monthly):
    d = monthly.copy()
    d["interval_nonconf_pct"] = 100 * d["nonconf_count"] / d["denominator_count"]
    d["period"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
    return d


def vvar_band_summary(vvar_site_year):
    counts = vvar_site_year[["adverse", "inactive", "shortfall",
                            "near_conformant", "surplus"]].sum()
    denominator = vvar_site_year["capability_assessable_count"].sum()
    return pd.DataFrame({
        "band": counts.index,
        "interval_count": counts.values,
        "pct_of_capability_assessable": 100 * counts.values / denominator,
        "counts_as_project_failure": [x in VVAR_FAILURE_BANDS for x in counts.index],
    })



def legacy_vw_normalized_nonconformance(
    legacy_site_df,
    meta,
    interval_h,
    capacity_col="ac_capacity_kw",
):
    """
    Reproduce the legacy (ARENA report) per-site normalised nonconformance.

    Formula per site:
        norm_nc_wh_kw = nc_sum_kw × interval_h × 1000
                        / (total_count × ac_capacity_kw)

    Where ``total_count`` in the legacy table is
    ``count(nonconformance_voltwattghi)`` — the count of non-NULL scored
    intervals, which includes intervals with a missing counterfactual
    (scored via the ``uncurtailed_P IS NULL`` path).  This is NOT the
    same as the v2 ``total_count`` (all voltage-exposed intervals).

    The group metric is the unweighted mean across sites, matching the
    colleague's ``("norm_nc_wh_kw", "mean")`` aggregation.
    """
    if meta["site_id"].duplicated().any():
        raise AssertionError("Metadata merge would fan out: duplicate site_id")

    d = legacy_site_df.merge(
        meta[["site_id", capacity_col]],
        on="site_id",
        how="left",
        validate="one_to_one",
    )

    d["norm_nc_wh_kw"] = np.where(
        (d["total_count"] > 0) & (d[capacity_col] > 0),
        d["nc_kw_sum"] * interval_h * 1000
        / (d["total_count"] * d[capacity_col]),
        np.nan,
    )

    return d


def legacy_vw_breakdown(
    legacy_site_df,
    meta,
    by,
    interval_h,
    capacity_col="ac_capacity_kw",
    threshold=0.10,
    min_sites=20,
):
    """
    Reproduce the legacy report's state/DNSP/OEM breakdown table.

    Returns total NC timestamps, total NC kWh, mean-across-sites
    normalised NC (Wh/kW/eligible-interval), and the 10% site
    nonconformance rate — all from the legacy ``conformance_voltwattghi``
    table's definitions.
    """
    d = legacy_vw_normalized_nonconformance(
        legacy_site_df, meta, interval_h, capacity_col
    )

    d = d.merge(
        meta[["site_id", by]],
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    d[by] = d[by].fillna("Unknown")

    d["nonconf_frac"] = np.where(
        d["total_count"] > 0,
        d["nc_count"] / d["total_count"],
        np.nan,
    )
    d["nonconformant"] = d["nonconf_frac"] >= threshold
    d["any_nonconf"] = d["nc_count"] > 0

    d["nc_kwh"] = d["nc_kw_sum"] * interval_h

    out = (
        d.groupby(by, dropna=False)
        .agg(
            n_sites=("site_id", "nunique"),
            total_nc_timestamps=("nc_count", "sum"),
            total_nc_kwh=("nc_kwh", "sum"),
            mean_norm_nc_wh_kw=("norm_nc_wh_kw", "mean"),
            pct_nonconf_10pct=("nonconformant", lambda s: s.mean() * 100),
            pct_any_nonconf=("any_nonconf", lambda s: s.mean() * 100),
        )
        .reset_index()
    )

    return out[out["n_sites"] >= min_sites].sort_values(
        "mean_norm_nc_wh_kw", ascending=False
    )


def reporting_tally(*tables):
    return pd.concat(tables, ignore_index=True, sort=False)
