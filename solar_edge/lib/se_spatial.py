"""
Is postcode actually explanatory of non-conformance?
====================================================

Deliverable D16. Two things live here: an honest test of whether geography
carries information about conformance, and an interactive map to explore it.

The question is harder than it looks
------------------------------------
"Does postcode explain non-conformance?" invites a regression of the site rate on
postcode, an R^2 near 0.9, and a conclusion that geography is the dominant driver.
That conclusion would be an artefact. The fleet has **1,600 sites across 507
postcodes** and the distribution is brutal::

    194 postcodes hold exactly 1 site
    115 hold 2
    median 2, mean 3.2

With one site in 38% of postcodes, "the postcode effect" and "that one site's
behaviour" are the same number. A 507-level categorical fitted to 1,600
observations will always absorb most of the variance, because it has enough free
parameters to memorise the data. High R^2 here is a statement about degrees of
freedom, not about geography.

So this module never reports a bare R^2. Every measure is compared against a
**permutation null**: shuffle the postcode labels between sites, recompute, and
ask how often chance alone does as well. That null automatically carries the same
group-size structure and the same overfitting capacity as the real data, so what
survives it is real signal.

Three complementary tests, because they can disagree
----------------------------------------------------
``variance_decomposition``
    Are sites in the same postcode more alike than sites in different postcodes?
    (ICC, permutation-tested.) A *label* question -- it treats postcodes as
    unordered categories and ignores where they are.

``morans_i``
    Are *nearby* postcodes more alike than distant ones? A *spatial* question.
    Postcode can fail the first test and pass this one, if the real gradient is
    regional rather than per-postcode.

``nested_explained_variance``
    Does postcode add anything once state, system size and phase cohort are
    known? Postcode correlates with all three; if they account for the same
    variance, postcode is a proxy, not a cause.

None of this establishes causation. Postcode is a stand-in for network topology,
transformer sizing, feeder length, installer, and inverter vintage -- none of
which are in this dataset. A confirmed postcode effect says "look at the network
here", not "the postcode caused it".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solar_edge.lib import se_params

__all__ = [
    "site_rates",
    "variance_decomposition",
    "morans_i",
    "nested_explained_variance",
    "postcode_effect_report",
    "interactive_postcode_map",
    "MIN_ASSESSABLE",
]

#: Sites with fewer assessable intervals than this have rates dominated by noise.
#: 500 five-minute intervals is roughly 40 hours of assessable operation.
MIN_ASSESSABLE = 500


# ═══════════════════════════════════════════════════════════════════════════
# INPUT
# ═══════════════════════════════════════════════════════════════════════════

def site_rates(con, site_day: pd.DataFrame, config=None,
               min_assessable: int = MIN_ASSESSABLE) -> pd.DataFrame:
    """
    One row per site: its non-conformance rate plus every candidate explanator.

    The unit is the **site, unweighted** -- one inverter, one observation -- which
    matches how the 10% rule treats them. Weighting by interval count would let a
    few heavily reporting sites define their postcode's rate, which is the
    opposite of what a between-postcode comparison needs.

    Sites below ``min_assessable`` are dropped, not zero-filled. A rate computed
    on 20 intervals is noise, and noise concentrated in sparse rural postcodes
    would manufacture exactly the geographic pattern this module is testing for.
    """
    from solar_edge.lib import se_conformance as cf

    config = (config or se_params.CONFIG).validate()
    enriched = cf.enrich_site_day(con, site_day)

    reduced = [f"{c}_count" for c in cf.REDUCED_NONCONF]
    per_site = enriched.groupby("site_alias", as_index=False).agg(
        postcode=("postcode", "first"),
        state=("site_state", "first"),
        capacity_band=("capacity_band", "first"),
        is_three_phase=("site_three_phase", "first"),
        s_99=("s_99", "first"),
        assessable=("total_count", "sum"),
        **{c: (c, "sum") for c in reduced},
    )
    per_site["reduced_nonconf"] = per_site[reduced].sum(axis=1)
    per_site["rate"] = 100.0 * per_site.reduced_nonconf / per_site.assessable.replace(0, np.nan)
    per_site["cohort"] = per_site.is_three_phase.map(
        {True: "three-phase", False: "single-phase"})

    kept = per_site[(per_site.assessable >= min_assessable)
                    & per_site.rate.notna()
                    & per_site.postcode.notna()].copy()

    dropped = len(per_site) - len(kept)
    print(f"{len(kept):,} sites with >= {min_assessable:,} assessable intervals "
          f"({dropped:,} dropped as too sparse to rate)")
    counts = kept.groupby("postcode").size()
    print(f"{counts.size:,} postcodes | median {counts.median():.0f} site(s) each | "
          f"{int((counts == 1).sum()):,} with only one")
    return kept


# ═══════════════════════════════════════════════════════════════════════════
# 1. IS THERE A POSTCODE EFFECT AT ALL?
# ═══════════════════════════════════════════════════════════════════════════

def _icc(values: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """
    One-way random-effects ICC: the share of variance that is between groups.

    Returns ``(icc, ms_between, ms_within)``. Groups of size 1 contribute to the
    between term but not the within term, which is exactly why the permutation
    null matters -- with 38% singleton postcodes the estimator is biased upward
    and only a like-for-like null can calibrate it.
    """
    frame = pd.DataFrame({"y": values, "g": groups})
    grand = frame.y.mean()
    stats = frame.groupby("g").y.agg(["mean", "count", "var"])

    k = len(stats)
    n = len(frame)
    if k < 2 or n <= k:
        return np.nan, np.nan, np.nan

    ss_between = float((stats["count"] * (stats["mean"] - grand) ** 2).sum())
    ss_within = float(((frame.y - frame.g.map(stats["mean"])) ** 2).sum())

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)

    # Adjusted average group size for unbalanced designs (Snedecor & Cochran).
    n0 = (n - (stats["count"] ** 2).sum() / n) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else np.nan
    return float(icc), float(ms_between), float(ms_within)


def variance_decomposition(rates: pd.DataFrame, n_permutations: int = 999,
                           seed: int = 0) -> dict:
    """
    Are two sites in the same postcode more alike than two sites picked at random?

    Reports the intraclass correlation and, crucially, the ICC obtained when the
    postcode labels are **shuffled between sites**. The shuffled null preserves the
    number of postcodes and the number of sites in each, so it reproduces the
    singleton-heavy structure and the associated upward bias. The observed ICC is
    only evidence of geography if it beats that null.

    ``p_value`` is the fraction of shuffles reaching the observed ICC, with the
    usual +1 correction so it can never be exactly zero.
    """
    values = rates.rate.to_numpy(float)
    groups = rates.postcode.to_numpy()

    observed, ms_b, ms_w = _icc(values, groups)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = _icc(values, rng.permutation(groups))[0]

    null = null[~np.isnan(null)]
    p = (1 + int((null >= observed).sum())) / (1 + len(null))
    return {
        "icc_observed": observed,
        "icc_null_mean": float(null.mean()),
        "icc_null_p95": float(np.quantile(null, 0.95)),
        "p_value": p,
        "ms_between": ms_b,
        "ms_within": ms_w,
        "n_sites": len(rates),
        "n_postcodes": int(rates.postcode.nunique()),
        "n_multi_site_postcodes": int((rates.groupby("postcode").size() > 1).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. IS THE EFFECT SPATIAL?
# ═══════════════════════════════════════════════════════════════════════════

def morans_i(rates: pd.DataFrame, centroids: pd.DataFrame, k: int = 8,
             n_permutations: int = 999, seed: int = 0,
             min_sites: int = 2) -> dict:
    """
    Moran's I on postcode-mean rates: are neighbouring postcodes alike?

    Weights are k-nearest-neighbour on the postcode representative points, row-
    standardised. kNN rather than a distance band because the fleet is three
    disconnected metro clusters plus sparse regional coverage -- any fixed radius
    either isolates the regional postcodes entirely or swamps the metro ones.

    Restricted to postcodes with at least ``min_sites`` sites. A postcode rate
    built from one site is that site's rate, and feeding those in measures
    autocorrelation of individual inverters dressed up as geography.

    I > 0 means nearby postcodes resemble each other, which is the signature of a
    network- or region-level driver. I ~ 0 means the variation is local to each
    postcode. Significance is again by permutation -- shuffling values across
    locations while holding the weights fixed.
    """
    grouped = rates.groupby("postcode").agg(rate=("rate", "mean"),
                                            n_sites=("site_alias", "size"))
    grouped = grouped[grouped.n_sites >= min_sites].reset_index()
    merged = grouped.merge(centroids, on="postcode", how="inner").dropna(
        subset=["lat", "lon"])
    n = len(merged)
    if n < k + 2:
        return {"morans_i": np.nan, "p_value": np.nan, "n_postcodes": n,
                "note": f"only {n} postcodes with >= {min_sites} sites; need > {k + 2}"}

    lat = np.radians(merged.lat.to_numpy(float))
    lon = np.radians(merged.lon.to_numpy(float))
    # Equirectangular approximation is ample at these separations and avoids a
    # scipy dependency; we only need the RANKING of distances for kNN.
    x = lon * np.cos(lat.mean())
    d2 = (x[:, None] - x[None, :]) ** 2 + (lat[:, None] - lat[None, :]) ** 2
    np.fill_diagonal(d2, np.inf)

    W = np.zeros((n, n))
    neighbours = np.argsort(d2, axis=1)[:, :k]
    np.put_along_axis(W, neighbours, 1.0, axis=1)
    W /= W.sum(axis=1, keepdims=True)          # row-standardised

    y = merged.rate.to_numpy(float)

    def _I(vals):
        z = vals - vals.mean()
        denom = (z ** 2).sum()
        return np.nan if denom == 0 else float(n / W.sum() * (z @ W @ z) / denom)

    observed = _I(y)
    rng = np.random.default_rng(seed)
    null = np.array([_I(rng.permutation(y)) for _ in range(n_permutations)])
    null = null[~np.isnan(null)]
    p = (1 + int((np.abs(null) >= abs(observed)).sum())) / (1 + len(null))

    return {"morans_i": observed, "expected_i": -1.0 / (n - 1),
            "null_sd": float(null.std()), "p_value": p,
            "n_postcodes": n, "k_neighbours": k}


# ═══════════════════════════════════════════════════════════════════════════
# 3. DOES POSTCODE ADD ANYTHING THE OTHERS DO NOT?
# ═══════════════════════════════════════════════════════════════════════════

def _r2_from_groups(values: np.ndarray, groups: np.ndarray) -> float:
    """R^2 of a group-means model: 1 - SS_within / SS_total."""
    frame = pd.DataFrame({"y": values, "g": groups})
    ss_tot = float(((frame.y - frame.y.mean()) ** 2).sum())
    if ss_tot == 0:
        return np.nan
    means = frame.groupby("g").y.transform("mean")
    return float(1 - ((frame.y - means) ** 2).sum() / ss_tot)


def nested_explained_variance(rates: pd.DataFrame, n_permutations: int = 199,
                              seed: int = 0) -> pd.DataFrame:
    """
    Variance explained by each candidate grouping, against its own random null.

    The ``r2_null_mean`` column is the whole point. A grouping with 507 levels
    achieves a large R^2 on 1,600 points **whatever the labels mean** -- shuffle
    them and it still does. The honest quantity is the excess::

        r2_excess = r2_observed - r2_null_mean

    which is what a grouping buys beyond its own capacity to overfit. Compare
    groupings on ``r2_excess``, never on ``r2_observed``.
    """
    values = rates.rate.to_numpy(float)
    rng = np.random.default_rng(seed)

    candidates = {
        "state": rates.state.astype(str),
        "capacity_band": rates.capacity_band.astype(str),
        "cohort": rates.cohort.astype(str),
        "state + capacity_band + cohort": (
            rates.state.astype(str) + "|" + rates.capacity_band.astype(str)
            + "|" + rates.cohort.astype(str)),
        "postcode": rates.postcode.astype(str),
    }

    rows = []
    for name, labels in candidates.items():
        g = labels.to_numpy()
        observed = _r2_from_groups(values, g)
        null = np.array([_r2_from_groups(values, rng.permutation(g))
                         for _ in range(n_permutations)])
        p = (1 + int((null >= observed).sum())) / (1 + len(null))
        rows.append({
            "grouping": name,
            "n_levels": int(pd.Series(g).nunique()),
            "r2_observed": round(observed, 4),
            "r2_null_mean": round(float(null.mean()), 4),
            "r2_excess": round(observed - float(null.mean()), 4),
            "p_value": round(p, 4),
        })
    return pd.DataFrame(rows).sort_values("r2_excess", ascending=False)


def postcode_effect_report(con, site_day: pd.DataFrame, config=None,
                           centroids: pd.DataFrame | None = None,
                           min_assessable: int = MIN_ASSESSABLE) -> dict:
    """Run all three tests and print a verdict that states its own limits."""
    from solar_edge.lib import se_plots as plots

    rates = site_rates(con, site_day, config, min_assessable)
    centroids = plots.postcode_centroids(con) if centroids is None else centroids

    vd = variance_decomposition(rates)
    mi = morans_i(rates, centroids)
    nested = nested_explained_variance(rates)

    print("\n" + "=" * 78)
    print("1. CLUSTERING  — are sites in the same postcode alike?")
    print("=" * 78)
    print(f"  ICC observed      {vd['icc_observed']:+.4f}")
    print(f"  ICC under shuffle {vd['icc_null_mean']:+.4f} "
          f"(95th pct {vd['icc_null_p95']:+.4f})")
    print(f"  permutation p     {vd['p_value']:.4f}")
    print(f"  {vd['n_multi_site_postcodes']:,} of {vd['n_postcodes']:,} postcodes "
          f"have more than one site — only those carry any within-postcode signal.")

    print("\n" + "=" * 78)
    print("2. SPATIAL  — are neighbouring postcodes alike?")
    print("=" * 78)
    if np.isnan(mi.get("morans_i", np.nan)):
        print(f"  not computable: {mi.get('note')}")
    else:
        print(f"  Moran's I  {mi['morans_i']:+.4f}   (expected under no pattern "
              f"{mi['expected_i']:+.4f})")
        print(f"  permutation p {mi['p_value']:.4f}   "
              f"over {mi['n_postcodes']:,} multi-site postcodes, k={mi['k_neighbours']}")

    print("\n" + "=" * 78)
    print("3. INCREMENTAL  — does postcode beat simpler explanations?")
    print("=" * 78)
    print(nested.to_string(index=False))
    print("\n  Compare r2_excess, NOT r2_observed. Postcode has ~500 levels and will")
    print("  score a high r2_observed on shuffled labels too; r2_excess subtracts that.")

    clustered = vd["p_value"] < 0.05
    spatial = (not np.isnan(mi.get("morans_i", np.nan))) and mi["p_value"] < 0.05
    idx = nested.set_index("grouping")
    pc, simple, state_only = idx.loc["postcode"], idx.loc["state + capacity_band + cohort"], idx.loc["state"]
    adds = pc.r2_excess > simple.r2_excess
    # Does postcode buy anything over STATE alone? That separates "geography
    # matters at postcode resolution" from "geography matters at state resolution".
    beats_state = pc.r2_excess > state_only.r2_excess * 1.25

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if clustered and spatial and adds:
        print("  Postcode carries real, spatially structured signal beyond state,")
        print("  system size and phase. Treat it as a pointer to network conditions")
        print("  worth investigating — not as a cause in itself.")
    elif clustered and spatial and not beats_state:
        print("  The variation IS spatially structured and highly significant")
        print(f"  (Moran's I {mi['morans_i']:+.3f}, p={mi['p_value']:.3f}) — but at a")
        print("  COARSER SCALE than postcode.")
        print(f"    state alone      r2_excess {state_only.r2_excess:.4f}  "
              f"({int(state_only.n_levels)} levels)")
        print(f"    postcode         r2_excess {pc.r2_excess:.4f}  "
              f"({int(pc.n_levels)} levels)")
        print("  ~500 postcode parameters buy almost nothing over 3 state ones, so the")
        print("  gradient is regional. Report conformance by state or metro area;")
        print("  per-postcode rates mostly resolve noise, not network conditions.")
    elif clustered and spatial:
        print("  Spatially structured clustering that also survives the simpler")
        print("  groupings. Postcode resolution is carrying information — worth")
        print("  following up against network topology.")
    elif clustered and adds:
        print("  Postcodes cluster, but the pattern is not spatially smooth: nearby")
        print("  postcodes are no more alike than distant ones. That points to a")
        print("  per-postcode driver (a feeder, an installer, a bulk deployment)")
        print("  rather than a regional one.")
    elif clustered:
        print("  Apparent clustering, but it does not survive controlling for state,")
        print("  system size and cohort. Postcode is most likely a proxy for those.")
    else:
        print("  No postcode effect detectable above chance. With a median of")
        print(f"  {rates.groupby('postcode').size().median():.0f} site(s) per postcode "
              "this is the expected result even")
        print("  if a real effect exists — the design cannot resolve it. Read this as")
        print("  'not shown', not as 'shown to be absent'.")

    print(f"\n  Overfitting check: shuffled postcode labels still score "
          f"r2 = {pc.r2_null_mean:.3f}.")
    print(f"  That is what {int(pc.n_levels)} free parameters buy on "
          f"{vd['n_sites']:,} sites before any signal.")
    print("\n  None of these tests establish causation. Postcode stands in for network")
    print("  topology, feeder length, transformer sizing, installer and inverter")
    print("  vintage — none of which are in this dataset.")

    return {"rates": rates, "variance": vd, "moran": mi, "nested": nested}


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE MAP
# ═══════════════════════════════════════════════════════════════════════════

def interactive_postcode_map(con, frame, value_col="pct_reduced_nonconf", *,
                             min_sites=1, centroids=None, out_path=None,
                             label="Volt-VAr reduced non-conformance",
                             zoom_start=5, cmap=("#1a9850", "#fee08b", "#d73027")):
    """
    A pannable, zoomable Leaflet map: rate as colour, installs as circle size.

    Answers the two questions together, which the static map can only hint at:
    hover any postcode for its rate **and** its install count, so a lurid red
    circle backed by two sites is immediately identifiable as noise rather than a
    finding.

    Radius scales with the SQUARE ROOT of the install count, so perceived area --
    not radius -- is proportional to the number of sites. Scaling radius directly
    would exaggerate large postcodes quadratically.

    ``folium`` is a soft dependency; it is only needed for this one function.
    Returns the map object, which renders inline in Jupyter, and writes a
    standalone HTML file if ``out_path`` is given.
    """
    try:
        import folium
        from branca.colormap import LinearColormap
    except ImportError as exc:      # pragma: no cover
        raise ImportError(
            "interactive_postcode_map needs folium.\n"
            "  conda install -c conda-forge folium   (or: pip install folium)"
        ) from exc

    from solar_edge.lib import se_plots as plots

    data = frame.copy()
    data["postcode"] = data.postcode.astype(str)
    centroids = plots.postcode_centroids(con) if centroids is None else centroids
    centroids = centroids.copy()
    centroids["postcode"] = centroids.postcode.astype(str)

    data = data.merge(centroids, on="postcode", how="left")
    data = data.dropna(subset=["lat", "lon", value_col])
    data = data[data.n_sites >= min_sites]
    if data.empty:
        raise ValueError(f"nothing to map for {value_col}")

    vmin, vmax = float(data[value_col].min()), float(data[value_col].max())
    scale = LinearColormap(list(cmap), vmin=vmin, vmax=vmax)
    scale.caption = f"{label} (%)"

    m = folium.Map(location=[float(data.lat.mean()), float(data.lon.mean())],
                   zoom_start=zoom_start, tiles="cartodbpositron")

    biggest = float(data.n_sites.max())
    for row in data.itertuples():
        # sqrt so that AREA tracks the install count
        radius = 4 + 16 * (row.n_sites / biggest) ** 0.5
        value = float(getattr(row, value_col))
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=radius,
            color="#444444", weight=0.6,
            fill=True, fill_color=scale(value), fill_opacity=0.82,
            tooltip=(f"<b>Postcode {row.postcode}</b><br>"
                     f"{label}: <b>{value:.1f}%</b><br>"
                     f"Installs in postcode: <b>{int(row.n_sites)}</b>"),
        ).add_to(m)

    scale.add_to(m)
    folium.LayerControl().add_to(m)

    if out_path is not None:
        from pathlib import Path

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_path))
        print(f"Interactive map -> {out_path}")
    return m
