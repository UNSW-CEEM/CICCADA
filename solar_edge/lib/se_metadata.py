"""
Site dimension and capacity proxies.
====================================

Deliverable D4. The SolarEdge analogue of `meta_up23c`, except that almost all of
it has to be *derived* rather than read: the delivery ships alias, postcode and
state, and nothing else.

Two tables:

* ``se_site``          one row per site: identity, geography, phase configuration,
                       observation coverage, and data-quality flags.
* ``se_site_capacity`` one row per site: the ``s_99`` empirical apparent-power
                       limit and its supporting counts, ported from
                       ``build_s99_estimates.py``.

They are kept apart deliberately. ``se_site`` is descriptive and stable;
``se_site_capacity`` is a *modelling choice* with a method version attached, and
is swept in the sensitivity notebook. Blending them would hide that distinction.

The missing nameplate
---------------------
Solar Analytics carried ``ac_capacity_kw`` from the provider and used ``s_99``
only as an alternative basis. Here there is no nameplate at all, so ``s_99`` is
the sole capacity basis and the AS/NZS 4777.2 tolerance is re-anchored to
``0.04 * s_99``. That substitution is labelled everywhere it appears and is a
first-class sensitivity axis, not a silent default.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from solar_edge.config import se_config as C

__all__ = [
    "build_site_dimension",
    "build_capacity_proxies",
    "attach_geography",
    "bom_grid_snap_sql",
    "site_summary",
    "check_site_dimension",
]

#: Provenance string written beside every capacity estimate, mirroring
#: `build_s99_estimates.METHOD_VERSION`.
CAPACITY_METHOD_VERSION = "observed_site_apparent_power_p99_v1"


# ═══════════════════════════════════════════════════════════════════════════
# SITE DIMENSION
# ═══════════════════════════════════════════════════════════════════════════

def build_site_dimension(
    con: duckdb.DuckDBPyConnection, write: bool = True
) -> pd.DataFrame:
    """
    Build ``se_site``: one row per site, from the store plus the alias mapping.

    Phase configuration is taken as the MAXIMUM number of phases ever seen
    reporting, not the modal count. A three-phase site whose phases 2 and 3 drop
    out for a week is still three-phase, and treating it as single-phase for that
    week would silently halve its apparent capacity.

    Coverage columns exist to support the cohort filters in D5: a site observed
    for three weeks of the year should not sit in the same denominator as one
    observed throughout, and the analysis layer needs to be able to say so.
    """
    frame = con.execute(
        f"""
        WITH per_site AS (
            SELECT
                site_alias,
                any_value(state)                        AS state,
                any_value(postcode)                     AS postcode,
                max(n_phases_reporting)                 AS n_phases,
                count(*)                                AS n_intervals,
                count(DISTINCT date_trunc('day', ts_aest)) AS n_days_observed,
                min(ts_aest)                            AS first_seen_aest,
                max(ts_aest)                            AS last_seen_aest,
                count(*) FILTER (WHERE P_kW > 0.1)      AS n_generating_intervals,
                count(*) FILTER (WHERE Q_kvar IS NOT NULL) AS n_reactive_intervals,
                count(*) FILTER (WHERE V_max IS NOT NULL)  AS n_voltage_intervals,
                count(*) FILTER (WHERE derating_active)    AS n_derating_intervals,
                count(*) FILTER (
                    WHERE (hour(ts_aest) >= 21 OR hour(ts_aest) < 4) AND P_kW > 0.5
                )                                       AS n_night_generation_rows
            FROM se_interval
            GROUP BY site_alias
        )
        SELECT
            p.site_alias,
            p.state,
            p.postcode,
            {_tz_case()}                                AS tz_name,
            p.n_phases,
            (p.n_phases >= 3)                           AS is_three_phase,
            p.first_seen_aest,
            p.last_seen_aest,
            p.n_intervals,
            p.n_days_observed,
            p.n_generating_intervals,
            p.n_reactive_intervals,
            p.n_voltage_intervals,
            p.n_derating_intervals,
            round(100.0 * p.n_derating_intervals / nullif(p.n_intervals, 0), 4)
                                                        AS pct_derating,
            -- Coverage against a full year of observation days, not intervals:
            -- most inverters stop reporting after dark, so an interval-based
            -- denominator would make every healthy site look 40% covered.
            round(100.0 * p.n_days_observed / 365.0, 2) AS pct_days_covered,
            p.n_night_generation_rows,
            (p.n_night_generation_rows > 0)             AS has_night_generation_anomaly,
            CAST(NULL AS DOUBLE)                        AS centroid_lat,
            CAST(NULL AS DOUBLE)                        AS centroid_lon,
            CAST(NULL AS DOUBLE)                        AS postcode_area_km2,
            CAST(NULL AS VARCHAR)                       AS geography_source
        FROM per_site p
        ORDER BY p.site_alias
        """
    ).df()

    if write:
        _write_single(con, frame, "se_site")
    return frame


def _tz_case() -> str:
    branches = "\n            ".join(
        f"WHEN p.state = '{state}' THEN '{tz}'"
        for state, tz in C.STATE_TIMEZONE.items()
    )
    return f"CASE\n            {branches}\n            END"


# ═══════════════════════════════════════════════════════════════════════════
# CAPACITY PROXIES
# ═══════════════════════════════════════════════════════════════════════════

def build_capacity_proxies(
    con: duckdb.DuckDBPyConnection,
    min_active_power_fraction: float = 0.0,
    write: bool = True,
) -> pd.DataFrame:
    """
    Build ``se_site_capacity``: the empirical apparent-power limit per site.

    Ported from ``build_s99_estimates.py``, with one forced substitution.

    The original filters the population with
    ``P_kw >= min_active_power_fraction * ac_capacity_kw``. There is no nameplate
    here, so the fraction is applied to the site's own ``p_99`` active power
    instead. At the default ``0.0`` the filter is inert and the two methods agree
    exactly; above zero they diverge, which is why the basis is recorded in
    ``capacity_basis`` on every row.

    ``s_99`` is an EMPIRICAL observation, not a manufacturer rating. It must never
    be presented as verified ``S_rated`` -- a site that never reached its inverter
    limit will have an ``s_99`` well below it, biasing any apparent-limit test
    toward finding symptoms. That is a stated limitation, and the reason the
    quantile is swept in D15.
    """
    if not 0.0 <= min_active_power_fraction < 1.0:
        raise ValueError("min_active_power_fraction must be in [0, 1)")

    frame = con.execute(
        f"""
        WITH s AS (
            SELECT site_alias,
                   P_kW,
                   sqrt(P_kW * P_kW + coalesce(Q_kvar, 0) * coalesce(Q_kvar, 0)) AS S_kVA,
                   ts_aest
            FROM se_interval
            WHERE P_kW IS NOT NULL
        ),
        ref AS (
            -- approx_quantile (T-Digest), not quantile_cont. Exact quantiles must
            -- hold every value per group in memory, which for 86.6 M rows across
            -- 1,602 groups exhausts a modest machine. It is also what the original
            -- does: build_s99_estimates.py uses Trino's approx_percentile, so this
            -- is the more faithful port as well as the cheaper one.
            SELECT site_alias, approx_quantile(P_kW, 0.99) AS p_99_ref
            FROM s GROUP BY site_alias
        ),
        filtered AS (
            SELECT s.*, r.p_99_ref
            FROM s JOIN ref r USING (site_alias)
            WHERE s.S_kVA > 0
              AND s.P_kW >= {min_active_power_fraction} * r.p_99_ref
        )
        SELECT
            site_alias,
            approx_quantile(S_kVA, 0.99)                AS s_99,
            approx_quantile(S_kVA, 0.95)                AS s_95,
            max(S_kVA)                                  AS s_max,
            approx_quantile(P_kW, 0.99)                 AS p_99,
            approx_quantile(P_kW, 0.95)                 AS p_95,
            max(P_kW)                                   AS p_max,
            count(*)                                    AS n_intervals,
            min(ts_aest)                                AS first_ts_aest,
            max(ts_aest)                                AS last_ts_aest,
            {min_active_power_fraction}                 AS min_active_power_fraction,
            'p_99_active_power'                         AS capacity_basis,
            '{CAPACITY_METHOD_VERSION}'                 AS method_version
        FROM filtered
        GROUP BY site_alias
        ORDER BY site_alias
        """
    ).df()

    if write:
        _write_single(con, frame, "se_site_capacity")
    return frame


def _write_single(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, logical: str) -> None:
    """Write a small dimension table as a single Parquet file and register it."""
    from solar_edge.lib import se_store

    path = C.store_path(logical)
    path.parent.mkdir(parents=True, exist_ok=True)
    con.register("_to_write", frame)
    con.execute(
        f"COPY _to_write TO '{path.as_posix()}' "
        f"(FORMAT PARQUET, COMPRESSION '{C.PARQUET_COMPRESSION}')"
    )
    con.unregister("_to_write")
    se_store.register_store_views(con)


# ═══════════════════════════════════════════════════════════════════════════
# GEOGRAPHY  (optional -- requires the ABS POA-2021 shapefile)
# ═══════════════════════════════════════════════════════════════════════════

def attach_geography(
    con: duckdb.DuckDBPyConnection,
    shapefile: Path | str | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """
    Add postcode centroid, polygon area and geography provenance to ``se_site``.

    Requires the ABS **POA 2021** shapefile (``POA_2021_AUST_GDA2020.shp``), the
    same one ``BOM_NCI/Get_ALL_postcodes_ABS.ipynb`` uses. Point
    `se_config.POA_SHAPEFILE` at it or pass the path.

    Why this is a separate, optional step
    -------------------------------------
    It is the only part of D4 that needs an external file and two extra packages,
    and D5/D6 do not depend on it. Keeping it separate means a missing shapefile
    delays the irradiance work at D12 without blocking the conformance work.

    What the centroid is and is not
    -------------------------------
    ``representative_point()`` is used rather than the geometric centroid, because
    a centroid can fall outside a concave or multi-part polygon -- and Australian
    postcodes include plenty of both.

    Even so, this is the largest methodological compromise in the port. Solar
    Analytics had per-site coordinates; this delivery has a postcode. A regional
    SA or QLD postcode can span tens of kilometres, so the point may sit well away
    from the actual array, and cloud fields decorrelate over that distance.
    ``postcode_area_km2`` is recorded precisely so that the resulting bias is
    auditable: expect the D12 counterfactual quality gate to reject large-area
    postcodes preferentially, which is correct behaviour but NOT random attrition.
    """
    shapefile = Path(shapefile or getattr(C, "POA_SHAPEFILE", "") or "")
    if not shapefile.exists():
        raise FileNotFoundError(
            "ABS POA-2021 shapefile not found.\n"
            f"  Looked for: {shapefile}\n"
            "  Expected file: POA_2021_AUST_GDA2020.shp (with its .dbf/.shx/.prj siblings)\n"
            "  This is the same shapefile BOM_NCI/Get_ALL_postcodes_ABS.ipynb uses.\n"
            "  Set CICCADA_SE_POA_SHAPEFILE or pass shapefile=... .\n"
            "  D5 and D6 do not need it; D12 (BOM irradiance) does."
        )

    import geopandas as gpd

    sites = con.execute("SELECT * FROM se_site").df()

    poa = gpd.read_file(shapefile)[["POA_CODE21", "geometry"]].to_crs("EPSG:4326")
    poa = poa[poa.POA_CODE21.isin(set(sites.postcode))].copy()

    # Area needs an equal-area projection; EPSG:3577 (GDA94 Australian Albers)
    # is the standard choice for continental Australia.
    poa["postcode_area_km2"] = poa.to_crs("EPSG:3577").area / 1e6
    points = poa.geometry.representative_point()
    poa["centroid_lat"] = points.y
    poa["centroid_lon"] = points.x

    lookup = poa[["POA_CODE21", "centroid_lat", "centroid_lon", "postcode_area_km2"]]
    enriched = sites.drop(
        columns=["centroid_lat", "centroid_lon", "postcode_area_km2", "geography_source"]
    ).merge(lookup, left_on="postcode", right_on="POA_CODE21", how="left")
    enriched = enriched.drop(columns=["POA_CODE21"])
    enriched["geography_source"] = f"ABS POA 2021 representative_point ({shapefile.name})"

    missing = int(enriched.centroid_lat.isna().sum())
    if missing:
        print(f"WARNING: {missing} sites have no POA-2021 polygon for their postcode.")

    if write:
        _write_single(con, enriched, "se_site")
    return enriched


def bom_grid_snap_sql(
    lat_col: str = "centroid_lat",
    lon_col: str = "centroid_lon",
    spacing: float | None = None,
) -> str:
    """
    SQL snapping a coordinate to the nearest BOM satellite grid node.

    UNCONFIRMED, and deliberately parameterised. ``BOM_NCI/process_bom.ipynb``
    matches grid points by rounding latitude and longitude to two decimal places,
    which implies a 0.01-0.02 degree grid, but the true node spacing of
    ``bom_nci.solar`` cannot be verified from this repository.

    Confirm it at D12a with::

        SELECT DISTINCT latitude FROM bom_nci.solar ORDER BY latitude LIMIT 20

    then set `se_config.BOM_GRID_SPACING_DEG` accordingly. Until then this exists
    so the mapping is written down and testable, not so it can be trusted.

    Note also that ``process_bom.ipynb`` ultimately AVERAGES all grid points within
    a postcode (``groupby(['time','postcode']).mean()``) rather than picking the
    nearest to a point. For SolarEdge, where the site location within the postcode
    is unknown, that average is arguably the better estimator -- prefer it at D12
    and use this snap only for diagnostics.
    """
    step = spacing if spacing is not None else getattr(C, "BOM_GRID_SPACING_DEG", 0.02)
    return (
        f"round({lat_col} / {step}) * {step} AS bom_grid_lat, "
        f"round({lon_col} / {step}) * {step} AS bom_grid_lon"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY AND CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def site_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Fleet composition by state and phase configuration."""
    return con.execute(
        """
        SELECT s.state,
               count(*)                                        AS n_sites,
               count(*) FILTER (WHERE NOT s.is_three_phase)     AS n_single_phase,
               count(*) FILTER (WHERE s.is_three_phase)         AS n_three_phase,
               count(DISTINCT s.postcode)                       AS n_postcodes,
               round(median(c.s_99), 2)                         AS median_s_99_kVA,
               round(median(c.p_99), 2)                         AS median_p_99_kW,
               round(median(s.pct_days_covered), 1)             AS median_pct_days,
               count(*) FILTER (WHERE s.has_night_generation_anomaly)
                                                                AS n_night_anomaly
        FROM se_site s
        LEFT JOIN se_site_capacity c USING (site_alias)
        GROUP BY s.state
        ORDER BY n_sites DESC
        """
    ).df()


def check_site_dimension(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """D4 acceptance checks."""
    site = con.execute(
        """
        SELECT count(*) n, count(DISTINCT site_alias) n_unique,
               count(*) FILTER (WHERE tz_name IS NULL) n_null_tz,
               count(*) FILTER (WHERE postcode IS NULL) n_null_pc,
               count(*) FILTER (WHERE n_phases NOT IN (1, 3)) n_odd_phases,
               count(*) FILTER (WHERE centroid_lat IS NOT NULL) n_with_geo
        FROM se_site
        """
    ).df().iloc[0]

    cap = con.execute(
        """
        SELECT count(*) n, count(DISTINCT site_alias) n_unique,
               count(*) FILTER (WHERE s_99 IS NULL OR s_99 <= 0) n_bad_s99,
               count(*) FILTER (WHERE s_99 < p_99 - 0.05) n_s99_below_p99,
               round(coalesce(max(p_99 - s_99), 0), 4) max_s99_p99_gap,
               round(median(s_99), 3) median_s_99
        FROM se_site_capacity
        """
    ).df().iloc[0]

    # Sites with no positive apparent power anywhere have no capacity to estimate
    # and are legitimately absent from se_site_capacity. Two such sites exist in
    # this delivery, each with a single all-zero interval.
    estimable = int(
        con.execute(
            """
            SELECT count(DISTINCT site_alias) FROM se_interval
            WHERE sqrt(P_kW * P_kW + coalesce(Q_kvar, 0) * coalesce(Q_kvar, 0)) > 0
            """
        ).fetchone()[0]
    )

    def row(check, expected, observed, passed, note=""):
        return {"check": check, "expected": expected, "observed": observed,
                "pass": bool(passed), "note": note}

    return pd.DataFrame([
        row("se_site has one row per site", f"{C.EXPECTED_N_SITES:,}",
            f"{int(site.n):,}", int(site.n) == C.EXPECTED_N_SITES),
        row("site_alias unique", f"{int(site.n):,}", f"{int(site.n_unique):,}",
            int(site.n) == int(site.n_unique)),
        row("every site has a timezone", "0 nulls", f"{int(site.n_null_tz)}",
            int(site.n_null_tz) == 0),
        row("every site has a postcode", "0 nulls", f"{int(site.n_null_pc)}",
            int(site.n_null_pc) == 0),
        row("phase count is 1 or 3", "0 others", f"{int(site.n_odd_phases)}",
            int(site.n_odd_phases) == 0,
            "2-phase would mean a dropped phase, not a real configuration"),
        row("se_site_capacity covers every estimable site", f"{estimable:,}",
            f"{int(cap.n):,}", int(cap.n) == estimable,
            f"{int(site.n) - estimable} site(s) have no interval with S > 0, so there "
            "is no capacity to estimate -- legitimately absent, not missing"),
        row("s_99 positive everywhere", "0 bad", f"{int(cap.n_bad_s99)}",
            int(cap.n_bad_s99) == 0),
        row("s_99 >= p_99 within tolerance", "0 violations > 0.05 kVA",
            f"{int(cap.n_s99_below_p99)} (max gap {cap.max_s99_p99_gap} kVA)",
            int(cap.n_s99_below_p99) == 0,
            "S = sqrt(P^2+Q^2) >= P exactly, but s_99 and p_99 come from separate "
            "T-Digest sketches, so they can cross by a hair where Q is small. "
            "Observed max gap 0.018 kVA (0.25%) -- approximation noise, not a data fault"),
        row("geography attached (optional)", "1,602 or 0 (pending shapefile)",
            f"{int(site.n_with_geo):,}", True,
            "D12 irradiance needs this; D5/D6 do not"),
    ])
