"""
BOM satellite irradiance: probe, map and extract.
=================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oem_analysis.config import se_config as C

__all__ = [
    "get_aq",
    "describe_bom",
    "probe_one_day",
    "probe_coverage",
    "probe_grid",
    "postcode_grid_points",
    "thin_mapping",
    "fit_mapping_to_athena",
    "extract_bom",
    "BOM_DATABASE",
    "BOM_TABLE",
]

BOM_DATABASE = "bom_nci"
BOM_TABLE = "bom_nci.solar"


def get_aq():
    """
    Return the project's Athena query helper.
    """
    C.bootstrap_sys_path()
    try:
        from bms_sa_review.shared.aws_config import aq
    except Exception as exc:  # pragma: no cover - depends on local AWS state
        raise RuntimeError(
            "Could not load the Athena helper from bms_sa_review.shared.aws_config.\n"
            "  1. aws sso login --profile ciccada\n"
            "  2. check awswrangler and boto3 are installed in the ciccada env\n"
            f"  original error: {exc}"
        ) from exc
    return aq


# ═══════════════════════════════════════════════════════════════════════════
# PROBES
# ═══════════════════════════════════════════════════════════════════════════
# Bounding box covering the OEM fleet (NSW, SA, QLD) with margin.
FLEET_BOX = dict(lat_min=-39.5, lat_max=-9.5, lon_min=129.0, lon_max=154.5)


def describe_bom(aq=None) -> pd.DataFrame:
    """
    Column names, types and PARTITION columns of ``bom_nci.solar``.
    """
    aq = aq or get_aq()
    return aq(f"DESCRIBE {BOM_TABLE}", database=BOM_DATABASE)


def probe_one_day(aq=None, day: str = "2025-06-15", **box) -> pd.DataFrame:
    """
    The cheapest possible sanity check: ONE day, fleet box only.
    """
    aq = aq or get_aq()
    b = {**FLEET_BOX, **box}
    return aq(
        f"""
        SELECT count(*)                                AS n_rows,
               count(DISTINCT (latitude, longitude))   AS n_grid_points,
               count(DISTINCT time)                    AS n_times,
               min(time) AS first_time, max(time) AS last_time,
               avg(surface_global_irradiance)          AS mean_ghi
        FROM {BOM_TABLE}
        WHERE time >= TIMESTAMP '{day} 00:00:00'
          AND time <  TIMESTAMP '{day} 00:00:00' + INTERVAL '1' DAY
          AND latitude BETWEEN {b['lat_min']} AND {b['lat_max']}
          AND longitude BETWEEN {b['lon_min']} AND {b['lon_max']}
        """,
        database=BOM_DATABASE,
    )


def probe_coverage(aq=None, year: int = C.STUDY_YEAR, months=None, **box) -> pd.DataFrame:
    aq = aq or get_aq()
    b = {**FLEET_BOX, **box}
    months = list(months) if months is not None else [6]
    month_list = ", ".join(str(int(m)) for m in months)

    return aq(
        f"""
        SELECT year(time)                                   AS year,
               month(time)                                  AS month,
               count(*)                                     AS n_rows,
               count(DISTINCT time)                         AS n_times,
               count(DISTINCT (latitude, longitude))        AS n_grid_points,
               min(time)                                    AS first_time,
               max(time)                                    AS last_time
        FROM {BOM_TABLE}
        WHERE time >= TIMESTAMP '{year}-01-01 00:00:00'
          AND time <  TIMESTAMP '{year + 1}-01-01 00:00:00'
          AND month(time) IN ({month_list})
          AND latitude BETWEEN {b['lat_min']} AND {b['lat_max']}
          AND longitude BETWEEN {b['lon_min']} AND {b['lon_max']}
        GROUP BY year(time), month(time)
        ORDER BY month
        """,
        database=BOM_DATABASE,
    )


def probe_grid(
    aq=None,
    lat_min: float = -45.0, lat_max: float = -9.0,
    lon_min: float = 112.0, lon_max: float = 154.0,
    year: int = C.STUDY_YEAR,
) -> pd.DataFrame:
    """
    How many grid nodes sit in the fleet's bounding box, and what is the node
    spacing?
    """
    aq = aq or get_aq()
    return aq(
        f"""
        SELECT count(DISTINCT (latitude, longitude))  AS n_grid_points,
               count(DISTINCT latitude)               AS n_distinct_lat,
               count(DISTINCT longitude)              AS n_distinct_lon,
               min(latitude)  AS lat_min, max(latitude)  AS lat_max,
               min(longitude) AS lon_min, max(longitude) AS lon_max
        FROM {BOM_TABLE}
        -- Explicit range on `time`, not `year(time)`: a function on the column
        -- blocks Athena partition pruning. Restricted to ONE month by default --
        -- the grid does not change month to month, so a year-wide scan buys
        -- nothing and costs twelve times as much.
        WHERE time >= TIMESTAMP '{year}-06-01 00:00:00'
          AND time <  TIMESTAMP '{year}-07-01 00:00:00'
          AND latitude BETWEEN {lat_min} AND {lat_max}
          AND longitude BETWEEN {lon_min} AND {lon_max}
        """,
        database=BOM_DATABASE,
    )


def probe_spacing(aq=None, year: int = C.STUDY_YEAR, n: int = 30) -> pd.DataFrame:
    """
    The first ``n`` distinct latitudes, so the node spacing can be read directly.
    Set ``se_config.BOM_GRID_SPACING_DEG`` from the result.
    """
    aq = aq or get_aq()
    return aq(
        f"""
        SELECT DISTINCT latitude FROM {BOM_TABLE}
        WHERE time >= TIMESTAMP '{year}-06-15 00:00:00'
          AND time <  TIMESTAMP '{year}-06-16 00:00:00'
          AND latitude BETWEEN -38 AND -33
        ORDER BY latitude LIMIT {n}
        """,
        database=BOM_DATABASE,
    )


# ═══════════════════════════════════════════════════════════════════════════
# POSTCODE -> GRID POINTS
# ═══════════════════════════════════════════════════════════════════════════

def postcode_grid_points(
    con,
    grid: pd.DataFrame,
    shapefile: Path | str | None = None,
) -> pd.DataFrame:
    """
    Map every fleet postcode to the BOM grid nodes inside its polygon.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    shapefile = Path(shapefile or C.POA_SHAPEFILE)
    if not shapefile.exists():
        raise FileNotFoundError(f"POA-2021 shapefile not found: {shapefile}")

    postcodes = con.execute("SELECT DISTINCT postcode FROM se_site").df().postcode
    poa = gpd.read_file(shapefile)[["POA_CODE21", "geometry"]].to_crs("EPSG:4326")
    poa = poa[poa.POA_CODE21.isin(set(postcodes))].copy()

    nodes = gpd.GeoDataFrame(
        grid.copy(),
        geometry=[Point(x, y) for x, y in zip(grid.longitude, grid.latitude)],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(nodes, poa, how="inner", predicate="within")
    within = joined[["POA_CODE21", "latitude", "longitude"]].copy()
    within["match_type"] = "within"

    covered = set(within.POA_CODE21)
    missing = poa[~poa.POA_CODE21.isin(covered)]

    rows = [within]
    if len(missing):
        # Project to metres for a meaningful nearest-neighbour distance.
        nodes_m = nodes.to_crs("EPSG:3577")
        pts = missing.copy()
        pts["geometry"] = pts.geometry.representative_point()
        nearest = gpd.sjoin_nearest(
            pts.to_crs("EPSG:3577"), nodes_m, how="left", distance_col="dist_m"
        )
        fallback = nearest[["POA_CODE21", "latitude", "longitude"]].copy()
        fallback["match_type"] = "nearest"
        rows.append(fallback)
        print(
            f"{len(missing)} postcode(s) contain no grid node and were matched to the "
            f"nearest one (median {nearest.dist_m.median() / 1000:.1f} km away)."
        )

    out = pd.concat(rows, ignore_index=True).rename(columns={"POA_CODE21": "postcode"})
    return out.drop_duplicates()


def fetch_grid_points(aq=None, con=None, buffer_deg: float = 0.5,
                      year: int = C.STUDY_YEAR) -> pd.DataFrame:
    """
    Distinct BOM grid nodes within the fleet's bounding box.
    """
    aq = aq or get_aq()
    box = (-39.5, -9.5, 129.0, 154.5)  # lat_min, lat_max, lon_min, lon_max
    if con is not None:
        got = con.execute(
            "SELECT min(centroid_lat), max(centroid_lat), "
            "min(centroid_lon), max(centroid_lon) FROM se_site"
        ).fetchone()
        if all(v is not None for v in got):
            box = (got[0] - buffer_deg, got[1] + buffer_deg,
                   got[2] - buffer_deg, got[3] + buffer_deg)

    return aq(
        f"""
        SELECT DISTINCT latitude, longitude
        FROM {BOM_TABLE}
        WHERE year(time) = {year}
          AND latitude BETWEEN {box[0]} AND {box[1]}
          AND longitude BETWEEN {box[2]} AND {box[3]}
        """,
        database=BOM_DATABASE,
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACT
# ═══════════════════════════════════════════════════════════════════════════

#: Athena rejects a StartQueryExecution whose queryString exceeds 262,144
#: characters. The inline VALUES mapping is by far the largest part of the
#: extract SQL, so the node list has to be kept under a budget.
MAX_ATHENA_SQL = 262_144
_SQL_BUDGET = 200_000        # leaves ~60 KB for the rest of the statement


def _postcode_values_sql(mapping: pd.DataFrame) -> str:
    """Render the postcode <-> node mapping as an inline Athena VALUES list."""
    rows = ", ".join(
        f"({r.latitude!r}, {r.longitude!r}, '{r.postcode}')"
        for r in mapping.itertuples()
    )
    return f"(VALUES {rows}) AS g (lat, lon, postcode)"


def thin_mapping(mapping: pd.DataFrame, max_per_postcode: int) -> pd.DataFrame:
    """
    Keep only the ``max_per_postcode`` nodes closest to each postcode's centre.
    """
    frame = mapping[["latitude", "longitude", "postcode"]].drop_duplicates().copy()
    centres = frame.groupby("postcode")[["latitude", "longitude"]].transform("mean")
    # Squared degrees is monotone in true distance at this scale; cos(lat)
    # weighting on longitude keeps east-west and north-south comparable.
    import numpy as np

    coslat = np.cos(np.radians(centres.latitude))
    frame["_d2"] = ((frame.latitude - centres.latitude) ** 2
                    + ((frame.longitude - centres.longitude) * coslat) ** 2)
    kept = (frame.sort_values(["postcode", "_d2"])
                 .groupby("postcode", as_index=False, group_keys=False)
                 .head(max_per_postcode)
                 .drop(columns="_d2")
                 .reset_index(drop=True))
    return kept


def fit_mapping_to_athena(mapping: pd.DataFrame, budget: int = _SQL_BUDGET) -> pd.DataFrame:
    """
    Thin the mapping until its inline VALUES clause fits Athena's query limit.
    """
    nodes = mapping[["latitude", "longitude", "postcode"]].drop_duplicates()
    size = len(_postcode_values_sql(nodes))
    if size <= budget:
        return nodes.reset_index(drop=True)

    for cap in (64, 32, 16, 12, 8, 6, 4, 3, 2, 1):
        thinned = thin_mapping(nodes, cap)
        new_size = len(_postcode_values_sql(thinned))
        if new_size <= budget:
            print(
                f"Mapping thinned to <= {cap} node(s) per postcode: "
                f"{len(nodes):,} -> {len(thinned):,} nodes "
                f"({size / 1024:.0f} KB -> {new_size / 1024:.0f} KB of SQL).\n"
                f"  Athena caps a query at {MAX_ATHENA_SQL:,} characters; the full "
                f"list does not fit.\n"
                f"  Nodes kept are those closest to each postcode's centre, so "
                f"large rural postcodes lose their far-flung nodes and dense urban "
                f"ones are unchanged."
            )
            return thinned.reset_index(drop=True)

    raise ValueError(
        f"Even one node per postcode renders to more than {budget:,} characters "
        f"({nodes.postcode.nunique():,} postcodes). Extract in postcode batches."
    )


def extract_bom(
    aq=None,
    mapping: pd.DataFrame | None = None,
    year: int = C.STUDY_YEAR,
    months: list[int] | None = None,
    out_path: Path | None = None,
    grid_points: pd.DataFrame | None = None,
    max_nodes_per_postcode: int | None = None,
) -> pd.DataFrame:
    """
    Pull ``bom_nci.solar`` for the fleet's postcodes and land it locally.
    """
    aq = aq or get_aq()
    out_path = Path(out_path or C.store_path("bom_solar"))
    months = months or list(range(1, 13))

    if mapping is None and grid_points is not None:
        raise ValueError(
            "extract_bom now takes `mapping` (postcode <-> node) rather than "
            "`grid_points` (a bare node list).\n"
            "  Run se_bom.postcode_grid_points(con, grid_points) first, then pass "
            "its result as mapping=.\n"
            "  A bare node list has no postcode column, and filtering on its "
            "bounding box scans most of eastern Australia."
        )
    if mapping is None or mapping.empty:
        raise ValueError("mapping is required - run postcode_grid_points() first")

    required = {"latitude", "longitude", "postcode"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"mapping must have columns {sorted(required)}")

    nodes = mapping[["latitude", "longitude", "postcode"]].drop_duplicates()
    if max_nodes_per_postcode is not None:
        nodes = thin_mapping(nodes, max_nodes_per_postcode)
    # Guard, not an optimisation: Athena rejects the whole statement above
    # 262,144 characters, and the failure comes back as an opaque
    # InvalidRequestException from StartQueryExecution.
    nodes = fit_mapping_to_athena(nodes)
    print(f"Extracting {nodes.postcode.nunique():,} postcodes "
          f"from {len(nodes):,} grid nodes, {len(months)} month(s).")
    print(f"Expect roughly {nodes.postcode.nunique() * 4464 / 1e6:.1f} M rows per month.\n")
    values = _postcode_values_sql(nodes)

    frames = []
    for month in months:
        frame = aq(
            f"""
            SELECT b.time,
                   g.postcode,
                   avg(b.surface_global_irradiance) AS GHI,
                   avg(b.cloud_type)                AS cloud_type,
                   count(*)                         AS n_nodes
            FROM {BOM_TABLE} b
            JOIN {values}
              ON b.latitude = g.lat AND b.longitude = g.lon
            WHERE b.time >= TIMESTAMP '{year}-{month:02d}-01 00:00:00'
              AND b.time <  date_add('month', 1, TIMESTAMP '{year}-{month:02d}-01 00:00:00')
              AND b.surface_global_irradiance IS NOT NULL
            GROUP BY b.time, g.postcode
            """,
            database=BOM_DATABASE,
        )
        print(f"  {year}-{month:02d}: {len(frame):,} rows", flush=True)
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {len(out):,} rows to {out_path} "
          f"({out_path.stat().st_size / 1024**2:.0f} MB)")
    return out
