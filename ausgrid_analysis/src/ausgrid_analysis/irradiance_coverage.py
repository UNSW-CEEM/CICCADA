"""BOM/NCI coverage queries and nearest-grid diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import FoundationConfig
from .db import connect, prepare_output_file
from .schemas import sql_string


@dataclass(frozen=True)
class GeographicBounds:
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float

    @classmethod
    def from_sites(
        cls,
        sites: pd.DataFrame,
        *,
        padding_degrees: float = 0.1,
    ) -> GeographicBounds:
        return cls(
            float(sites["sub_lat"].min()) - padding_degrees,
            float(sites["sub_lat"].max()) + padding_degrees,
            float(sites["sub_long"].min()) - padding_degrees,
            float(sites["sub_long"].max()) + padding_degrees,
        )

    def predicate(self) -> str:
        return (
            f"latitude BETWEEN {self.minimum_latitude:.8f} "
            f"AND {self.maximum_latitude:.8f} "
            f"AND longitude BETWEEN {self.minimum_longitude:.8f} "
            f"AND {self.maximum_longitude:.8f}"
        )


def _month_predicate(months: Sequence[tuple[int, int]]) -> str:
    if not months:
        raise ValueError("At least one (year, month) pair is required")
    return " OR ".join(
        f"(year = {int(year)} AND month = {int(month)})"
        for year, month in months
    )


def bom_inventory_sql(
    bounds: GeographicBounds,
    months: Sequence[tuple[int, int]],
) -> str:
    """Return a partition-filtered Athena query with no raw-row download."""

    return f"""
        SELECT
            year,
            month,
            count(*) AS n_rows,
            count(DISTINCT concat(
                cast(latitude AS varchar), '|', cast(longitude AS varchar)
            )) AS n_grid_points,
            min(time) AS first_time,
            max(time) AS last_time,
            count_if(surface_global_irradiance IS NULL) AS null_ghi,
            count_if(quality_mask IS NULL) AS null_quality_mask
        FROM solar
        WHERE ({_month_predicate(months)})
          AND {bounds.predicate()}
        GROUP BY year, month
        ORDER BY year, month
    """.strip()


def bom_locations_sql(
    bounds: GeographicBounds,
    months: Sequence[tuple[int, int]],
) -> str:
    return f"""
        SELECT DISTINCT
            latitude,
            longitude,
            postcode
        FROM solar
        WHERE ({_month_predicate(months)})
          AND {bounds.predicate()}
        ORDER BY latitude, longitude, postcode
    """.strip()


def _haversine_km(
    latitude: float,
    longitude: float,
    grid_latitudes: np.ndarray,
    grid_longitudes: np.ndarray,
) -> np.ndarray:
    earth_radius_km = 6371.0088
    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)
    lat2 = np.radians(grid_latitudes)
    lon2 = np.radians(grid_longitudes)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def nearest_grid_mapping(
    sites: pd.DataFrame,
    grid_locations: pd.DataFrame,
    *,
    good_distance_km: float = 5.0,
    review_distance_km: float = 20.0,
) -> pd.DataFrame:
    """Map site/substation coordinates to the nearest available BOM grid point."""

    if grid_locations.empty:
        raise ValueError("No BOM grid locations were supplied")
    required_site = {"serial", "sub_lat", "sub_long"}
    required_grid = {"latitude", "longitude"}
    if not required_site.issubset(sites.columns):
        raise ValueError(f"Sites need columns: {sorted(required_site)}")
    if not required_grid.issubset(grid_locations.columns):
        raise ValueError(f"Grid locations need columns: {sorted(required_grid)}")

    grid = grid_locations.drop_duplicates(["latitude", "longitude"]).reset_index(drop=True)
    grid_lat = grid["latitude"].astype(float).to_numpy()
    grid_lon = grid["longitude"].astype(float).to_numpy()
    rows: list[dict[str, object]] = []
    for site in sites.itertuples(index=False):
        distances = _haversine_km(
            float(site.sub_lat), float(site.sub_long), grid_lat, grid_lon
        )
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance <= good_distance_km:
            quality = "good"
        elif distance <= review_distance_km:
            quality = "review"
        else:
            quality = "poor"
        rows.append(
            {
                "serial": str(site.serial),
                "site_latitude_source": "substation_metadata",
                "site_latitude": float(site.sub_lat),
                "site_longitude": float(site.sub_long),
                "bom_latitude": float(grid.iloc[index]["latitude"]),
                "bom_longitude": float(grid.iloc[index]["longitude"]),
                "distance_km": distance,
                "spatial_mapping_quality": quality,
            }
        )
    return pd.DataFrame(rows).sort_values("serial").reset_index(drop=True)


def _point_predicate(points: Iterable[tuple[float, float]]) -> str:
    clauses = [
        f"(latitude = {float(lat):.8f} AND longitude = {float(lon):.8f})"
        for lat, lon in points
    ]
    if not clauses:
        raise ValueError("At least one grid point is required")
    return " OR ".join(clauses)


def bom_mapped_coverage_sql(
    points: Sequence[tuple[float, float]],
    months: Sequence[tuple[int, int]],
) -> str:
    """Return monthly completeness diagnostics for a small point chunk."""

    return f"""
        SELECT
            latitude,
            longitude,
            year,
            month,
            count(*) AS n_rows,
            count(DISTINCT time) AS n_timestamps,
            min(time) AS first_time,
            max(time) AS last_time,
            count_if(surface_global_irradiance IS NULL) AS null_ghi,
            count_if(quality_mask IS NULL) AS null_quality_mask,
            min(quality_mask) AS minimum_quality_mask,
            max(quality_mask) AS maximum_quality_mask,
            approx_distinct(quality_mask) AS n_quality_mask_values,
            count_if(quality_mask = 1) AS quality_mask_1_rows,
            count_if(surface_global_irradiance IS NOT NULL) AS usable_ghi_rows
        FROM solar
        WHERE ({_month_predicate(months)})
          AND ({_point_predicate(points)})
        GROUP BY latitude, longitude, year, month
        ORDER BY latitude, longitude, year, month
    """.strip()


def irradiance_location_map_path(config: FoundationConfig) -> Path:
    return config.paths.derived_root / "irradiance" / "site_to_bom_grid.parquet"


def irradiance_coverage_path(config: FoundationConfig) -> Path:
    return config.paths.derived_root / "irradiance" / "bom_monthly_coverage.parquet"


def write_frame_parquet(
    config: FoundationConfig,
    frame: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = prepare_output_file(config, path, overwrite=overwrite)
    connection = connect(config)
    try:
        connection.register("_output_frame", frame)
        connection.execute(
            f"""COPY (SELECT * FROM _output_frame)
            TO {sql_string(output)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )
        connection.unregister("_output_frame")
    finally:
        connection.close()
    return output
