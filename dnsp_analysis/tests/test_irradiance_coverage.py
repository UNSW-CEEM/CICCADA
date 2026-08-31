from __future__ import annotations

import pandas as pd

from ausgrid_analysis.irradiance_coverage import (
    GeographicBounds,
    bom_inventory_sql,
    bom_mapped_coverage_sql,
    nearest_grid_mapping,
)


def test_nearest_grid_mapping_labels_distance_quality() -> None:
    sites = pd.DataFrame(
        [{"serial": "1", "sub_lat": -33.0, "sub_long": 151.0}]
    )
    grid = pd.DataFrame(
        [
            {"latitude": -33.0, "longitude": 151.0, "postcode": 2000},
            {"latitude": -34.0, "longitude": 150.0, "postcode": 2500},
        ]
    )
    result = nearest_grid_mapping(sites, grid).iloc[0]
    assert result["bom_latitude"] == -33.0
    assert result["bom_longitude"] == 151.0
    assert result["distance_km"] == 0
    assert result["spatial_mapping_quality"] == "good"
    assert result["site_latitude_source"] == "substation_metadata"


def test_athena_sql_is_month_partition_filtered() -> None:
    bounds = GeographicBounds(-34, -32, 150, 152)
    inventory = bom_inventory_sql(bounds, [(2024, 8), (2025, 7)])
    coverage = bom_mapped_coverage_sql(
        [(-33.0, 151.0)], [(2024, 8), (2025, 7)]
    )
    for query in (inventory, coverage):
        assert "year = 2024 AND month = 8" in query
        assert "year = 2025 AND month = 7" in query
    assert "latitude = -33.00000000" in coverage

