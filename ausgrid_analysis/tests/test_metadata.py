from __future__ import annotations

import pandas as pd
import pytest

from ausgrid_analysis.metadata import canonicalize_metadata_frame


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unique Number ID": ["810000001", "810000002"],
            "Controlled Load": ["No", "Yes"],
            "DER_Type": ["Solar", "Solar_Battery"],
            "Install Phase": [1, 3],
            "Approved Capacity (kW)": [10, 30],
            "Solar_kW (total capacity)": [6.6, 13.2],
            "Solar Manufacturer": ["Example", "Example"],
            "Solar Model": ["INV-5", "INV-10"],
            "Battery_kWh": [0, 10],
            "Battery Manufacturer": [0, "Example"],
            "Battery Model": [0, "BAT-10"],
            "Battery Inverter Capacity (kW)": [0, 5],
            "Solar Install Year": [2022, 2024],
            "Battery Install Year": [None, 2025],
            "Zone External ID": [1, 2],
            "Sub External ID": [10, 20],
            "Sub Lat": [-33.8, -33.9],
            "Sub Long": [151.1, 151.2],
            "FY26 EVM Test": [0, 1],
        }
    )


def test_metadata_is_normalised_without_inventing_s_rated() -> None:
    result = canonicalize_metadata_frame(_source())
    assert result["serial"].tolist() == ["810000001", "810000002"]
    assert result["analysis_cohort"].tolist() == ["solar_only", "solar_battery"]
    assert result["s_rated_kva"].isna().all()
    assert result["s_rated_source"].eq("unavailable").all()


def test_duplicate_metadata_ids_are_rejected() -> None:
    source = _source()
    source.loc[1, "Unique Number ID"] = source.loc[0, "Unique Number ID"]
    with pytest.raises(ValueError, match="not unique"):
        canonicalize_metadata_frame(source)

