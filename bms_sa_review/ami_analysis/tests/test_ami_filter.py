"""Unit tests for `ami_filter` -- residential-scope filtering and DNSP/OEM key remapping."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bms_sa_review.ami_analysis.lib import ami_filter as Filter


# --------------------------------------------------------------------------- #
# flag_oversized_capacity
# --------------------------------------------------------------------------- #

def test_flag_oversized_capacity_flags_above_threshold():
    frame = pd.DataFrame({"ac_capacity_kw": [5.0, 30.0, 30.001, 45.0]})
    flag = Filter.flag_oversized_capacity(frame)
    assert list(flag) == [False, False, True, True]


def test_flag_oversized_capacity_does_not_flag_null():
    # No PV capacity on record (e.g. a load-only site) is not evidence the
    # site is oversized -- must not be flagged for lack of the very
    # metadata the filter depends on.
    frame = pd.DataFrame({"ac_capacity_kw": [5.0, np.nan]})
    flag = Filter.flag_oversized_capacity(frame)
    assert list(flag) == [False, False]


def test_flag_oversized_capacity_missing_column_returns_all_false():
    frame = pd.DataFrame({"other_col": [1, 2]})
    assert list(Filter.flag_oversized_capacity(frame)) == [False, False]


def test_flag_oversized_capacity_custom_threshold_and_column():
    frame = pd.DataFrame({"cap": [10.0, 20.0]})
    flag = Filter.flag_oversized_capacity(frame, capacity_column="cap", max_kw=15.0)
    assert list(flag) == [False, True]


# --------------------------------------------------------------------------- #
# build_key_mapping
# --------------------------------------------------------------------------- #

def test_build_key_mapping_builds_dict():
    mapping_frame = pd.DataFrame({
        "dnsp": ["Ausgrid", "SAPN", "None"],
        "dnsp_id": ["DNSP_8", "DNSP_5", "DNSP_17"],
    })
    result = Filter.build_key_mapping(mapping_frame, key_column="dnsp", value_column="dnsp_id")
    assert result == {"Ausgrid": "DNSP_8", "SAPN": "DNSP_5", "None": "DNSP_17"}


def test_build_key_mapping_drops_null_keys():
    mapping_frame = pd.DataFrame({"dnsp": ["Ausgrid", None], "dnsp_id": ["DNSP_8", "DNSP_99"]})
    result = Filter.build_key_mapping(mapping_frame, key_column="dnsp", value_column="dnsp_id")
    assert result == {"Ausgrid": "DNSP_8"}


def test_build_key_mapping_raises_on_ambiguous_key():
    mapping_frame = pd.DataFrame({
        "dnsp": ["Ausgrid", "Ausgrid"],
        "dnsp_id": ["DNSP_8", "DNSP_9"],
    })
    with pytest.raises(ValueError, match="Ambiguous"):
        Filter.build_key_mapping(mapping_frame, key_column="dnsp", value_column="dnsp_id")


def test_build_key_mapping_same_key_same_value_twice_is_not_ambiguous():
    mapping_frame = pd.DataFrame({
        "dnsp": ["Ausgrid", "Ausgrid"],
        "dnsp_id": ["DNSP_8", "DNSP_8"],
    })
    result = Filter.build_key_mapping(mapping_frame, key_column="dnsp", value_column="dnsp_id")
    assert result == {"Ausgrid": "DNSP_8"}


# --------------------------------------------------------------------------- #
# apply_key_mapping
# --------------------------------------------------------------------------- #

def test_apply_key_mapping_maps_known_values():
    series = pd.Series(["Ausgrid", "SAPN"])
    mapped, unmapped = Filter.apply_key_mapping(series, {"Ausgrid": "DNSP_8", "SAPN": "DNSP_5"})
    assert list(mapped) == ["DNSP_8", "DNSP_5"]
    assert unmapped == []


def test_apply_key_mapping_coalesces_null_to_null_key_before_lookup():
    series = pd.Series(["Ausgrid", None, np.nan])
    mapped, unmapped = Filter.apply_key_mapping(
        series, {"Ausgrid": "DNSP_8", "None": "DNSP_17"},
    )
    assert list(mapped) == ["DNSP_8", "DNSP_17", "DNSP_17"]
    assert unmapped == []


def test_apply_key_mapping_keeps_original_value_when_unmapped():
    # A real value with no entry in the mapping file (e.g. a casing
    # mismatch, or a manufacturer the CSV hasn't caught up to) must keep
    # its original value, not become NaN, and must be reported.
    series = pd.Series(["Ausgrid", "Some New DNSP"])
    mapped, unmapped = Filter.apply_key_mapping(series, {"Ausgrid": "DNSP_8"})
    assert list(mapped) == ["DNSP_8", "Some New DNSP"]
    assert unmapped == ["Some New DNSP"]


def test_apply_key_mapping_unmapped_values_are_deduplicated_and_sorted():
    series = pd.Series(["Zeta Corp", "Alpha Corp", "Zeta Corp"])
    mapped, unmapped = Filter.apply_key_mapping(series, {})
    assert unmapped == ["Alpha Corp", "Zeta Corp"]
