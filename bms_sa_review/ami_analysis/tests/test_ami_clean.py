"""Unit tests for `ami_clean` -- the `ami_meter` quality-check/cleaning module."""
from __future__ import annotations

import pandas as pd
import pytest

from bms_sa_review.ami_analysis.lib import ami_clean as Clean


def _base_frame(n=4):
    return pd.DataFrame({
        "site_id": [1] * n,
        "circuit_id": [10] * n,
        "t_stamp": pd.date_range("2025-06-01", periods=n, freq="5min"),
        "V": [240.0] * n,
        "P_kw": [1.0] * n,
        "Q_kvar": [0.1] * n,
        "S_kva": [1.0] * n,
        "power_factor": [0.99] * n,
        "current_a": [4.2] * n,
    })


# --------------------------------------------------------------------------- #
# flag_voltage_outliers
# --------------------------------------------------------------------------- #

def test_flag_voltage_outliers_flags_zero_and_negative():
    frame = _base_frame(3)
    frame.loc[1, "V"] = 0.0
    frame.loc[2, "V"] = -5.0
    flag = Clean.flag_voltage_outliers(frame)
    assert list(flag) == [False, True, True]


def test_flag_voltage_outliers_flags_above_hard_max_but_not_within_envelope():
    frame = _base_frame(3)
    frame.loc[1, "V"] = 265.0   # extreme but within AS/NZS 4777.2-relevant envelope -- NOT flagged
    frame.loc[2, "V"] = 305.0   # beyond the hard 300V cutoff -- flagged
    flag = Clean.flag_voltage_outliers(frame)
    assert list(flag) == [False, False, True]


def test_flag_voltage_outliers_flags_null():
    frame = _base_frame(2)
    frame.loc[1, "V"] = None
    assert list(Clean.flag_voltage_outliers(frame)) == [False, True]


# --------------------------------------------------------------------------- #
# flag_power_factor_outliers
# --------------------------------------------------------------------------- #

def test_flag_power_factor_outliers_flags_above_one():
    frame = _base_frame(3)
    frame.loc[1, "power_factor"] = 1.5
    frame.loc[2, "power_factor"] = -1.2
    assert list(Clean.flag_power_factor_outliers(frame)) == [False, True, True]


def test_flag_power_factor_outliers_missing_column_returns_all_false():
    frame = _base_frame(2).drop(columns=["power_factor"])
    assert list(Clean.flag_power_factor_outliers(frame)) == [False, False]


# --------------------------------------------------------------------------- #
# flag_negative_current
# --------------------------------------------------------------------------- #

def test_flag_negative_current():
    frame = _base_frame(2)
    frame.loc[1, "current_a"] = -1.0
    assert list(Clean.flag_negative_current(frame)) == [False, True]


# --------------------------------------------------------------------------- #
# flag_apparent_power_inconsistency
# --------------------------------------------------------------------------- #

def test_flag_apparent_power_inconsistency_flags_real_mismatch():
    frame = _base_frame(2)
    # row 1: V*I/1000 = 240*10/1000 = 2.4 kVA, but S_kva stored as 1.0 -- big mismatch
    frame.loc[1, "current_a"] = 10.0
    frame.loc[1, "S_kva"] = 1.0
    flag = Clean.flag_apparent_power_inconsistency(frame)
    assert list(flag) == [False, True]


def test_flag_apparent_power_inconsistency_consistent_row_not_flagged():
    frame = _base_frame(1)
    frame.loc[0, "current_a"] = 4.1667  # 240 * 4.1667 / 1000 ~= 1.0 kVA, matches S_kva
    frame.loc[0, "S_kva"] = 1.0
    assert list(Clean.flag_apparent_power_inconsistency(frame)) == [False]


def test_flag_apparent_power_inconsistency_skips_low_current_rows():
    frame = _base_frame(1)
    frame.loc[0, "current_a"] = 0.05   # below min_current_a -- not evaluated even though S_kva=1.0
    frame.loc[0, "S_kva"] = 1.0
    assert list(Clean.flag_apparent_power_inconsistency(frame)) == [False]


def test_flag_apparent_power_inconsistency_does_not_flag_low_power_quantization_noise():
    # V*I quantization at low, but not skipped, current: a small ABSOLUTE
    # mismatch that would blow up into a huge RELATIVE one if S_kva were
    # used alone as the denominator -- must NOT flag on relative error alone
    # (regression test for the real-data bug found 2026-09: a relative-only
    # version of this check flagged ~22% of a real month almost entirely on
    # near-zero-power rows like this one).
    frame = _base_frame(1)
    frame.loc[0, "current_a"] = 1.2          # implied kVA = 240*1.2/1000 = 0.288
    frame.loc[0, "S_kva"] = 0.20              # ~44% relative error, but only 0.088 kVA absolute
    assert list(Clean.flag_apparent_power_inconsistency(frame)) == [False]


def test_flag_apparent_power_inconsistency_flags_when_both_absolute_and_relative_are_large():
    frame = _base_frame(1)
    frame.loc[0, "current_a"] = 20.0         # implied kVA = 240*20/1000 = 4.8
    frame.loc[0, "S_kva"] = 1.0               # 3.8 kVA absolute, 79% relative -- both large
    assert list(Clean.flag_apparent_power_inconsistency(frame)) == [True]


# --------------------------------------------------------------------------- #
# flag_extreme_power_magnitude
# --------------------------------------------------------------------------- #

def test_flag_extreme_power_magnitude_flags_hard_ceiling():
    frame = _base_frame(2)
    frame.loc[1, "P_kw"] = 150.0
    flag = Clean.flag_extreme_power_magnitude(frame)
    assert list(flag) == [False, True]


def test_flag_extreme_power_magnitude_flags_per_circuit_robust_outlier():
    frame = pd.DataFrame({
        "circuit_id": [10] * 10,
        "P_kw": [1.0] * 9 + [40.0],   # last row far from this circuit's own median
    })
    flag = Clean.flag_extreme_power_magnitude(frame)
    assert list(flag) == [False] * 9 + [True]


def test_flag_extreme_power_magnitude_different_circuits_dont_cross_contaminate():
    frame = pd.DataFrame({
        "circuit_id": [10] * 5 + [20] * 5,
        # circuit 20 normally runs much higher -- its own values should NOT
        # flag just because they're far from circuit 10's median.
        "P_kw": [1.0] * 5 + [20.0] * 5,
    })
    flag = Clean.flag_extreme_power_magnitude(frame)
    assert not flag.any()


# --------------------------------------------------------------------------- #
# flag_duplicate_readings
# --------------------------------------------------------------------------- #

def test_flag_duplicate_readings_flags_repeats_keeping_first():
    frame = _base_frame(3)
    frame.loc[2, "t_stamp"] = frame.loc[0, "t_stamp"]  # duplicate of row 0's key
    flag = Clean.flag_duplicate_readings(frame)
    assert list(flag) == [False, False, True]


def test_flag_duplicate_readings_missing_key_column_returns_all_false():
    frame = _base_frame(2).drop(columns=["circuit_id"])
    assert list(Clean.flag_duplicate_readings(frame)) == [False, False]


# --------------------------------------------------------------------------- #
# flag_missing_critical_fields
# --------------------------------------------------------------------------- #

def test_flag_missing_critical_fields():
    frame = _base_frame(3)
    frame.loc[1, "V"] = None
    frame.loc[2, "P_kw"] = None
    flag = Clean.flag_missing_critical_fields(frame)
    assert list(flag) == [False, True, True]


# --------------------------------------------------------------------------- #
# build_quality_report
# --------------------------------------------------------------------------- #

def test_build_quality_report_counts_and_classifies():
    frame = _base_frame(4)
    flags = {
        "voltage_implausible": pd.Series([True, False, False, False]),
        "power_magnitude_extreme": pd.Series([False, True, False, False]),
        "something_else": pd.Series([False, False, False, True]),
    }
    report = Clean.build_quality_report(frame, flags).set_index("flag")
    assert report.loc["voltage_implausible", "kind"] == "hard"
    assert report.loc["voltage_implausible", "n_flagged"] == 1
    assert report.loc["voltage_implausible", "share_flagged"] == pytest.approx(0.25)
    assert report.loc["power_magnitude_extreme", "kind"] == "soft"
    assert report.loc["something_else", "kind"] == "unclassified"


# --------------------------------------------------------------------------- #
# apply_cleaning
# --------------------------------------------------------------------------- #

def test_apply_cleaning_drops_hard_flagged_rows_only():
    frame = _base_frame(4)
    flags = {
        "voltage_implausible": pd.Series([True, False, False, False]),
        "current_negative": pd.Series([False, False, True, False]),
    }
    clean, removed = Clean.apply_cleaning(frame, flags)
    assert len(clean) == 2
    assert len(removed) == 2
    assert set(removed.index.tolist()) == {0, 1}  # reset_index -- positions 0,1 in `removed`


def test_apply_cleaning_keeps_soft_flagged_rows_and_tags_them():
    frame = _base_frame(3)
    flags = {
        "power_magnitude_extreme": pd.Series([False, True, False]),
    }
    clean, removed = Clean.apply_cleaning(frame, flags)
    assert len(clean) == 3
    assert len(removed) == 0
    assert list(clean["power_magnitude_extreme"]) == [False, True, False]


def test_apply_cleaning_no_flags_keeps_everything():
    frame = _base_frame(3)
    clean, removed = Clean.apply_cleaning(frame, {})
    assert len(clean) == 3
    assert len(removed) == 0


def test_apply_cleaning_row_flagged_hard_and_soft_is_dropped_not_tagged():
    frame = _base_frame(2)
    flags = {
        "voltage_implausible": pd.Series([True, False]),
        "power_magnitude_extreme": pd.Series([True, False]),
    }
    clean, removed = Clean.apply_cleaning(frame, flags)
    assert len(clean) == 1
    assert len(removed) == 1
    assert list(clean["power_magnitude_extreme"]) == [False]
