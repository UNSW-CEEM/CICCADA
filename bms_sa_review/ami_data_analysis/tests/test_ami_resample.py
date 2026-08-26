"""
The pure unit-verification and resample logic behind 03_signal_taxonomy: which
source column is power vs energy, verified against physical plausibility
rather than trusted from a docstring, and the resample rule that follows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bms_sa_review.ami_data_analysis.lib import ami_resample as R


# ── interval_energy_kwh ─────────────────────────────────────────────────────

def test_interval_energy_matches_hand_calculation():
    # 1200 W for 5 minutes = 1200/1000 kW * (5/60) h = 0.1 kWh
    result = R.interval_energy_kwh(pd.Series([1200.0]), source_interval_minutes=5)
    assert result.iloc[0] == pytest.approx(0.1)


def test_interval_energy_scales_with_interval_length():
    half_hour = R.interval_energy_kwh(pd.Series([1000.0]), source_interval_minutes=30)
    five_min = R.interval_energy_kwh(pd.Series([1000.0]), source_interval_minutes=5)
    assert half_hour.iloc[0] == pytest.approx(6 * five_min.iloc[0])


# ── verify_column_units ──────────────────────────────────────────────────────

def _units_sample(power_w, reactive_raw, n=200):
    return pd.DataFrame({
        "power": np.full(n, power_w),
        "energy_reactive": np.full(n, reactive_raw),
    })


def test_hypothesis_a_preferred_when_only_it_lands_in_the_plausible_band():
    # ratio_b = |10| / |1000| = 0.01  -- below the 0.02 floor, implausible under B.
    # ratio_a = |10*12| / |1000| = 0.12 -- comfortably inside 0.02..5.0.
    sample = _units_sample(power_w=1000.0, reactive_raw=10.0)
    result = R.verify_column_units(sample)
    assert result["verdict"] == "A (5-minute kvarh -- x12 needed)"
    assert result["share_implausible_hypothesis_A"] == pytest.approx(0.0)
    assert result["share_implausible_hypothesis_B"] == pytest.approx(1.0)


def test_hypothesis_b_preferred_when_x12_would_overshoot_the_band():
    # ratio_b = |1000| / |1000| = 1.0  -- comfortably inside 0.02..5.0.
    # ratio_a = |1000*12| / |1000| = 12.0 -- above the 5.0 ceiling, implausible under A.
    sample = _units_sample(power_w=1000.0, reactive_raw=1000.0)
    result = R.verify_column_units(sample)
    assert result["verdict"] == "B (already instantaneous kvar -- no x12)"
    assert result["share_implausible_hypothesis_A"] == pytest.approx(1.0)
    assert result["share_implausible_hypothesis_B"] == pytest.approx(0.0)


def test_both_hypotheses_equally_implausible_is_a_tie_not_a_guess():
    # ratio_b = |0.0001| / |1000| ~ 1e-7, ratio_a = 12x that -- both far below 0.02.
    sample = _units_sample(power_w=1000.0, reactive_raw=0.0001)
    result = R.verify_column_units(sample)
    assert result["verdict"] is None
    assert "tie" in result["reason"]


def test_no_sample_returns_none_verdict_not_a_guess():
    result = R.verify_column_units(pd.DataFrame())
    assert result["verdict"] is None
    assert "no sample" in result["reason"]


def test_all_zero_power_returns_none_verdict():
    sample = _units_sample(power_w=0.0, reactive_raw=100.0)
    result = R.verify_column_units(sample)
    assert result["verdict"] is None
    assert "nonzero power" in result["reason"]


# ── confirm_energy_matches_power ─────────────────────────────────────────────

def test_energy_matching_power_times_interval_is_confirmed():
    # Real `ts` sample shape: energy [Wh] = power [W] * (5/60) exactly.
    power = pd.Series([208.8633, 93.0567, -565.5367])
    sample = pd.DataFrame({"power": power, "energy": power * (5.0 / 60.0)})
    result = R.confirm_energy_matches_power(sample, interval_minutes=5.0)
    assert result["confirmed"] is True
    assert result["n_mismatched"] == 0


def test_energy_not_matching_power_times_interval_is_not_confirmed():
    sample = pd.DataFrame({
        "power": [1000.0] * 10,
        "energy": [999.0] * 10,  # way off from 1000 * 5/60 = 83.33
    })
    result = R.confirm_energy_matches_power(sample, interval_minutes=5.0, tolerance_wh=0.5)
    assert result["confirmed"] is False
    assert result["n_mismatched"] == 10
    assert result["share_mismatched"] == pytest.approx(1.0)


def test_tolerance_absorbs_small_rounding_noise():
    sample = pd.DataFrame({
        "power": [1200.0] * 5,
        # expected = 1200 * 5/60 = 100.0 Wh; off by 0.1, under the 0.5 default tolerance
        "energy": [100.1] * 5,
    })
    result = R.confirm_energy_matches_power(sample, interval_minutes=5.0)
    assert result["confirmed"] is True
    assert result["n_mismatched"] == 0


def test_confirm_energy_empty_sample_returns_none():
    result = R.confirm_energy_matches_power(pd.DataFrame())
    assert result["confirmed"] is None
    assert "no sample" in result["reason"]


# ── resample_to_interval ─────────────────────────────────────────────────────

def test_instantaneous_power_resamples_via_energy_sum_not_mean_of_power():
    # 6 x 5-min intervals at 1200 W each -> 6 * 0.1 kWh = 0.6 kWh in the 30-min bucket.
    times = pd.date_range("2025-06-01 00:00", periods=6, freq="5min")
    frame = pd.DataFrame({
        "site_id": [1] * 6,
        "t_stamp": times,
        "power": np.full(6, 1200.0),
    })
    out = R.resample_to_interval(
        frame, time_column="t_stamp", group_columns=["site_id"],
        energy_like_columns={"power": False},
        source_interval_minutes=5, target_interval_minutes=30,
    )
    assert len(out) == 1
    assert out["power"].iloc[0] == pytest.approx(0.6)


def test_already_energy_column_is_summed_directly():
    times = pd.date_range("2025-06-01 00:00", periods=6, freq="5min")
    frame = pd.DataFrame({
        "site_id": [1] * 6,
        "t_stamp": times,
        "energy_kwh": np.full(6, 0.1),
    })
    out = R.resample_to_interval(
        frame, time_column="t_stamp", group_columns=["site_id"],
        energy_like_columns={"energy_kwh": True},
        source_interval_minutes=5, target_interval_minutes=30,
    )
    assert out["energy_kwh"].iloc[0] == pytest.approx(0.6)


def test_non_multiple_target_interval_raises():
    frame = pd.DataFrame({
        "site_id": [1], "t_stamp": pd.to_datetime(["2025-06-01"]), "power": [100.0],
    })
    with pytest.raises(ValueError, match="whole multiple"):
        R.resample_to_interval(
            frame, time_column="t_stamp", group_columns=["site_id"],
            energy_like_columns={"power": False},
            source_interval_minutes=5, target_interval_minutes=7,
        )


def test_resample_coerces_non_datetime_time_column():
    """
    A caller can hand this an untyped time column -- e.g. concatenating
    several Athena pulls where one came back with zero rows for a real data
    gap leaves the merged column as dtype=object rather than datetime64.
    `.dt.floor` used to raise "Can only use .dt accessor with datetimelike
    values" on that; this must resample normally instead.
    """
    times = pd.date_range("2025-06-01 00:00", periods=6, freq="5min")
    frame = pd.DataFrame({
        "site_id": [1] * 6,
        "t_stamp": pd.Series([str(t) for t in times], dtype=object),
        "power": np.full(6, 1200.0),
    })
    out = R.resample_to_interval(
        frame, time_column="t_stamp", group_columns=["site_id"],
        energy_like_columns={"power": False},
        source_interval_minutes=5, target_interval_minutes=30,
    )
    assert len(out) == 1
    assert out["power"].iloc[0] == pytest.approx(0.6)


def test_resample_keeps_groups_separate():
    times = pd.date_range("2025-06-01 00:00", periods=6, freq="5min")
    frame = pd.DataFrame({
        "site_id": [1] * 3 + [2] * 3,
        "t_stamp": list(times[:3]) * 2,
        "power": [1000.0] * 3 + [2000.0] * 3,
    })
    out = R.resample_to_interval(
        frame, time_column="t_stamp", group_columns=["site_id"],
        energy_like_columns={"power": False},
        source_interval_minutes=5, target_interval_minutes=15,
    ).set_index("site_id")
    assert out.loc[2, "power"] == pytest.approx(2 * out.loc[1, "power"])
