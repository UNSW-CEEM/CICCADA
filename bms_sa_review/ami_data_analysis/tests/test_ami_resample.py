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


def test_confirm_energy_breakdown_by_group_isolates_the_bad_type():
    sample = pd.DataFrame({
        "power": [1000.0] * 5 + [1000.0] * 5,
        "circuit_type": ["load_pool"] * 5 + ["pv_site_net"] * 5,
        # load_pool matches perfectly; pv_site_net is way off.
        "energy": [83.333333] * 5 + [1.0] * 5,
    })
    result = R.confirm_energy_matches_power(
        sample, interval_minutes=5.0, group_column="circuit_type"
    )
    assert result["confirmed"] is False
    breakdown = result["share_mismatched_by_group"]
    assert breakdown["pv_site_net"] == pytest.approx(1.0)
    assert breakdown["load_pool"] == pytest.approx(0.0)


# ── confirm_energy_matches_power_actual_interval ─────────────────────────────

def test_missed_poll_gap_is_not_a_mismatch_when_using_actual_interval():
    """
    Reproduces the real-fleet shape: one circuit reports every 5 minutes
    except for one missed poll, so the gap into the NEXT reading is really
    10 minutes. `energy` on that row correctly reflects the full 10 minutes
    (a genuinely-measured energy register would) -- comparing it against the
    NOMINAL 5-minute interval wrongly flags it; comparing it against the
    ACTUAL (t_stamp-derived) interval should not.
    """
    times = pd.to_datetime([
        "2025-06-01 00:00:00", "2025-06-01 00:05:00",
        "2025-06-01 00:15:00",  # missed the 00:10 poll -- a real 10-min gap
        "2025-06-01 00:20:00",
    ])
    power = pd.Series([600.0, 600.0, 600.0, 600.0])
    # energy correctly reflects each row's REAL elapsed time since the last
    # reading (first row has no prior reading, so its value is irrelevant
    # here -- the function drops rows with no actual interval to compute).
    intervals_min = [5.0, 5.0, 10.0, 5.0]
    energy = [p * (m / 60.0) for p, m in zip(power, intervals_min)]
    sample = pd.DataFrame({
        "circuit_id": [1] * 4, "t_stamp": times, "power": power, "energy": energy,
    })

    nominal = R.confirm_energy_matches_power(sample, interval_minutes=5.0)
    actual = R.confirm_energy_matches_power_actual_interval(sample)

    # Against the nominal (constant) 5-minute assumption, the missed-poll
    # row's energy (600 * 10/60 = 100 Wh) doesn't match 600 * 5/60 = 50 Wh.
    assert nominal["confirmed"] is False
    assert nominal["n_mismatched"] == 1

    # Against each row's own actual gap, every row matches.
    assert actual["confirmed"] is True
    assert actual["n_mismatched"] == 0
    # 4 rows in, but the first has no earlier reading to diff against.
    assert actual["n_rows"] == 3


def test_actual_interval_check_still_flags_a_genuine_mismatch():
    times = pd.to_datetime(["2025-06-01 00:00:00", "2025-06-01 00:05:00"])
    sample = pd.DataFrame({
        "circuit_id": [1, 1], "t_stamp": times,
        "power": [1000.0, 1000.0],
        "energy": [1000.0, 999.0],  # expected 1000*5/60=83.33 -- way off
    })
    result = R.confirm_energy_matches_power_actual_interval(sample)
    assert result["confirmed"] is False
    assert result["n_mismatched"] == 1


def test_actual_interval_check_keeps_circuits_independent():
    # Two circuits' own gaps must not bleed into each other via a naive
    # groupby-less diff (which would compute a bogus interval between the
    # last row of one circuit and the first row of the next).
    sample = pd.DataFrame({
        "circuit_id": [1, 1, 2, 2],
        "t_stamp": pd.to_datetime([
            "2025-06-01 00:00:00", "2025-06-01 00:05:00",
            "2025-06-01 00:00:00", "2025-06-01 00:05:00",
        ]),
        "power": [600.0, 600.0, 300.0, 300.0],
        "energy": [50.0, 50.0, 25.0, 25.0],  # each = power * 5/60
    })
    result = R.confirm_energy_matches_power_actual_interval(sample)
    assert result["confirmed"] is True
    assert result["n_rows"] == 2  # one first-reading dropped per circuit


def test_actual_interval_check_single_reading_per_circuit_returns_none():
    sample = pd.DataFrame({
        "circuit_id": [1], "t_stamp": pd.to_datetime(["2025-06-01 00:00:00"]),
        "power": [600.0], "energy": [50.0],
    })
    result = R.confirm_energy_matches_power_actual_interval(sample)
    assert result["confirmed"] is None
    assert "no row has an earlier reading" in result["reason"]


def test_actual_interval_check_empty_sample_returns_none():
    result = R.confirm_energy_matches_power_actual_interval(pd.DataFrame())
    assert result["confirmed"] is None
    assert "no sample" in result["reason"]


def test_actual_interval_check_breakdown_by_group():
    sample = pd.DataFrame({
        "circuit_id": [1, 1, 2, 2],
        "circuit_type": ["load_pool", "load_pool", "pv_site_net", "pv_site_net"],
        "t_stamp": pd.to_datetime([
            "2025-06-01 00:00:00", "2025-06-01 00:05:00",
            "2025-06-01 00:00:00", "2025-06-01 00:05:00",
        ]),
        "power": [600.0, 600.0, 1000.0, 1000.0],
        "energy": [50.0, 50.0, 1.0, 1.0],  # load_pool matches, pv_site_net doesn't
    })
    result = R.confirm_energy_matches_power_actual_interval(
        sample, group_column="circuit_type"
    )
    breakdown = result["share_mismatched_by_group"]
    assert breakdown["pv_site_net"] == pytest.approx(1.0)
    assert breakdown["load_pool"] == pytest.approx(0.0)


# ── energy_granularity_and_implied_interval ──────────────────────────────────

def test_clean_circuit_shows_exact_nominal_interval_and_no_integer_snapping():
    times = pd.date_range("2025-06-01 00:00", periods=6, freq="5min")
    sample = pd.DataFrame({
        "circuit_id": [1] * 6,
        "power": [601.2, 431.9, 812.4, 250.7, 999.3, 111.1],
        # exactly power * 5/60 -- a genuinely continuous energy column
        "energy": [p * (5.0 / 60.0) for p in [601.2, 431.9, 812.4, 250.7, 999.3, 111.1]],
    })
    sample["t_stamp"] = times
    result = R.energy_granularity_and_implied_interval(sample)
    row = result[result.circuit_id == 1].iloc[0]
    assert row.share_integer_energy == pytest.approx(0.0)
    assert row.implied_interval_minutes == pytest.approx(5.0)


def test_integer_energy_register_with_shorter_true_interval_is_flagged():
    # energy is always a whole Wh, and consistently implies ~4.9 minutes,
    # not the logged 5-minute cadence -- the device/meter-model signature
    # `confirm_energy_matches_power_actual_interval` cannot see (its
    # `t_stamp` gaps are still exactly 5 minutes).
    powers = [700.0, 900.0, 1200.0, 500.0, 1500.0]
    true_interval_minutes = 4.9
    sample = pd.DataFrame({
        "circuit_id": [2] * 5,
        "power": powers,
        "energy": [round(p * (true_interval_minutes / 60.0)) for p in powers],
    })
    result = R.energy_granularity_and_implied_interval(sample)
    row = result[result.circuit_id == 2].iloc[0]
    assert row.share_integer_energy == pytest.approx(1.0)
    assert row.implied_interval_minutes == pytest.approx(true_interval_minutes, abs=0.05)


def test_low_power_rows_excluded_from_the_ratio_by_default():
    # Below the 200W default floor -- would still count toward
    # share_integer_energy, but must NOT be used for the interval ratio,
    # where small-power quantization noise dominates.
    sample = pd.DataFrame({
        "circuit_id": [3] * 3,
        "power": [5.0, 8.0, 900.0],
        "energy": [1.0, 1.0, 900.0 * (5.0 / 60.0)],
    })
    result = R.energy_granularity_and_implied_interval(sample)
    row = result[result.circuit_id == 3].iloc[0]
    assert row.n_rows == 3
    assert row.n_rows_used_for_ratio == 1
    assert row.implied_interval_minutes == pytest.approx(5.0)


def test_distinguishes_circuits_within_the_same_sample():
    clean_times = pd.date_range("2025-06-01 00:00", periods=3, freq="5min")
    sample = pd.DataFrame({
        "circuit_id": [1, 1, 1, 2, 2, 2],
        "power": [601.2, 803.5, 1002.9, 601.2, 803.5, 1002.9],
        "energy": [
            601.2 * (5.0 / 60.0), 803.5 * (5.0 / 60.0), 1002.9 * (5.0 / 60.0),
            round(601.2 * (4.9 / 60.0)), round(803.5 * (4.9 / 60.0)), round(1002.9 * (4.9 / 60.0)),
        ],
    })
    result = R.energy_granularity_and_implied_interval(sample).set_index("circuit_id")
    assert result.loc[1, "share_integer_energy"] == pytest.approx(0.0)
    assert result.loc[2, "share_integer_energy"] == pytest.approx(1.0)
    assert result.loc[1, "implied_interval_minutes"] > result.loc[2, "implied_interval_minutes"]


def test_empty_sample_returns_empty_frame_with_expected_columns():
    result = R.energy_granularity_and_implied_interval(pd.DataFrame())
    assert list(result.columns) == [
        "circuit_id", "n_rows", "share_integer_energy",
        "implied_interval_minutes", "n_rows_used_for_ratio",
    ]
    assert len(result) == 0


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
