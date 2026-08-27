"""
The pure sign-verification and signal-mapping logic behind
03_signal_taxonomy: does `circuit_polarity` behave as assumed, and does
`build_signal_map` route proven aggregates and proven storage circuits out of
`gross_load` rather than trusting a name to do it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bms_sa_review.ami_data_analysis.lib import ami_signal as S


# ── classify_storage_circuits ───────────────────────────────────────────────

def test_battery_and_ev_names_are_flagged():
    result = S.classify_storage_circuits(["load_pool", "battery_1", "ev_charger", "load_ev"])
    assert result["battery_1"] is True
    assert result["ev_charger"] is True
    assert result["load_ev"] is True
    assert result["load_pool"] is False


# ── verify_polarity_makes_positive ──────────────────────────────────────────

def test_consistently_positive_circuit_is_not_bidirectional():
    sample = pd.DataFrame({
        "circuit_type": ["load_pool"] * 20,
        "power": np.full(20, 500.0),
        "circuit_polarity": np.full(20, 1),
    })
    out = S.verify_polarity_makes_positive(sample).set_index("circuit_type")
    assert out.loc["load_pool", "share_negative"] == 0.0
    assert out.loc["load_pool", "bidirectional"] == False  # noqa: E712


def test_mixed_sign_circuit_is_flagged_bidirectional():
    sample = pd.DataFrame({
        "circuit_type": ["battery_1"] * 20,
        "power": [500.0] * 10 + [-500.0] * 10,
        "circuit_polarity": np.full(20, 1),
    })
    out = S.verify_polarity_makes_positive(sample).set_index("circuit_type")
    assert out.loc["battery_1", "bidirectional"] == True  # noqa: E712
    assert out.loc["battery_1", "share_negative"] == pytest.approx(0.5)


def test_rare_negative_noise_below_threshold_is_not_bidirectional():
    # 1 negative out of 20 = 5%, at the default threshold -- must be > not >=.
    sample = pd.DataFrame({
        "circuit_type": ["load_pool"] * 20,
        "power": [500.0] * 19 + [-1.0],
        "circuit_polarity": np.full(20, 1),
    })
    out = S.verify_polarity_makes_positive(
        sample, bidirectional_share_threshold=0.05
    ).set_index("circuit_type")
    assert out.loc["load_pool", "bidirectional"] == False  # noqa: E712


def test_negative_polarity_flips_correction_sign():
    sample = pd.DataFrame({
        "circuit_type": ["load_x"] * 5,
        "power": np.full(5, 100.0),
        "circuit_polarity": np.full(5, -1),
    })
    out = S.verify_polarity_makes_positive(sample).set_index("circuit_type")
    assert out.loc["load_x", "share_negative"] == 1.0


def test_empty_sample_returns_empty_frame():
    assert S.verify_polarity_makes_positive(pd.DataFrame()).empty
    assert S.verify_polarity_makes_positive(None).empty


def test_small_magnitude_noise_is_not_bidirectional_with_default_floor():
    """
    A real fleet run without a magnitude floor flagged 17 of 24 load types
    (lighting, hot water, a fridge...) as bidirectional -- physically
    implausible for most of them. The cause: small negative readings from
    CT/power-factor noise around an idle baseline clear a bare sign-count
    threshold at fleet scale without the circuit ever really reversing.
    This reproduces that shape at small scale: 20% genuinely small negative
    noise (-5W, well under the 50W default floor) plus 80% real positive
    draw -- must NOT be flagged, even though the RAW share easily would be.
    """
    sample = pd.DataFrame({
        "circuit_type": ["load_lighting"] * 20,
        "power": [500.0] * 16 + [-5.0] * 4,
        "circuit_polarity": np.full(20, 1),
    })
    out = S.verify_polarity_makes_positive(sample).set_index("circuit_type")
    assert out.loc["load_lighting", "share_negative_raw"] == pytest.approx(0.2)
    assert out.loc["load_lighting", "share_negative"] == 0.0
    assert out.loc["load_lighting", "bidirectional"] == False  # noqa: E712


def test_large_magnitude_reversal_still_flagged_bidirectional():
    # Genuine reversal (well above the noise floor) must still be caught --
    # the floor should not blind the check entirely.
    sample = pd.DataFrame({
        "circuit_type": ["battery_1"] * 20,
        "power": [500.0] * 10 + [-500.0] * 10,
        "circuit_polarity": np.full(20, 1),
    })
    out = S.verify_polarity_makes_positive(sample).set_index("circuit_type")
    assert out.loc["battery_1", "bidirectional"] == True  # noqa: E712
    assert out.loc["battery_1", "share_negative"] == pytest.approx(0.5)


def test_noise_floor_is_configurable():
    sample = pd.DataFrame({
        "circuit_type": ["load_x"] * 20,
        "power": [500.0] * 16 + [-5.0] * 4,
        "circuit_polarity": np.full(20, 1),
    })
    out = S.verify_polarity_makes_positive(
        sample, noise_floor_w=1.0
    ).set_index("circuit_type")
    # With a 1W floor, the -5W readings now clear it and count as negative.
    assert out.loc["load_x", "share_negative"] == pytest.approx(0.2)
    assert out.loc["load_x", "bidirectional"] == True  # noqa: E712


# ── build_signal_map ─────────────────────────────────────────────────────────

def test_aggregate_type_is_excluded():
    census = pd.DataFrame({"circuit_type": ["ac_load_net"], "is_pv": [False]})
    mapping = S.build_signal_map(census, aggregate_types={"ac_load_net"})
    assert "EXCLUDE" in mapping["ac_load_net"]
    assert "aggregate" in mapping["ac_load_net"].lower()


def test_storage_type_is_excluded_from_gross_load():
    census = pd.DataFrame({"circuit_type": ["battery_1"], "is_pv": [False]})
    mapping = S.build_signal_map(census, aggregate_types=set(), storage_types={"battery_1"})
    assert "EXCLUDE" in mapping["battery_1"]
    assert "storage" in mapping["battery_1"].lower()


def test_pv_type_maps_to_pv_generation():
    census = pd.DataFrame({"circuit_type": ["pv_site"], "is_pv": [True]})
    mapping = S.build_signal_map(census, aggregate_types=set())
    assert mapping["pv_site"] == "pv_generation"


def test_ordinary_load_type_keeps_its_own_name():
    census = pd.DataFrame({"circuit_type": ["load_pool"], "is_pv": [False]})
    mapping = S.build_signal_map(census, aggregate_types=set())
    assert mapping["load_pool"] == "load_pool"


def test_aggregate_rule_takes_priority_over_is_pv():
    census = pd.DataFrame({"circuit_type": ["pv_site_net"], "is_pv": [True]})
    mapping = S.build_signal_map(census, aggregate_types={"pv_site_net"})
    assert "EXCLUDE" in mapping["pv_site_net"]


def test_duplicate_circuit_types_in_census_collapse_to_one_entry():
    census = pd.DataFrame({
        "circuit_type": ["load_pool", "load_pool"],
        "is_pv": [False, False],
    })
    mapping = S.build_signal_map(census, aggregate_types=set())
    assert mapping == {"load_pool": "load_pool"}


# ── night_window_stats ───────────────────────────────────────────────────────

def _night_sample(hours, values, power_column="power_signed"):
    times = pd.to_datetime([f"2025-06-01 {h:02d}:00:00" for h in hours])
    return pd.DataFrame({"t_stamp": times, power_column: values})


def test_night_window_stats_computes_mean_and_median_inside_window():
    sample = _night_sample([0, 1, 2, 3, 4, 12], [10.0, 20.0, 30.0, 40.0, 50.0, 999.0])
    out = S.night_window_stats(sample, night_hour_start=1, night_hour_end=4)
    assert out["n_night_intervals"] == 3  # hours 1, 2, 3 -- end is exclusive
    assert out["night_mean"] == pytest.approx(30.0)
    assert out["night_median"] == pytest.approx(30.0)
    assert out["reason"] is None


def test_night_window_stats_no_intervals_in_window():
    sample = _night_sample([12, 13, 14], [100.0, 100.0, 100.0])
    out = S.night_window_stats(sample, night_hour_start=1, night_hour_end=4)
    assert out["n_night_intervals"] == 0
    assert out["night_mean"] is None
    assert "no intervals" in out["reason"]


def test_night_window_stats_empty_sample_returns_none():
    out = S.night_window_stats(pd.DataFrame())
    assert out["night_mean"] is None
    assert "no sample" in out["reason"]


def test_night_window_stats_missing_time_column_returns_none():
    out = S.night_window_stats(pd.DataFrame({"power_signed": [1.0]}))
    assert out["night_mean"] is None
    assert "missing time column" in out["reason"]


# ── classify_pv_night_behaviour ─────────────────────────────────────────────

def test_small_night_value_is_generation_like():
    # -10 W at night is a plausible inverter standby draw, not load consumption.
    sample = _night_sample([1, 2, 3], [-10.0, -12.0, -8.0])
    out = S.classify_pv_night_behaviour(sample, net_like_threshold_w=100.0)
    assert out["verdict"] == "generation-like"
    assert "standby" in out["reason"]


def test_large_negative_night_value_is_net_like():
    # -800 W at night, with no sunlight to produce it, looks like load draw.
    sample = _night_sample([1, 2, 3], [-800.0, -820.0, -790.0])
    out = S.classify_pv_night_behaviour(sample, net_like_threshold_w=100.0)
    assert out["verdict"] == "net-like"
    assert "load consumption" in out["reason"]


def test_value_exactly_at_threshold_is_generation_like_not_net_like():
    sample = _night_sample([1, 2], [100.0, 100.0])
    out = S.classify_pv_night_behaviour(sample, net_like_threshold_w=100.0)
    assert out["verdict"] == "generation-like"  # <=, not <


def test_classify_pv_night_behaviour_no_data_returns_none_verdict():
    sample = _night_sample([12, 13], [500.0, 500.0])
    out = S.classify_pv_night_behaviour(sample, night_hour_start=1, night_hour_end=4)
    assert out["verdict"] is None
    assert "no intervals" in out["reason"]


# ── compare_pv_night_to_load ─────────────────────────────────────────────────

def test_comparison_skipped_when_pv_verdict_is_not_net_like():
    pv_stats = {"verdict": "generation-like", "night_mean": -10.0}
    load_stats = {"night_mean": -500.0}
    out = S.compare_pv_night_to_load(pv_stats, load_stats)
    assert out["corroborated"] is None
    assert "not net-like" in out["reason"]


def test_comparison_skipped_when_no_load_data_available():
    pv_stats = {"verdict": "net-like", "night_mean": -800.0}
    out = S.compare_pv_night_to_load(pv_stats, {"night_mean": None})
    assert out["corroborated"] is None
    assert "no co-located load sample" in out["reason"]


def test_similar_magnitude_is_corroborated():
    pv_stats = {"verdict": "net-like", "night_mean": -800.0}
    load_stats = {"night_mean": 700.0}  # ratio 800/700 ~= 1.14, within (0.3, 2.0)
    out = S.compare_pv_night_to_load(pv_stats, load_stats)
    assert out["corroborated"] is True
    assert out["ratio"] == pytest.approx(800.0 / 700.0)


def test_very_different_magnitude_is_not_corroborated():
    pv_stats = {"verdict": "net-like", "night_mean": -800.0}
    load_stats = {"night_mean": 10.0}  # ratio 80 -- wildly outside the band
    out = S.compare_pv_night_to_load(pv_stats, load_stats)
    assert out["corroborated"] is False


def test_zero_load_night_mean_cannot_corroborate():
    pv_stats = {"verdict": "net-like", "night_mean": -800.0}
    load_stats = {"night_mean": 0.0}
    out = S.compare_pv_night_to_load(pv_stats, load_stats)
    assert out["corroborated"] is False
    assert "~0" in out["reason"]


# ── sites_with_storage_circuits ─────────────────────────────────────────────

def test_sites_with_storage_circuits_flags_name_matches_only():
    meta = pd.DataFrame({
        "site_id": [1, 2, 3],
        "circuit_type": ["load_battery", "ac_load_net", "load_pool"],
    })
    assert S.sites_with_storage_circuits(meta) == [1]


def test_sites_with_storage_circuits_empty_input():
    assert S.sites_with_storage_circuits(pd.DataFrame()) == []
    assert S.sites_with_storage_circuits(None) == []


# ── reconstruct_gross_load ──────────────────────────────────────────────────

def _interval_row(site_id, circuit_id, circuit_type, t_stamp, power):
    return {"site_id": site_id, "circuit_id": circuit_id, "circuit_type": circuit_type,
            "t_stamp": t_stamp, "power": power}


def test_reconstruct_gross_load_sums_multi_phase_and_adds_pv_back():
    t = pd.Timestamp("2025-06-01 12:00:00")
    interval_table = pd.DataFrame([
        # 3-phase load: raw power positive on each phase, polarity +1 -> import
        _interval_row(1, 11, "ac_load_net", t, 200.0),
        _interval_row(1, 12, "ac_load_net", t, 150.0),
        _interval_row(1, 13, "ac_load_net", t, 130.0),
        # PV: raw power reads negative for generation, polarity -1 flips it positive
        _interval_row(1, 14, "pv_site_net", t, -3000.0),
    ])
    circuit_polarity = pd.DataFrame({
        "circuit_id": [11, 12, 13, 14],
        "circuit_polarity": [1, 1, 1, -1],
    })
    out = S.reconstruct_gross_load(interval_table, circuit_polarity)
    row = out.iloc[0]
    assert row.load_signed == pytest.approx(480.0)     # 200+150+130, polarity +1
    assert row.pv_signed == pytest.approx(3000.0)       # -3000 * -1
    assert row.reconstructed_load == pytest.approx(3480.0)


def test_reconstruct_gross_load_empty_input():
    out = S.reconstruct_gross_load(pd.DataFrame(), pd.DataFrame())
    assert out.empty
    assert "reconstructed_load" in out.columns


def test_reconstruct_gross_load_missing_pv_side_leaves_nan_not_zero():
    t = pd.Timestamp("2025-06-01 12:00:00")
    interval_table = pd.DataFrame([_interval_row(1, 11, "ac_load_net", t, 500.0)])
    circuit_polarity = pd.DataFrame({"circuit_id": [11], "circuit_polarity": [1]})
    out = S.reconstruct_gross_load(interval_table, circuit_polarity)
    assert out.iloc[0].pv_signed != out.iloc[0].pv_signed  # NaN != NaN
    assert out.iloc[0].reconstructed_load != out.iloc[0].reconstructed_load


# ── evaluate_load_reconstruction ────────────────────────────────────────────

def test_evaluate_load_reconstruction_flags_negative_night_values():
    night = pd.Timestamp("2025-06-01 02:00:00")   # inside default 1-4am window
    day = pd.Timestamp("2025-06-01 12:00:00")
    reconstructed = pd.DataFrame({
        "site_id": [1, 1, 2, 2],
        "t_stamp": [night, day, night, day],
        "reconstructed_load": [-500.0, 300.0, 400.0, 350.0],
    })
    out = S.evaluate_load_reconstruction(reconstructed).set_index("site_id")
    assert out.loc[1, "likely_storage_or_sign_issue"] == True
    assert out.loc[1, "share_negative_night"] == pytest.approx(1.0)
    assert out.loc[2, "likely_storage_or_sign_issue"] == False
    assert out.loc[2, "share_negative_night"] == pytest.approx(0.0)


def test_evaluate_load_reconstruction_drops_nan_rows_before_scoring():
    t = pd.Timestamp("2025-06-01 12:00:00")
    reconstructed = pd.DataFrame({
        "site_id": [1, 1], "t_stamp": [t, t], "reconstructed_load": [300.0, float("nan")],
    })
    out = S.evaluate_load_reconstruction(reconstructed)
    assert out.iloc[0]["n_intervals"] == 1


def test_evaluate_load_reconstruction_empty_input():
    out = S.evaluate_load_reconstruction(pd.DataFrame())
    assert out.empty
    assert "likely_storage_or_sign_issue" in out.columns
