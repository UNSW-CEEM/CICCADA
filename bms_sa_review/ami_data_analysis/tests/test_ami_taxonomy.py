"""
The pure taxonomy/aggregation-proof logic behind 03_signal_taxonomy.

The one rule this file exists to protect: a circuit_type is never treated as
an aggregate because its name suggests it (`flag_suspected_aggregates` is a
hypothesis generator, nothing more) -- only `check_aggregation`'s arithmetic
comparison, on both mean AND worst-interval difference, may call it proven.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bms_sa_review.ami_data_analysis.lib import ami_taxonomy as T


# ── summarise_circuit_types ─────────────────────────────────────────────────

def test_share_within_is_pv_sums_to_one_per_group():
    raw = pd.DataFrame({
        "circuit_type": ["pv_site", "pv_site_net", "load_pool", "ac_load_net"],
        "is_pv": [True, True, False, False],
        "n_circuits": [8000, 2000, 6000, 4000],
    })
    out = T.summarise_circuit_types(raw)
    totals = out.groupby("is_pv")["share_within_is_pv"].sum()
    assert totals.loc[True] == pytest.approx(1.0)
    assert totals.loc[False] == pytest.approx(1.0)


def test_summarise_empty_input_returns_empty_frame():
    assert T.summarise_circuit_types(pd.DataFrame()).empty
    assert T.summarise_circuit_types(None).empty


# ── flag_suspected_aggregates ───────────────────────────────────────────────

def test_net_and_site_names_are_flagged_suspect():
    census = pd.DataFrame({"circuit_type": ["ac_load_net", "pv_site", "load_pool"]})
    out = T.flag_suspected_aggregates(census)
    flags = out.set_index("circuit_type")["suspected_aggregate"]
    assert bool(flags["ac_load_net"]) is True
    assert bool(flags["pv_site"]) is True
    assert bool(flags["load_pool"]) is False


def test_flagging_is_case_insensitive():
    census = pd.DataFrame({"circuit_type": ["AC_LOAD_NET"]})
    assert T.flag_suspected_aggregates(census)["suspected_aggregate"].iloc[0]


# ── cohort_completeness ─────────────────────────────────────────────────────

def test_cohort_completeness_pivots_counts_per_site():
    meta = pd.DataFrame({
        "site_id": [1, 1, 1, 2, 2],
        "circuit_type": ["ac_load_net", "load_pool", "load_hot_water", "pv_site", "pv_site"],
        "circuit_id": [10, 11, 12, 20, 21],
    })
    out = T.cohort_completeness(meta).set_index("site_id")
    assert out.loc[1, "ac_load_net"] == 1
    assert out.loc[1, "load_pool"] == 1
    assert out.loc[1, "load_hot_water"] == 1
    assert out.loc[2, "pv_site"] == 2
    # site 1 has no pv_site column value other than the fill default
    assert out.loc[1, "pv_site"] == 0


# ── check_aggregation ────────────────────────────────────────────────────────

def _long_frame(candidate_series, component_a_series, component_b_series, times):
    rows = []
    for t, c, a, b in zip(times, candidate_series, component_a_series, component_b_series):
        rows.append({"t_stamp": t, "circuit_id": 1, "power_signed": c})
        rows.append({"t_stamp": t, "circuit_id": 2, "power_signed": a})
        rows.append({"t_stamp": t, "circuit_id": 3, "power_signed": b})
    return pd.DataFrame(rows)


def test_exact_duplicate_is_confirmed_aggregate():
    times = pd.date_range("2025-06-01", periods=30, freq="5min")
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 500, size=30)
    b = rng.uniform(0, 500, size=30)
    candidate = a + b  # candidate IS the sum, exactly
    long = _long_frame(candidate, a, b, times)

    result = T.check_aggregation(
        long, site_id=99, candidate_circuit_ids=[1], component_circuit_ids=[2, 3],
        candidate_type="ac_load_net", component_types=("load_pool", "load_hot_water"),
    )
    assert result.is_aggregate is True
    assert result.n_intervals_compared == 30
    assert result.max_abs_diff < 1e-6


def test_independent_circuit_is_not_confirmed_aggregate():
    times = pd.date_range("2025-06-01", periods=30, freq="5min")
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 500, size=30)
    b = rng.uniform(0, 500, size=30)
    candidate = rng.uniform(0, 500, size=30)  # unrelated to a + b
    long = _long_frame(candidate, a, b, times)

    result = T.check_aggregation(
        long, site_id=99, candidate_circuit_ids=[1], component_circuit_ids=[2, 3],
        candidate_type="load_stove", component_types=("load_pool", "load_hot_water"),
    )
    assert result.is_aggregate is False


def test_small_but_real_average_mismatch_with_huge_outlier_is_not_aggregate():
    """Passing on average alone must not be enough -- one bad interval fails it."""
    times = pd.date_range("2025-06-01", periods=30, freq="5min")
    a = np.full(30, 100.0)
    b = np.full(30, 100.0)
    candidate = a + b
    candidate[5] += 5000.0  # one wild outlier interval
    long = _long_frame(candidate, a, b, times)

    result = T.check_aggregation(
        long, site_id=99, candidate_circuit_ids=[1], component_circuit_ids=[2, 3],
        candidate_type="ac_load_net", component_types=("load_pool", "load_hot_water"),
        mean_relative_tolerance=0.5,  # generous on the mean...
    )
    assert result.is_aggregate is False  # ...but the max-diff check still catches it


def test_too_few_overlapping_intervals_is_inconclusive_not_false():
    times = pd.date_range("2025-06-01", periods=5, freq="5min")
    a = np.full(5, 100.0)
    b = np.full(5, 100.0)
    candidate = a + b
    long = _long_frame(candidate, a, b, times)

    result = T.check_aggregation(
        long, site_id=99, candidate_circuit_ids=[1], component_circuit_ids=[2, 3],
        candidate_type="ac_load_net", component_types=("load_pool", "load_hot_water"),
        min_intervals=20,
    )
    assert result.is_aggregate is None
    assert "only 5" in result.reason


def test_all_zero_components_and_matching_candidate_is_aggregate():
    times = pd.date_range("2025-06-01", periods=25, freq="5min")
    zeros = np.zeros(25)
    long = _long_frame(zeros, zeros, zeros, times)
    result = T.check_aggregation(
        long, site_id=1, candidate_circuit_ids=[1], component_circuit_ids=[2, 3],
        candidate_type="x", component_types=("y", "z"),
    )
    assert result.is_aggregate is True


def test_all_zero_components_but_nonzero_candidate_is_not_aggregate():
    times = pd.date_range("2025-06-01", periods=25, freq="5min")
    zeros = np.zeros(25)
    candidate = np.full(25, 50.0)
    long = _long_frame(candidate, zeros, zeros, times)
    result = T.check_aggregation(
        long, site_id=1, candidate_circuit_ids=[1], component_circuit_ids=[2, 3],
        candidate_type="x", component_types=("y", "z"),
    )
    assert result.is_aggregate is False


def test_missing_columns_returns_inconclusive():
    result = T.check_aggregation(
        pd.DataFrame({"foo": [1, 2]}), site_id=1,
        candidate_circuit_ids=[1], component_circuit_ids=[2],
        candidate_type="x", component_types=("y",),
    )
    assert result.is_aggregate is None
    assert "missing required columns" in result.reason


# ── pick_aggregation_test_site ──────────────────────────────────────────────

def test_pick_aggregation_test_site_orders_by_sibling_count_descending():
    cohort = pd.DataFrame({
        "site_id": [1, 2, 3],
        "ac_load_net": [1, 1, 0],       # site 3 has no candidate -- excluded
        "load_pool": [1, 2, 5],
        "load_hot_water": [0, 3, 5],
    })
    ranked = T.pick_aggregation_test_site(
        cohort, candidate_type="ac_load_net",
        component_types=["load_pool", "load_hot_water"],
    )
    assert ranked == [2, 1]  # site 2 has 5 sibling circuits, site 1 has 1


def test_pick_aggregation_test_site_respects_min_components():
    cohort = pd.DataFrame({
        "site_id": [1, 2],
        "ac_load_net": [1, 1],
        "load_pool": [0, 2],
    })
    ranked = T.pick_aggregation_test_site(
        cohort, candidate_type="ac_load_net", component_types=["load_pool"],
        min_components=1,
    )
    assert ranked == [2]


def test_pick_aggregation_test_site_missing_columns_returns_empty():
    cohort = pd.DataFrame({"site_id": [1]})
    assert T.pick_aggregation_test_site(cohort, candidate_type="x", component_types=["y"]) == []


# ── signal_coverage_summary ──────────────────────────────────────────────────

def test_coverage_summary_counts_sites_with_both_sides():
    cohort = pd.DataFrame({
        "site_id": [1, 2, 3, 4],
        "pv_site": [1, 0, 1, 0],
        "load_pool": [1, 1, 0, 0],
    })
    out = T.signal_coverage_summary(cohort, pv_types=["pv_site"], load_types=["load_pool"])
    assert out["n_sites"] == 4
    assert out["n_with_both"] == 1        # site 1 only
    assert out["n_pv_only"] == 1          # site 3
    assert out["n_load_only"] == 1        # site 2
    assert out["n_neither"] == 1          # site 4
    assert out["share_with_both"] == pytest.approx(0.25)


def test_coverage_summary_missing_type_columns_is_reported_not_guessed():
    cohort = pd.DataFrame({"site_id": [1], "load_pool": [1]})
    out = T.signal_coverage_summary(cohort, pv_types=["pv_site"], load_types=["load_pool"])
    assert out["n_with_both"] is None
    assert "reason" in out


def test_coverage_summary_empty_cohort():
    out = T.signal_coverage_summary(pd.DataFrame(), pv_types=["pv_site"], load_types=["load_pool"])
    assert out["n_with_both"] is None
    assert out["n_sites"] == 0


# ── find_duplicate_circuits ──────────────────────────────────────────────────

def _duplicate_test_frame():
    rng = np.random.default_rng(0)
    times = pd.date_range("2025-06-01", periods=100, freq="5min")
    base = rng.uniform(-5000, 5000, size=len(times))
    independent = rng.uniform(-1000, 1000, size=len(times))
    rows = []
    for cid, values in (
        (1, base),                    # site 1: original
        (2, -base + 0.01),            # site 1: near-exact mirror (opposite sign)
        (3, independent),              # site 1: genuinely independent
        (4, base),                    # site 2: a different site -- must NOT match circuit 1
    ):
        site_id = 1 if cid != 4 else 2
        for t, v in zip(times, values):
            rows.append({"site_id": site_id, "circuit_id": cid, "t_stamp": t, "power": v})
    return pd.DataFrame(rows)


def test_find_duplicate_circuits_flags_mirrored_pair():
    out = T.find_duplicate_circuits(_duplicate_test_frame())
    pair = out[out.circuit_id_a.isin([1, 2]) & out.circuit_id_b.isin([1, 2])]
    assert len(pair) == 1
    assert pair.iloc[0]["sign"] == "opposite"
    assert pair.iloc[0]["correlation"] == pytest.approx(-1.0, abs=1e-3)


def test_find_duplicate_circuits_does_not_flag_independent_circuit():
    out = T.find_duplicate_circuits(_duplicate_test_frame())
    involves_3 = out[(out.circuit_id_a == 3) | (out.circuit_id_b == 3)]
    assert len(involves_3) == 0


def test_find_duplicate_circuits_does_not_cross_sites():
    # Circuit 1 (site 1) and circuit 4 (site 2) carry identical values, but a
    # duplicate reading only means something WITHIN one site -- two different
    # sites coincidentally matching is not a data-tagging bug to flag.
    out = T.find_duplicate_circuits(_duplicate_test_frame())
    assert not ((out.circuit_id_a == 4) | (out.circuit_id_b == 4)).any()


def test_find_duplicate_circuits_empty_input():
    out = T.find_duplicate_circuits(pd.DataFrame())
    assert len(out) == 0
    assert "correlation" in out.columns


def test_find_duplicate_circuits_respects_threshold():
    # Same generator, but a threshold above 1.0 can never be met.
    out = T.find_duplicate_circuits(_duplicate_test_frame(), correlation_threshold=1.5)
    assert len(out) == 0
