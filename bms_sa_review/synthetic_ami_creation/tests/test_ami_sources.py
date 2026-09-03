"""
The pure comparison logic behind 02_source_selection.

The one rule this file exists to protect: a candidate qualifies only if it has
BOTH signals and can tell them apart -- cost and cleanliness never override
that, so a cheap, clean, disqualified candidate can never look like a
contender in `recommend()`'s output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bms_sa_review.synthetic_ami_creation.lib import ami_sources as S


# ── verify_is_pv_only ───────────────────────────────────────────────────────

def test_schema_without_is_pv_is_flagged_is_pv_only():
    schema = pd.DataFrame({"column_name": ["site_id", "t_stamp", "p_kw_norm"]})
    result = S.verify_is_pv_only(schema)
    assert result["is_pv_only"] is True
    assert "no `is_pv` column" in result["reason"]


def test_schema_with_is_pv_is_not_flagged():
    schema = pd.DataFrame({"column_name": ["circuit_id", "t_stamp", "is_pv", "power"]})
    result = S.verify_is_pv_only(schema)
    assert result["is_pv_only"] is False


def test_is_pv_column_name_matching_is_case_insensitive():
    schema = pd.DataFrame({"column_name": ["IS_PV", "power"]})
    assert S.verify_is_pv_only(schema)["is_pv_only"] is False


@pytest.mark.parametrize("schema", [None, pd.DataFrame(), pd.DataFrame({"other": [1]})])
def test_missing_schema_gives_none_not_a_false_pass(schema):
    """None here must never be read as 'verified is_pv_only=False'."""
    result = S.verify_is_pv_only(schema)
    assert result["is_pv_only"] is None
    assert "cannot assess" in result["reason"]


# ── compare_circuit_and_row_shares ──────────────────────────────────────────

def test_shares_split_correctly_with_boolean_is_pv():
    circuit_counts = pd.DataFrame({
        "is_pv": [False, True],
        "n_circuits": [90_000, 81_411],
        "n_sites": [35_000, 30_000],
    })
    out = S.compare_circuit_and_row_shares(circuit_counts, ts_rows_false=8_000, ts_rows_true=2_000)
    row = out.set_index("label")
    assert row.loc["is_pv=false (load)", "n_circuits"] == 90_000
    assert row.loc["is_pv=false (load)", "share_of_circuits"] == pytest.approx(90_000 / 171_411)
    assert row.loc["is_pv=false (load)", "n_rows_in_ts"] == 8_000
    assert row.loc["is_pv=false (load)", "share_of_ts_rows"] == pytest.approx(0.8)


def test_shares_handle_string_booleans_defensively():
    circuit_counts = pd.DataFrame({"is_pv": ["true", "false"], "n_circuits": [5, 5], "n_sites": [3, 3]})
    out = S.compare_circuit_and_row_shares(circuit_counts, ts_rows_false=1, ts_rows_true=1)
    assert set(out.label) == {"is_pv=true (pv)", "is_pv=false (load)"}


def test_an_unrecognised_is_pv_value_gets_its_own_row_not_dropped():
    """A null or malformed is_pv value is a data-quality signal, not noise to discard."""
    circuit_counts = pd.DataFrame({"is_pv": [True, None], "n_circuits": [10, 2], "n_sites": [5, 1]})
    out = S.compare_circuit_and_row_shares(circuit_counts, ts_rows_false=1, ts_rows_true=1)
    assert len(out) == 2
    assert any("unknown" in label for label in out.label)


def test_empty_input_gives_empty_output():
    assert S.compare_circuit_and_row_shares(pd.DataFrame(), 1, 1).empty
    assert S.compare_circuit_and_row_shares(None, 1, 1).empty


def test_shares_of_circuits_sum_to_one():
    circuit_counts = pd.DataFrame({"is_pv": [True, False], "n_circuits": [30, 70], "n_sites": [1, 1]})
    out = S.compare_circuit_and_row_shares(circuit_counts, ts_rows_false=1, ts_rows_true=1)
    assert out.share_of_circuits.sum() == pytest.approx(1.0)


# ── build_comparison_table ──────────────────────────────────────────────────

def _candidate(**overrides) -> S.SourceCandidate:
    defaults = dict(
        name="test", grain="circuit", has_load_signal=True, has_pv_signal=True,
        decomposable=True, n_rows=1000, size_bytes=1024 ** 3,
    )
    defaults.update(overrides)
    return S.SourceCandidate(**defaults)


def test_comparison_table_has_one_row_per_candidate():
    table = S.build_comparison_table([_candidate(name="a"), _candidate(name="b")])
    assert list(table.candidate) == ["a", "b"]


def test_comparison_table_computes_cost_from_size():
    table = S.build_comparison_table([_candidate(size_bytes=1024 ** 4)])  # exactly 1 TB
    from bms_sa_review.synthetic_ami_creation.config import ami_config as C
    assert table.loc[0, "full_scan_cost_aud"] == pytest.approx(C.ATHENA_PRICE_PER_TB, abs=0.01)


def test_comparison_table_handles_missing_size_without_raising():
    table = S.build_comparison_table([_candidate(size_bytes=None, n_rows=None)])
    assert table.loc[0, "full_scan_cost_aud"] is None
    assert table.loc[0, "size"] == ""


def test_comparison_table_preserves_boolean_columns_for_recommend_to_use():
    table = S.build_comparison_table([_candidate(has_load_signal=False)])
    assert table.loc[0, "has_load_signal"] == False  # noqa: E712 -- checking the literal value


# ── recommend: the rule that must never be fooled by cost or cleanliness ──

def test_the_only_qualifying_candidate_is_recommended():
    circuit_level = _candidate(name="ts", has_load_signal=True, has_pv_signal=True, decomposable=True)
    site_level = _candidate(name="structured_data", has_load_signal=False, decomposable=False)
    verdict = S.recommend([circuit_level, site_level])
    assert verdict["qualifying"] == ["ts"]
    assert "no load signal" in verdict["excluded"]["structured_data"]
    assert "not decomposable" in verdict["excluded"]["structured_data"]


def test_a_cheap_clean_but_disqualified_candidate_is_never_recommended():
    """
    The rule this test exists to enforce: cost/cleanliness must NEVER let a
    disqualified candidate outrank a qualifying one. A candidate that is cheap
    and clean but missing a signal is still excluded.
    """
    expensive_but_qualifying = _candidate(
        name="ts", has_load_signal=True, has_pv_signal=True, decomposable=True,
        size_bytes=1024 ** 4 * 10,  # 10 TB -- deliberately huge
    )
    cheap_but_disqualified = _candidate(
        name="structured_data", has_load_signal=False, has_pv_signal=True,
        decomposable=False, size_bytes=1024,  # 1 KB -- deliberately tiny
        cleanliness_notes="pristine",
    )
    verdict = S.recommend([expensive_but_qualifying, cheap_but_disqualified])
    assert verdict["qualifying"] == ["ts"]
    assert "structured_data" in verdict["excluded"]


def test_zero_qualifying_candidates_is_reported_not_guessed():
    verdict = S.recommend([
        _candidate(name="a", has_load_signal=False),
        _candidate(name="b", has_pv_signal=False),
    ])
    assert verdict["qualifying"] == []
    assert len(verdict["excluded"]) == 2


def test_multiple_qualifying_candidates_are_all_returned():
    verdict = S.recommend([_candidate(name="a"), _candidate(name="b")])
    assert verdict["qualifying"] == ["a", "b"]


def test_excluded_reason_lists_every_missing_criterion():
    verdict = S.recommend([_candidate(has_load_signal=False, has_pv_signal=False, decomposable=False)])
    reason = list(verdict["excluded"].values())[0]
    assert "no load signal" in reason
    assert "no PV signal" in reason
    assert "not decomposable" in reason
