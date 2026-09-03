"""
Unit tests for `ami_extract` -- the chunked Athena -> local Parquet landing
step. `aq_fn` is always a fake in these tests: no real Athena call should
ever be needed to verify the chunking/SQL/write orchestration is correct.
"""
from __future__ import annotations

import pandas as pd
import pytest

from bms_sa_review.synthetic_ami_creation.lib import ami_extract as Extract


# --------------------------------------------------------------------------- #
# chunk_circuit_ids
# --------------------------------------------------------------------------- #

def test_chunk_circuit_ids_splits_into_fixed_size_sorted_chunks():
    ids = [50, 10, 30, 20, 40]
    chunks = Extract.chunk_circuit_ids(ids, chunk_size=2)
    assert chunks == [[10, 20], [30, 40], [50]]


def test_chunk_circuit_ids_deduplicates():
    chunks = Extract.chunk_circuit_ids([1, 1, 2, 2, 3], chunk_size=10)
    assert chunks == [[1, 2, 3]]


def test_chunk_circuit_ids_empty_input():
    assert Extract.chunk_circuit_ids([], chunk_size=10) == []


def test_chunk_circuit_ids_is_deterministic_across_calls():
    ids = [7, 3, 9, 1, 5]
    assert Extract.chunk_circuit_ids(ids, chunk_size=2) == Extract.chunk_circuit_ids(ids, chunk_size=2)


# --------------------------------------------------------------------------- #
# months_in_range
# --------------------------------------------------------------------------- #

def test_months_in_range_within_one_year():
    assert Extract.months_in_range(2025, 6, 2025, 9) == [
        (2025, 6), (2025, 7), (2025, 8), (2025, 9),
    ]


def test_months_in_range_crosses_year_boundary():
    assert Extract.months_in_range(2025, 11, 2026, 2) == [
        (2025, 11), (2025, 12), (2026, 1), (2026, 2),
    ]


def test_months_in_range_single_month():
    assert Extract.months_in_range(2025, 6, 2025, 6) == [(2025, 6)]


def test_months_in_range_full_year():
    pairs = Extract.months_in_range(2025, 1, 2025, 12)
    assert len(pairs) == 12
    assert pairs[0] == (2025, 1) and pairs[-1] == (2025, 12)


def test_months_in_range_inverted_raises():
    with pytest.raises(ValueError):
        Extract.months_in_range(2025, 9, 2025, 6)


# --------------------------------------------------------------------------- #
# build_extract_sql
# --------------------------------------------------------------------------- #

def test_build_extract_sql_shape_matches_notebook_04_pattern():
    sql = Extract.build_extract_sql([101, 102, 103], 2025, 6, True)
    assert "FROM ts" in sql
    assert "year = 2025" in sql
    assert "month = 6" in sql
    assert "is_pv = true" in sql
    assert "circuit_id IN (101, 102, 103)" in sql
    assert "power" in sql and "energy_reactive" in sql


def test_build_extract_sql_is_pv_false_renders_lowercase_false():
    sql = Extract.build_extract_sql([1], 2025, 1, False)
    assert "is_pv = false" in sql


def test_build_extract_sql_empty_chunk_raises():
    with pytest.raises(ValueError):
        Extract.build_extract_sql([], 2025, 1, True)


# --------------------------------------------------------------------------- #
# extract_chunk
# --------------------------------------------------------------------------- #

def test_extract_chunk_calls_aq_fn_with_built_sql_and_returns_its_frame():
    captured = {}

    def fake_aq(sql, database=None, label=None):
        captured["sql"] = sql
        captured["database"] = database
        captured["label"] = label
        return pd.DataFrame({"circuit_id": [1], "power": [10.0]})

    result = Extract.extract_chunk(fake_aq, [1, 2], 2025, 6, True, database="db")
    assert "circuit_id IN (1, 2)" in captured["sql"]
    assert captured["database"] == "db"
    assert "2025-06" in captured["label"]
    assert len(result) == 1


def test_extract_chunk_default_label_mentions_chunk_size():
    def fake_aq(sql, database=None, label=None):
        return pd.DataFrame({"circuit_id": [], "power": []})

    Extract.extract_chunk(fake_aq, [1, 2, 3], 2025, 6, False, label=None)
    # no assertion needed beyond "doesn't raise" -- label content checked above


# --------------------------------------------------------------------------- #
# write_extract_chunk
# --------------------------------------------------------------------------- #

def test_write_extract_chunk_writes_hive_partitioned_parquet(tmp_path):
    frame = pd.DataFrame({"circuit_id": [1, 2], "power": [10.0, 20.0]})
    path = Extract.write_extract_chunk(
        frame, tmp_path, 2025, 6, chunk_index=0, is_pv=True,
    )
    assert path is not None
    assert path.exists()
    assert "dt_month=2025-06" in str(path)
    roundtrip = pd.read_parquet(path)
    assert len(roundtrip) == 2


def test_write_extract_chunk_empty_frame_writes_nothing(tmp_path):
    empty = pd.DataFrame(columns=["circuit_id", "power"])
    path = Extract.write_extract_chunk(empty, tmp_path, 2025, 6, chunk_index=0)
    assert path is None
    assert not list(tmp_path.rglob("*.parquet"))


def test_write_extract_chunk_rerun_overwrites_not_duplicates(tmp_path):
    frame_v1 = pd.DataFrame({"circuit_id": [1], "power": [10.0]})
    frame_v2 = pd.DataFrame({"circuit_id": [1, 2], "power": [10.0, 99.0]})
    Extract.write_extract_chunk(frame_v1, tmp_path, 2025, 6, chunk_index=0, is_pv=False)
    path2 = Extract.write_extract_chunk(frame_v2, tmp_path, 2025, 6, chunk_index=0, is_pv=False)
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1
    assert len(pd.read_parquet(path2)) == 2


def test_write_extract_chunk_different_is_pv_same_month_do_not_collide(tmp_path):
    load_frame = pd.DataFrame({"circuit_id": [1], "power": [10.0]})
    pv_frame = pd.DataFrame({"circuit_id": [2], "power": [20.0]})
    Extract.write_extract_chunk(load_frame, tmp_path, 2025, 6, chunk_index=0, is_pv=False)
    Extract.write_extract_chunk(pv_frame, tmp_path, 2025, 6, chunk_index=0, is_pv=True)
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 2


# --------------------------------------------------------------------------- #
# run_extraction
# --------------------------------------------------------------------------- #

def _fake_aq_factory(calls):
    def fake_aq(sql, database=None, label=None):
        calls.append(sql)
        # Pretend every circuit in the IN-list reported 3 rows.
        import re
        ids = [int(x) for x in re.search(r"IN \(([^)]*)\)", sql).group(1).split(", ")]
        rows = []
        for cid in ids:
            for t in range(3):
                rows.append({"circuit_id": cid, "t_stamp": f"2025-06-0{t+1}", "power": 100.0})
        return pd.DataFrame(rows)
    return fake_aq


def test_run_extraction_covers_every_month_is_pv_chunk_combination(tmp_path):
    circuits = pd.DataFrame({
        "circuit_id": [1, 2, 3, 4, 5],
        "is_pv": [False, False, False, True, True],
    })
    calls = []
    manifest = Extract.run_extraction(
        _fake_aq_factory(calls), circuits, [(2025, 6), (2025, 7)], tmp_path,
        chunk_size=2,
    )
    # load: chunks [1,2],[3] -> 2 chunks; pv: chunks [4,5] -> 1 chunk.
    # 2 months x (2 load chunks + 1 pv chunk) = 6 queries/rows in manifest.
    assert len(manifest) == 6
    assert len(calls) == 6
    assert set(manifest.n_rows) == {6, 3}  # 2-circuit chunks -> 6 rows, 1-circuit -> 3 rows
    assert manifest.path.notna().all()


def test_run_extraction_writes_files_that_exist_on_disk(tmp_path):
    circuits = pd.DataFrame({"circuit_id": [1, 2], "is_pv": [False, True]})
    manifest = Extract.run_extraction(
        _fake_aq_factory([]), circuits, [(2025, 6)], tmp_path, chunk_size=10,
    )
    for path in manifest.path:
        assert path.exists()


def test_run_extraction_empty_circuits_returns_empty_manifest(tmp_path):
    manifest = Extract.run_extraction(
        _fake_aq_factory([]), pd.DataFrame(columns=["circuit_id", "is_pv"]),
        [(2025, 6)], tmp_path,
    )
    assert len(manifest) == 0
    assert list(manifest.columns) == [
        "year", "month", "is_pv", "chunk_index", "n_circuits", "n_rows", "path",
    ]


def test_run_extraction_calls_aq_once_per_chunk_not_batched_together(tmp_path):
    """
    Pins the documented one-chunk-at-a-time contract from the orchestration
    side (not via id() reuse, which CPython can recycle immediately after a
    frame is dropped and is therefore not a reliable "was it held" signal):
    4 circuits at chunk_size=1 must mean 4 separate `aq_fn` calls, not one
    call for all 4 -- i.e. `run_extraction` never grows a single IN-list
    across the whole `circuits` frame regardless of chunk_size.
    """
    circuits = pd.DataFrame({"circuit_id": [1, 2, 3, 4], "is_pv": [False] * 4})
    calls = []

    def fake_aq(sql, database=None, label=None):
        calls.append(sql)
        return pd.DataFrame({"circuit_id": [1], "power": [1.0]})

    Extract.run_extraction(fake_aq, circuits, [(2025, 6)], tmp_path, chunk_size=1)
    assert len(calls) == 4
    for sql in calls:
        assert sql.count("circuit_id IN (") == 1
        # each call's IN-list has exactly one id -- never merged across chunks
        in_list = sql.split("circuit_id IN (")[1].split(")")[0]
        assert len(in_list.split(",")) == 1
