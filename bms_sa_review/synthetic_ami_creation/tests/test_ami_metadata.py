"""
Unit tests for `ami_metadata` -- the two Phase 6 dimension tables
(`ami_site_metadata`, `ami_circuit_metadata`) built from `meta_up23c` and
scoped to whatever is actually landed in the local Parquet store.

`aq_fn` is always a fake here (mirrors `test_ami_extract.py`'s convention):
no real Athena call should ever be needed to verify the query-building,
chunking, or aggregation logic is correct. `determine_landed_scope`/
`determine_landed_circuit_scope` are tested against small real Parquet
fixtures via a real in-memory `duckdb.connect()`, since their whole job is
to read real files correctly.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from bms_sa_review.synthetic_ami_creation.lib import ami_metadata as Meta


# --------------------------------------------------------------------------- #
# build_site_level_query / build_circuit_level_query
# --------------------------------------------------------------------------- #

def test_build_site_level_query_sorts_dedupes_and_lists_columns():
    sql = Meta.build_site_level_query([30, 10, 10, 20])
    assert "WHERE site_id IN (10,20,30)" in sql
    for col in Meta.SITE_LEVEL_COLUMNS:
        assert col in sql
    assert "circuit_id" not in sql


def test_build_circuit_level_query_sorts_dedupes_and_lists_columns():
    sql = Meta.build_circuit_level_query([300, 100, 100, 200])
    assert "WHERE circuit_id IN (100,200,300)" in sql
    for col in Meta.CIRCUIT_LEVEL_COLUMNS:
        assert col in sql


# --------------------------------------------------------------------------- #
# pull_site_level_meta / pull_circuit_level_meta
# --------------------------------------------------------------------------- #

def test_pull_site_level_meta_chunks_and_concatenates():
    calls = []

    def fake_aq(sql, database=None, label=None):
        calls.append(sql)
        site_id = int(sql.split("IN (")[1].split(")")[0].split(",")[0])
        return pd.DataFrame({col: [None] for col in Meta.SITE_LEVEL_COLUMNS} | {"site_id": [site_id]})

    result = Meta.pull_site_level_meta(fake_aq, [1, 2, 3], chunk_size=1)
    assert len(calls) == 3
    assert sorted(result.site_id) == [1, 2, 3]


def test_pull_site_level_meta_deduplicates_on_site_id():
    def fake_aq(sql, database=None, label=None):
        return pd.DataFrame({"site_id": [1, 1], "state": ["NSW", "NSW"]})

    result = Meta.pull_site_level_meta(fake_aq, [1], chunk_size=800)
    assert len(result) == 1


def test_pull_site_level_meta_empty_input_makes_no_calls():
    calls = []

    def fake_aq(sql, database=None, label=None):
        calls.append(sql)
        return pd.DataFrame()

    result = Meta.pull_site_level_meta(fake_aq, [], chunk_size=800)
    assert calls == []
    assert list(result.columns) == list(Meta.SITE_LEVEL_COLUMNS)
    assert len(result) == 0


def test_pull_circuit_level_meta_chunks_and_deduplicates():
    calls = []

    def fake_aq(sql, database=None, label=None):
        calls.append(sql)
        return pd.DataFrame({"circuit_id": [10, 10], "site_id": [1, 1]})

    result = Meta.pull_circuit_level_meta(fake_aq, [10], chunk_size=800)
    assert len(calls) == 1
    assert len(result) == 1


def test_pull_circuit_level_meta_empty_input_makes_no_calls():
    calls = []

    def fake_aq(sql, database=None, label=None):
        calls.append(sql)
        return pd.DataFrame()

    result = Meta.pull_circuit_level_meta(fake_aq, [], chunk_size=800)
    assert calls == []
    assert list(result.columns) == list(Meta.CIRCUIT_LEVEL_COLUMNS)
    assert len(result) == 0


# --------------------------------------------------------------------------- #
# determine_landed_scope / determine_landed_circuit_scope
# --------------------------------------------------------------------------- #

def _write_partitioned(tmp_path, name, rows):
    frame = pd.DataFrame(rows)
    for (year, month), part in frame.groupby(["year", "month"]):
        d = tmp_path / name / f"dt_month={year:04d}-{month:02d}"
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / "part.parquet", index=False)


def test_determine_landed_scope_reads_real_parquet(tmp_path):
    _write_partitioned(tmp_path, "ami_raw", [
        {"site_id": 1, "year": 2025, "month": 1},
        {"site_id": 2, "year": 2025, "month": 1},
    ])
    _write_partitioned(tmp_path, "ami_meter", [
        {"site_id": 2, "year": 2025, "month": 1},
        {"site_id": 3, "year": 2025, "month": 1},
    ])
    con = duckdb.connect()
    table_paths = {
        "ami_raw": (tmp_path / "ami_raw" / "dt_month=*" / "*.parquet").as_posix(),
        "ami_meter": (tmp_path / "ami_meter" / "dt_month=*" / "*.parquet").as_posix(),
        "ami_missing": (tmp_path / "ami_missing" / "dt_month=*" / "*.parquet").as_posix(),
    }
    result = Meta.determine_landed_scope(con, table_paths)
    assert result["ami_raw"] == {1, 2}
    assert result["ami_meter"] == {2, 3}
    assert result["ami_missing"] == set()


def test_determine_landed_circuit_scope_unions_across_tables(tmp_path):
    _write_partitioned(tmp_path, "ami_meter", [
        {"circuit_id": 10, "year": 2025, "month": 1},
        {"circuit_id": 20, "year": 2025, "month": 1},
    ])
    _write_partitioned(tmp_path, "ami_raw_phaseseparate", [
        {"circuit_id": 20, "year": 2025, "month": 1},
        {"circuit_id": 30, "year": 2025, "month": 1},
    ])
    con = duckdb.connect()
    table_paths = {
        "ami_meter": (tmp_path / "ami_meter" / "dt_month=*" / "*.parquet").as_posix(),
        "ami_raw_phaseseparate": (tmp_path / "ami_raw_phaseseparate" / "dt_month=*" / "*.parquet").as_posix(),
    }
    result = Meta.determine_landed_circuit_scope(con, table_paths)
    assert result == {10, 20, 30}


# --------------------------------------------------------------------------- #
# build_ami_site_metadata
# --------------------------------------------------------------------------- #

def _site_meta(rows):
    return pd.DataFrame(rows)


def _circuit_meta(rows):
    return pd.DataFrame(rows)


def test_build_ami_site_metadata_derives_phase_counts_and_capacity():
    site_meta = _site_meta([
        {"site_id": 1, "state": "NSW", "dnsp_name": "Ausgrid"},
    ])
    circuit_meta = _circuit_meta([
        {"site_id": 1, "circuit_id": 10, "circuit_type": "ac_load_net", "s_99": 5.0,
         "min_time": pd.Timestamp("2024-01-01"), "max_time": pd.Timestamp("2025-06-01")},
        {"site_id": 1, "circuit_id": 20, "circuit_type": "ac_load_net", "s_99": 4.0,
         "min_time": pd.Timestamp("2024-02-01"), "max_time": pd.Timestamp("2025-05-01")},
        {"site_id": 1, "circuit_id": 11, "circuit_type": "pv_site_net", "s_99": 7.5,
         "min_time": pd.Timestamp("2024-01-15"), "max_time": pd.Timestamp("2025-07-01")},
    ])
    result = Meta.build_ami_site_metadata(site_meta, circuit_meta)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.n_load_phases == 2
    assert row.n_pv_phases == 1
    assert row.s_99 == pytest.approx(7.5)
    assert row.first_seen == pd.Timestamp("2024-01-01")
    assert row.last_seen == pd.Timestamp("2025-07-01")


def test_build_ami_site_metadata_adds_presence_flags():
    site_meta = _site_meta([{"site_id": 1}, {"site_id": 2}])
    circuit_meta = _circuit_meta([])
    presence = {"ami_raw": {1}, "ami_meter": {1, 2}}
    result = Meta.build_ami_site_metadata(site_meta, circuit_meta, presence).set_index("site_id")
    assert bool(result.loc[1, "in_ami_raw"]) is True
    assert bool(result.loc[2, "in_ami_raw"]) is False
    assert bool(result.loc[1, "in_ami_meter"]) is True
    assert bool(result.loc[2, "in_ami_meter"]) is True


def test_build_ami_site_metadata_site_with_no_circuit_detail_still_gets_a_row():
    site_meta = _site_meta([{"site_id": 1}, {"site_id": 2}])
    circuit_meta = _circuit_meta([
        {"site_id": 1, "circuit_id": 10, "circuit_type": "ac_load_net", "s_99": 5.0,
         "min_time": pd.Timestamp("2024-01-01"), "max_time": pd.Timestamp("2025-01-01")},
    ])
    result = Meta.build_ami_site_metadata(site_meta, circuit_meta).set_index("site_id")
    assert result.loc[2, "n_load_phases"] == 0
    assert result.loc[2, "n_pv_phases"] == 0
    assert pd.isna(result.loc[2, "s_99"])


def test_build_ami_site_metadata_deduplicates_site_meta():
    site_meta = _site_meta([{"site_id": 1}, {"site_id": 1}])
    result = Meta.build_ami_site_metadata(site_meta, _circuit_meta([]))
    assert len(result) == 1


def test_build_ami_site_metadata_empty_site_meta():
    result = Meta.build_ami_site_metadata(pd.DataFrame(), pd.DataFrame())
    assert len(result) == 0
    for col in ("n_load_phases", "n_pv_phases", "s_99", "first_seen", "last_seen"):
        assert col in result.columns


# --------------------------------------------------------------------------- #
# build_ami_circuit_metadata
# --------------------------------------------------------------------------- #

def test_build_ami_circuit_metadata_selects_orders_and_deduplicates():
    circuit_meta = _circuit_meta([
        {"site_id": 2, "circuit_id": 20, "circuit_type": "ac_load_net", "extra_col": "drop_me"},
        {"site_id": 1, "circuit_id": 10, "circuit_type": "ac_load_net", "extra_col": "drop_me"},
        {"site_id": 1, "circuit_id": 10, "circuit_type": "ac_load_net", "extra_col": "dup"},
    ])
    result = Meta.build_ami_circuit_metadata(circuit_meta)
    assert list(result.circuit_id) == [10, 20]
    assert "extra_col" not in result.columns
    assert len(result) == 2


def test_build_ami_circuit_metadata_empty_input():
    result = Meta.build_ami_circuit_metadata(pd.DataFrame())
    assert len(result) == 0
    assert list(result.columns) == list(Meta.CIRCUIT_LEVEL_COLUMNS)


# --------------------------------------------------------------------------- #
# write_metadata_table
# --------------------------------------------------------------------------- #

def test_write_metadata_table_writes_single_parquet_file(tmp_path):
    frame = pd.DataFrame({"site_id": [1, 2], "state": ["NSW", "VIC"]})
    out_path = tmp_path / "nested" / "ami_site_metadata.parquet"
    result_path = Meta.write_metadata_table(frame, out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert out_path.is_file()
    round_trip = pd.read_parquet(out_path)
    assert sorted(round_trip.site_id) == [1, 2]
