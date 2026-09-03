"""
The pure frame handling in `ami_inventory`.

`$partitions` comes back in one of three shapes depending on engine version and
table format, and the count columns are named differently between them. The
normaliser has to cope with all of it and, where it cannot, return fewer columns
rather than raise. These tests are the synthetic frames it was written against.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bms_sa_review.synthetic_ami_creation.lib import ami_inventory as I


# ── synthetic $partitions frames ───────────────────────────────────────────

def iceberg_struct_frame() -> pd.DataFrame:
    """Athena Iceberg `$partitions`, partition column returned as a dict."""
    return pd.DataFrame({
        "partition": [
            {"year": 2025, "month": 1, "is_pv": True},
            {"year": 2025, "month": 1, "is_pv": False},
            {"year": 2025, "month": 2, "is_pv": True},
        ],
        "record_count": [1_000_000, 250_000, 900_000],
        "file_count": [10, 4, 9],
        "total_size": [50_000_000, 12_000_000, 45_000_000],
    })


def iceberg_string_frame() -> pd.DataFrame:
    """Same, but the struct arrived as its string rendering."""
    return pd.DataFrame({
        "partition": [
            "{year=2025, month=1, is_pv=true}",
            "{year=2025, month=2, is_pv=true}",
        ],
        "record_count": [1_000_000, 900_000],
        "file_count": [10, 9],
        "total_size": [50_000_000, 45_000_000],
    })


def hive_frame() -> pd.DataFrame:
    """Hive `$partitions`: values only, no counts."""
    return pd.DataFrame({
        "year": [2024, 2024, 2025],
        "month": [11, 12, 1],
    })


def alternate_names_frame() -> pd.DataFrame:
    """Engine variant with different column names for the same things."""
    return pd.DataFrame({
        "year": [2025], "month": [3],
        "row_count": [42], "data_file_count": [2],
        "file_size_in_bytes": [1024],
    })


# ── normalise_partitions ───────────────────────────────────────────────────

def test_normalises_a_dict_struct():
    tidy = I.normalise_partitions(iceberg_struct_frame())
    assert set(["year", "month", "is_pv", "n_rows", "n_files", "size_bytes"]) <= set(tidy.columns)
    assert tidy.n_rows.sum() == 2_150_000
    assert list(tidy.year.unique()) == [2025]


def test_normalises_a_stringified_struct():
    tidy = I.normalise_partitions(iceberg_string_frame())
    assert list(tidy.month) == [1, 2]
    assert tidy.n_rows.sum() == 1_900_000
    # 'true' survives as a string; it is a value to display, not to compute with.
    assert set(tidy.is_pv) == {"true"}


def test_normalises_a_hive_frame_without_counts():
    tidy = I.normalise_partitions(hive_frame())
    assert list(tidy.columns) == ["year", "month"]
    assert len(tidy) == 3
    assert "n_rows" not in tidy.columns


def test_normalises_alternate_column_names():
    tidy = I.normalise_partitions(alternate_names_frame())
    assert tidy.n_rows.iloc[0] == 42
    assert tidy.n_files.iloc[0] == 2
    assert tidy.size_bytes.iloc[0] == 1024


def test_sorts_by_partition_key():
    frame = pd.DataFrame({"year": [2025, 2024], "month": [1, 12], "record_count": [1, 2]})
    tidy = I.normalise_partitions(frame)
    assert list(zip(tidy.year, tidy.month)) == [(2024, 12), (2025, 1)]


def test_year_and_month_are_nullable_integers_not_floats():
    """A partition printed as '2025.0' in a summary table looks like a bug."""
    tidy = I.normalise_partitions(iceberg_struct_frame())
    assert str(tidy.year.dtype) == "Int64"
    assert str(tidy.month.dtype) == "Int64"


@pytest.mark.parametrize("frame", [None, pd.DataFrame()])
def test_empty_input_gives_empty_output(frame):
    assert I.normalise_partitions(frame).empty


def test_unrecognised_shape_does_not_raise():
    frame = pd.DataFrame({"something_else": [1, 2], "partition": [None, None]})
    I.normalise_partitions(frame)  # must not raise


# ── partition_totals ───────────────────────────────────────────────────────

def test_totals_from_a_full_iceberg_frame():
    totals = I.partition_totals(I.normalise_partitions(iceberg_struct_frame()))
    assert totals["n_partitions"] == 3
    assert totals["n_rows"] == 2_150_000
    assert totals["size_bytes"] == 107_000_000
    assert totals["first_partition"] == "2025-01"
    assert totals["last_partition"] == "2025-02"
    assert set(totals["is_pv_values"]) == {"True", "False"}


def test_totals_from_a_hive_frame_have_no_counts_but_do_have_coverage():
    totals = I.partition_totals(I.normalise_partitions(hive_frame()))
    assert totals["n_partitions"] == 3
    assert totals["n_rows"] is None
    assert totals["first_partition"] == "2024-11"
    assert totals["last_partition"] == "2025-01"


def test_totals_of_nothing():
    assert I.partition_totals(pd.DataFrame()) == {"n_partitions": 0, "partition_columns": []}


def test_is_pv_breakdown_is_what_phase_2_hangs_on():
    """
    If `ts` has no is_pv = False partitions there are no load circuits, and the
    whole exercise changes shape. Phase 1 must surface that, so the roll-up has
    to carry it.
    """
    totals = I.partition_totals(I.normalise_partitions(iceberg_struct_frame()))
    assert "is_pv_values" in totals


# ── summary_table ──────────────────────────────────────────────────────────

def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"database": "solar_analytics_iceberg", "table": "ts", "table_type": "EXTERNAL_TABLE",
         "format": "ICEBERG", "is_iceberg": True, "n_columns": 12,
         "partition_keys": "year, month, is_pv", "location": "s3://x/ts/", "updated": None},
        {"database": "solar_analytics_iceberg", "table": "circuits", "table_type": "EXTERNAL_TABLE",
         "format": "", "is_iceberg": False, "n_columns": 5,
         "partition_keys": "", "location": "s3://x/circuits/", "updated": None},
    ])


def test_summary_table_joins_totals_and_orders_by_size():
    totals = {
        "solar_analytics_iceberg.ts": I.partition_totals(
            I.normalise_partitions(iceberg_struct_frame())
        )
    }
    summary = I.summary_table(catalog_frame(), totals)
    assert list(summary.table) == ["ts", "circuits"]
    assert summary.loc[0, "fmt"] == "iceberg"
    assert summary.loc[0, "coverage"] == "2025-01 .. 2025-02"
    assert summary.loc[0, "size"].endswith("MB")


def test_summary_table_keeps_unprobed_tables_rather_than_dropping_them():
    summary = I.summary_table(catalog_frame(), {})
    assert len(summary) == 2
    assert summary.n_rows.isna().all()


def test_summary_table_of_an_empty_catalog():
    assert I.summary_table(pd.DataFrame(), {}).empty


# ── candidate shortlist ────────────────────────────────────────────────────

def test_shortlist_flags_but_never_drops():
    catalog = pd.DataFrame({
        "database": ["d"] * 3,
        "table": ["ts", "meta_up23c", "unrelated_thing"],
    })
    out = I.guess_candidates(catalog)
    assert len(out) == 3
    assert out.set_index("table").shortlisted.to_dict() == {
        "ts": True, "meta_up23c": True, "unrelated_thing": False
    }


# ── should_probe_partitions: the Iceberg-hides-partitioning-from-Glue bug ──
#
# Real failure this guards: `ts` is genuinely partitioned on (year, month,
# is_pv) per `aws_config.py`, but Glue's declared `PartitionKeys` for it came
# back empty (`entry.partition_keys == ""`) -- a known Iceberg-on-Glue quirk,
# where the partition spec lives in Iceberg's own metadata rather than in
# Glue's Hive-style field. `probe_partitions` used to gate on the declared
# keys, which meant `$partitions` for `ts` was never queried and every
# downstream cell that expected `ts` partition data got nothing.

def test_should_probe_an_iceberg_table_with_no_declared_keys():
    """This is exactly the `ts` case: Iceberg, Glue says "no partition keys"."""
    assert I.should_probe_partitions(is_iceberg=True, declared_partition_keys="") is True


def test_should_probe_an_iceberg_table_with_declared_keys_too():
    assert I.should_probe_partitions(is_iceberg=True, declared_partition_keys="year, month") is True


def test_should_not_probe_an_unpartitioned_hive_table():
    """A Hive table with no declared keys genuinely has no $partitions to read."""
    assert I.should_probe_partitions(is_iceberg=False, declared_partition_keys="") is False


def test_should_probe_a_partitioned_hive_table():
    assert I.should_probe_partitions(is_iceberg=False, declared_partition_keys="year") is True


def test_should_probe_treats_none_like_empty_string():
    assert I.should_probe_partitions(is_iceberg=False, declared_partition_keys=None) is False


# ── partition_totals reports the ACTUAL columns, not what Glue declared ────

def test_totals_report_actual_partition_columns():
    totals = I.partition_totals(I.normalise_partitions(iceberg_struct_frame()))
    assert totals["partition_columns"] == ["year", "month", "is_pv"]


def test_totals_of_nothing_still_has_the_key():
    """So callers can do `stats.get("partition_columns", [])` uniformly."""
    assert I.partition_totals(pd.DataFrame())["partition_columns"] == []


# ── summary_table surfaces a Glue-hidden partitioning rather than showing "" ─

def test_summary_table_shows_recovered_columns_when_glue_declares_none():
    catalog = pd.DataFrame([{
        "database": "solar_analytics_iceberg", "table": "ts", "table_type": "EXTERNAL_TABLE",
        "format": "ICEBERG", "is_iceberg": True, "n_columns": 12,
        "partition_keys": "",  # <- Glue declares nothing, exactly the real `ts` case
        "location": "s3://x/ts/", "updated": None,
    }])
    totals = {
        "solar_analytics_iceberg.ts": I.partition_totals(
            I.normalise_partitions(iceberg_struct_frame())
        )
    }
    summary = I.summary_table(catalog, totals)
    assert "year, month, is_pv" in summary.loc[0, "partitioned_by"]
    assert "not declared in Glue" in summary.loc[0, "partitioned_by"]
