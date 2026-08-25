"""
The pure arithmetic and SQL parsing in `ami_athena`.

Nothing here touches AWS. Everything here is logic that, if wrong, produces a
plausible-looking number or silently lets an expensive query through -- which is
exactly the class of defect a notebook cell will not catch.
"""

from __future__ import annotations

import pytest

from bms_sa_review.ami_data_analysis.config import ami_config as C
from bms_sa_review.ami_data_analysis.lib import ami_athena as A


# ── byte formatting ────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (0, "0 B"),
    (512, "512 B"),
    (1024, "1.00 KB"),
    (1024 ** 2, "1.00 MB"),
    (1024 ** 3, "1.00 GB"),
    (1024 ** 4, "1.00 TB"),
    (5 * 1024 ** 4, "5.00 TB"),
    (None, "?"),
])
def test_fmt_bytes(value, expected):
    assert A.fmt_bytes(value) == expected


def test_fmt_bytes_is_binary_not_decimal():
    """1e9 bytes is 0.93 GiB, not 1 GB. Getting this wrong understates a bill."""
    assert A.fmt_bytes(1_000_000_000).startswith("953")


# ── cost ───────────────────────────────────────────────────────────────────

def test_billed_bytes_applies_the_10mb_minimum():
    assert A.billed_bytes(0) == C.ATHENA_MIN_SCAN_BYTES
    assert A.billed_bytes(1) == C.ATHENA_MIN_SCAN_BYTES
    assert A.billed_bytes(50 * 1024 ** 2) == 50 * 1024 ** 2
    assert A.billed_bytes(None) is None


def test_estimate_cost_of_one_tb_is_the_headline_price():
    assert A.estimate_cost(1024 ** 4) == pytest.approx(C.ATHENA_PRICE_PER_TB)


def test_estimate_cost_can_ignore_the_minimum():
    tiny = A.estimate_cost(1, apply_minimum=False)
    assert tiny is not None and tiny < A.estimate_cost(1)


def test_estimate_cost_of_none_is_none():
    assert A.estimate_cost(None) is None


# ── SQL noise stripping ────────────────────────────────────────────────────

def test_strip_noise_removes_comments_and_literals():
    sql = """
        SELECT a  -- year = 2025 in a comment
        /* month = 1 in a block comment */
        FROM t WHERE s = 'is_pv = True'
    """
    stripped = A._strip_noise(sql)
    assert "comment" not in stripped
    assert "is_pv" not in stripped
    assert "SELECT a" in stripped


def test_strip_noise_handles_doubled_quotes():
    stripped = A._strip_noise("SELECT 'it''s year 2025' AS x FROM t")
    assert "2025" not in stripped
    assert "FROM t" in stripped


# ── table reference extraction ─────────────────────────────────────────────

@pytest.mark.parametrize("sql,expected", [
    ("SELECT * FROM ts", ["ts"]),
    ("SELECT * FROM solar_analytics_iceberg.ts", ["ts"]),
    ('SELECT * FROM "ts"', ["ts"]),
    ('SELECT * FROM "ts$partitions"', ["ts$partitions"]),
    ("SELECT * FROM a JOIN b ON a.x = b.x", ["a", "b"]),
    ("SELECT * FROM ts JOIN meta_up23c m ON ts.circuit_id = m.circuit_id",
     ["ts", "meta_up23c"]),
    ("SELECT * FROM (SELECT 1) x", []),
])
def test_referenced_tables(sql, expected):
    assert A._referenced_tables(sql) == expected


def test_referenced_tables_ignores_table_names_inside_strings():
    assert A._referenced_tables("SELECT 'from ts' AS note FROM circuits") == ["circuits"]


# ── the partition guard ────────────────────────────────────────────────────

def test_guard_blocks_a_bare_scan_of_ts():
    problems = A.check_partition_filters("SELECT * FROM ts")
    assert problems and "ts" in problems[0]


def test_guard_blocks_an_aggregate_with_no_partition_predicate():
    assert A.check_partition_filters("SELECT count(*) FROM ts")


def test_guard_blocks_a_limit_only_query():
    """LIMIT does not bound the scan once there is a join or an aggregate."""
    assert A.check_partition_filters("SELECT * FROM ts LIMIT 5")


@pytest.mark.parametrize("sql", [
    "SELECT * FROM ts WHERE year = 2025 AND month = 1 LIMIT 5",
    "SELECT * FROM ts WHERE is_pv = True AND year = 2025",
    "SELECT * FROM ts WHERE t_stamp >= TIMESTAMP '2025-01-01 00:00:00'",
    "SELECT * FROM ts WHERE t_stamp BETWEEN a AND b",
])
def test_guard_passes_a_partitioned_query(sql):
    assert A.check_partition_filters(sql) == []


def test_guard_exempts_iceberg_metadata_tables():
    assert A.check_partition_filters('SELECT * FROM "ts$partitions"') == []
    assert A.check_partition_filters('SELECT * FROM "ts$files"') == []


def test_guard_ignores_small_tables():
    assert A.check_partition_filters("SELECT * FROM circuits") == []
    assert A.check_partition_filters("SELECT * FROM meta_up23c") == []


def test_guard_catches_the_bom_table():
    """bom_nci.solar is the whole Himawari disc; se_bom learned this the hard way."""
    assert A.check_partition_filters("SELECT * FROM bom_nci.solar")


def test_guard_is_not_fooled_by_a_partition_word_in_a_comment():
    assert A.check_partition_filters("SELECT * FROM ts -- year = 2025")


def test_guard_covers_the_structured_data_tables():
    for table in ("structured_data_v2_flex_included", "all_uncurtailedpv_v2_flex_included"):
        assert A.check_partition_filters(f"SELECT * FROM {table}"), table


def test_assert_partition_filters_raises_with_a_usable_message():
    with pytest.raises(ValueError, match="allow_full_scan"):
        A.assert_partition_filters("SELECT * FROM ts")


def test_assert_partition_filters_is_silent_when_safe():
    A.assert_partition_filters("SELECT * FROM ts WHERE year = 2025 AND month = 1")


# ── scan metadata extraction ───────────────────────────────────────────────

class _Frame:
    """Stand-in for the DataFrame awswrangler decorates with query_metadata."""
    def __init__(self, metadata=None):
        if metadata is not None:
            self.query_metadata = metadata


def test_scan_bytes_reads_the_raw_payload():
    frame = _Frame({"Statistics": {"DataScannedInBytes": 12345}})
    assert A._scan_bytes_from_frame(frame) == (12345.0, "query_metadata")


def test_scan_bytes_reads_a_nested_payload():
    frame = _Frame({"QueryExecution": {"Statistics": {"DataScannedInBytes": 7}}})
    assert A._scan_bytes_from_frame(frame) == (7.0, "query_metadata")


@pytest.mark.parametrize("metadata", [
    None,                                   # attribute absent entirely
    {},                                     # present but empty
    {"Statistics": {}},                     # no scan figure
    {"Statistics": "not a dict"},
    "a string",
])
def test_scan_bytes_degrades_instead_of_raising(metadata):
    assert A._scan_bytes_from_frame(_Frame(metadata)) == (None, "unavailable")


# ── the scan log ───────────────────────────────────────────────────────────

def test_scan_report_is_empty_and_typed_before_any_query():
    A.reset_scan_log()
    report = A.scan_report(verbose=False)
    assert report.empty
    assert "scanned_bytes" in report.columns


def test_scan_report_totals_and_flags_unknowns():
    A.reset_scan_log()
    A.SCAN_LOG.append(A.ScanRecord("a", "db", 1, 0.1, 1024 ** 3, "query_metadata"))
    A.SCAN_LOG.append(A.ScanRecord("b", "db", 1, 0.1, None, "unavailable"))
    report = A.scan_report(verbose=False)
    assert len(report) == 2
    assert report.scanned_bytes.sum() == 1024 ** 3
    assert report.scanned_bytes.isna().sum() == 1
    # The report rounds to 4 decimal places for display; assert against that
    # rather than the raw figure, so a change to either one shows up here.
    assert report.loc[0, "cost"] == round(A.estimate_cost(1024 ** 3), 4)
    assert A.estimate_cost(1024 ** 3) == pytest.approx(C.ATHENA_PRICE_PER_TB / 1024)
    A.reset_scan_log()
