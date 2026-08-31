from __future__ import annotations

from datetime import datetime, timedelta

from dnsp_analysis.schemas import utc_to_local


def test_sydney_dst_is_applied_in_january() -> None:
    local = utc_to_local(datetime(2025, 1, 15, 0, 0), "Australia/Sydney")
    assert local.utcoffset() == timedelta(hours=11)
    assert local.hour == 11


def test_sydney_standard_time_is_applied_in_july() -> None:
    local = utc_to_local(datetime(2025, 7, 15, 0, 0), "Australia/Sydney")
    assert local.utcoffset() == timedelta(hours=10)
    assert local.hour == 10

