"""
Deliverable D2: timestamp and DST resolution.

The tests that matter here are the cross-validation ones. `resolve_timestamp()`
(Python `zoneinfo`) and `ts_utc_sql()` (DuckDB ICU) are two independent
implementations of the same conversion; if they agree across the year and at both
2025 daylight-saving boundaries, the conversion is right.

A DST error at ingest would silently duplicate an hour every April and delete one
every October, and would never announce itself downstream. That is why this runs
before any data is rewritten.

Run:  pytest solar_edge/tests/test_se_ingest.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solar_edge.config import se_config as C  # noqa: E402
from solar_edge.lib import se_ingest as ing  # noqa: E402

duckdb = pytest.importorskip("duckdb")

SYDNEY = "Australia/Sydney"
ADELAIDE = "Australia/Adelaide"
BRISBANE = "Australia/Brisbane"

# 2025 transitions: DST ends Sun 6 April, DST starts Sun 5 October.
DST_END = "2025-04-06"
DST_START = "2025-10-05"


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect(database=":memory:")
    connection.execute("PRAGMA disable_progress_bar")
    try:
        connection.execute("LOAD icu")
    except duckdb.Error:
        pytest.skip("DuckDB ICU extension unavailable; timezone support required")
    yield connection
    connection.close()


def sql_utc(con, naive_local: str, tz_name: str) -> datetime:
    """Run the production SQL expression for a single value."""
    expression = ing.ts_utc_sql("ts", "tz")
    return con.execute(
        f"SELECT {expression} AS out FROM (SELECT ?::TIMESTAMP AS ts, ?::VARCHAR AS tz)",
        [naive_local, tz_name],
    ).fetchone()[0]


# ═══════════════════════════════════════════════════════════════════════════
# The timezone map
# ═══════════════════════════════════════════════════════════════════════════

def test_every_state_in_the_delivery_has_a_timezone():
    assert set(C.STATE_TIMEZONE) == {
        "New South Wales",
        "South Australia",
        "Queensland",
    }


def test_site_timezone_rejects_unknown_states():
    assert ing.site_timezone("Queensland") == BRISBANE
    with pytest.raises(KeyError):
        ing.site_timezone("Victoria")


def test_tz_case_sql_covers_every_state_and_defaults_to_null():
    sql = ing.tz_case_sql("state")
    for state, tz in C.STATE_TIMEZONE.items():
        assert state in sql and tz in sql
    assert "ELSE NULL" in sql, "an unmapped state must yield NULL, not a wrong zone"


# ═══════════════════════════════════════════════════════════════════════════
# Standard offsets
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "local, tz, expected_utc",
    [
        # Winter: no DST anywhere. AEST = UTC+10, ACST = UTC+9:30.
        ("2025-06-15 12:00:00", SYDNEY, "2025-06-15 02:00:00"),
        ("2025-06-15 12:00:00", BRISBANE, "2025-06-15 02:00:00"),
        ("2025-06-15 12:00:00", ADELAIDE, "2025-06-15 02:30:00"),
        # Summer: NSW and SA on DST, Queensland is not.
        ("2025-01-15 12:00:00", SYDNEY, "2025-01-15 01:00:00"),
        ("2025-01-15 12:00:00", BRISBANE, "2025-01-15 02:00:00"),
        ("2025-01-15 12:00:00", ADELAIDE, "2025-01-15 01:30:00"),
    ],
)
def test_known_offsets(con, local, tz, expected_utc):
    expected = datetime.fromisoformat(expected_utc)
    assert sql_utc(con, local, tz) == expected
    assert ing.resolve_timestamp(local, tz) == expected


def test_queensland_offset_is_identical_in_summer_and_winter(con):
    """The control case: Queensland has no DST, so the offset never moves."""
    for local in ("2025-01-15 12:00:00", "2025-06-15 12:00:00"):
        naive = datetime.fromisoformat(local)
        assert (naive - sql_utc(con, local, BRISBANE)) == timedelta(hours=10)


# ═══════════════════════════════════════════════════════════════════════════
# DST boundaries
# ═══════════════════════════════════════════════════════════════════════════

def test_april_ambiguous_hour_resolves_to_standard_time(con):
    """
    On 2025-04-06 the local hour 02:00-02:59 occurs twice. Policy: standard time,
    i.e. the second occurrence (UTC+10), matching zoneinfo fold=1.
    """
    got = sql_utc(con, f"{DST_END} 02:30:00", SYDNEY)
    assert got == datetime.fromisoformat("2025-04-05 16:30:00")
    assert (datetime.fromisoformat(f"{DST_END} 02:30:00") - got) == timedelta(hours=10)
    assert got == ing.resolve_timestamp(f"{DST_END} 02:30:00", SYDNEY)


def test_april_boundary_is_monotonic_and_gapless(con):
    """
    Stepping through the transition in local time must produce strictly increasing
    UTC instants. Anything else means an hour was duplicated or reordered.
    """
    locals_ = [f"{DST_END} 0{h}:{m:02d}:00" for h in (1, 2, 3) for m in (0, 30)]
    utcs = [sql_utc(con, ts, SYDNEY) for ts in locals_]
    assert utcs == sorted(utcs)
    assert len(set(utcs)) == len(utcs), "an instant was produced twice"


def test_october_nonexistent_hour_shifts_forward(con):
    """
    On 2025-10-05 the local hour 02:00-02:59 does not exist. ICU shifts the wall
    clock forward by the gap and applies the post-transition offset, reading local
    02:30 as 03:30 AEDT.

    Note this is where ICU and bare `zoneinfo` diverge: `zoneinfo` keeps the wall
    clock and lands an hour earlier. `resolve_timestamp` reproduces ICU explicitly
    so it remains a true reference implementation.
    """
    got = sql_utc(con, f"{DST_START} 02:30:00", SYDNEY)
    assert got == datetime.fromisoformat("2025-10-04 16:30:00")
    assert got == ing.resolve_timestamp(f"{DST_START} 02:30:00", SYDNEY)


def test_local_time_kinds_are_classified_correctly():
    assert ing.local_time_kind(datetime(2025, 4, 6, 2, 30), SYDNEY) == "ambiguous"
    assert ing.local_time_kind(datetime(2025, 10, 5, 2, 30), SYDNEY) == "nonexistent"
    assert ing.local_time_kind(datetime(2025, 6, 15, 12, 0), SYDNEY) == "normal"
    # Queensland never transitions, so nothing is ever ambiguous or skipped there.
    for month, day in ((4, 6), (10, 5), (6, 15)):
        assert ing.local_time_kind(datetime(2025, month, day, 2, 30), BRISBANE) == "normal"


def test_october_gap_times_collide_with_real_times(con):
    """
    Shift-forward semantics mean a nonexistent local time maps onto the SAME
    instant as a real one: local 02:00 and local 03:00 both become 16:00 UTC.

    This is inherent, not a bug, and it is exactly why `build_month` compares
    distinct local keys against distinct UTC keys and refuses to write when they
    differ. Deduplicating such a collision would discard a genuine reading.

    It is theoretical for this delivery -- `dst_hazards()` measures zero NSW/SA rows
    in the gap -- but the guard must exist for future deliveries.
    """
    gap_time = sql_utc(con, f"{DST_START} 02:00:00", SYDNEY)
    real_time = sql_utc(con, f"{DST_START} 03:00:00", SYDNEY)
    assert gap_time == real_time, "expected the gap hour to collide with the real hour"


def test_october_boundary_is_monotonic_outside_the_gap(con):
    """Real local times, either side of the gap, must map to increasing instants."""
    locals_ = [
        f"{DST_START} 00:30:00",
        f"{DST_START} 01:30:00",
        f"{DST_START} 03:30:00",
        f"{DST_START} 04:30:00",
    ]
    utcs = [sql_utc(con, ts, SYDNEY) for ts in locals_]
    assert utcs == sorted(utcs)
    assert len(set(utcs)) == len(utcs)


def test_queensland_is_unaffected_by_both_boundaries(con):
    for date in (DST_END, DST_START):
        for minute in (0, 30):
            local = f"{date} 02:{minute:02d}:00"
            offset = datetime.fromisoformat(local) - sql_utc(con, local, BRISBANE)
            assert offset == timedelta(hours=10), f"Queensland moved at {local}"


# ═══════════════════════════════════════════════════════════════════════════
# Cross-validation across the whole study year
# ═══════════════════════════════════════════════════════════════════════════

def test_sql_matches_zoneinfo_across_the_year(con):
    """
    Two independent implementations, every 6 hours through 2025, in all three
    zones. This is the test that would catch a silent ICU or policy change.
    """
    cursor = datetime(2025, 1, 1, 0, 0)
    end = datetime(2026, 1, 1, 0, 0)
    samples = []
    while cursor < end:
        samples.append(cursor)
        cursor += timedelta(hours=6)

    for tz in (SYDNEY, ADELAIDE, BRISBANE):
        rows = [(ts, tz) for ts in samples]
        con.execute("CREATE OR REPLACE TEMP TABLE _s (ts TIMESTAMP, tz VARCHAR)")
        con.executemany("INSERT INTO _s VALUES (?, ?)", rows)
        got = con.execute(
            f"SELECT ts, {ing.ts_utc_sql('ts', 'tz')} AS utc FROM _s ORDER BY ts"
        ).fetchall()

        mismatches = [
            (local, produced, ing.resolve_timestamp(local, tz))
            for local, produced in got
            if produced != ing.resolve_timestamp(local, tz)
        ]
        assert not mismatches, f"{tz}: {len(mismatches)} mismatches, first {mismatches[0]}"


def test_analysis_frame_is_a_fixed_ten_hour_offset(con):
    """
    ts_aest must be ts_utc + 10 h with no seasonal movement. The whole point of the
    fixed frame is that solar physics does not observe daylight saving.
    """
    got = con.execute(
        f"""
        SELECT ts, {ing.ts_aest_sql('ts')} AS aest
        FROM (VALUES (TIMESTAMP '2025-01-15 02:00:00'),
                     (TIMESTAMP '2025-06-15 02:00:00'),
                     (TIMESTAMP '{DST_END} 16:30:00'),
                     (TIMESTAMP '{DST_START} 16:30:00')) t(ts)
        """
    ).fetchall()
    for utc, aest in got:
        assert aest - utc == timedelta(hours=10)


# ═══════════════════════════════════════════════════════════════════════════
# Conventions
# ═══════════════════════════════════════════════════════════════════════════

def test_reactive_power_sign_is_flipped_and_active_is_not():
    """
    The single most consequential constant in the port. SolarEdge reports reactive
    power in the load convention (positive = absorbing); CICCADA and AS/NZS 4777.2
    use the generator convention (negative = absorbing).
    """
    assert C.ACTIVE_POWER_SIGN == +1.0
    assert C.REACTIVE_POWER_SIGN == -1.0


def test_reactive_power_is_not_scaled_like_solar_analytics():
    """
    Solar Analytics stores `energy_reactive` and multiplies by 12 to reach kvar.
    SolarEdge reports instantaneous var, so that factor must NOT appear.
    """
    assert C.REACTIVE_IS_INSTANTANEOUS is True
    assert C.VAR_TO_KVAR == 1.0 / 1000.0


def test_interval_matches_the_solar_analytics_pipeline():
    assert C.INTERVAL_MINUTES == 5
    assert C.INTERVAL_H == pytest.approx(5 / 60)
    assert C.INTERVAL_H == pytest.approx(C.as4777()["INTERVAL_H"])
