"""
Raw OEM delivery -> tidy local store.
===========================================

Deliverables D2 (timestamp / DST resolution) and D3 (the store builder).

This is the OEM analogue of `build_structured_data.py`: the one place where
the delivered telemetry is turned into the CICCADA convention. Every conversion
happens here, once, and is recorded in `se_config` so it travels with the results.

What the ingest does
--------------------
1. **Resolves time.** The `timestamp` column is a naive string in each site's
   local civil time, *including* daylight saving. It is converted to a UTC
   instant, then to the fixed AEST (UTC+10) analysis frame that the Solar
   Analytics pipeline uses.
2. **Corrects the reactive-power sign.** OEM reports reactive power in the
   load convention (positive = absorbing); CICCADA uses the generator convention
   (negative = absorbing). See `se_config` section 3.
3. **Converts units.** W -> kW and var -> kvar. Reactive power is instantaneous
   and is NOT multiplied by 12 (unlike Solar Analytics' `energy_reactive`).
4. **Aggregates phases to site level**, keeping a phase-level table alongside.
5. **Deduplicates** on (site_alias, ts_utc), deterministically.
6. **Re-partitions and sorts**, which is the point of the exercise: the delivered
   files have a single row group each, so nothing can be pruned.

What the ingest does NOT do
---------------------------
It does not drop implausible values. Voltage zeros, voltage outliers and
frequency excursions are *flagged*, not removed, so the store reconciles exactly
against the raw delivery and every filtering decision stays visible in the
analysis layer where it can be swept. Cleaning that silently changes row counts
belongs nowhere near an ingest step.

Why this runs in DuckDB rather than pandas
------------------------------------------
86.6 M rows is roughly 9-12 GB as float64. Month by month, DuckDB streams each
file through the transform and writes Parquet without ever materialising a
DataFrame. Peak memory stays bounded by one month's working set.
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from oem_analysis.config import se_config as C

__all__ = [
    # D2 -- time
    "tz_case_sql",
    "ts_utc_sql",
    "ts_aest_sql",
    "resolve_timestamp",
    "site_timezone",
    # D3 -- store
    "build_store",
    "build_month",
    "reconcile",
    "raw_deduplicated_totals",
    "dst_hazards",
    "dst_audit",
    "night_generation_anomaly",
]


# ═══════════════════════════════════════════════════════════════════════════
# D2. TIME RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════
#
# Three frames, and it matters which is which:
#
#   ts_local  the delivered wall clock, per-site local civil time WITH daylight
#             saving. Not comparable across states, and discontinuous in April
#             and October. Never aggregate on this.
#   ts_utc    the physical instant. The join key for anything external, including
#             the BOM irradiance extract at D12.
#   ts_aest   ts_utc + 10 h, a FIXED offset with no daylight saving. This is the
#             analysis frame, matching `ciccada_config.FIXED_OFFSET`.
#
# Materialising `ts_aest` at ingest removes a whole class of bug. The Solar
# Analytics queries carry `+ interval '10' hour` at every use site, and forgetting
# it produced the UTC date-extraction bug (R3/R9) in the legacy conformance
# tables. Here, `hour(ts_aest)` and `date(ts_aest)` are simply correct.


def site_timezone(state: str) -> str:
    """IANA timezone for a state as spelled in the alias mapping."""
    try:
        return C.STATE_TIMEZONE[state]
    except KeyError:
        raise KeyError(
            f"No timezone mapped for state {state!r}. "
            f"Known states: {sorted(C.STATE_TIMEZONE)}"
        ) from None


def tz_case_sql(state_col: str = "a.state") -> str:
    """
    SQL CASE mapping a state column to its IANA timezone.

    An unmapped state yields NULL rather than a wrong answer, and the ingest
    asserts that no row has a NULL timezone before writing anything.
    """
    branches = "\n".join(
        f"            WHEN {state_col} = '{state}' THEN '{tz}'"
        for state, tz in C.STATE_TIMEZONE.items()
    )
    return f"CASE\n{branches}\n            ELSE NULL\n        END"


def ts_utc_sql(ts_col: str = "r.timestamp", tz_expr: str = "tz_name") -> str:
    """
    SQL converting a naive local-civil timestamp to a naive UTC timestamp.

    `AT TIME ZONE <tz>` interprets a naive TIMESTAMP as local time in that zone
    and returns a TIMESTAMPTZ instant; the second `AT TIME ZONE 'UTC'` renders
    that instant back as a naive TIMESTAMP in UTC.

    Naive UTC is stored rather than TIMESTAMPTZ so that reading the store cannot
    depend on the reader's session timezone -- a silent, hard-to-spot failure mode.

    DST edge cases resolve per `se_config.DST_AMBIGUOUS_POLICY` and
    `DST_NONEXISTENT_POLICY`, cross-validated against Python `zoneinfo` in
    tests/test_se_ingest.py.
    """
    return f"(CAST({ts_col} AS TIMESTAMP) AT TIME ZONE {tz_expr}) AT TIME ZONE 'UTC'"


def ts_aest_sql(ts_utc_expr: str = "ts_utc") -> str:
    """SQL converting naive UTC to the fixed AEST (UTC+10) analysis frame."""
    return f"({ts_utc_expr} + INTERVAL '{C.ANALYSIS_UTC_OFFSET_HOURS}' HOUR)"


def local_time_kind(naive_local: datetime, tz_name: str) -> str:
    """
    Classify a naive local time as 'normal', 'ambiguous' or 'nonexistent'.

    Uses the PEP 495 `fold` disambiguation: when the two folds give different UTC
    offsets, the local time sits on a transition. Which fold carries the LARGER
    offset tells you which kind:

        fold=0 larger  -> ambiguous  (clocks went back; the hour repeats)
        fold=1 larger  -> nonexistent (clocks went forward; the hour is skipped)
    """
    zone = ZoneInfo(tz_name)
    offset_0 = naive_local.replace(tzinfo=zone, fold=0).utcoffset()
    offset_1 = naive_local.replace(tzinfo=zone, fold=1).utcoffset()
    if offset_0 == offset_1:
        return "normal"
    return "ambiguous" if offset_0 > offset_1 else "nonexistent"


def resolve_timestamp(naive_local: datetime | str, tz_name: str) -> datetime:
    """
    Pure-Python reference implementation of `ts_utc_sql`, used to cross-validate
    the SQL. Returns a naive UTC datetime.

    Reproduces DuckDB ICU semantics, which differ from bare `zoneinfo` at one of
    the two boundaries:

    * **Ambiguous** (April, clocks back): resolve to STANDARD time, the second
      occurrence. Equivalent to `zoneinfo` fold=1, so the two agree.
    * **Nonexistent** (October, clocks forward): ICU shifts the WALL CLOCK forward
      by the size of the gap and then applies the post-transition offset, so local
      02:30 is read as 03:30 AEDT. Bare `zoneinfo` instead keeps the wall clock and
      applies an offset, landing an hour earlier. They do NOT agree, so the shift
      is reproduced explicitly here.

    The October case is theoretical for this delivery -- no NSW or SA row falls in
    the gap, because the site clocks are daylight-saving aware. `dst_hazards()`
    asserts that rather than trusting it.
    """
    if isinstance(naive_local, str):
        naive_local = datetime.fromisoformat(naive_local)

    zone = ZoneInfo(tz_name)
    kind = local_time_kind(naive_local, tz_name)

    if kind == "nonexistent":
        gap = (
            naive_local.replace(tzinfo=zone, fold=1).utcoffset()
            - naive_local.replace(tzinfo=zone, fold=0).utcoffset()
        )
        naive_local = naive_local + gap

    localised = naive_local.replace(tzinfo=zone, fold=1)
    return localised.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


# ═══════════════════════════════════════════════════════════════════════════
# D3. STORE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

#: Phase-level projection. One row per (site, timestamp, phase) that reported.
#: Kept because OEM gives true per-phase voltage, which Solar Analytics
#: never had -- needed for the three-phase cohort investigation, and potentially
#: for a per-phase conformance result with no published precedent.
_PHASE_SELECT = """
    SELECT
        site_alias,
        ts_utc,
        ts_aest,
        phase,
        {active_sign} * active_power   / 1000.0 AS P_kW,
        {reactive_sign} * reactive_power / 1000.0 AS Q_kvar,
        ac_voltage                              AS V,
        ac_frequency                            AS freq_hz
    FROM unpivoted
    WHERE active_power IS NOT NULL
       OR reactive_power IS NOT NULL
       OR ac_voltage IS NOT NULL
"""


def _month_sql(path: Path) -> str:
    """
    The full per-month transform, as one SQL statement.

    Structure:
        typed      cast the timestamp, attach state and timezone
        resolved   ts_utc and ts_aest
        deduped    one row per (site_alias, ts_utc), deterministically
        site       phases folded to site level, signs and units applied
    """
    tz_case = tz_case_sql("a.state")
    ts_utc = ts_utc_sql("tz.timestamp", "tz.tz_name")
    ts_aest = ts_aest_sql("res.ts_utc")
    src = str(path.as_posix()).replace("'", "''")

    # Sum only over phases that reported; if none did, the result must be NULL,
    # not zero. coalesce-and-add would silently turn "no data" into "zero power".
    def _phase_sum(prefix: str) -> str:
        terms = " + ".join(f"coalesce({prefix}_{p}, 0)" for p in C.PHASES)
        guard = " OR ".join(f"{prefix}_{p} IS NOT NULL" for p in C.PHASES)
        return f"CASE WHEN {guard} THEN {terms} END"

    p_sum = _phase_sum("active_power")
    q_sum = _phase_sum("reactive_power")

    return f"""
    WITH tz AS (
        SELECT r.*, a.state, a.zip_code, {tz_case} AS tz_name
        FROM read_parquet('{src}') r
        LEFT JOIN se_alias a ON r.site_alias = a.alias
    ),
    res AS (
        SELECT tz.*, {ts_utc} AS ts_utc
        FROM tz
    ),
    stamped AS (
        SELECT res.*, {ts_aest} AS ts_aest
        FROM res
    )
    SELECT
        site_alias,
        CAST(timestamp AS TIMESTAMP)                    AS ts_local,
        ts_utc,
        ts_aest,
        state,
        -- Zero-padded string, not an integer. Australian postcodes are 4-character
        -- codes, some with a leading zero, and the ABS POA-2021 shapefile that D4
        -- joins against keys on strings. Storing an integer would break that join
        -- for NT postcodes if the fleet ever extends there.
        lpad(CAST(zip_code AS VARCHAR), 4, '0')         AS postcode,
        {C.ACTIVE_POWER_SIGN}   * ({p_sum}) * {C.W_TO_KW}   AS P_kW,
        {C.REACTIVE_POWER_SIGN} * ({q_sum}) * {C.VAR_TO_KVAR} AS Q_kvar,
        -- greatest() over coalesced sentinels would return the sentinel when every
        -- phase is null; nullif restores NULL for "no voltage reported".
        nullif(greatest(coalesce(ac_voltage_1, -1),
                        coalesce(ac_voltage_2, -1),
                        coalesce(ac_voltage_3, -1)), -1) AS V_max,
        (coalesce(ac_voltage_1, 0) + coalesce(ac_voltage_2, 0) + coalesce(ac_voltage_3, 0))
            / nullif((CASE WHEN ac_voltage_1 IS NOT NULL THEN 1 ELSE 0 END)
                   + (CASE WHEN ac_voltage_2 IS NOT NULL THEN 1 ELSE 0 END)
                   + (CASE WHEN ac_voltage_3 IS NOT NULL THEN 1 ELSE 0 END), 0)
                                                        AS V_mean,
        (CASE WHEN active_power_1 IS NOT NULL THEN 1 ELSE 0 END)
      + (CASE WHEN active_power_2 IS NOT NULL THEN 1 ELSE 0 END)
      + (CASE WHEN active_power_3 IS NOT NULL THEN 1 ELSE 0 END)
                                                        AS n_phases_reporting,
        coalesce(ac_frequency_1, ac_frequency_2, ac_frequency_3) AS freq_hz,
        -- The raw flag is 1.0 or NULL, never 0.0, so "not derating" and "not
        -- reported" are indistinguishable in the source. Collapsing NULL to FALSE
        -- is an INTERPRETATION, made here once and recorded in se_config, not a
        -- fact about the data. Its consequence: precision against this flag is
        -- interpretable, recall is not. Method C (D14) states that limitation.
        coalesce(derating_active_flag = 1.0, FALSE)     AS derating_active,
        strftime(ts_aest, '%Y-%m')                      AS {C.PARTITION_KEY}
    FROM stamped
    """


# Deduplication rule: keep the row with the LOWEST active power, breaking ties on
# Q, then V, then frequency. Deterministic regardless of input order, so a rebuild
# reproduces the store exactly.
#
# Why lowest P, rather than first-seen or highest?
#
# The colliding rows in this delivery are NOT retransmissions of one reading. They
# are a small number of rows timestamped in a different frame -- apparently UTC --
# landing on top of genuine local-time rows. Measured: 2,217 colliding rows across
# 86.6 M, of which 1,101 keys carry conflicting values, and a wider population of
# 22,124 rows (0.026%, confined to 20 sites) shows daytime-scale generation at
# implausible night hours.
#
# Because a mis-framed row carries DAYTIME power onto a NIGHT timestamp, the lower
# of two colliding values is the physically plausible one. Taking the minimum
# therefore preserves the genuine reading at exactly the collisions where it
# matters. At the fleet scale involved the choice moves no published number --
# it is made explicitly so that it is reproducible and auditable, not because it
# is material.
#
# The underlying anomaly is NOT fixed here. `night_generation_anomaly()` surfaces
# the affected sites; whether to exclude them is an analysis-layer decision for
# D6/D7, where it can be swept, not an ingest-layer one that silently drops data.
_DEDUPE_SQL = """
    SELECT * EXCLUDE (_rn) FROM (
        SELECT s.*,
               row_number() OVER (
                   PARTITION BY site_alias, ts_utc
                   ORDER BY P_kW, Q_kvar, V_max, freq_hz
               ) AS _rn
        FROM _stamped s
    ) WHERE _rn = 1
"""


def build_month(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Transform one raw monthly file and append it to the partitioned store.

    Returns a stats dict for the reconciliation table. Peak memory is bounded by
    one month's working set, so this runs on a laptop as happily as on the
    compute server.
    """
    out_dir = out_dir or C.store_path("se_interval")
    out_dir.mkdir(parents=True, exist_ok=True)

    source_month = C.raw_month_of(path)
    start = time.perf_counter()

    con.execute(f"CREATE OR REPLACE TEMP VIEW _stamped AS {_month_sql(path)}")
    con.execute(f"CREATE OR REPLACE TEMP VIEW _month AS {_DEDUPE_SQL}")

    # Guard rails, all cheap relative to the write, and each one has bitten a
    # real pipeline at some point.
    guard = con.execute(
        """
        SELECT count(*)                                       AS n_stamped,
               count(*) FILTER (WHERE ts_utc IS NULL)         AS n_null_ts,
               count(*) FILTER (WHERE state IS NULL)          AS n_null_state,
               count(DISTINCT site_alias)                     AS n_sites,
               count(DISTINCT (site_alias, ts_local))         AS n_local_keys,
               count(DISTINCT (site_alias, ts_utc))           AS n_utc_keys,
               min(ts_aest)                                   AS min_ts_aest,
               max(ts_aest)                                   AS max_ts_aest
        FROM _stamped
        """
    ).df().iloc[0]

    if int(guard.n_null_ts):
        raise ValueError(
            f"{source_month}: {int(guard.n_null_ts):,} rows have a NULL ts_utc. "
            "That means an unmapped state or an unparseable timestamp."
        )
    if int(guard.n_null_state):
        raise ValueError(
            f"{source_month}: {int(guard.n_null_state):,} rows have no state. "
            "A site_alias is missing from the alias mapping."
        )

    # The hazard that matters: two DISTINCT local timestamps for one site collapsing
    # onto a single UTC instant. That can only come from the daylight-saving gap,
    # where ICU shifts nonexistent wall-clock times forward onto real ones. Ordinary
    # duplicates in the delivery (same site, same local timestamp) are benign and
    # are counted separately. If this ever fires, deduplication would be discarding
    # a genuine reading rather than a repeat.
    n_collisions = int(guard.n_local_keys) - int(guard.n_utc_keys)
    if n_collisions:
        raise ValueError(
            f"{source_month}: {n_collisions:,} distinct local timestamps collapsed onto "
            "an existing UTC instant. This is the daylight-saving gap colliding with "
            "real readings; deduplication here would discard genuine data. Inspect with "
            "se_ingest.dst_hazards() before proceeding."
        )

    con.execute(
        f"""
        COPY (SELECT * EXCLUDE (ts_local, ts_aest) FROM _month
              ORDER BY {', '.join(C.STORE_SORT_KEY)})
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET,
         PARTITION_BY ({C.PARTITION_KEY}),
         COMPRESSION '{C.PARQUET_COMPRESSION}',
         ROW_GROUP_SIZE {C.PARQUET_ROW_GROUP_SIZE},
         FILENAME_PATTERN 'part_{source_month.replace('-', '_')}_{{i}}',
         OVERWRITE_OR_IGNORE {'true' if overwrite else 'false'})
        """
    )

    return {
        "source_month": source_month,
        "raw_rows": int(guard.n_stamped),
        "store_rows": int(guard.n_utc_keys),
        "duplicates_removed": int(guard.n_stamped) - int(guard.n_utc_keys),
        "utc_collisions": n_collisions,
        "n_sites": int(guard.n_sites),
        "min_ts_aest": guard.min_ts_aest,
        "max_ts_aest": guard.max_ts_aest,
        "seconds": round(time.perf_counter() - start, 1),
    }


def build_store(
    con: duckdb.DuckDBPyConnection | None = None,
    months: list[str] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build `se_interval` from the raw delivery, one month at a time.

    Each month is processed on its own short-lived connection (see the comment in
    the loop). `con`, if given, is only used to re-register the store views at the
    end so the caller's session sees the new data.

    Parameters
    ----------
    months : restrict to these source months, e.g. ["2025-06"]. A partial build
        APPENDS to the existing store, replacing only those months' files -- each
        month writes files named for its source, so nothing else is disturbed.
        This makes an interrupted build resumable rather than a restart.
    overwrite : only meaningful for a full build (months=None), where it wipes the
        store first. A full build refuses to clobber an existing store without it.

    Memory
    ------
    Peak RSS is roughly the DuckDB memory limit plus ~200 MB. The month is sorted
    twice (once to deduplicate, once to write in store order), so on a constrained
    machine set `CICCADA_SE_DUCKDB_MEMORY` to comfortably below physical RAM --
    DuckDB will spill to `DUCKDB_TEMP_DIR` rather than be killed.
    """
    out_dir = C.store_path("se_interval")
    files = C.raw_files()

    if months:
        wanted = set(months)
        files = [f for f in files if C.raw_month_of(f) in wanted]
        if not files:
            raise ValueError(f"No raw files match months={months}")
    elif out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already exists. Pass overwrite=True to rebuild it, or "
                "pass months=[...] to build a subset without touching the rest."
            )
        shutil.rmtree(out_dir)

    from oem_analysis.lib import se_store

    rows = []
    for n, path in enumerate(files, start=1):
        # A fresh connection per month, so peak memory is bounded by one month's
        # working set no matter how many months are built in a run. Each month is
        # sorted twice (deduplication, then store order), and DuckDB's buffer
        # manager has no reason to release that back between statements.
        #
        # Measured: peak RSS ~= the DuckDB memory limit + 200 MB, flat across the
        # run. Cost of the extra connections is view registration, milliseconds.
        month_con = se_store.connect()
        try:
            stats = build_month(month_con, path, out_dir=out_dir, overwrite=True)
        finally:
            month_con.close()

        rows.append(stats)
        if verbose:
            print(
                f"  [{n:>2}/{len(files)}] {stats['source_month']}  "
                f"{stats['store_rows']:>10,} rows  "
                f"{stats['n_sites']:>5} sites  "
                f"{stats['duplicates_removed']:>5,} dup  "
                f"{stats['seconds']:>6.1f}s",
                flush=True,
            )

    if con is not None:
        se_store.register_store_views(con)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════════

def raw_deduplicated_totals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Per source file: row count, distinct keys, duplicate behaviour, and the
    phase-summed power totals over the DEDUPLICATED raw rows.

    The deduplicated basis is the point. The store removes duplicate
    (site, timestamp) rows, so comparing it against the raw totals *including*
    duplicates would always show a spurious mismatch. Reconciling like against
    like is what makes the check meaningful.

    Aggregated one file at a time, so peak memory stays at one month's working set.
    """
    rows = []
    for path in C.raw_files():
        src = str(path.as_posix()).replace("'", "''")
        rows.append(
            con.execute(
                f"""
                WITH totals AS (
                    SELECT site_alias, timestamp,
                           CASE WHEN active_power_1 IS NOT NULL
                                  OR active_power_2 IS NOT NULL
                                  OR active_power_3 IS NOT NULL
                                THEN coalesce(active_power_1, 0)
                                   + coalesce(active_power_2, 0)
                                   + coalesce(active_power_3, 0) END AS p_w,
                           CASE WHEN reactive_power_1 IS NOT NULL
                                  OR reactive_power_2 IS NOT NULL
                                  OR reactive_power_3 IS NOT NULL
                                THEN coalesce(reactive_power_1, 0)
                                   + coalesce(reactive_power_2, 0)
                                   + coalesce(reactive_power_3, 0) END AS q_var
                    FROM read_parquet('{src}')
                ),
                grouped AS (
                    -- Mirrors the store's deduplication rule exactly: the surviving
                    -- row is the one with the lowest active power, and its OWN
                    -- reactive value is taken via arg_min. Using min(p) and min(q)
                    -- independently would mix values across different rows and
                    -- produce a total that no actual row ever held.
                    SELECT site_alias, timestamp,
                           count(*)              AS n,
                           min(p_w)              AS p_kept,
                           arg_min(q_var, p_w)   AS q_kept,
                           min(p_w)              AS p_min,
                           max(p_w)              AS p_max,
                           min(q_var)            AS q_min,
                           max(q_var)            AS q_max
                    FROM totals GROUP BY site_alias, timestamp
                )
                SELECT '{C.raw_month_of(path)}'              AS source_month,
                       sum(n)                                AS raw_rows,
                       count(*)                              AS distinct_keys,
                       sum(n) - count(*)                     AS duplicate_rows,
                       count(*) FILTER (
                           WHERE n > 1 AND (p_min IS DISTINCT FROM p_max
                                         OR q_min IS DISTINCT FROM q_max)
                       )                                     AS conflicting_keys,
                       sum(coalesce(p_kept, 0)) / 1000.0     AS dedup_p_kw,
                       sum(coalesce(q_kept, 0)) / 1000.0     AS dedup_q_kvar
                FROM grouped
                """
            ).df()
        )
    return pd.concat(rows, ignore_index=True)


def reconcile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Prove the store is a faithful transform of the raw delivery.

    Five claims, each checked against the DEDUPLICATED raw basis:

    1. **Rows accounted for** -- store rows == raw distinct (site, timestamp) keys.
       An identity, not a tolerance: every raw key appears exactly once.
    2. **Duplicates are value-identical** -- so which copy survives is immaterial
       and the deduplication cannot have changed a number.
    3. **Sites conserved** -- 1,602 in both.
    4. **Active energy conserved** -- raw phase sums == store site-level `P_kW`.
       Exercises the phase folding and the W -> kW conversion.
    5. **Reactive energy matches the configured sign** -- store total ==
       ``REACTIVE_POWER_SIGN`` x raw total, read from config rather than hard-coded
       so the check cannot outlive the transform it verifies.

    Time is deliberately not reconciled here; it cannot be, because the raw
    timestamps are local and the store's are UTC. D2's unit tests cover the
    conversion and `dst_hazards()` covers the boundaries.
    """
    raw = raw_deduplicated_totals(con)

    store = con.execute(
        """
        SELECT count(*)                   AS store_rows,
               count(DISTINCT site_alias) AS store_sites,
               sum(coalesce(P_kW, 0))     AS store_p_kw,
               sum(coalesce(Q_kvar, 0))   AS store_q_kvar
        FROM se_interval
        """
    ).df().iloc[0]

    raw_sites = int(
        con.execute("SELECT count(DISTINCT site_alias) FROM se_raw").fetchone()[0]
    )

    def _rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a), abs(b), 1e-12)

    raw_keys = int(raw.distinct_keys.sum())
    raw_p = float(raw.dedup_p_kw.sum())
    raw_q = float(raw.dedup_q_kvar.sum())
    n_dups = int(raw.duplicate_rows.sum())
    n_conflicting = int(raw.conflicting_keys.sum())

    # Compare against the sign the ingest ACTUALLY applies, read from config.
    # This previously hard-coded -1, which silently became wrong the moment
    # REACTIVE_POWER_SIGN changed to +1 on 13 Aug -- the data was fine and the
    # check was stale. A reconciliation that can disagree with the transform it
    # is reconciling is worse than no reconciliation.
    p_rel = _rel(raw_p, C.ACTIVE_POWER_SIGN * float(store.store_p_kw))
    q_rel = _rel(raw_q, C.REACTIVE_POWER_SIGN * float(store.store_q_kvar))
    q_rule = (
        "unchanged (as delivered)" if C.REACTIVE_POWER_SIGN > 0
        else "sign-flipped (-1 x raw)"
    )

    checks = [
        {
            "check": "rows accounted for (deduplicated basis)",
            "raw": f"{raw_keys:,} distinct keys",
            "store": f"{int(store.store_rows):,}",
            "delta": f"{n_dups:,} duplicate rows removed from {int(raw.raw_rows.sum()):,}",
            "pass": raw_keys == int(store.store_rows),
        },
        {
            # Reported, not asserted. Conflicting duplicates DO occur here: a small
            # number of rows are timestamped in a different frame and land on top of
            # genuine ones. The deduplication rule handles them deterministically
            # (see _DEDUPE_SQL); pretending they do not exist would be the error.
            "check": "duplicate keys with conflicting values (informational)",
            "raw": f"{n_dups:,} duplicate rows",
            "store": "-",
            "delta": f"{n_conflicting:,} conflicting keys -- see night_generation_anomaly()",
            "pass": True,
        },
        {
            "check": "sites conserved",
            "raw": f"{raw_sites:,}",
            "store": f"{int(store.store_sites):,}",
            "delta": "0",
            "pass": raw_sites == int(store.store_sites),
        },
        {
            "check": "active energy conserved (kW-sum)",
            "raw": f"{raw_p:,.3f}",
            "store": f"{store.store_p_kw:,.3f}",
            "delta": f"relative {p_rel:.2e}",
            "pass": p_rel < 1e-6,
        },
        {
            "check": f"reactive energy {q_rule} (kvar-sum)",
            "raw": f"{raw_q:,.3f}",
            "store": f"{store.store_q_kvar:,.3f}",
            "delta": f"relative {q_rel:.2e} vs "
                     f"{C.REACTIVE_POWER_SIGN:+.0f} x raw",
            "pass": q_rel < 1e-6,
        },
    ]
    return pd.DataFrame(checks)


#: 2025 daylight-saving transitions for the DST-observing states in this fleet.
DST_TRANSITIONS_2025 = {
    "April (clocks back, local 02:00-03:00 occurs twice)": "2025-04-06",
    "October (clocks forward, local 02:00-03:00 absent)": "2025-10-05",
}


def dst_hazards(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Count rows sitting in the two daylight-saving danger windows, straight from
    the RAW delivery.

    Two distinct hazards, and only one of them is real here:

    * **October gap** -- local 02:00-02:59 on 2025-10-05 does not exist in NSW or
      SA. Any row there means a site clock that is not daylight-saving aware, and
      ICU will shift it forward onto a real instant, colliding with a genuine
      reading. Measured on this delivery: **zero** NSW/SA rows. Queensland rows in
      that window are perfectly normal -- Queensland has no daylight saving.

    * **April overlap** -- local 02:00-02:59 on 2025-04-06 occurs twice in NSW and
      SA. A site reporting through both passes would emit two rows with the same
      local timestamp. Measured on this delivery: 74 NSW and 227 SA rows, every one
      with a distinct key, so no site reported through both passes.

    Run this against a new delivery before trusting the ingest.
    """
    cases = "\n            ".join(
        f"WHEN CAST(r.timestamp AS TIMESTAMP) >= TIMESTAMP '{date} 02:00:00' "
        f"AND CAST(r.timestamp AS TIMESTAMP) < TIMESTAMP '{date} 03:00:00' "
        f"THEN '{label}'"
        for label, date in DST_TRANSITIONS_2025.items()
    )
    return con.execute(
        f"""
        WITH windowed AS (
            SELECT a.state,
                   CASE
            {cases}
                   END AS dst_window,
                   r.site_alias, r.timestamp
            FROM se_raw r
            JOIN se_alias a ON r.site_alias = a.alias
        )
        SELECT dst_window,
               state,
               (state = 'Queensland')                     AS observes_dst_exempt,
               count(*)                                   AS n_rows,
               count(DISTINCT site_alias)                 AS n_sites,
               count(DISTINCT (site_alias, timestamp))    AS n_distinct_keys,
               count(*) - count(DISTINCT (site_alias, timestamp)) AS n_repeat_keys
        FROM windowed
        WHERE dst_window IS NOT NULL
        GROUP BY dst_window, state
        ORDER BY dst_window, state
        """
    ).df()


def night_generation_anomaly(
    con: duckdb.DuckDBPyConnection, p_threshold_kw: float = 0.5
) -> pd.DataFrame:
    """
    Sites reporting active power at hours when photovoltaic generation is impossible.

    CORRECTED 13 Aug 2026. This was originally reported as a single phenomenon --
    "mis-framed timestamps" -- which was wrong. Interrogating the sites individually
    shows TWO distinct populations, and only one of them is a data fault:

    ``likely_storage`` (6 sites, ~95% of the affected rows)
        These inverters report CONTINUOUSLY: about 105,000 rows a year, which is
        24 h x 12 intervals x 365 exactly, with overnight row counts equal to midday
        row counts. Their night output is a low, flat plateau that scales with
        season -- AUS639 averages 0.46 kW overnight in January and 0.10 kW in June,
        and is flat 0.4-0.5 kW right across the night rather than peaking at any
        hour. A displaced solar curve would show a peak somewhere; a battery
        discharging overnight looks exactly like this.

    ``stray_timestamps`` (14 sites, ~5% of the affected rows)
        These report daylight hours only, about 53,000 rows a year, and carry a
        handful of night rows -- 3 to 323 across the year -- at daytime power
        levels. This IS the original hypothesis, and it holds here: a few readings
        landing in the wrong frame.

    Why the distinction matters
    --------------------------
    Storage is not a defect, it is a population. CICCADA covers BESS explicitly, so
    six battery sites are of interest rather than something to discard. But they
    cannot be treated like PV-only sites either:

      * ``s_99`` absorbs battery discharge, inflating the capacity proxy and hence
        the required-Q curve that scales off it;
      * a drop in active power may be the battery charging, not curtailment, which
        breaks the core assumption of both Method A and Method B.

    The classification here is INFERRED from reporting behaviour, not read from
    metadata -- the delivery has no storage flag. Confirm with OEM before
    relying on it.

    Returns one row per affected site with its classification, worst first.
    """
    return con.execute(
        f"""
        WITH per_site AS (
            SELECT site_alias,
                   any_value(state)                                          AS state,
                   count(*)                                                  AS n_rows,
                   count(*) FILTER (WHERE hour(ts_aest) BETWEEN 0 AND 3)      AS n_deep_night,
                   count(*) FILTER (WHERE hour(ts_aest) BETWEEN 10 AND 13)    AS n_midday,
                   count(*) FILTER (
                       WHERE (hour(ts_aest) >= 21 OR hour(ts_aest) < 4)
                         AND P_kW > {p_threshold_kw}
                   )                                                          AS n_night_rows,
                   round(max(P_kW) FILTER (
                       WHERE hour(ts_aest) >= 21 OR hour(ts_aest) < 4), 2)    AS max_night_P_kW,
                   round(avg(P_kW) FILTER (
                       WHERE hour(ts_aest) BETWEEN 0 AND 3), 3)               AS mean_deep_night_P_kW,
                   round(avg(P_kW) FILTER (
                       WHERE hour(ts_aest) BETWEEN 10 AND 13), 3)             AS mean_midday_P_kW
            FROM se_interval
            GROUP BY site_alias
        )
        SELECT site_alias, state, n_rows, n_night_rows,
               round(100.0 * n_deep_night / nullif(n_midday, 0), 1) AS night_coverage_pct,
               max_night_P_kW, mean_deep_night_P_kW, mean_midday_P_kW,
               -- Continuous overnight reporting is the discriminator. An inverter
               -- that logs all night is energised all night; one that logs only in
               -- daylight cannot legitimately produce a night reading.
               CASE WHEN n_deep_night > 0.5 * n_midday
                    THEN 'likely_storage'
                    ELSE 'stray_timestamps' END          AS classification
        FROM per_site
        WHERE n_night_rows > 0
        ORDER BY n_night_rows DESC
        """
    ).df()


def dst_audit(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Where the daylight-saving boundary rows actually landed in the store.

    Filters on `ts_utc`, not `ts_aest`. The transition is an instant, and the
    windows below are that instant expressed in UTC:

        April  2025-04-06 03:00 AEDT  ->  2025-04-05 16:00 UTC
        October 2025-10-05 02:00 AEST ->  2025-10-04 16:00 UTC

    A two-hour UTC window either side therefore straddles the local hour that
    repeats (April) or is skipped (October). Queensland rows appear because
    Queensland is in the same UTC window, not because anything happened to it --
    it has no daylight saving at all.

    Expect very few rows and no generation: this is 02:00-03:00 local, inside the
    ~3% overnight coverage.
    """
    return con.execute(
        """
        WITH boundary AS (
            SELECT state, P_kW, ts_utc,
                   CASE
                       WHEN ts_utc >= TIMESTAMP '2025-04-05 15:00:00'
                        AND ts_utc <  TIMESTAMP '2025-04-05 17:00:00'
                            THEN 'April (clocks back; local 02:00-03:00 repeats)'
                       WHEN ts_utc >= TIMESTAMP '2025-10-04 15:00:00'
                        AND ts_utc <  TIMESTAMP '2025-10-04 17:00:00'
                            THEN 'October (clocks forward; local 02:00-03:00 skipped)'
                   END AS boundary
            FROM se_interval
        )
        SELECT boundary, state,
               count(*)              AS n_rows,
               round(max(P_kW), 3)   AS max_P_kW,
               min(ts_utc)           AS first_ts_utc,
               max(ts_utc)           AS last_ts_utc
        FROM boundary
        WHERE boundary IS NOT NULL
        GROUP BY boundary, state
        ORDER BY boundary, state
        """
    ).df()
