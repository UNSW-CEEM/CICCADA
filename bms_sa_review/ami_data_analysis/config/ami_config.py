"""
Shared constants and paths for the synthetic AMI dataset.
=========================================================

This is the ONLY place that writes down:
  * where the local Parquet store lives and what its tables are called,
  * the Athena cost constants and the scan budget the notebooks work to,
  * the sign convention of the synthetic meter,
  * the circuit -> signal mapping,
  * the source/target interval constants and which source column is power and
    which is energy.

Database names, the AEST offset and the AS/NZS 4777.2 set-points are NOT
restated here. They are imported from `bms_sa_review.shared.ciccada_config` so
that this study and the conformance work cannot drift apart.

READ THIS BEFORE TRUSTING ANY CONSTANT BELOW
--------------------------------------------
Written before Phase 1. Everything in sections 3, 4 and 5 marked UNRESOLVED is a
placeholder awaiting evidence from notebooks 02 and 03. They are declared now,
rather than added later, so that their absence is visible in `manifest()`
instead of being silently absent. `describe_conventions()` prints the resolution
state of every one of them.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 0. REPOSITORY ROOT AND IMPORT BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards until we find the directory containing `bms_sa_review`."""
    current = (start or Path(__file__)).resolve()
    for path in (current, *current.parents):
        if (path / "bms_sa_review").is_dir():
            return path
    raise RuntimeError(
        "Could not locate the CICCADA repository root (expected a directory "
        f"containing 'bms_sa_review'). Searched upwards from {current}."
    )


REPO_ROOT = find_repo_root()


def bootstrap_sys_path() -> Path:
    """
    Put the repository root on `sys.path` so `bms_sa_review.*` imports cleanly
    from a notebook regardless of where Jupyter was started. Mirrors
    `oem_analysis.config.se_config.bootstrap_sys_path`.
    """
    for entry in (REPO_ROOT, REPO_ROOT / "bms_sa_review"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    return REPO_ROOT


bootstrap_sys_path()

# Imported, never restated. `ciccada_config` pulls in `pytz` and nothing else,
# so this is safe at import time -- unlike `aws_config`, which builds a boto3
# session and is therefore always imported lazily (see `lib/ami_athena.py`).
from bms_sa_review.shared.ciccada_config import (  # noqa: E402
    AS4777,
    FIXED_OFFSET,
    REBUILT,
    SA,
    SAI,
    TABLES,
)

#: The BOM satellite irradiance database. Not used until/unless a daylight test
#: in Phase 6 needs irradiance; declared here so the name has one home.
BOM_DB = "bom_nci"

__all__ = [
    "SA", "SAI", "BOM_DB", "TABLES", "REBUILT", "AS4777", "FIXED_OFFSET",
    "REPO_ROOT", "bootstrap_sys_path", "find_repo_root",
    "DATA_ROOT", "STORE_DIR", "ARTEFACT_DIR", "store_path",
    "describe_conventions", "describe_paths", "duckdb_path_list",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. ATHENA COST
# ═══════════════════════════════════════════════════════════════════════════
#
# Athena bills by DATA SCANNED, not rows returned. Every number here exists so
# that a notebook can print what a query cost instead of guessing.

#: Sydney on-demand price. The figure quoted in `shared/aws_config.py`
#: ("~AUD $8 per TB in Sydney"). Override with CICCADA_AMI_ATHENA_PRICE if the
#: account is on a different rate or you want USD.
ATHENA_PRICE_PER_TB = float(os.environ.get("CICCADA_AMI_ATHENA_PRICE", 8.0))
ATHENA_PRICE_CURRENCY = os.environ.get("CICCADA_AMI_ATHENA_CURRENCY", "AUD")

#: Athena rounds every query up to a 10 MB minimum scan. A hundred free-looking
#: metadata queries are therefore not free; they are about a cent.
ATHENA_MIN_SCAN_BYTES = 10 * 1024**2

#: "Tell me before anything you expect to scan more than a few GB." This is that
#: threshold, in one place, so the guard and the prose agree.
SCAN_WARN_BYTES = 5 * 1024**3

#: Tables large enough that an unfiltered query is a real cost event. The guard
#: in `ami_athena.check_partition_filters` refuses to run against these without
#: a partition predicate.
BIG_TABLES: frozenset[str] = frozenset({
    "ts",
    "structured_data",
    "structured_data_v2",
    "structured_data_v2_flex_included",
    "all_uncurtailedpv",
    "all_uncurtailedpv_v2_flex_included",
    "solar",          # bom_nci.solar -- the whole Himawari disc
})

#: Partition columns of `ts`, in order. A query against a BIG_TABLE must mention
#: at least one of these (or be a metadata query) to pass the guard.
TS_PARTITION_COLUMNS = ("year", "month", "is_pv")

#: Dimension tables small enough to query freely.
SMALL_TABLES: frozenset[str] = frozenset({
    "circuits", "sites", "partition_lookup", "meta_up23c",
    "meta_single_inverters",
})


# ═══════════════════════════════════════════════════════════════════════════
# 2. PATHS
# ═══════════════════════════════════════════════════════════════════════════
#
# The extracted Parquet lives OUTSIDE the git repository. It is far too large to
# commit, it is regenerable from Athena, and putting it in OneDrive would mean
# re-uploading gigabytes on every rebuild. Same reasoning, same default location
# and same override mechanism as `oem_analysis.config.se_config`.


def _default_store_dir() -> Path:
    """
    Windows      -> %LOCALAPPDATA%\\ciccada\\ami_store
    Linux/macOS  -> $XDG_DATA_HOME/ciccada/ami_store, else
                    ~/.local/share/ciccada/ami_store

    Override with CICCADA_AMI_STORE_DIR. The override is passed through
    `expandvars`/`expanduser`, so both `%LOCALAPPDATA%\\...` and `~/...` work.
    """
    override = os.environ.get("CICCADA_AMI_STORE_DIR")
    if override:
        return Path(os.path.expandvars(override)).expanduser()

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "ciccada" / "ami_store"
    return Path.home() / ".local" / "share" / "ciccada" / "ami_store"


STORE_DIR = _default_store_dir()

#: Alias kept for symmetry with se_config, where DATA_ROOT is a separate raw
#: delivery. Here there is no delivery: everything derives from the extract.
DATA_ROOT = STORE_DIR

#: Small, human-reviewable outputs that DO belong in the repository.
ARTEFACT_DIR = REPO_ROOT / "bms_sa_review" / "ami_data_analysis" / "artefacts"

#: Spill directory for DuckDB's out-of-core sorts. Deliberately not inside
#: STORE_DIR: spill files are transient, can run to gigabytes, and would make
#: any sync client churn on every rebuild.
DUCKDB_TEMP_DIR = Path(
    os.environ.get(
        "CICCADA_AMI_DUCKDB_TEMP", Path(tempfile.gettempdir()) / "ciccada_ami_duckdb"
    )
)

DUCKDB_MEMORY_LIMIT = os.environ.get("CICCADA_AMI_DUCKDB_MEMORY") or None
_threads = os.environ.get("CICCADA_AMI_DUCKDB_THREADS")
DUCKDB_THREADS = int(_threads) if _threads else None

#: S3 prefix that Phase 4 UNLOADs into before download. Under the same bucket
#: `aws_config` already stages Athena results in.
S3_EXTRACT_PREFIX = os.environ.get(
    "CICCADA_AMI_S3_PREFIX", "s3://project-ciccada/ami-extract/"
)


# ═══════════════════════════════════════════════════════════════════════════
# 3. STORE REGISTRY  (Phase 4/5)
# ═══════════════════════════════════════════════════════════════════════════
#
# Logical name -> path within STORE_DIR. The ONLY place these are written down,
# mirroring `ciccada_config.TABLES` and `se_config.STORE_TABLES`.

STORE_TABLES: dict[str, str] = {
    # Phase 4: landed exactly as Athena returned it, nothing derived.
    "ami_extract":   "ami_extract",          # circuit x interval, Hive-partitioned
    "ami_meta":      "ami_meta.parquet",     # circuit dimension, from meta_up23c
    "ami_provenance": "ami_provenance.parquet",  # per-chunk SQL, counts, timestamps
    # Phase 5: the deliverable trio.
    "ami_raw":              "ami_raw",              # site x interval, ground truth
    "ami_meter":            "ami_meter",            # site x AMI interval, net only, per phase
    "ami_raw_phaseseparate": "ami_raw_phaseseparate",  # per-phase ground truth, PV-allocated
}

PARTITIONED_TABLES = frozenset({"ami_extract", "ami_raw", "ami_meter"})

#: Partition key used when writing the local store.
PARTITION_KEY = "dt_month"

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_SIZE = 1_000_000
STORE_SORT_KEY = ("site_id", "t_stamp")


def store_path(logical_name: str) -> Path:
    """Resolve a logical store table name to its path."""
    if logical_name not in STORE_TABLES:
        raise KeyError(
            f"Unknown store table {logical_name!r}. Known: {sorted(STORE_TABLES)}"
        )
    return STORE_DIR / STORE_TABLES[logical_name]


def duckdb_path_list(paths) -> str:
    """Render a list of paths as a DuckDB SQL array literal (forward slashes)."""
    rendered = [str(Path(p).as_posix()).replace("'", "''") for p in paths]
    return "[" + ", ".join(f"'{p}'" for p in rendered) + "]"


# ═══════════════════════════════════════════════════════════════════════════
# 4. TIME AND INTERVALS
# ═══════════════════════════════════════════════════════════════════════════
#
# `ts.t_stamp` is UTC, and `ts` is partitioned on UTC year/month. Every AEST
# rendering in the Stage 1 pipeline is `t_stamp + interval '10' hour`. Getting
# that wrong is exactly the R3/R9 bug class in the legacy conformance tables, so
# the offset is imported from ciccada_config rather than typed again.

ANALYSIS_UTC_OFFSET_HOURS = 10           # AEST, fixed, no DST
ANALYSIS_TZ_LABEL = "AEST (UTC+10, fixed)"

#: Native reporting interval of `ts`.
SOURCE_INTERVAL_MINUTES = 5
SOURCE_INTERVAL_H = SOURCE_INTERVAL_MINUTES / 60.0

#: Target interval of the synthetic meter. UNRESOLVED -- Phase 3 recommends and
#: you confirm. 30 is the placeholder because it is the dominant Australian NEM
#: settlement and retail-billing interval for residential AMI; 15 is the common
#: alternative and 5 exists in newer meters.
TARGET_INTERVAL_MINUTES = 30
TARGET_INTERVAL_RESOLVED = False

#: How each source column resamples. THIS IS THE ONE THAT WILL BITE.
#:
#: The Stage 1 pipeline treats them differently and they must stay different:
#:
#:   power            INSTANTANEOUS W. To energy over one 5-min interval:
#:                      kWh = power / 1000 * (5/60)
#:                    Resampling to 30 min is a SUM of those interval energies,
#:                    equivalently mean(power) * 0.5 h. Never a sum of powers.
#:
#:   energy_reactive  5-MINUTE kvarh, already an energy. Stage 1 multiplies by
#:                    12 to render it as an average kvar. Resampling to 30 min is
#:                    a plain SUM; multiplying by 12 first and then summing is
#:                    wrong by a factor of 6.
#:
#: UNRESOLVED: this is what the Stage 1 code IMPLIES the columns are. Phase 3
#: verifies it against the data before anything is built on it.
SOURCE_COLUMN_UNITS: dict[str, str] = {
    "power":           "instantaneous W (per Stage 1: /1000 -> kW)",
    "energy_reactive": "5-minute kvarh (per Stage 1: /1000*12 -> average kvar)",
    "voltage":         "instantaneous V",
}
SOURCE_COLUMN_UNITS_RESOLVED = False

RESAMPLE_RULE: dict[str, str] = {
    "power":           "to interval energy first, then SUM",
    "energy_reactive": "SUM as delivered (do NOT pre-multiply by 12)",
    "voltage":         "not carried into the meter; mean if ever needed",
}


# ═══════════════════════════════════════════════════════════════════════════
# 5. SIGNAL COMPOSITION AND SIGN  (Phase 3 resolves all of this)
# ═══════════════════════════════════════════════════════════════════════════
#
# ---------------------------------------------------------------------------
# THE SIGN CONVENTION OF THE SYNTHETIC METER
# ---------------------------------------------------------------------------
# A real AMI meter records energy at the connection point in the LOAD (consumer)
# convention: import positive. That is deliberately the OPPOSITE of the
# generator convention CICCADA uses for AS/NZS 4777.2 conformance work
# (`shared/as4777_curves.py`, `se_config` section 3), where +P = generating.
#
# Both are used in this package, in different tables, on purpose:
#
#   ami_raw    components as MAGNITUDES, each positive in its own natural sense:
#              pv_generation >= 0 (generated), gross_load >= 0 (consumed).
#              This makes the composition literally true:
#
#                  net_import = gross_load - pv_generation
#
#   ami_meter  LOAD convention. net_import_kwh > 0 means the site drew from the
#              grid; net_export_kwh > 0 means it pushed back. Real meters carry
#              the two as separate non-negative registers, so the synthetic one
#              does too:
#
#                  net_import_kwh = max(net, 0)
#                  net_export_kwh = max(-net, 0)
#
# `circuit_polarity` from `meta_up23c` is applied BEFORE any of this, at
# extraction, exactly as `build_structured_data.py` does
# (`sum(power * circuit_polarity)`). It normalises each circuit's own wiring
# direction and is not a convention choice -- it is a correction. Phase 3
# confirms what it actually does per circuit_type rather than assuming.
NET_SIGN_CONVENTION = (
    "load/consumer: net_import > 0 = drawn from grid, net_export > 0 = pushed to grid"
)
COMPONENT_SIGN_CONVENTION = (
    "magnitudes: pv_generation >= 0 (generated), gross_load >= 0 (consumed)"
)
#: Deliberately LEFT UNRESOLVED (2026-08-27), not flipped alongside
#: CIRCUIT_SIGNAL_MAP_RESOLVED above: Phase 3 Section 5's real-fleet
#: bidirectional-circuit_type check flagged `ac_load_net` itself (not just
#: sub-loads) as bidirectional (share_negative/share_positive both above the
#: noise-floor threshold) -- i.e. real sites export through `ac_load_net`,
#: not just import. That is in real tension with `gross_load >= 0` above.
#: Phase 4's interval-level table sidesteps this (it emits raw/corrected
#: `power`, not a sign-applied `gross_load` column -- see `ami_resolution`),
#: so it does not need this resolved to proceed, but whoever applies
#: `circuit_polarity` and builds `gross_load` (originally sketched as
#: Phase 5's `ami_build`) must revisit this before trusting the magnitude
#: convention for `ac_load_net` specifically.
SIGN_CONVENTION_RESOLVED = False

# ---------------------------------------------------------------------------
# CIRCUIT -> SIGNAL MAPPING
# ---------------------------------------------------------------------------
# UNRESOLVED. Phase 3 fills this from the actual `circuit_type` census.
#
# What is already known from the existing code, and must be CONFIRMED not
# assumed: `SolA2024_Analysis/NetMeter.ipynb` splits circuits three ways --
# `circuit_type LIKE 'pv_%'`, `circuit_type = 'ac_load_net'`, and everything
# else -- which means at least one whole-site NET circuit type exists in this
# fleet. Notebook output elsewhere in the repository shows values including
# `ac_load`, `ac_load_net`, `pv_site`, `pv_site_net`, `load_air_conditioner`,
# `load_hot_water`, `load_machine`, `load_pool`, `load_stove`.
#
# Note `pv_site_net` and `ac_load_net`. If a site carries `ac_load_net` AND its
# sub-loads, summing all of them double-counts the sub-loads. On a site whose
# sub-loads dominate, that roughly doubles gross_load -- and the result looks
# entirely plausible. This is the single highest-risk defect in the build, which
# is why Phase 3 must PROVE aggregate status arithmetically (does the candidate
# aggregate equal the sum of the others, interval by interval?) rather than
# inferring it from the substring "net".
# RESOLVED, 2026-08-27, for the CORE two signals only (Phase 3's real-fleet
# aggregation-check + PV-night-diagnostic findings): `ac_load_net` is a
# genuine whole-site load signal, not a sum to reconstruct (0/200 sites
# `is_aggregate=True`), and `pv_site_net` is confirmed generation-only
# (0/36 tested sites showed net-of-load behaviour at night). Named sub-loads
# (`load_pool`, `load_hot_water`, `load_other`, etc.) are bonus/optional --
# NOT yet assigned a signal name here, since Phase 4's core ground-truth
# table does not require them (see `bms_sa_review/ami_data_analysis`
# Phase 4 kickoff doc). Add them here if/when a later phase needs them.
CIRCUIT_SIGNAL_MAP: dict[str, str] = {
    "ac_load_net": "gross_load",
    "pv_site_net": "pv_generation",
}
#: No circuit_type has been proven an aggregate of its own siblings at fleet
#: scale (Phase 3, Section 4: 0/200 sites `is_aggregate=True`) -- stays
#: empty, not UNRESOLVED-empty; this IS the resolved answer.
AGGREGATE_CIRCUIT_TYPES: frozenset[str] = frozenset()
CIRCUIT_SIGNAL_MAP_RESOLVED = True

#: Signals the build emits. Sub-loads are appended once Phase 3 names them.
CORE_SIGNALS = ("pv_generation", "gross_load")

#: Do battery and EV circuits belong in `gross_load`? A battery is not a load;
#: it is a second controllable resource, and folding its discharge into
#: gross_load makes the ground truth wrong in a way disaggregation cannot
#: recover. UNRESOLVED -- Phase 3 establishes whether such circuits exist here
#: at all before the question needs answering.
STORAGE_HANDLING = "UNRESOLVED"

#: Source dataset. RESOLVED, 2026-08-26, in notebook 02 -- raw `ts` (joined to
#: `meta_up23c` for circuit metadata). `structured_data` (all variants) was
#: ruled out on two independent grounds, either one sufficient alone: it has
#: no `is_pv` column at all (`Sources.verify_is_pv_only` confirms it can only
#: ever hold one signal), and even where load rows exist elsewhere the single
#: `p_kw_norm` scalar per site is not decomposable back into components. See
#: PHASE 2 FINDINGS below for the full comparison.
SOURCE_CHOICE = "ts"


# ═══════════════════════════════════════════════════════════════════════════
# 6. PHASE 1 FINDINGS  (measured 2026-08-26, via Inventory.probe_partitions)
# ═══════════════════════════════════════════════════════════════════════════
#
# Facts about the catalogue, not methodological choices -- recorded here so
# Phase 2 onward read them rather than repeating a `$partitions` scan that
# already answered them for free. Re-measure and update if the catalogue
# changes (a re-load, a new Stage 1 rebuild, ...).

#: THE number Phase 2 needed. `ts` row counts by its is_pv partition key --
#: load circuits (is_pv=false) are NOT a discarded minority, they are in fact
#: slightly the larger half of the table.
TS_ROWS_BY_IS_PV: dict[str, int] = {
    "is_pv=false (load)": 8_521_715_460,
    "is_pv=true (pv)":    7_823_538_598,
}
TS_TOTAL_ROWS = sum(TS_ROWS_BY_IS_PV.values())            # 16,345,254,058

#: Approximate -- reconstructed from the 2-decimal-place GB `Athena.fmt_bytes`
#: printed, not the raw byte counts, so treat these as +/- a few MB.
TS_SIZE_GB_BY_IS_PV: dict[str, float] = {
    "is_pv=false (load)": 252.02,
    "is_pv=true (pv)":    204.91,
}
TS_TOTAL_SIZE_GB = sum(TS_SIZE_GB_BY_IS_PV.values())      # ~456.93 GB compressed

#: `ts` reports 816 total partitions -- RESOLVED, 2026-08-26, from the raw
#: `$partitions` frame. The initial guess here (Iceberg partition-spec
#: evolution -- day granularity early, month granularity later) was WRONG; the
#: real explanation is a FOURTH partition dimension this module did not know
#: about. The raw `partition` struct reads
#: `{year=..., month=..., is_pv=..., postcode_bu...}` (the fourth key is
#: truncated in pandas' display -- confirm the exact name with
#: `list(ts_tidy.columns)` before relying on it by name). Arithmetic proof:
#:     24 months x 2 is_pv values x 17 (postcode buckets) = 816   exact
#: and the by-month row counts sum to exactly TS_TOTAL_ROWS. So `ts` is
#: partitioned on (year, month, is_pv, <postcode bucket>), at MONTH
#: granularity throughout -- there is no day-level data and no partition-spec
#: evolution. `normalise_partitions` already handles this correctly: it
#: expands every key in the struct generically, so `ts_tidy` carries the
#: postcode-bucket column even though nothing here names it explicitly, and
#: every groupby used so far (by is_pv, by year/month) sums across it
#: correctly. It matters for Phase 4: chunking `ts` by (year, month) alone
#: undercounts the true partition count by 17x, which changes the resumability
#: math.
TS_PARTITION_COUNT_ANOMALY_RESOLVED = True

#: Confirmed date coverage, from the coverage-by-month table: 24 consecutive
#: months, no gaps.
TS_COVERAGE = ("2024-01", "2025-12")
TS_COVERAGE_MONTHS = 24

#: `structured_data` (all variants) confirmed PV-only by schema (no `is_pv`
#: column at all -- it does not need one, because `ts.is_pv = True` is baked
#: into `build_structured_data.py` before the table is ever written) and by
#: partitioning (year, month only -- no is_pv key, consistent with a table
#: that only ever holds one is_pv value). NOT a candidate source for
#: gross_load. Confirms the concern in the original brief.
STRUCTURED_DATA_IS_PV_ONLY = True

#: circuits (dimension table, presumed canonical circuit list): 171,411 rows.
N_CIRCUITS_DIM_TABLE = 171_411
#: meta_up23c: 423,990 rows -- ~2.47x more than `circuits`. meta_up23c is
#: therefore NOT one row per circuit_id. `build_structured_data.py` already
#: guards this with `GROUP BY circuit_id` + `max(...)` over every metadata
#: column before joining (see `_insert_sql`'s inner subquery) -- that is the
#: established convention and `ami_signal` (Phase 3) must follow it, not
#: rediscover it. WHY meta_up23c fans out (time-versioned rows? re-registration
#: history?) is not yet established -- a Phase 3 question.
N_META_UP23C_ROWS = 423_990
N_SITES = 41_393

#: Databases confirmed OUT OF SCOPE for this project -- not Solar Analytics
#: data, or scratch/infrastructure artefacts. Excluded from every later phase
#: without re-justifying it each time.
OUT_OF_SCOPE_DATABASES = frozenset({
    "sapn2022",     # a different DNSP's dataset (South Australia Power Networks)
    "test_db",      # scratch tables (evm_batch_*)
    "elb_logdb",    # AWS load-balancer access logs
})
#: Individual tables out of scope within an otherwise in-scope database.
OUT_OF_SCOPE_TABLES = frozenset({
    "solar_analytics.test_sola_2025_7",
    "solar_analytics.test_sola_2025_8",
    "solar_analytics.test_sola_2025_9",
    "solar_analytics.test_sola_2025_12",
})


def phase1_findings():
    """The measured facts above, as one table. Printable from any later notebook."""
    import pandas as pd

    return pd.DataFrame([
        ("ts: rows, is_pv=false (load)", f"{TS_ROWS_BY_IS_PV['is_pv=false (load)']:,}"),
        ("ts: rows, is_pv=true (pv)",    f"{TS_ROWS_BY_IS_PV['is_pv=true (pv)']:,}"),
        ("ts: total rows",               f"{TS_TOTAL_ROWS:,}"),
        ("ts: is_pv=false share of rows",
         f"{TS_ROWS_BY_IS_PV['is_pv=false (load)'] / TS_TOTAL_ROWS:.1%}"),
        ("ts: total size (compressed, approx)", f"{TS_TOTAL_SIZE_GB:.1f} GB"),
        ("ts: partition count anomaly resolved", str(TS_PARTITION_COUNT_ANOMALY_RESOLVED)),
        ("ts: date coverage", f"{TS_COVERAGE[0]} .. {TS_COVERAGE[1]} ({TS_COVERAGE_MONTHS} months, no gaps)"),
        ("structured_data: is PV-only",  str(STRUCTURED_DATA_IS_PV_ONLY)),
        ("circuits (dimension table)",   f"{N_CIRCUITS_DIM_TABLE:,} rows"),
        ("meta_up23c",                   f"{N_META_UP23C_ROWS:,} rows "
                                          f"({N_META_UP23C_ROWS / N_CIRCUITS_DIM_TABLE:.2f}x "
                                          "circuits -- fans out, GROUP BY circuit_id required)"),
        ("sites",                        f"{N_SITES:,} rows"),
        ("out-of-scope databases",       ", ".join(sorted(OUT_OF_SCOPE_DATABASES))),
    ], columns=["finding", "value"])


# ═══════════════════════════════════════════════════════════════════════════
# 6b. PHASE 2 FINDINGS  (measured 2026-08-26, notebook 02_source_selection)
# ═══════════════════════════════════════════════════════════════════════════
#
# The source-selection decision, and the evidence behind it -- recorded so
# Phase 3 onward can cite it rather than re-deriving it. Live re-checks in
# notebook 02 (schema check, fresh partition probe, row-count cross-check
# against Phase 1's numbers) all passed without discrepancy on this run.

#: `meta_up23c` circuit counts, by is_pv -- for reference against `ts`'s row
#: split (TS_ROWS_BY_IS_PV above). Not identical shares, and not expected to
#: be: circuits differ in how much history each has, so a circuit-count share
#: and a row-count share are different things. Recorded here as a sanity
#: check, not as a value future code should index into by name.
N_CIRCUITS_BY_IS_PV: dict[str, int] = {
    "is_pv=false (load)": 33_462,
    "is_pv=true (pv)":    27_108,
}
N_SITES_BY_IS_PV: dict[str, int] = {
    "is_pv=false (load)": 15_167,
    "is_pv=true (pv)":    16_148,
}

#: `structured_data_v2_flex_included` (the live `Config.TABLES["structured_data"]`
#: target), measured fresh in notebook 02 -- matches the shape implied by
#: Phase 1's inventory (year, month partitioning; no is_pv key).
STRUCTURED_DATA_V2_FLEX_N_ROWS = 871_655_350
STRUCTURED_DATA_V2_FLEX_SIZE_GB = 36.62

#: Full-table scan costs, AUD, as computed by `Sources.build_comparison_table`
#: -- the cost of building from EVERY row, before any partition/date pruning
#: Phase 4 will actually apply. Included so a future re-read of this file
#: doesn't have to re-run the notebook to see why cost never overrode the
#: decision: `ts` costs more and was chosen anyway, because it is the only
#: candidate with both signals and a decomposable grain.
TS_FULL_SCAN_COST_AUD = 3.57
STRUCTURED_DATA_V2_FLEX_FULL_SCAN_COST_AUD = 0.29

#: `Sources.recommend()`'s verdict on the two candidates it was given.
QUALIFYING_SOURCE = "raw `ts` + `meta_up23c`"
EXCLUDED_SOURCES: dict[str, str] = {
    "`structured_data_v2_flex_included` (site-level, PV-only)":
        "no load signal, not decomposable at this grain",
}

#: Total Athena cost actually incurred running notebook 02 end to end --
#: the sample-circuit-pruning trick (per-file min/max `circuit_id` stats)
#: worked as intended: 8 queries, well under the ~8.5 GB worst case quoted
#: to the user before the notebook was run.
PHASE2_NOTEBOOK_SCAN_MB = 74.61
PHASE2_NOTEBOOK_SCAN_COST_AUD = 0.0011


def phase2_findings():
    """The Phase 2 source-selection evidence, as one table."""
    import pandas as pd

    return pd.DataFrame([
        ("source dataset chosen", SOURCE_CHOICE),
        ("qualifying candidate", QUALIFYING_SOURCE),
        ("excluded candidate(s)", "; ".join(
            f"{name} -- {reason}" for name, reason in EXCLUDED_SOURCES.items())),
        ("meta_up23c circuits, is_pv=false (load)",
         f"{N_CIRCUITS_BY_IS_PV['is_pv=false (load)']:,}"),
        ("meta_up23c circuits, is_pv=true (pv)",
         f"{N_CIRCUITS_BY_IS_PV['is_pv=true (pv)']:,}"),
        ("structured_data_v2_flex_included: rows",
         f"{STRUCTURED_DATA_V2_FLEX_N_ROWS:,}"),
        ("structured_data_v2_flex_included: size",
         f"{STRUCTURED_DATA_V2_FLEX_SIZE_GB:.2f} GB"),
        ("ts: full-scan cost (reference, not what Phase 4 will pay)",
         f"AUD {TS_FULL_SCAN_COST_AUD:.2f}"),
        ("notebook 02 actual cost",
         f"{PHASE2_NOTEBOOK_SCAN_MB:.2f} MB, AUD {PHASE2_NOTEBOOK_SCAN_COST_AUD:.4f}"),
    ], columns=["finding", "value"])


# ═══════════════════════════════════════════════════════════════════════════
# 7. REPORTING
# ═══════════════════════════════════════════════════════════════════════════

#: (label, value, resolved) for every convention above. `resolved=False` means
#: the value is a placeholder and any result depending on it is provisional.
def _convention_rows() -> list[tuple[str, object, bool]]:
    return [
        ("source dataset", SOURCE_CHOICE, SOURCE_CHOICE != "UNRESOLVED"),
        ("source interval", f"{SOURCE_INTERVAL_MINUTES} min", True),
        ("target AMI interval", f"{TARGET_INTERVAL_MINUTES} min", TARGET_INTERVAL_RESOLVED),
        ("analysis frame", ANALYSIS_TZ_LABEL, True),
        ("t_stamp frame in `ts`", "UTC; partitions are UTC year/month", True),
        ("power column units", SOURCE_COLUMN_UNITS["power"], SOURCE_COLUMN_UNITS_RESOLVED),
        ("energy_reactive units", SOURCE_COLUMN_UNITS["energy_reactive"],
         SOURCE_COLUMN_UNITS_RESOLVED),
        ("resample: power", RESAMPLE_RULE["power"], SOURCE_COLUMN_UNITS_RESOLVED),
        ("resample: energy_reactive", RESAMPLE_RULE["energy_reactive"],
         SOURCE_COLUMN_UNITS_RESOLVED),
        ("circuit_polarity", "applied at extraction, per build_structured_data.py", True),
        ("component signs", COMPONENT_SIGN_CONVENTION, SIGN_CONVENTION_RESOLVED),
        ("meter signs", NET_SIGN_CONVENTION, SIGN_CONVENTION_RESOLVED),
        ("composition", "net_import = gross_load - pv_generation", SIGN_CONVENTION_RESOLVED),
        ("circuit -> signal map",
         f"{len(CIRCUIT_SIGNAL_MAP)} entries", CIRCUIT_SIGNAL_MAP_RESOLVED),
        ("aggregate circuit types",
         sorted(AGGREGATE_CIRCUIT_TYPES) or "none identified yet",
         CIRCUIT_SIGNAL_MAP_RESOLVED),
        ("battery / EV handling", STORAGE_HANDLING, STORAGE_HANDLING != "UNRESOLVED"),
        ("Athena price", f"{ATHENA_PRICE_CURRENCY} {ATHENA_PRICE_PER_TB:.2f} / TB scanned", True),
    ]


def describe_conventions():
    """
    Every convention this package depends on, with its resolution state.

    Printed by `00_connection_check` and folded into `manifest()` from Phase 5
    onward, so these choices travel with every result rather than living only in
    a comment. `resolved = False` is the important column: it marks a value that
    is a placeholder, not a decision.
    """
    import pandas as pd

    return pd.DataFrame(
        [{"convention": name, "value": value, "resolved": resolved}
         for name, value, resolved in _convention_rows()]
    )


def unresolved() -> list[str]:
    """Names of every convention still awaiting evidence."""
    return [name for name, _, resolved in _convention_rows() if not resolved]


def describe_paths() -> list[tuple[str, str, bool]]:
    """(label, path, exists) for every path this module defines."""
    entries = [
        ("REPO_ROOT", REPO_ROOT),
        ("STORE_DIR", STORE_DIR),
        ("ARTEFACT_DIR", ARTEFACT_DIR),
        ("DUCKDB_TEMP_DIR", DUCKDB_TEMP_DIR),
    ]
    return [(label, str(path), Path(path).exists()) for label, path in entries]
