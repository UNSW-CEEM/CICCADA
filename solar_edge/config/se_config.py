"""
Shared constants and paths for the SolarEdge extension.
=======================================================

This is the ONLY place that writes down:
  * where the raw data and the derived store live,
  * the raw schema contract and expected inventory,
  * the sign / unit conventions used to translate SolarEdge telemetry into the
    CICCADA convention,
  * the per-state timezone map and the DST resolution policy,
  * the logical -> physical name registry for the derived store.

AS/NZS 4777.2:2020 set-points are NOT restated here. They are imported from
`bms_sa_review.shared.ciccada_config` so that the SolarEdge and Solar Analytics
results are comparable by construction.
"""

from __future__ import annotations

import os
import re
import tempfile
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 0. REPOSITORY ROOT AND IMPORT BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════

def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards until we find the directory containing `bms_sa_review`."""
    current = (start or Path(__file__)).resolve()
    for path in (current, *current.parents):
        if (path / "bms_sa_review").is_dir() and (path / "solar_edge").is_dir():
            return path
    raise RuntimeError(
        "Could not locate the CICCADA repository root (expected a directory "
        f"containing both 'bms_sa_review' and 'solar_edge'). Searched upwards from {current}."
    )


REPO_ROOT = find_repo_root()


def bootstrap_sys_path() -> Path:
    """
    Put the repository root on sys.path so `bms_sa_review.*` and `solar_edge.*`
    both import cleanly from a notebook. Mirrors the bootstrap cell used by the
    `bms_sa_review` notebooks.
    """
    for entry in (REPO_ROOT, REPO_ROOT / "bms_sa_review"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    return REPO_ROOT


# ═══════════════════════════════════════════════════════════════════════════
# 1. PATHS
# ═══════════════════════════════════════════════════════════════════════════
#
# The raw data lives OUTSIDE the git repository, in OneDrive. Override with the
# CICCADA_SE_DATA_ROOT environment variable if your layout differs.

_DEFAULT_DATA_ROOT = (
    Path.home() / "OneDrive - UNSW" / "Documents" / "CICCADA - Data" / "solar edge"
)

DATA_ROOT = Path(os.environ.get("CICCADA_SE_DATA_ROOT", _DEFAULT_DATA_ROOT))

#: The extracted SolarEdge delivery.
DELIVERY_DIR = DATA_ROOT / "OneDrive_1_6-18-2026"

#: Raw monthly Parquet files, as delivered.
RAW_DIR = DELIVERY_DIR / "data_2025_01_12_alias_zipped"

#: Alias -> postcode/state mapping (the entire available site metadata).
ALIAS_MAPPING_CSV = DELIVERY_DIR / "alias_mapping_alias_only.csv"

def _default_store_dir() -> Path:
    """
    Where the derived store lives by default: a LOCAL application-data directory,
    deliberately NOT inside OneDrive and NOT inside the git repository.

    The store is ~1.5 GB and is rewritten on every rebuild. Inside a synced folder
    that means OneDrive re-uploading the whole thing each time, competing for the
    file handles DuckDB is writing through. It is also fully regenerable from the
    raw delivery, so it is exactly the kind of thing that should not be synced or
    versioned.

    Windows      -> %LOCALAPPDATA%\\ciccada\\solar_edge_store
    Linux/macOS  -> $XDG_DATA_HOME/ciccada/solar_edge_store, else
                    ~/.local/share/ciccada/solar_edge_store

    Override with CICCADA_SE_STORE_DIR. The override is passed through
    `expandvars`/`expanduser`, so both `%LOCALAPPDATA%\\...` and `~/...` work.
    """
    override = os.environ.get("CICCADA_SE_STORE_DIR")
    if override:
        return Path(os.path.expandvars(override)).expanduser()

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "ciccada" / "solar_edge_store"
    return Path.home() / ".local" / "share" / "ciccada" / "solar_edge_store"


STORE_DIR = _default_store_dir()

#: Small, human-reviewable outputs that DO belong in the repository.
ARTEFACT_DIR = REPO_ROOT / "solar_edge" / "artefacts"

#: ABS POA-2021 postcode boundaries, needed only for D4 geography and D12 irradiance.
#: The same shapefile `BOM_NCI/Get_ALL_postcodes_ABS.ipynb` reads. Not in this
#: repository; set CICCADA_SE_POA_SHAPEFILE to wherever your copy lives.
POA_SHAPEFILE = Path(
    os.environ.get(
        "CICCADA_SE_POA_SHAPEFILE",
        DATA_ROOT.parent / "POA_2021_AUST_GDA2020_SHP" / "POA_2021_AUST_GDA2020.shp",
    )
)

#: Node spacing of the `bom_nci.solar` satellite grid, in degrees.
#: UNCONFIRMED -- inferred from BOM_NCI/process_bom.ipynb rounding lat/lon to 2dp.
#: Verify at D12a with `SELECT DISTINCT latitude FROM bom_nci.solar ORDER BY 1 LIMIT 20`.
BOM_GRID_SPACING_DEG = 0.02

#: Filename pattern of the raw files, e.g. "data_2025_06_alias_zipped (1).parquet".
RAW_FILENAME_RE = re.compile(
    r"^data_(?P<year>\d{4})_(?P<month>\d{2})_alias_zipped.*\.parquet$", re.IGNORECASE
)


def raw_files(raw_dir: Path | None = None) -> list[Path]:
    """Return the raw monthly Parquet files, sorted by (year, month)."""
    raw_dir = raw_dir or RAW_DIR
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw SolarEdge directory not found: {raw_dir}\n"
            "Set the CICCADA_SE_DATA_ROOT environment variable if your data lives elsewhere."
        )
    found = []
    for path in raw_dir.iterdir():
        match = RAW_FILENAME_RE.match(path.name)
        if match:
            found.append((int(match["year"]), int(match["month"]), path))
    if not found:
        raise FileNotFoundError(f"No files matching {RAW_FILENAME_RE.pattern!r} in {raw_dir}")
    return [path for _, _, path in sorted(found)]


def raw_month_of(path: Path) -> str:
    """'data_2025_06_alias_zipped (1).parquet' -> '2025-06'."""
    match = RAW_FILENAME_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"Not a recognised raw SolarEdge filename: {path}")
    return f"{match['year']}-{match['month']}"


def duckdb_path_list(paths) -> str:
    """
    Render a list of paths as a DuckDB SQL array literal.

    Forward slashes are used throughout: DuckDB accepts them on Windows, and they
    avoid backslash-escaping problems. Single quotes are doubled.
    """
    rendered = [str(Path(p).as_posix()).replace("'", "''") for p in paths]
    return "[" + ", ".join(f"'{p}'" for p in rendered) + "]"


# ═══════════════════════════════════════════════════════════════════════════
# 2. RAW SCHEMA CONTRACT
# ═══════════════════════════════════════════════════════════════════════════
#
# All 12 delivered files carry this schema, verified byte-identical by the
# schema hash in se_diagnostics.raw_inventory(). If it ever changes, D1 fails
# loudly rather than the analysis failing quietly downstream.

RAW_COLUMNS: dict[str, str] = {
    "site_alias":           "string",   # anonymised site id, 'AUS001'...
    "timestamp":            "string",   # 'YYYY-MM-DD HH:MM:SS.ffffff', LOCAL CIVIL TIME
    "active_power_1":       "float",    # W,   phase 1
    "active_power_2":       "float",    # W,   phase 2 (null for single-phase sites)
    "active_power_3":       "float",    # W,   phase 3 (null for single-phase sites)
    "reactive_power_1":     "float",    # var, phase 1   -- see SIGN CONVENTION below
    "reactive_power_2":     "float",    # var, phase 2
    "reactive_power_3":     "float",    # var, phase 3
    "ac_voltage_1":         "float",    # V,   phase 1
    "ac_voltage_2":         "float",    # V,   phase 2
    "ac_voltage_3":         "float",    # V,   phase 3
    "ac_frequency_1":       "float",    # Hz,  phase 1
    "ac_frequency_2":       "float",    # Hz,  phase 2
    "ac_frequency_3":       "float",    # Hz,  phase 3
    "derating_active_flag": "float",    # 1.0 when derating active, else NULL (never 0.0)
}

PHASES = (1, 2, 3)

#: Expected row count per delivered file. Measured from the Parquet footers on
#: 12 Aug 2026. D1 asserts against this.
EXPECTED_RAW_ROWS: dict[str, int] = {
    "2025-01": 8_232_278,
    "2025-02": 7_090_717,
    "2025-03": 7_218_954,
    "2025-04": 6_619_100,
    "2025-05": 6_439_017,
    "2025-06": 6_074_226,
    "2025-07": 6_422_544,
    "2025-08": 6_783_051,
    "2025-09": 7_059_218,
    "2025-10": 7_864_708,
    "2025-11": 8_118_656,
    "2025-12": 8_720_716,
}

EXPECTED_RAW_TOTAL_ROWS = sum(EXPECTED_RAW_ROWS.values())

#: Rows in alias_mapping_alias_only.csv (excluding the header).
EXPECTED_N_SITES = 1_602

#: Study period.
STUDY_YEAR = 2025
STUDY_MONTHS = tuple(EXPECTED_RAW_ROWS)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SIGN AND UNIT CONVENTIONS
# ═══════════════════════════════════════════════════════════════════════════
#
# ---------------------------------------------------------------------------
# THE TWO CONVENTIONS
# ---------------------------------------------------------------------------
# Generator (source) convention  -- current flowing OUT of the device is positive
#     +P = generating / exporting
#     +Q = SUPPLYING (injecting) reactive power
#     -Q = ABSORBING (consuming) reactive power
#   This is what AS/NZS 4777.2:2020 Fig 3.2 uses, and what CICCADA uses
#   throughout (`bms_sa_review/shared/as4777_curves.py` states it explicitly).
#   In the 240-258 V band the standard REQUIRES Q < 0.
#
# Load (consumer / sink) convention -- current flowing INTO the device is positive
#     +P = consuming / importing
#     +Q = ABSORBING (inductive)
#     -Q = SUPPLYING (capacitive)
#   This is what the Ausgrid AMI dataset uses, and what SolarEdge appears to use
#   for reactive power.
#
# ---------------------------------------------------------------------------
# WHAT SOLAREDGE ACTUALLY REPORTS -- A MIXED CONVENTION
# ---------------------------------------------------------------------------
# Active power:   reported as a PRODUCTION MAGNITUDE. Over the whole 2025
#   dataset `active_power_*` has min = 0 and ZERO negative values, so it is
#   already generator-positive. No sign change is required.
#
# Reactive power: reported in the LOAD CONVENTION, i.e. POSITIVE = ABSORBING.
#   A sign flip IS required to reach the CICCADA generator convention.
#
# Mixed reporting like this is common in inverter telemetry, where "production"
# and "reactive power" come from different registers.
#
# ---------------------------------------------------------------------------
# EVIDENCE FOR THE REACTIVE-POWER SIGN
# ---------------------------------------------------------------------------
# No SolarEdge documentation was available. The convention was established
# empirically and confirmed by Bernardo Mendonca Severiano on 12 Aug 2026, then
# frozen here. The evidence, from the 2025-06 file, single-phase sites, P > 200 W:
#
#   * Per-site median Q at V < 235 vs V > 250, sites with >= 50 samples in both
#     bands (n = 53): Q RISES with voltage in 50 of 53 sites; fleet median
#     delta = +56.7 var.
#   * Strongest responders show a textbook Volt-VAr curve:
#         AUS765   +136 var  ->  +3,314 var
#         AUS351    +78 var  ->  +2,451 var
#         AUS989   +347 var  ->  +1,648 var
#   * Fleet-wide binned medians run from about -90 var at 210-215 V to +227 var
#     at 255 V, i.e. supplying at low voltage and absorbing at high voltage --
#     the shape of the AS/NZS 4777.2 Australia A Volt-VAr curve.
#
# Under the generator convention that pattern would mean the entire single-phase
# fleet SUPPLIES more reactive power as voltage rises, which is the wrong
# direction under a mandatory standard and is not physically credible at fleet
# scale. Hence: SolarEdge positive Q = absorbing = load convention.
#
# CAVEAT ON RECORD: the 405 three-phase sites show a flat median phase Q of
# about -90 var across every voltage bin -- no Volt-VAr signature at all -- while
# their derating flag still responds to voltage. The sign convention above is
# inferred from the single-phase cohort. The three-phase cohort is characterised
# separately in 02_fleet_eda before the two are pooled.
#
# ---------------------------------------------------------------------------
# THE TRANSFORM
# ---------------------------------------------------------------------------
#     P_kW   = ACTIVE_POWER_SIGN   * sum(active_power_*)   * W_TO_KW
#     Q_kvar = REACTIVE_POWER_SIGN * sum(reactive_power_*) * VAR_TO_KVAR
#
# After this, +Q = supplying and -Q = absorbing, matching `as4777_curves`, and
# the Method A / Method B `Q_kvar < 0` absorbing tests port unchanged.

#: +1.0 -- SolarEdge active power is already generator-positive.
ACTIVE_POWER_SIGN = +1.0

# ---------------------------------------------------------------------------
# REACTIVE SIGN -- REVISED 13 Aug 2026. The store now holds the value AS
# DELIVERED. Read the reasoning before changing this.
# ---------------------------------------------------------------------------
# This was -1.0 from 12 to 13 Aug, on the strength of a sample of 53 single-phase
# sites whose median raw Q rose +56.7 var between the low- and high-voltage bands.
# That sample was dominated by the ~800 sites which barely respond at all (fleet
# median power factor 0.995-0.997); their tens-of-var wobble is noise around a
# small standing offset, not a Volt-VAr signature. The sites that actually
# implement the curve move by KILOvars, and they move in BOTH directions.
#
# The fleet-wide test (se_sign.fleet_orientation_fit, all 1,590 assessable sites)
# compares measured Q against the required curve across the 241-253 V ramp in both
# orientations, against the +/-4% tolerance band:
#
#     cohort          neither fits   raw fits   stored (flipped) fits
#     single-phase             944        127                     105
#     three-phase              327         86                       1
#     all                    1,271        213                     106
#
# So: as-delivered fits twice as many sites as flipped, and the three-phase cohort
# is 86:1 in its favour. Hence +1.0.
#
# BUT NOTE WHAT THAT TABLE ALSO SAYS. It is not a clean answer. Single-phase splits
# 127/105 -- the reported polarity is inconsistent WITHIN a cohort, and no single
# constant is correct for the whole fleet. 106 sites are misoriented under this
# setting, just as 213 were under the previous one.
#
# The residual is handled downstream rather than pretended away:
#   * se_analysis orientation is a CONFIG PARAMETER (SEAnalysisConfig.
#     reactive_orientation), so it can be swept without another ingest;
#   * se_adverse classifies every adverse-direction site by whether its MAGNITUDE
#     tracks the required curve, which separates a likely polarity artefact from
#     genuinely adverse behaviour.
#
# Direction-based conformance remains formally unresolved until SolarEdge confirms
# the convention. Magnitude-based conformance is orientation-independent and is
# not affected.
REACTIVE_POWER_SIGN = +1.0

#: Human-readable labels, printed by `manifest()` so the choice travels with every result.
ACTIVE_POWER_SOURCE_CONVENTION = "generator (production magnitude, always >= 0)"
REACTIVE_POWER_SOURCE_CONVENTION = (
    "AS DELIVERED -- majority of curve-following sites are already generator "
    "convention (negative = absorbing)"
)
TARGET_CONVENTION = "generator (AS/NZS 4777.2 Fig 3.2; negative Q = absorbing)"
SIGN_CONVENTION_BASIS = (
    "fleet_orientation_fit over all 1,590 assessable sites, 2026-08-13: "
    "213 fit as-delivered, 106 fit flipped, 1,271 fit neither. "
    "NOT unanimous -- 106 sites remain misoriented; see se_adverse"
)

W_TO_KW = 1.0 / 1000.0
VAR_TO_KVAR = 1.0 / 1000.0

#: Reactive power is INSTANTANEOUS var. Unlike Solar Analytics' `energy_reactive`,
#: it must NOT be multiplied by 12.
REACTIVE_IS_INSTANTANEOUS = True

# --- derating_active_flag ---------------------------------------------------
# The raw column is 1.0 or NULL, never 0.0, so the source cannot distinguish
# "not derating" from "not reported". The ingest collapses NULL to FALSE, which
# is the only usable reading, but it is an INTERPRETATION and is recorded as one.
#
# Consequence for Method C (D14): precision against this flag is interpretable,
# recall is not. Any claim of the form "Method A missed N derating intervals" is
# unsound; "of the intervals flagged as derating, Method A caught N" is sound.
DERATING_NULL_INTERPRETATION = "NULL -> FALSE (interpretation; source has no explicit 0)"


# ═══════════════════════════════════════════════════════════════════════════
# 4. TIME
# ═══════════════════════════════════════════════════════════════════════════
#
# The `timestamp` column is a naive string in each site's LOCAL CIVIL TIME,
# INCLUDING daylight saving. Established from the power-weighted centroid of the
# diurnal profile by state:
#
#     State  June centroid  January centroid   Interpretation
#     QLD        11.79 h        11.96 h        no DST, stable  (Brisbane 11:48 AEST)
#     NSW        11.91 h        13.07 h        DST applied     (Sydney  11:55 AEST)
#     SA         12.30 h        13.48 h        DST applied     (Adelaide 12:16 ACST)
#
# A single fixed AEST frame would put the SA June centroid near 12.77 h. It does
# not. So the timestamps are per-site local civil time, and the October/April
# DST transitions are real discontinuities that must be resolved at ingest.

#: State (as spelled in alias_mapping_alias_only.csv) -> IANA timezone.
STATE_TIMEZONE: dict[str, str] = {
    "New South Wales": "Australia/Sydney",     # AEST / AEDT
    "South Australia": "Australia/Adelaide",   # ACST / ACDT
    "Queensland":      "Australia/Brisbane",   # AEST year-round, no DST
}

#: Fixed analysis frame, matching `ciccada_config.FIXED_OFFSET`. Solar physics does
#: not observe daylight saving, and a fixed offset avoids the DST discontinuity.
ANALYSIS_UTC_OFFSET_HOURS = 10  # AEST
ANALYSIS_TZ_LABEL = "AEST (UTC+10, fixed)"

# --- DST resolution policy -------------------------------------------------
# Resolution is delegated to DuckDB's ICU `AT TIME ZONE`, and these constants
# record what it does so the behaviour is documented rather than merely inherited.
# Verified against Python's `zoneinfo` in tests/test_se_ingest.py.
#
# April (DST ends, 2025-04-06): 02:00-02:59 local occurs TWICE. ICU resolves to
#   STANDARD time, i.e. the SECOND occurrence. Equivalent to zoneinfo fold=1.
#   Consequence: a site reporting through the whole overlap yields two rows that
#   map to the same UTC instant. Those are counted and deduplicated at ingest,
#   never silently collapsed.
#
# October (DST starts, 2025-10-05): 02:00-02:59 local does NOT exist. ICU shifts
#   forward to 03:00-03:59 AEDT. This lands on the same UTC instant that zoneinfo
#   produces for the same input, so the two implementations agree.
#
# Both are deterministic and neither drops data. The window is 02:00-03:00 on one
# night, which falls in the ~3% overnight coverage and carries no generation, so
# the practical impact is nil -- but it is asserted rather than assumed.
DST_AMBIGUOUS_POLICY = "standard_time (second occurrence; zoneinfo fold=1)"
DST_NONEXISTENT_POLICY = "shift_forward (to the post-transition offset)"
DST_ENGINE = "DuckDB ICU AT TIME ZONE"

#: Nominal reporting interval. Modal inter-sample gap is 300 s.
INTERVAL_MINUTES = 5
INTERVAL_H = INTERVAL_MINUTES / 60.0

#: Timestamps are NOT aligned to a common 5-minute grid; each site has its own
#: phase offset (e.g. 10:05:01, 14:42:04). Any cross-site or external join needs
#: an explicit alignment rule -- see D3.
TIMESTAMPS_ARE_GRID_ALIGNED = False


# ═══════════════════════════════════════════════════════════════════════════
# 5. STORE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
#
# Logical name -> path within STORE_DIR. This is the ONLY place these are written
# down, mirroring `ciccada_config.TABLES`.
#
# `se_interval` is the SolarEdge analogue of `structured_data`.

STORE_TABLES: dict[str, str] = {
    "se_interval":       "se_interval",         # site-level tidy facts, partitioned by month
    "se_interval_phase": "se_interval_phase",   # phase-level facts, partitioned by month
    "se_site":           "se_site.parquet",     # site dimension
    "se_site_capacity":  "se_site_capacity.parquet",
    "bom_solar":         "bom_solar_2025.parquet",
    "se_structured":     "se_structured",       # se_interval + GHI / GHI_cs
    "se_ghi_model":      "se_ghi_model.parquet",
    "se_uncurtailedpv":  "se_uncurtailedpv",
    # Stage 2 conformance, site-day grain, matching conformance_voltvar_v2 /
    # conformance_voltwatt_v2 so the two studies compare directly.
    "se_conformance_voltvar":  "se_conformance_voltvar.parquet",
    "se_conformance_voltwatt": "se_conformance_voltwatt.parquet",
}

#: Which store tables exist as Hive-partitioned directories rather than single files.
PARTITIONED_TABLES = {"se_interval", "se_interval_phase", "se_structured", "se_uncurtailedpv"}

#: Columns computed by the registered view rather than stored on disk.
#:
#: `ts_aest` is exactly `ts_utc + 10 h`. Measured at 21 MB per month -- about 250 MB
#: across the store -- for a value DuckDB derives for free. Computing it in the view
#: keeps the ergonomic win (queries say `hour(ts_aest)`, never `+ interval '10' hour`,
#: which is the bug class that produced R3/R9 in the legacy conformance tables) at
#: zero storage cost.
#:
#: Contrast with `state` and `postcode`, which ARE stored: dictionary encoding makes
#: them cost 0.004 MB and 0.009 MB per month, so normalising them away would save
#: nothing and add a join to every query.
STORE_VIEW_PROJECTION: dict[str, str] = {
    "se_interval": f"*, ts_utc + INTERVAL '{ANALYSIS_UTC_OFFSET_HOURS}' HOUR AS ts_aest",
}

#: Partition key used when writing.
PARTITION_KEY = "dt_month"

# --- Write settings (D3) ---------------------------------------------------
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
#: The delivered files have ONE row group of 8.2M rows, so nothing can be pruned.
#: Re-writing with bounded row groups is the highest-value ingest step.
PARQUET_ROW_GROUP_SIZE = 1_000_000
#: Sort within each partition so site- and time-filtered queries prune effectively.
STORE_SORT_KEY = ("site_alias", "ts_utc")


def store_path(logical_name: str) -> Path:
    """Resolve a logical store table name to its path."""
    if logical_name not in STORE_TABLES:
        raise KeyError(
            f"Unknown store table {logical_name!r}. Known: {sorted(STORE_TABLES)}"
        )
    return STORE_DIR / STORE_TABLES[logical_name]


# ═══════════════════════════════════════════════════════════════════════════
# 6. DUCKDB RUNTIME
# ═══════════════════════════════════════════════════════════════════════════
#
# The dataset is 1.57 GB compressed on disk (~12 GB as float64 in memory), so
# nothing is ever loaded whole. These bounds keep DuckDB comfortably out-of-core.

# Both default to None, meaning "let DuckDB decide" -- it sizes itself from the
# machine (roughly 80% of RAM, all cores). Hard-coding a limit is worse than no
# limit: too high and the process is killed on a small machine, too low and it
# spills needlessly on a large one. Override explicitly if you are sharing the
# machine, e.g. the CEEM compute server:
#     CICCADA_SE_DUCKDB_MEMORY=8GB  CICCADA_SE_DUCKDB_THREADS=4
DUCKDB_MEMORY_LIMIT = os.environ.get("CICCADA_SE_DUCKDB_MEMORY") or None
_threads = os.environ.get("CICCADA_SE_DUCKDB_THREADS")
DUCKDB_THREADS = int(_threads) if _threads else None

#: Spill directory for out-of-core operations (the large sorts during ingest).
#:
#: Deliberately NOT inside STORE_DIR. Spill files are transient and can run to
#: gigabytes; putting them next to the store would make OneDrive sync churn on
#: every rebuild, and on read-only or permission-restricted mounts DuckDB cannot
#: clean them up at all. The system temp directory is local and fast, which is
#: what a spill target should be.
DUCKDB_TEMP_DIR = Path(
    os.environ.get("CICCADA_SE_DUCKDB_TEMP", Path(tempfile.gettempdir()) / "ciccada_se_duckdb")
)


# ═══════════════════════════════════════════════════════════════════════════
# 7. RE-EXPORT THE STANDARD
# ═══════════════════════════════════════════════════════════════════════════
#
# Imported, never restated. If this import fails, the repository root is not on
# sys.path -- call bootstrap_sys_path() first.

def as4777():
    """Return the AS/NZS 4777.2:2020 constants dict from the shared CICCADA config."""
    bootstrap_sys_path()
    from bms_sa_review.shared.ciccada_config import AS4777
    return AS4777


def describe_conventions():
    """
    The sign, unit and time conventions, as a DataFrame.

    Printed by `00_environment_check` and folded into `manifest()` from D5 onward,
    so these choices travel with every result rather than living only in a comment.
    """
    import pandas as pd

    rows = [
        ("active power: source convention", ACTIVE_POWER_SOURCE_CONVENTION),
        ("active power: sign applied", f"{ACTIVE_POWER_SIGN:+.0f} (no change)"),
        ("reactive sign: sites fitting", "213 as-delivered / 106 flipped / 1,271 neither"),
        ("reactive power: source convention", REACTIVE_POWER_SOURCE_CONVENTION),
        ("reactive power: sign applied", f"{REACTIVE_POWER_SIGN:+.0f} (no change)"),
        ("target convention", TARGET_CONVENTION),
        ("basis for the reactive sign", SIGN_CONVENTION_BASIS),
        ("active power units", "W -> kW"),
        ("reactive power units", "var -> kvar (instantaneous; NOT multiplied by 12)"),
        ("raw timestamp frame", "per-site local civil time, INCLUDING daylight saving"),
        ("state -> timezone", ", ".join(f"{k} = {v}" for k, v in STATE_TIMEZONE.items())),
        ("analysis frame", ANALYSIS_TZ_LABEL),
        ("DST ambiguous (April overlap)", DST_AMBIGUOUS_POLICY),
        ("DST nonexistent (October gap)", DST_NONEXISTENT_POLICY),
        ("nominal interval", f"{INTERVAL_MINUTES} min (INTERVAL_H = {INTERVAL_H:.6f})"),
        ("timestamps grid-aligned", str(TIMESTAMPS_ARE_GRID_ALIGNED)),
        ("capacity basis", "s_99 only (no nameplate exists in this delivery)"),
        ("derating flag NULL handling", DERATING_NULL_INTERPRETATION),
    ]
    return pd.DataFrame(rows, columns=["convention", "value"])


def describe_paths() -> list[tuple[str, str, bool]]:
    """(label, path, exists) for every path this module defines. Used by 00_environment_check."""
    entries = [
        ("REPO_ROOT", REPO_ROOT),
        ("DATA_ROOT", DATA_ROOT),
        ("DELIVERY_DIR", DELIVERY_DIR),
        ("RAW_DIR", RAW_DIR),
        ("ALIAS_MAPPING_CSV", ALIAS_MAPPING_CSV),
        ("STORE_DIR", STORE_DIR),
        ("ARTEFACT_DIR", ARTEFACT_DIR),
    ]
    return [(label, str(path), Path(path).exists()) for label, path in entries]
