"""
Shared constants and paths for the OEM extension.
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
    # Walk upwards until we find the directory containing `bms_sa_review`
    current = (start or Path(__file__)).resolve()
    for path in (current, *current.parents):
        if (path / "bms_sa_review").is_dir() and (path / "oem_analysis").is_dir():
            return path
    raise RuntimeError(
        "Could not locate the CICCADA repository root (expected a directory "
        f"containing both 'bms_sa_review' and 'oem_analysis'). Searched upwards from {current}."
    )

REPO_ROOT = find_repo_root()

def bootstrap_sys_path() -> Path:
    """
    Put the repository root on sys.path so `bms_sa_review.*` and `oem_analysis.*`
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

_DEFAULT_DATA_ROOT = None  # no fallback. CICCADA_SE_DATA_ROOT must be set

DATA_ROOT = Path(os.environ["CICCADA_SE_DATA_ROOT"])

#: The extracted OEM zip contining hte parquets
DELIVERY_DIR = DATA_ROOT / "OneDrive_1_6-18-2026"

#: Raw monthly Parquet files
RAW_DIR = DELIVERY_DIR / "data_2025_01_12_alias_zipped"

# Alias -> postcode/state mapping 
ALIAS_MAPPING_CSV = DELIVERY_DIR / "alias_mapping_alias_only.csv"

def _default_store_dir() -> Path:
    """
    Where the derived store lives by default: 
    a LOCAL application-data directory

    Windows      -> %LOCALAPPDATA%\\ciccada\\oem_analysis_store
    Linux/macOS  -> $XDG_DATA_HOME/ciccada/oem_analysis_store, else
                    ~/.local/share/ciccada/oem_analysis_store

    """
    override = os.environ.get("CICCADA_SE_STORE_DIR")
    if override:
        return Path(os.path.expandvars(override)).expanduser()

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "ciccada" / "oem_analysis_store"
    return Path.home() / ".local" / "share" / "ciccada" / "oem_analysis_store"


STORE_DIR = _default_store_dir()

# artefacts (should be gitgnored)
ARTEFACT_DIR = REPO_ROOT / "oem_analysis" / "artefacts"

#: ABS POA-2021 postcode boundaries, needed for geography and irradiance.
POA_SHAPEFILE = Path(
    os.environ.get(
        "CICCADA_SE_POA_SHAPEFILE",
        DATA_ROOT.parent / "POA_2021_AUST_GDA2020_SHP" / "POA_2021_AUST_GDA2020.shp",
    )
)

#: Node spacing of the `bom_nci.solar` satellite grid, in degrees.
#: UNCONFIRMED -- inferred from BOM_NCI/process_bom.ipynb rounding lat/lon to 2dp.
#: Verify with `SELECT DISTINCT latitude FROM bom_nci.solar ORDER BY 1 LIMIT 20`.
BOM_GRID_SPACING_DEG = 0.02

# Filename pattern of the raw files
RAW_FILENAME_RE = re.compile(
    r"^data_(?P<year>\d{4})_(?P<month>\d{2})_alias_zipped.*\.parquet$", re.IGNORECASE
)

def raw_files(raw_dir: Path | None = None) -> list[Path]:
    """Return the raw monthly Parquet files, sorted by (year, month)."""
    raw_dir = raw_dir or RAW_DIR
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw directory not found: {raw_dir}\n"
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
        raise ValueError(f"Not a recognised raw filename: {path}")
    return f"{match['year']}-{match['month']}"


def duckdb_path_list(paths) -> str:
    """
    Render a list of paths as a DuckDB SQL array literal.
    """
    rendered = [str(Path(p).as_posix()).replace("'", "''") for p in paths]
    return "[" + ", ".join(f"'{p}'" for p in rendered) + "]"


# ═══════════════════════════════════════════════════════════════════════════
# 2. RAW SCHEMA CONTRACT
# ═══════════════════════════════════════════════════════════════════════════

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

#: Expected row count per delivered file. Measured from the Parquet footers on 12 Aug 2026.
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

# Rows in alias_mapping_alias_only.csv (excluding the header).
EXPECTED_N_SITES = 1_602

# period.
STUDY_YEAR = 2025
STUDY_MONTHS = tuple(EXPECTED_RAW_ROWS)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SIGN AND UNIT CONVENTIONS
# ═══════════════════════════════════════════════════════════════════════════
#
# ---------------------------------------------------------------------------
# CONVENTIONS
# ---------------------------------------------------------------------------
# Generator (source) convention  -- current flowing OUT of the device is positive
#     +P = generating / exporting
#     +Q = SUPPLYING (injecting) reactive power
#     -Q = ABSORBING (consuming) reactive power
#   This is what AS/NZS 4777.2:2020 Fig 3.2 uses, and what CICCADA uses
#   throughout (see `bms_sa_review/shared/as4777_curves.py`).
#
# Load (consumer / sink) convention -- current flowing INTO the device is positive
#     +P = consuming / importing
#     +Q = ABSORBING (inductive)
#     -Q = SUPPLYING (capacitive)
#   If needed, can use this convention
#
# ---------------------------------------------------------------------------
# WHAT THIS OEM ACTUALLY REPORTS
# ---------------------------------------------------------------------------
# Active power:   
#   Reported as a PRODUCTION MAGNITUDE. 
#   Over the whole 2025 
#   Dataset `active_power_*` has min = 0 and ZERO negative values, so it is already generator-positive. No sign change is required.
#
# Reactive power:
#   Seem to be reported in the LOAD CONVENTION, i.e. POSITIVE = ABSORBING.
#   A sign flip IS required to reach the CICCADA generator convention.
#
# ---------------------------------------------------------------------------
# EVIDENCE FOR THE REACTIVE-POWER SIGN
# ---------------------------------------------------------------------------
# No OEM documentation was available. The convention was established empirically
# The evidence, from the 2025-06 file, single-phase sites, P > 200 W:
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
# scale. Hence: OEM positive Q = absorbing = load convention.
#
# ---------------------------------------------------------------------------
# THE TRANSFORM
# ---------------------------------------------------------------------------
#     P_kW   = ACTIVE_POWER_SIGN   * sum(active_power_*)   * W_TO_KW
#     Q_kvar = REACTIVE_POWER_SIGN * sum(reactive_power_*) * VAR_TO_KVAR
#
# After this, +Q = supplying and -Q = absorbing, matching `as4777_curves`, and
# the Method A / Method B `Q_kvar < 0` absorbing tests port unchanged.

#: +1.0 -> OEM active power is already generator-positive.
ACTIVE_POWER_SIGN = +1.0

# ---------------------------------------------------------------------------
# REACTIVE SIGN. The store now holds the value AS DELIVERED.
# ---------------------------------------------------------------------------
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

# Reactive power is INSTANTANEOUS var. Unlike OEM Analytics' `energy_reactive`, it must NOT be multiplied by 12.
REACTIVE_IS_INSTANTANEOUS = True

# derating_active_flag
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

# State (as spelled in alias_mapping_alias_only.csv) -> timezone.
STATE_TIMEZONE: dict[str, str] = {
    "New South Wales": "Australia/Sydney",     # AEST / AEDT
    "South Australia": "Australia/Adelaide",   # ACST / ACDT
    "Queensland":      "Australia/Brisbane",   # AEST year-round, no DST
}

ANALYSIS_UTC_OFFSET_HOURS = 10  # AEST
ANALYSIS_TZ_LABEL = "AEST (UTC+10, fixed)"

# DST resolution policy
# Resolution is delegated to DuckDB's ICU `AT TIME ZONE`, 
# and these constants record what it does so the behaviour is documented
# Verified against Python's `zoneinfo` in tests/test_se_ingest.py.
DST_AMBIGUOUS_POLICY = "standard_time (second occurrence; zoneinfo fold=1)"
DST_NONEXISTENT_POLICY = "shift_forward (to the post-transition offset)"
DST_ENGINE = "DuckDB ICU AT TIME ZONE"

#: Nominal reporting interval. 
# Modal inter-sample gap is 300 s.
INTERVAL_MINUTES = 5
INTERVAL_H = INTERVAL_MINUTES / 60.0

# Timestamps are NOT aligned to a common 5-minute grid; each site has its own phase offset (e.g. 10:05:01, 14:42:04). 
# Any cross-site or external join needs an explicit alignment rule 
TIMESTAMPS_ARE_GRID_ALIGNED = False

# ═══════════════════════════════════════════════════════════════════════════
# 5. STORE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
#
# Logical name -> path within STORE_DIR.
#
# `se_interval` is the OEM analogue of `structured_data`.

STORE_TABLES: dict[str, str] = {
    "se_interval":       "se_interval",         # site-level tidy facts, partitioned by month
    "se_interval_phase": "se_interval_phase",   # phase-level facts, partitioned by month
    "se_site":           "se_site.parquet",     # site dimension
    "se_site_capacity":  "se_site_capacity.parquet",
    "bom_solar":         "bom_solar_2025.parquet",
    "se_structured":     "se_structured",       # se_interval + GHI / GHI_cs
    "se_ghi_model":      "se_ghi_model.parquet",
    "se_uncurtailedpv":  "se_uncurtailedpv",
    # Stage 2 conformance, site-day grain
    "se_conformance_voltvar":  "se_conformance_voltvar.parquet",
    "se_conformance_voltwatt": "se_conformance_voltwatt.parquet",
}

# Which store tables exist as Hive-partitioned directories rather than single files.
PARTITIONED_TABLES = {"se_interval", "se_interval_phase", "se_structured", "se_uncurtailedpv"}

# Columns computed by the registered view rather than stored on disk.
# `ts_aest` is exactly `ts_utc + 10 h`.
STORE_VIEW_PROJECTION: dict[str, str] = {
    "se_interval": f"*, ts_utc + INTERVAL '{ANALYSIS_UTC_OFFSET_HOURS}' HOUR AS ts_aest",
}

# Partition key used when writing.
PARTITION_KEY = "dt_month"

# Write settings
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_SIZE = 1_000_000
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
# CICCADA_SE_DUCKDB_MEMORY=8GB  CICCADA_SE_DUCKDB_THREADS=4
DUCKDB_MEMORY_LIMIT = os.environ.get("CICCADA_SE_DUCKDB_MEMORY") or None
_threads = os.environ.get("CICCADA_SE_DUCKDB_THREADS")
DUCKDB_THREADS = int(_threads) if _threads else None

# Spill directory for out-of-core operations
DUCKDB_TEMP_DIR = Path(
    os.environ.get("CICCADA_SE_DUCKDB_TEMP", Path(tempfile.gettempdir()) / "ciccada_se_duckdb")
)

# ═══════════════════════════════════════════════════════════════════════════
# 7. RE-EXPORT THE STANDARD
# ═══════════════════════════════════════════════════════════════════════════

def as4777():
    """Return the AS/NZS 4777.2:2020 constants dict from the shared CICCADA config."""
    bootstrap_sys_path()
    from bms_sa_review.shared.ciccada_config import AS4777
    return AS4777


def describe_conventions():
    """
    The sign, unit and time conventions, as a DataFrame.
    Printed by `00_environment_check` and folded into `manifest()`
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
