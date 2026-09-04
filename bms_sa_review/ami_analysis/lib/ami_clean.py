"""
`ami_meter` quality checks and cleaning -- Phase 7 (`bms_sa_review/ami_analysis`).

This module deliberately keeps two kinds of "outlier" apart, because
CICCADA's own research question makes them opposite things to do with:

* **HARD flags** -- physically impossible or internally inconsistent
  readings (negative current, voltage <= 0V or > 300V, |power factor| > 1,
  duplicate/missing key fields, a stored apparent power that doesn't match
  V x I for the same row). These are sensor or pipeline faults, never a real
  grid state. `apply_cleaning` drops rows carrying any hard flag.
* **SOFT flags** -- readings that are extreme relative to a circuit's own
  history (via a robust per-circuit MAD threshold) but not physically
  impossible. A single-phase circuit briefly at 270V during a PV export
  event, or a load spike from an EV charger, is exactly the kind of
  AS/NZS 4777.2 Volt-Watt/Volt-VAr-relevant event this project exists to
  study -- blanket-removing it as "noise" would delete the signal, not the
  fault. `apply_cleaning` keeps these rows, tagged with their flag column,
  so downstream analysis can see them without having to trust every one.

The 0V/300V hard voltage bound and the general shape of these checks follow
a convention this project has already validated on real data (see
`claude/ami_phase3_status_and_phase4_plan.md`: a colleague's
`structured_data` build applies a `voltage > 0 and voltage < 300` sanity
filter, described there as "a reasonable pattern to borrow directly").
300V is well outside AS/NZS 4777.2's sustained operating envelope
(nominally ~180-270V) on the high side, and 0V/negative readings cannot
represent a live, reporting circuit -- both sides of that band are pipeline
faults, not grid events, which is why the bound sits there rather than
tighter around the standard's own thresholds.

Every function here is pure (DataFrame/Series in, DataFrame/Series/dict
out) and stateless -- no Athena/DuckDB import in this module, mirroring
`synthetic_ami_creation`'s own `ami_*` modules. A notebook wires in the
real `ami_meter` table; tests wire in small synthetic frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "flag_voltage_outliers",
    "flag_power_factor_outliers",
    "flag_negative_current",
    "flag_apparent_power_inconsistency",
    "flag_extreme_power_magnitude",
    "flag_duplicate_readings",
    "flag_missing_critical_fields",
    "HARD_FLAGS",
    "SOFT_FLAGS",
    "build_quality_report",
    "apply_cleaning",
]


def flag_voltage_outliers(
    frame: pd.DataFrame, *, voltage_column: str = "V",
    hard_min: float = 0.0, hard_max: float = 300.0,
) -> pd.Series:
    """
    HARD flag: `voltage_column` <= `hard_min` or > `hard_max`. Physically
    impossible for a live, reporting circuit -- not a bound on genuine grid
    voltage excursions (AS/NZS 4777.2's own operating envelope, nominally
    ~180-270V, is narrower than this and deliberately NOT enforced here;
    values inside 0..300V but outside that envelope are exactly the
    compliance-relevant events this project studies and must stay in the
    cleaned table, not be flagged).
    """
    v = frame[voltage_column]
    return (v <= hard_min) | (v > hard_max) | v.isna()


def flag_power_factor_outliers(
    frame: pd.DataFrame, *, pf_column: str = "power_factor", tol: float = 1e-3,
) -> pd.Series:
    """HARD flag: |power factor| > 1 + `tol`. cos(phi) cannot exceed 1 in magnitude."""
    if pf_column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[pf_column].abs() > (1.0 + tol)


def flag_negative_current(frame: pd.DataFrame, *, current_column: str = "current_a") -> pd.Series:
    """HARD flag: negative current. `current_a` is a magnitude, never signed."""
    if current_column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[current_column] < 0


def flag_apparent_power_inconsistency(
    frame: pd.DataFrame, *,
    voltage_column: str = "V", current_column: str = "current_a",
    apparent_power_column: str = "S_kva", rel_tol: float = 0.2,
    min_current_a: float = 0.5,
) -> pd.Series:
    """
    HARD flag: the row's own V x I (converted to kVA) disagrees with its
    stored `S_kva` by more than `rel_tol` (relative). These two are
    independently sourced -- `S_kva` is derived upstream from
    `P_kw`/`Q_kvar` (a power-side measurement), while V and `current_a` are
    a separate voltage/current-side measurement of the same row -- so a
    genuine mismatch flags a metering/pipeline fault (e.g. a wrong CT
    ratio) rather than a real grid event. Skipped (flag False) below
    `min_current_a`, where V x I noise dominates and a relative comparison
    is meaningless.
    """
    if not {voltage_column, current_column, apparent_power_column} <= set(frame.columns):
        return pd.Series(False, index=frame.index)
    implied_kva = (frame[voltage_column] * frame[current_column]) / 1000.0
    stable = frame[current_column] >= min_current_a
    denom = frame[apparent_power_column].abs().clip(lower=1e-9)
    rel_error = (implied_kva - frame[apparent_power_column]).abs() / denom
    return stable & (rel_error > rel_tol)


def flag_extreme_power_magnitude(
    frame: pd.DataFrame, *,
    power_column: str = "P_kw", circuit_column: str = "circuit_id",
    hard_abs_limit_kw: float = 100.0, mad_k: float = 8.0, min_mad_kw: float = 0.05,
) -> pd.Series:
    """
    SOFT flag: `power_column` is either beyond a generous absolute physical
    ceiling (`hard_abs_limit_kw` -- no residential single circuit draws
    100kW) OR far (`mad_k` median-absolute-deviations) from ITS OWN
    circuit's median. Per-circuit, robust (median/MAD, not mean/std) rather
    than one fleet-wide threshold, because a normally-small circuit's real
    anomaly and a normally-large EV-charging circuit's normal operation
    would otherwise be indistinguishable under a single cutoff.
    `min_mad_kw` floors the MAD so a near-constant circuit (MAD ~ 0)
    doesn't flag every tiny wobble.
    """
    p = frame[power_column]
    hard = p.abs() > hard_abs_limit_kw

    grouped = p.groupby(frame[circuit_column])
    median = grouped.transform("median")
    mad = (p - median).abs().groupby(frame[circuit_column]).transform("median")
    mad = mad.clip(lower=min_mad_kw)
    robust_z = (p - median).abs() / mad
    return hard | (robust_z > mad_k)


def flag_duplicate_readings(
    frame: pd.DataFrame, *, key_columns=("site_id", "circuit_id", "t_stamp"),
) -> pd.Series:
    """HARD flag: every row sharing `key_columns` with an earlier row (keep=first kept unflagged)."""
    key_columns = list(key_columns)
    if not set(key_columns) <= set(frame.columns):
        return pd.Series(False, index=frame.index)
    return frame.duplicated(subset=key_columns, keep="first")


def flag_missing_critical_fields(
    frame: pd.DataFrame, *, columns=("V", "P_kw"),
) -> pd.Series:
    """HARD flag: any of `columns` is null. These are the fields every other check here depends on."""
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series(False, index=frame.index)
    return frame[present].isna().any(axis=1)


#: Flags that mark a row as a data-quality fault -- dropped by `apply_cleaning`.
HARD_FLAGS = (
    "voltage_implausible",
    "power_factor_implausible",
    "current_negative",
    "apparent_power_inconsistent",
    "duplicate_reading",
    "missing_critical_field",
)

#: Flags that mark a row as extreme-but-physically-possible -- kept, tagged.
SOFT_FLAGS = (
    "power_magnitude_extreme",
)


def build_quality_report(frame: pd.DataFrame, flags: dict[str, pd.Series]) -> pd.DataFrame:
    """
    One row per flag in `flags`: `n_flagged`, `share_flagged`, and which
    bucket (`hard`/`soft`) it belongs to per `HARD_FLAGS`/`SOFT_FLAGS`.
    Pure summary -- does not touch `frame` beyond `len`.
    """
    n_total = len(frame)
    rows = []
    for name, flag in flags.items():
        n_flagged = int(flag.sum())
        rows.append({
            "flag": name,
            "kind": "hard" if name in HARD_FLAGS else ("soft" if name in SOFT_FLAGS else "unclassified"),
            "n_flagged": n_flagged,
            "share_flagged": (n_flagged / n_total) if n_total else 0.0,
        })
    return pd.DataFrame(rows, columns=["flag", "kind", "n_flagged", "share_flagged"])


def apply_cleaning(
    frame: pd.DataFrame, flags: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split `frame` into `(clean, removed)` using `flags`: a row is removed
    if ANY flag in `HARD_FLAGS` present in `flags` is True for it; every
    SOFT flag present in `flags` is instead added to `clean` as its own
    boolean column (never used to drop a row) so it survives into
    downstream analysis. Flags not classified in either tuple are ignored
    here (caller error if that wasn't intended -- `build_quality_report`
    surfaces them as `kind="unclassified"` so this doesn't fail silently).
    """
    hard_present = [name for name in HARD_FLAGS if name in flags]
    if hard_present:
        drop_mask = pd.concat([flags[name] for name in hard_present], axis=1).any(axis=1)
    else:
        drop_mask = pd.Series(False, index=frame.index)

    removed = frame.loc[drop_mask].copy()
    clean = frame.loc[~drop_mask].copy()

    for name in SOFT_FLAGS:
        if name in flags:
            clean[name] = flags[name].loc[~drop_mask].values

    return clean.reset_index(drop=True), removed.reset_index(drop=True)
