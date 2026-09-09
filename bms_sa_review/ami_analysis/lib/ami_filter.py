"""
Residential-scope filtering and DNSP/manufacturer key remapping -- Phase 7
(`bms_sa_review/ami_analysis`), notebook 03.

Two independent, unrelated operations live in this module, both applied to
`ami_meter_clean`/`ami_site_metadata` to produce the notebook's outputs:

* **Residential-scope filtering**: this project's analysis is about
  residential PV/load behaviour, not utility-/commercial-scale systems, so
  sites whose `ac_capacity_kw` exceeds a residential ceiling are dropped
  entirely (see `flag_oversized_capacity`).
* **Key remapping**: DNSP and manufacturer names are replaced with the
  locally-held anonymised ID codes (`DNSP.csv`/`OEM.csv`, held OUTSIDE this
  repository -- see notebook 03's own path constant) before the data is
  used further, so the real DNSP/manufacturer names don't need to travel
  with every downstream file (see `build_key_mapping`/`apply_key_mapping`).

Every function here is pure (DataFrame/Series/dict in, DataFrame/Series/dict
out) and stateless, mirroring `ami_clean.py`'s own convention -- no file I/O
or DuckDB import in this module. A notebook wires in the real CSVs/tables;
tests wire in small synthetic frames.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "RESIDENTIAL_MAX_AC_CAPACITY_KW",
    "flag_oversized_capacity",
    "build_key_mapping",
    "apply_key_mapping",
]


#: Residential/commercial size boundary used to scope this analysis. There is
#: no single authoritative Australian residential/commercial PV threshold,
#: but 30kW ac sits comfortably above virtually every genuine single-phase
#: or three-phase residential installation and below small-commercial
#: systems, so it is used here as a practical, documented cutoff rather than
#: a regulatory one.
RESIDENTIAL_MAX_AC_CAPACITY_KW = 30.0


def flag_oversized_capacity(
    frame: pd.DataFrame, *,
    capacity_column: str = "ac_capacity_kw", max_kw: float = RESIDENTIAL_MAX_AC_CAPACITY_KW,
) -> pd.Series:
    """
    True where `capacity_column` exceeds `max_kw` -- sites flagged True are
    dropped by notebook 03 to keep this analysis to residential-scale PV.

    A NULL capacity is deliberately NOT flagged. It means "no PV capacity
    on record" (e.g. a load-only site with no PV circuit at all), which is
    not evidence the site is oversized -- a size filter has nothing to
    measure there, so it must not silently drop a site for lack of the
    very metadata the filter depends on.
    """
    if capacity_column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[capacity_column] > max_kw


def build_key_mapping(mapping_frame: pd.DataFrame, *, key_column: str, value_column: str) -> dict:
    """
    Build a `{key: value}` dict from a loaded mapping CSV (e.g. `DNSP.csv`'s
    `dnsp`/`dnsp_id` columns, `OEM.csv`'s `manufacturer`/`m_id` columns).

    Rows with a null `key_column` are dropped (nothing to map from). Raises
    `ValueError` if the same key maps to more than one distinct value in the
    file -- a genuinely ambiguous mapping the caller must fix in the CSV,
    not something this function should silently resolve one way.
    """
    frame = mapping_frame[[key_column, value_column]].dropna(subset=[key_column])
    ambiguous = frame.groupby(key_column)[value_column].nunique()
    ambiguous = sorted(ambiguous[ambiguous > 1].index.tolist())
    if ambiguous:
        raise ValueError(f"Ambiguous key mapping -- these keys map to more than one value: {ambiguous}")
    return dict(zip(frame[key_column], frame[value_column]))


def apply_key_mapping(
    series: pd.Series, mapping: dict, *, null_key: str = "None",
) -> tuple[pd.Series, list]:
    """
    Map `series` through `mapping`, returning `(mapped, unmapped_values)`.

    Null/NaN entries are first coalesced to `null_key` before lookup --
    this project's key-mapping CSVs represent "no DNSP"/"no manufacturer"
    as a literal `"None"` row rather than leaving it blank (see `DNSP.csv`),
    so a genuinely-missing value must be looked up the same way as any other
    key, not silently skipped.

    Any value with no entry in `mapping` (a real name found in the data but
    absent from the CSV -- e.g. a casing difference, or a manufacturer the
    key file hasn't caught up to yet) keeps its ORIGINAL value rather than
    becoming NaN, so a mapping gap never silently deletes the underlying
    column's identity. `unmapped_values` lists every distinct value this
    happened for, so the caller can surface it instead of it going
    unnoticed.
    """
    filled = series.fillna(null_key)
    mapped = filled.map(mapping)
    unmapped_mask = mapped.isna()
    unmapped_values = sorted(filled[unmapped_mask].unique().tolist())
    return mapped.where(~unmapped_mask, filled), unmapped_values
