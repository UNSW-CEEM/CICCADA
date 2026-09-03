"""
CICCADA OEM extension
===========================

Local-first, out-of-core reproduction of the `bms_sa_review` Volt-VAr / Volt-Watt
conformance and curtailment analysis on the OEM fleet dataset.

Design rules (mirroring `bms_sa_review`):

* Logic lives in `.py` modules; notebooks are thin orchestrators.
* Separation of concerns: queries -> metrics -> plots.
* AS/NZS 4777.2:2020 set-points have exactly one definition, imported from
  `bms_sa_review.shared.as4777_curves` / `ciccada_config`. Nothing here restates them.
* Every methodological choice is a validated dataclass field and appears in `manifest()`.
* The data-access layer is DuckDB over local Parquet. There is no AWS runtime dependency;
  the single Athena touch (the BOM irradiance extract, deliverable D12a) is a one-off
  that lands a local Parquet file.
"""

__version__ = "0.1.0"
