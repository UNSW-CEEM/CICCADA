# Delivery 1 release notes

## Scope

This release provides the local foundation layer only. It does not score
Volt-VAr, Volt-Watt or AS/NZS 4777 behavior.

The primary orchestrator is now the step-by-step Jupyter notebook
`notebooks/01_foundation_pipeline.ipynb`. The CLI uses the same underlying
modules and remains available for batch reruns.

## Inputs verified

- The configured parquet path exists and exposes the nine expected columns.
- The metadata workbook contains `Cust_DER_Network Data`.
- That metadata sheet contains 1,282 data rows and all 19 expected fields.
- The workbook data dictionary identifies `MeasureTime` as GMT.
- The workbook describes active and reactive power as instantaneous samples at
  the measurement interval.

No full telemetry scan was performed while preparing this delivery.

## Working assumptions

- Negative raw active power means export.
- Negative raw reactive power means absorption.
- Source timestamps are GMT/UTC.
- The AMI measurement location remains unknown.
- Inverter rated apparent power remains unavailable.

All provisional assumptions are configuration values. Raw power fields are
retained so a later sign correction does not alter the source data.

## Verification performed

- Python syntax compilation passed.
- Thirteen tests passed.
- The test suite includes a synthetic end-to-end pipeline with unique rows,
  identical duplicates, conflicting duplicates, unmatched metadata, timezone
  conversion, canonical output and exact row accounting.
- The real parquet schema and workbook headers were checked read-only.
- The notebook structure, empty-output state, stage headings, safety gate and
  Python code cells are tested.

## First user-run checkpoint

Run one April 2025 site bucket first. Review the inventory, metadata
reconciliation, duplicate summary, conflicting rows and canonical validation
before increasing scope.
