# Architecture

## Design principles

- Raw files are immutable.
- Every transformation is local and reproducible.
- Large telemetry operations run in DuckDB, not pandas.
- Raw and normalised sign conventions coexist.
- Duplicate resolution is explicit and auditable.
- Conflicting measurements are quarantined, not averaged.
- Sample outputs cannot contaminate full-run outputs.
- A result is not considered valid merely because a script completed.
- Jupyter notebooks are the primary review-and-run orchestrators.
- Notebooks call tested modules; calculations are not duplicated inside plots.

## Pipeline

```text
source parquet ──────┐
                     ├─ inventory
metadata workbook ───┘

metadata workbook ───── metadata canonicalisation ── ID reconciliation

source parquet ── duplicate-key audit ── canonical phase parquet
                                              │
                                              └─ foundation validation

notebook orchestrator ── runs and inspects each stage above
```

## Generated layout

```text
derived/
├── _duckdb/
│   ├── foundation.duckdb
│   └── tmp/
├── _manifests/
│   ├── source_inventory.json
│   ├── metadata_summary.json
│   └── foundation_run_<timestamp>.json
├── metadata/
│   ├── metadata_canonical.parquet
│   └── id_reconciliation.csv
├── audit/
│   ├── duplicate_key_audit.parquet
│   ├── duplicate_summary.json
│   └── canonical_validation.json
├── canonical_phase/
│   └── year_utc=YYYY/month_utc=M/site_bucket=N/*.parquet
└── samples/
    └── <scope>/
        ├── audit/
        └── canonical_phase/
```

Scoped runs use `derived/samples/<scope>` and never write to the full-run
`audit`, reconciliation or `canonical_phase` paths. The small canonical
metadata parquet is shared, but telemetry-ID reconciliation always uses the
active scope.

## Duplicate policy

The canonical key is:

```text
serial + MeasureTime + Vphase
```

For every repeated key, the audit records:

- number of source rows;
- number of distinct physical payloads;
- number of source files;
- duplicate classification.

Physical payload:

```text
Volts + Curr + ReactPow + ActivePow
```

Classification:

- `identical_duplicate`: multiple rows, one physical payload;
- `conflicting_duplicate`: multiple physical payloads for the same key.

Canonical treatment:

- unique keys: retained;
- identical duplicates: one row retained with `duplicate_count`;
- conflicting duplicates: all versions excluded from canonical telemetry and
  retained in the audit.

The configured floating-point tolerance is applied by rounding physical values
before payload hashing. This avoids classifying insignificant serialisation
noise as a substantive conflict.

## Timestamp contract

The loader removed timezone metadata from the parquet, but the workbook states
that `MeasureTime` is GMT. The pipeline therefore:

1. interprets the stored naive timestamp as UTC;
2. stores a timezone-aware `timestamp_utc`;
3. derives `timestamp_local` using `Australia/Sydney`;
4. uses DST-aware local time rather than a fixed UTC+10 offset.

UTC year/month are used for output partitioning.

## Sign contract

Provider values remain unchanged:

```text
active_power_raw_w
reactive_power_raw_var
```

Normalised fields are calculated from configuration:

```text
p_export_w       = active_power_raw_w × active_export_sign
q_absorbing_var  = reactive_power_raw_var × reactive_absorbing_sign
q_generator_var  = -q_absorbing_var
```

With the initial `-1` assumptions:

- negative raw active power becomes positive export;
- negative raw reactive power becomes positive absorption;
- generator-convention Q remains negative when absorbing.

## Scope and resumability

A scope can be:

- one month and one site bucket;
- one month and all site buckets;
- the complete dataset.

Every scope has its own duplicate audit, canonical output and validation
report. The orchestrator requires `--overwrite` before replacing an existing
canonical directory.

## Delivery 1 boundary

Delivery 1 does not:

- infer inverter-connected phases;
- aggregate phases into an inverter output;
- join BOM irradiance;
- infer `S_rated`;
- score AS/NZS 4777 conformance;
- estimate energy or uncurtailed generation.

## Notebook contract

Each delivery notebook must provide:

- a deterministic small-slice run;
- tabular and visual output inspection;
- assertions between stages;
- bounded pandas samples for plots;
- explicit full-run confirmation;
- reusable pipeline logic imported from `src`, not hidden notebook-only
  transformations.
