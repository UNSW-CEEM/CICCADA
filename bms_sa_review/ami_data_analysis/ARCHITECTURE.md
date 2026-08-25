# Synthetic AMI dataset — module sketch

**Status: DRAFT, written before Phase 1.** This is the plan the modules grow to,
not a description of what exists. Revise it as the phases land; a sketch that is
never corrected is worse than none.

## The thing being built

Two paired tables, joinable on `(site_id, t_stamp)`:

| Table | Grain | Content | Role |
|---|---|---|---|
| `ami_raw` | site x source interval | one column per component signal — `pv_generation`, `gross_load`, each available sub-load | ground truth |
| `ami_meter` | site x AMI interval | `net_import_kwh` / `net_export_kwh` only | what a real meter would have recorded |

`ami_meter` is a lossy projection of `ami_raw`. That asymmetry is the whole
point: disaggregation algorithms see only `ami_meter` and are scored against
`ami_raw`.

## Layout

Mirrors `solar_edge/`, which is the working precedent for a locally-run,
AWS-free analysis in this repository.

```
bms_sa_review/ami_data_analysis/
├── ARCHITECTURE.md          this file
├── config/
│   └── ami_config.py        THE one place for: paths, store registry, sign
│                            convention, interval constants, circuit->signal
│                            mapping, Athena cost constants
├── lib/
│   ├── ami_athena.py        [P0] Athena access, credential diagnosis,
│   │                             partition-predicate guard, per-query scan
│   │                             accounting in bytes and AUD
│   ├── ami_diagnostics.py   [P0] check/summarise table helpers, environment report
│   ├── ami_inventory.py     [P1] catalog tally: Glue tables, real schemas,
│   │                             Iceberg $partitions metadata, coverage
│   ├── ami_sources.py       [P2] candidate-source comparison; the is_pv
│   │                             question; granularity/cleanliness/cost matrix
│   ├── ami_taxonomy.py      [P3] circuit_type census, AGGREGATE DETECTION,
│   │                             battery/EV classification, cohort completeness
│   ├── ami_signal.py        [P3] THE circuit->signal mapping and THE sign
│   │                             convention. One place. Both bugs live here.
│   ├── ami_resample.py      [P3] 5-min -> AMI interval. Energy sums, not power
│   │                             averages. Power vs energy columns handled apart.
│   ├── ami_extract.py       [P4] Athena UNLOAD/CTAS, chunked and resumable,
│   │                             provenance sidecar. TOUCHES AWS. Runs once.
│   ├── ami_store.py         [P4] DuckDB connect + register views over local
│   │                             Parquet. Port of se_store.py. NO boto3.
│   ├── ami_params.py        [P5] AmiConfig dataclass + validate() + with_changes()
│   ├── ami_contract.py      [P5] manifest() — every methodological choice,
│   │                             printed alongside every result
│   ├── ami_build.py         [P5] ami_raw + ami_meter. Local only.
│   ├── ami_degrade.py       [P5] quantisation, dropouts, clock skew.
│   │                             Explicit, off by default, never in ground truth.
│   ├── ami_validate.py      [P6] invariants: sum-to-net, energy conservation,
│   │                             impossible values, coverage/gap stats
│   └── ami_plots.py         [P3/6] example-site composition and validation plots
├── notebooks/               00..06, orchestrate and narrate only
├── tests/
│   ├── test_notebook_names.py   ported from solar_edge
│   └── test_ami_*.py            one per module with arithmetic to get wrong
└── artefacts/               small, human-reviewable CSVs that belong in git
```

## The boundary that matters

```
   AWS                                    LOCAL
   ─────────────────────────────────      ─────────────────────────────────
   ami_athena  ─┐                         ami_store ─┬─ ami_build
   ami_inventory ├─ ami_extract ──────►   Parquet ───┤   ami_validate
   ami_sources  ─┘   (once)               on disk    └─ ami_plots
   ami_taxonomy ─┘
```

Everything left of the arrow imports `boto3`. Nothing right of it does, and
`tests/` asserts that for `ami_build` and `ami_validate`. `ami_taxonomy` and
`ami_signal` straddle: the *census* queries Athena, the *mapping* is a plain
dict that Phase 5 applies locally.

## Conventions this package must pin down, and where

| Decision | Home | Settled in |
|---|---|---|
| Which circuits compose which signal | `ami_config.CIRCUIT_SIGNAL_MAP` | Phase 3 |
| Which circuits are already aggregates | `ami_config.AGGREGATE_CIRCUIT_TYPES` | Phase 3 |
| Sign convention of the synthetic meter | `ami_config.NET_SIGN_CONVENTION` | Phase 3 |
| `power` vs `energy_reactive` unit handling | `ami_config.SOURCE_COLUMN_UNITS` | Phase 3 |
| Target AMI interval and resample rule | `ami_config.TARGET_INTERVAL_MINUTES` | Phase 3 |
| Cohort completeness criterion | `ami_params.AmiConfig.min_*` | Phase 3 |
| Source dataset | `ami_config.SOURCE_CHOICE` | Phase 2 |
| Where the local Parquet lives | `ami_config.STORE_DIR` | Phase 4 |

Nothing above is decided yet except by placeholder. Every placeholder is marked
`UNRESOLVED` in `ami_config` and every one of them is printed by `manifest()`,
so an unresolved choice cannot quietly become a silent default.

## Phase status

| Phase | Notebook | Modules | State |
|---|---|---|---|
| 0 | `00_connection_check` | `ami_athena`, `ami_diagnostics` | written |
| 1 | `01_data_lake_inventory` | `ami_inventory` | written |
| 2 | `02_source_selection` | `ami_sources` | not started |
| 3 | `03_signal_taxonomy` | `ami_taxonomy`, `ami_signal`, `ami_resample` | not started |
| 4 | `04_extract` | `ami_extract`, `ami_store` | not started |
| 5 | `05_build_ami` | `ami_params`, `ami_contract`, `ami_build`, `ami_degrade` | not started |
| 6 | `06_validate` | `ami_validate` | not started |
