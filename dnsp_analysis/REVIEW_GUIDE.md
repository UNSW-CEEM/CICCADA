# Delivery 1 review guide

This guide is the recommended order for reviewing the package together. Do not
start with the full 303-million-row run.

## Session 1: contract and configuration

Review:

1. `README.md`
2. `notebooks/01_foundation_pipeline.ipynb`
3. `config/analysis.example.toml`
4. `DATA_CONTRACT.md`
5. `QUESTIONS_FOR_AUSGRID.md`

Confirm the input paths, provisional signs, GMT/UTC timestamp interpretation,
resource limits and the location of generated outputs.

## Session 2: source and metadata foundation

Review:

1. `src/dnsp_analysis/config.py`
2. `src/dnsp_analysis/schemas.py`
3. `src/dnsp_analysis/inventory.py`
4. `src/dnsp_analysis/metadata.py`
5. `scripts/00_inventory_sources.py`
6. `scripts/01_prepare_metadata.py`

Expected outputs:

- source schema and coverage inventory;
- canonical metadata parquet;
- telemetry/metadata ID reconciliation;
- explicit solar-only and solar-plus-battery cohorts;
- null `s_rated_kva` until a defensible rating source is available.

## Session 3: duplicate policy

Review:

1. the duplicate section in `ARCHITECTURE.md`;
2. `src/dnsp_analysis/duplicates.py`;
3. `tests/test_duplicates.py`.

The key decision is intentionally conservative: identical repeated records are
collapsed, while conflicting repeated records are quarantined rather than
averaged or selected arbitrarily.

## Session 4: canonical telemetry

Review:

1. `src/dnsp_analysis/canonical.py`;
2. `src/dnsp_analysis/validation.py`;
3. `tests/test_sign_conventions.py`;
4. `tests/test_timezones.py`;
5. `tests/test_foundation_e2e.py`.

Focus on:

- preservation of raw P and Q;
- configurable normalized signs;
- UTC and daylight-saving-aware local timestamps;
- metadata availability flags;
- exact source-to-canonical row accounting.

## Session 5: orchestrator

Review:

1. `notebooks/01_foundation_pipeline.ipynb`;
2. `src/dnsp_analysis/foundation.py`;
3. `scripts/run_foundation_pipeline.py`;
4. `scripts/02_build_canonical_phase.py`.

The notebook is the primary orchestrator. It presents each build and validation
stage separately and requires the exact `RUN FULL DATASET` confirmation before
the full section. The command-line orchestrator is the batch equivalent.

## Session 6: first controlled run

Install the package, copy the example configuration to `analysis.toml`, then
run:

```powershell
pytest
jupyter lab
```

Open `notebooks/01_foundation_pipeline.ipynb`, retain the default April
2025/bucket-0 scope, and run one cell at a time.

Review these small-scope outputs before widening the run:

```text
derived/samples/month_2025_04__bucket_0_of_32/_manifests/
derived/samples/month_2025_04__bucket_0_of_32/audit/
derived/samples/month_2025_04__bucket_0_of_32/canonical_phase/
```

Only after the row accounting and duplicate reports make sense should the run
be widened to a complete month, then to `--full`.

## Delivery 2 gate

Do not implement Volt-VAr or Volt-Watt classification until Delivery 1 has
established:

- trustworthy keys and duplicate handling;
- timestamp behavior;
- phase availability;
- metadata coverage;
- sign behavior observed in the data.

Irradiance matching, connected-phase inference and inverter-rating questions
remain deliberately open for the next delivery.
