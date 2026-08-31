# Delivery 1 foundation pipeline

This package builds an auditable local foundation for the DNSP AMI DER
dataset. It does **not** calculate Volt‑VAr or Volt‑Watt conformance yet.

The primary user interface is
`notebooks/01_foundation_pipeline.ipynb`. It runs each stage separately,
displays validation tables and plots, and stops at explicit gates before the
next stage. The command-line orchestrator remains available for repeatable
batch runs after the notebook workflow has been reviewed.

Delivery 1:

1. inventories the source parquet;
2. normalises and validates the metadata workbook;
3. reconciles telemetry IDs against metadata IDs;
4. classifies repeated `(serial, timestamp, phase)` keys;
5. collapses identical duplicates and quarantines conflicting duplicates;
6. writes canonical phase-level parquet;
7. validates row accounting, uniqueness, units, timestamps, signs and metadata
   coverage.

The source parquet and workbook are read-only. Derived outputs are created only
when **you** run the pipeline.

## Why DuckDB

The source parquet has more than 303 million rows. DuckDB queries the parquet
out of core and writes partitioned parquet without materialising the complete
dataset in pandas.

Pandas is used only for the 1,282-row metadata workbook and small reports.

## Installation

Python 3.11 or newer is required.

```powershell
cd path\to\dnsp_analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

Copy the example configuration:

```powershell
Copy-Item config\analysis.example.toml analysis.toml
```

Review `analysis.toml`. Its default paths match the supplied DNSP files.

## Working assumptions

The initial configuration assumes:

- negative `ActivePow` means export;
- negative `ReactPow` means absorbing reactive power;
- `MeasureTime` represents a UTC/GMT timestamp;
- P and Q are instantaneous samples;
- the AMI measurement location is not yet confirmed.

Raw P/Q fields are always retained. Changing a sign later requires rebuilding
the derived canonical parquet, not the raw source.

## First run: notebook, one small site bucket

Start Jupyter from the project folder:

```powershell
jupyter lab
```

Open `notebooks/01_foundation_pipeline.ipynb` and run it one cell at a time.
It defaults to April 2025 and one of the configured 32 deterministic site
buckets. Each stage displays its outputs and asserts the required gate before
the next stage.

Sample outputs are isolated under:

```text
derived\samples\month_2025_04__bucket_0_of_32\
```

Its ID reconciliation is scoped to the same sample. The complete
1,342-telemetry-ID reconciliation is produced only by an explicit full run.

The full-run section is locked until the notebook variable
`FULL_RUN_CONFIRMATION` is set to the exact text `RUN FULL DATASET`.

## Command-line alternative

After validating the notebook flow, the same sample can be run non-interactively:

```powershell
dnsp-foundation --config analysis.toml --month 2025-04 --site-bucket 0
```

Run all buckets for one month:

```powershell
dnsp-foundation --config analysis.toml --month 2025-04
```

Only after scoped results pass validation, run the full dataset explicitly:

```powershell
dnsp-foundation --config analysis.toml --full
```

The orchestrator refuses an unscoped run unless `--full` is present.

## Overwriting outputs

Existing canonical output directories are not replaced by default:

```powershell
dnsp-foundation --config analysis.toml --month 2025-04 --site-bucket 0 --overwrite
```

`--overwrite` is restricted to the configured derived-data directory.

## Individual stages

```powershell
python scripts\00_inventory_sources.py --config analysis.toml
python scripts\01_prepare_metadata.py --config analysis.toml
python scripts\02_build_canonical_phase.py --config analysis.toml --month 2025-04 --site-bucket 0
```

## Tests

```powershell
pytest
```

The tests use small synthetic tables. They do not read or modify the full
DNSP dataset.

## Important output fields

- `active_power_raw_w`: unchanged provider value.
- `reactive_power_raw_var`: unchanged provider value.
- `p_export_w`: raw active power multiplied by `active_export_sign`.
- `q_absorbing_var`: raw reactive power multiplied by
  `reactive_absorbing_sign`; positive means absorbing.
- `q_generator_var`: generator convention; negative means absorbing.
- `duplicate_status`: `unique` or `identical_duplicate`.
- `duplicate_count`: number of source rows collapsed into the canonical row.

Conflicting duplicate keys are excluded from canonical telemetry and retained
in the duplicate audit.

## Output layout

See `ARCHITECTURE.md` for the complete layout and stage dependencies.

## Suggested review sequence

See `REVIEW_GUIDE.md` for a file-by-file walkthrough and the recommended
sequence for reviewing and testing the orchestrator together.

`DELIVERY_ROADMAP.md` records the notebook-first requirement for Deliveries
2–5, including the final Volt-VAr/Volt-Watt conformance and curtailment
visualisation notebook.

## What comes next

After Delivery 1 passes:

1. phase and site-interval feature construction;
2. BOM/NCI irradiance coverage and matching;
3. scale-free Volt‑VAr behaviour analysis;
4. Volt‑Watt observability;
5. optional magnitude and counterfactual work once rating and measurement
   questions are resolved.
