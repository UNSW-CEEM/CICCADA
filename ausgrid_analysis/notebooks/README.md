# Notebook workflow

The notebooks are the primary user-facing orchestrators. The Python modules in
`src/ausgrid_analysis` contain the reusable pipeline logic; notebooks call
those modules one stage at a time and expose validation outputs before the
next stage.

Delivery 1 contains:

- `01_foundation_pipeline.ipynb`: source inventory, metadata reconciliation,
  duplicate audit, canonical phase telemetry, visual inspection and final
  validation.

Run the notebook from top to bottom. It defaults to April 2025 and one of 32
site buckets. The full-dataset section requires the exact confirmation text
`RUN FULL DATASET`.

Future notebook names and responsibilities are recorded in
`../DELIVERY_ROADMAP.md`.

