from pathlib import Path

# Copy this file to `local_paths.py`. The real `local_paths.py` is ignored by
# Git because it contains machine-specific paths to external SAPN/EVM/BOM data.
# Replace each `None` with a `Path(...)` for your machine.

# Root folder containing `Nov2022/`, `All Results/`, and `updated results/`.
SAPN_ROOT = None

# Root folder containing `site_metadata.csv`, `circuit_metadata.csv`, and the
# `curtailment training data parquet/` directory for the EVM training export.
EVM_ROOT = None

# Root folder containing the BOM daily parquet files.
BOM_ROOT = None

# CSV mapping BOM postcodes to point locations.
BOM_POINTS_CSV = None

# Phase B timestamp detail CSV used by the legacy exact-timestamp metrics
# script and the timestamp-debug notebook.
PHASE_B_TIMESTAMP_DETAIL_PATH = None

# Alternate Phase B timestamp detail CSV used by debugTimestamps.ipynb. If you
# only keep one local export, this can point at the same CSV as above.
PHASE_B_TIMESTAMP_DETAIL_ALT_PATH = None
