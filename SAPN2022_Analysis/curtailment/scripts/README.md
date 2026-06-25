# SAPN2022 Curtailment Scripts

This folder contains the local SAPN2022 curtailment workflow and a few small support/checking utilities.

## Local path setup

- External SAPN, EVM, and BOM locations are intentionally kept out of Git.
- Copy `local_paths.example.py` to `local_paths.py` in this folder and fill in the paths for your machine.
- `local_paths.py` is ignored by Git, while `local_paths.example.py` is the tracked template that documents each variable.

## Main pipeline scripts

- `build_structured_5m.py`: Builds the canonical 5-minute structured dataset from local EVM, SAPN, and BOM inputs; this is the default upstream for the later 5-minute model and curtailment scripts.
- `build_structured_high_resolution.py`: Builds the higher-resolution structured dataset and also provides many of the shared builder utilities reused by `build_structured_5m.py`.
- `fit_ghi_model.py`: Fits the per-site, per-time-bin GHI-normalised linear model from the structured training rows.
- `write_all_uncurtailedPV.py`: Applies the fitted GHI model to validation rows to estimate uncurtailed PV at each timestamp.
- `run_sapn2022_metrics.py`: Older exact-timestamp Phase B curtailment summary that joins responsibility flags to `all_uncurtailedPV`.
- `run_sapn2022_metrics_5m.py`: Main 5-minute curtailment summary that joins eligible LOS/OV1 buckets to `all_uncurtailedPV_5m`.
- `plot_site_day_evm.py`: Plots one site/day of actual power, estimated uncurtailed power, nonconformance, and voltage.
- `plot_aggregates.ipynb`: Interactive notebook for daily aggregate curtailed-power and kWh share pies from the 5-minute summary parquet.

## Supporting scripts used by the pipeline

- `funcs_from_SAPN2022updated.py`: Shared helpers for metrology cleaning, polarity handling, duplicate cleanup, and site-level aggregation.
- `sapn2022_metrics_5m_data_checks.py`: Shared uniqueness and join-coverage checks used by the 5-minute metrics and plotting scripts.
- `prepare_confidence_tier_site_cohort.py`: Writes the `confidence_tier_site_ids.csv` cohort consumed by the build scripts.
- `parquet2csv.py`: Small manual utility for exporting selected parquet outputs to CSV.

## Diagnosis / checking

- `write_structured_diagnostics.py`: Regenerates diagnostics CSVs from an existing structured parquet without rerunning the full build.
- `debugTimestamps.ipynb`: Checks timestamp alignment between Phase B detail, source data, and uncurtailed outputs.
- `assess sites with clear sky day.ipynb`: Investigates missing clear-sky reference days and missing daylight bins in the structured data.
