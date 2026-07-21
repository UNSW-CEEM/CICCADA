# Proposed `data_query` rebuild — installation and change manifest

This bundle does not alter Stage 1, Stage 2, Athena tables, or the existing
GitHub working tree. Copy files only after reviewing them.

## Copy these notebooks

Replace:

- `data_query/02_conformance_curtailment_analysis.ipynb`
- `data_query/03_voltvar_curtailment_detection.ipynb`

with the two notebooks at the root of this bundle.

## Add this module

- Add `lib/analysis_contract.py`.

It is the single run manifest for years, database, tables, capacity bases,
voltage aggregation, flexible-export selection, capability profile, interval
duration, and the project site threshold.

## Replace these modules completely

- `lib/conformance_queries.py`
- `lib/conformance_metrics.py`
- `lib/conformance_plots.py`
- `lib/voltvar_params.py`
- `lib/voltvar_queries.py`
- `lib/voltvar_metrics.py`
- `lib/voltvar_plots.py`

Do not merge individual old functions into these replacements. The old public
API and several old meanings are intentionally removed.

## Leave these modules unchanged for now

- `lib/explore_plots.py` — used by the exploratory notebooks.
- `lib/fleet_eda_diagnostics.py` — used by notebook 01b; not imported by the
  rebuilt 02/03 notebooks.
- `lib/site_selection.py` — used by notebook 01b; not used for conformance
  examples because it selects a single circuit.

## Stop using these old analysis paths

- `lib/voltvar_diagnostics.py` is no longer imported by notebook 03. Its useful
  coverage check is replaced by `voltvar_queries.fetch_input_coverage`; its
  raw-telemetry funnel and sign checks do not use the same site-level voltage
  construction as the current builders. The file may remain for legacy
  reproduction, but mark it obsolete or move it out of `lib` after checking
  that no other notebook imports it.
- Remove all imports of `OBSOLETEciccada_config` from active analysis code.
- Do not query the original `conformance_voltwatt`,
  `conformance_voltwattghi`, or `conformance_voltvar` tables except through the
  explicit legacy-reconciliation section.
- Do not use protective-function tables in notebook 02.

## Important semantic replacements

1. `total_count` in `conformance_voltwatt_v2` is the primary Volt-Watt
   maximum-output denominator.
2. `assessable_count` in `conformance_voltwattghi_v2` is renamed conceptually
   to `response_supported_count`; it is not the primary compliance population.
3. Missing counterfactual intervals remain missing and are reported as
   coverage loss. They are never converted to zero potential generation.
4. Site conformance is `nonconf_frac <= 0.10`; nonconformance is strictly
   `nonconf_frac > 0.10`.
5. Volt-VAr project failures are adverse + inactive + significant shortfall.
   Near-conformant and surplus remain separate response categories.
6. Five-minute kW/kvar sums are multiplied by `1/12` only when calculating
   energy.
7. Metadata is collapsed to one row per site and conflict counts are shown
   before any state/DNSP/OEM merge.
8. Notebook 03 uses `structured_data_v2`, which already has site-level circuit
   aggregation and the stored average-voltage convention. It does not filter
   individual circuit voltages before aggregation.
9. `S_99` is labelled as an empirical operating-limit proxy. It is not called
   manufacturer `S_rated`.
10. Method A's headroom result is called a displacement proxy, not curtailed
    energy. Only Method B reports counterfactual-supported attributed energy.

## First execution

1. Restart the notebook kernel after copying the modules.
2. Run notebook 02 from top to bottom.
3. Confirm the stored-provenance table shows the expected run choices.
4. Confirm metadata `site_id` is unique and review every non-zero metadata
   conflict count.
5. Confirm the population funnel is consistent with the known populations
   (approximately 4,809 voltage-exposed and 1,878 strict response-supported
   sites for the current run; exact values depend on the selected years).
6. Review the legacy reconciliation before quoting any comparison with the
   original analysis.
7. Run notebook 03 from top to bottom with `RUN_SENSITIVITY = False`.
8. Only after the baseline is accepted, set `RUN_SENSITIVITY = True`. This
   performs multiple additional Athena scans.

## No builder rerun required

These changes are read-only analysis changes. Do not recreate Stage 1 or Stage
2 tables merely to install this bundle. A builder rerun is required only if the
stored provenance does not match the run being reported, or if a requested
sensitivity population was removed upstream (for example, flexible-export
sites cannot be restored analytically when Stage 1 was built with them
excluded).
