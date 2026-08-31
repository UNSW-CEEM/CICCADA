# Notebook-first delivery roadmap

This is a standing requirement for Deliveries 1–5:

- reusable calculations live in tested Python modules;
- every delivery has a Jupyter orchestrator;
- the orchestrator runs one stage at a time;
- a small deterministic slice is run before any full build;
- each stage displays tables and/or plots plus explicit assertions;
- full runs require deliberate opt-in;
- generated result tables are inspected in notebooks, not treated as correct
  merely because a build completed.

## Delivery 1 — foundation

Notebook: `notebooks/01_foundation_pipeline.ipynb`

Build and inspect source inventory, canonical metadata, ID reconciliation,
duplicate classification, canonical phase telemetry and row accounting.

## Delivery 2 — structured site intervals

Planned notebook: `notebooks/02_structured_intervals.ipynb`

Build and inspect phase mapping, phase-to-site aggregation, solar/battery
cohorts, interval eligibility and structured-data tables.

## Delivery 3 — irradiance and counterfactual

Planned notebook: `notebooks/03_irradiance_counterfactual.ipynb`

Assess BOM/NCI coverage, perform spatial/temporal matching, build quality-gated
solar-resource and uncurtailed-power estimates, and visualise model diagnostics.

This delivery remains optional until irradiance suitability is established.

## Mechanism result tables

Notebook: `notebooks/04_build_mechanism_results.ipynb`

Run an empirical P/Q sign review, verify Python/SQL curve parity, and build
separate Volt-VAr net-meter-proxy, Volt-Watt net-meter-proxy and response-
observability tables as deterministic-slice and deliberate full runs. Validate
denominators, result keys, coverage and provenance. Magnitude results use only
verified `s_rated_kva`; absent ratings remain `not_assessable`. Counterfactual-
supported curtailment is not built while methodology gate 7 is unmet.

## Results analysis

Notebook: `notebooks/05_analyse_mechanism_results.ipynb`

Read and visualise Volt-VAr proxy results, Volt-Watt proxy results and response
observability independently at fleet, site, phase, voltage, month and cohort
levels, via `src/dnsp_analysis/result_views.py` and `result_plots.py`. Does
not blend conformance, observability or future counterfactual-supported
curtailment into one score; curtailment is shown as an explicit `unavailable
— methodology gate 7 unmet` panel. Inspects both `phase_scope_basis` tracks
(`der_inferred` and `all_phases`) side by side wherever both exist.


## Irradiance and decomposition

Notebooks:

- `notebooks/03a_build_analysis_cohort.ipynb`
- `notebooks/03b_assess_irradiance_coverage.ipynb`

Build an auditable modelling cohort, assess BOM/NCI temporal coverage, map
substation coordinates to BOM grid points, and stop at an explicit suitability
gate before implementing load–PV decomposition. The previous dedicated
PV-circuit GHI model is not applied directly to Ausgrid net-meter power.

