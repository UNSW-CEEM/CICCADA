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

## Delivery 4 — mechanism result tables

Planned notebook: `notebooks/04_build_conformance_results.ipynb`

Build test-slice and full result tables for Volt-VAr, Volt-Watt, response
observability and counterfactual-supported curtailment. Every result table will
have denominator, uniqueness, coverage and provenance checks.

## Delivery 5 — results analysis

Planned notebook: `notebooks/05_results_analysis.ipynb`

Read and visualise Volt-VAr and Volt-Watt conformance and curtailment results at
fleet, site, phase, voltage, month and cohort levels. Keep conformance,
observability and counterfactual-supported curtailment as separate questions.

## Irradiance and decomposition

Notebooks:

- `notebooks/03a_build_analysis_cohort.ipynb`
- `notebooks/03b_assess_irradiance_coverage.ipynb`

Build an auditable modelling cohort, assess BOM/NCI temporal coverage, map
substation coordinates to BOM grid points, and stop at an explicit suitability
gate before implementing load–PV decomposition. The previous dedicated
PV-circuit GHI model is not applied directly to Ausgrid net-meter power.

