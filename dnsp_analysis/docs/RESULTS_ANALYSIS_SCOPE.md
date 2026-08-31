# Results analysis scope

The results-analysis increment (Delivery 5) is built and read-only:

```text
src/ausgrid_analysis/result_views.py
src/ausgrid_analysis/result_plots.py
notebooks/05_analyse_mechanism_results.ipynb
tests/test_result_views.py
tests/test_results_notebook_contract.py
```

It reads the three Delivery 4 result tables (`voltvar_proxy_results.parquet`,
`voltwatt_proxy_results.parquet`, `response_observability.parquet`) and never
calls `build_voltvar_results`, `build_voltwatt_results` or
`build_response_observability`.

It also reads both `phase_scope_basis` tracks (`der_inferred`, the production
default, and `all_phases`, the sensitivity/comparison track added alongside
Delivery 4 -- see `MECHANISM_RESULTS.md`) side by side, extending
`result_views.py`'s functions with an optional `mechanism` argument beyond
what this document originally specified, for that reason.
`response_observability.parquet` is the one exception: it is always the
single shared, unnamespaced table regardless of track, so
`result_views.result_context` reports its methodology id separately
(`response_observability_methodology_matches_curve_tables`) instead of
raising when it legitimately differs from an `all_phases` curve-table run.

## Separate analysis tracks

### Volt-VAr proxy results

`voltvar_denominator_view`/`voltvar_status_view` show denominators, assessable
coverage and proxy curve-status counts at fleet, site, mapped-phase scope,
voltage-bin, UTC month and cohort levels. With null `s_rated_kva`, `n_assessable`
is 0 everywhere today; the views and Notebook 05 show that honestly rather than
inventing a rate.

### Volt-Watt proxy results

`voltwatt_denominator_view`/`voltwatt_status_view` show activated/exporting
denominators and curve-ceiling exceedance evidence at the same dimensions.
`proxy_does_not_exceed_curve_ceiling` is never relabeled as inverter
conformance -- the column name is preserved verbatim through the views, plot
labels and notebook.

### Response observability

`observability_status_view`/`observability_metric_view` show excitation
coverage, slopes, correlations and direction statuses at fleet, site, actual
telemetry phase, voltage exposure, month and cohort levels. These remain
observability/association evidence, never conformance results.

### Counterfactual-supported curtailment

`result_plots.plot_curtailment_unavailable` renders only an explicit
`unavailable — methodology gate 7 unmet` panel; `result_views.py` has no
curtailment view at all. No curtailment estimate, rate, energy or blended
score is computed anywhere in this increment. That remains blocked until the
user chooses one of these separate future paths:

1. Validate a load–PV decomposition and uncertainty-aware counterfactual first.
2. Authorise a clearly named sensitivity-only empirical-envelope method whose
   default interpretation remains `not_assessable`.

## Non-negotiable checks

- Every view reconciles to its own source table's assessed denominator
  (`result_views.validate_result_views`, exercised in Notebook 05 Stages 1
  and 7).
- No query combines conformance and observability into one classification.
- No local timestamp is used as a unique key -- the result tables carry no
  `timestamp_local` column at all; dimensions are UTC `year_utc`/`month_utc`.
- Every figure subtitle states measurement basis, voltage location, voltage
  basis, capacity basis and both sign-review states (`result_plots._subtitle`).
- Site/phase plots preserve low-denominator warnings and never rank an
  unassessable site as poor-performing.
