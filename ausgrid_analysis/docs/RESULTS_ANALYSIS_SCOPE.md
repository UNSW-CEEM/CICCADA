# Results analysis scope

The next results-analysis increment will be read-only. It will add:

```text
src/ausgrid_analysis/result_views.py
src/ausgrid_analysis/result_plots.py
notebooks/05_analyse_mechanism_results.ipynb
tests/test_result_views.py
tests/test_results_notebook_contract.py
```

## Separate analysis tracks

### Volt-VAr proxy results

Show denominators, assessable coverage and proxy curve-status counts at fleet,
site, mapped-phase scope, voltage-bin, UTC month and cohort levels. With null
`s_rated_kva`, views will show why magnitude assessment is unavailable rather
than inventing a rate.

### Volt-Watt proxy results

Show activated/exporting denominators and curve-ceiling exceedance evidence at
the same dimensions. `proxy_does_not_exceed_curve_ceiling` will never be
relabeled as inverter conformance.

### Response observability

Show excitation coverage, slopes, correlations and direction statuses at
fleet, site, actual telemetry phase, voltage exposure, month and cohort levels.
These plots remain observability evidence, not conformance results.

### Counterfactual-supported curtailment

Render an explicit `unavailable — methodology gate 7 unmet` panel only. No
curtailment estimate, rate, energy or blended score will be computed until the
user chooses one of these separate future paths:

1. Validate a load–PV decomposition and uncertainty-aware counterfactual first.
2. Authorise a clearly named sensitivity-only empirical-envelope method whose
   default interpretation remains `not_assessable`.

## Non-negotiable checks

- Every view reconciles to its own source table's assessed denominator.
- No query combines conformance and observability into one classification.
- No local timestamp is used as a unique key.
- Every figure subtitle states measurement basis, voltage location, voltage
  aggregate, capacity basis and sign-review state.
- Site/phase plots preserve low-denominator warnings and never rank an
  unassessable site as poor-performing.
