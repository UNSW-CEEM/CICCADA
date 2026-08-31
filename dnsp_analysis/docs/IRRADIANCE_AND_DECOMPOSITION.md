# Irradiance and decomposition stage

## Why this stage starts with gates

The previous `bms_sa` GHI model was trained on dedicated PV-circuit power
(`ts.is_pv = true`). Ausgrid supplies revenue-meter net P/Q:

```text
net meter export = PV generation - household demand
```

Therefore the old model cannot be applied directly to Ausgrid net export and
called uncurtailed PV. The safe sequence is:

1. Build a transparent site-eligibility table.
2. Verify BOM/NCI temporal coverage.
3. Map Ausgrid substation coordinates to BOM grid points and quantify distance.
4. Confirm what the BOM `quality_mask` values mean.
5. Only then compare candidate load–PV decomposition methods.
6. Validate decomposition uncertainty before estimating uncurtailed PV.

## Primary cohort

Notebook `03a_build_analysis_cohort.ipynb` independently records:

- solar-only metadata;
- battery exclusion;
- controlled-load exclusion;
- accepted phase-mapping confidence;
- minimum P/Q coverage across inferred DER phases;
- location availability;
- capacity availability.

It writes:

```text
derived/analysis_cohort/site_eligibility.parquet
derived/audit/site_eligibility_summary.json
```

No site is deleted from structured telemetry. Eligibility is a downstream view
with explicit pass/fail columns and exclusion reasons.

## Irradiance coverage

Notebook `03b_assess_irradiance_coverage.ipynb` uses partition-filtered Athena
queries against `bom_nci.solar`. It writes:

```text
derived/irradiance/site_to_bom_grid.parquet
derived/irradiance/bom_monthly_coverage.parquet
```

Ausgrid coordinates are labelled `substation_metadata`. They are not customer
coordinates. A short distance to a BOM grid point does not prove that the
irradiance represents roof-plane irradiance at the customer.

## Decomposition remains experimental

The next implementation increment should compare at least:

- a local-time load baseline;
- an irradiance-conditioned net-export model;
- a conservative uncertainty envelope.

Battery and controlled-load sites remain excluded from the primary experiment.
Results must remain labelled `net_meter_proxy` until a decomposition method is
validated well enough to estimate inverter quantities.

