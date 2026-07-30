# Delivery 2 — explore and structure canonical telemetry

## Run order

1. Open `notebooks/02a_explore_canonical.ipynb`.
2. `CONFIG_PATH` uses your existing project-root `analysis.toml`.
3. Select a month/site and explore bounded extracts. This notebook writes nothing.
4. Open `notebooks/02b_build_structured_intervals.ipynb`.
5. Run the same one-month/one-bucket sample used in Delivery 1.
6. Review phase mappings, battery/cohort counts, interval previews and validation.
7. Only then type the exact full-run confirmation.

Your existing `analysis.toml` remains valid. The optional `[delivery2]` section in
`analysis.example.toml` exposes the phase-profile thresholds. Because Ausgrid has
confirmed the voltage is at the revenue meter, setting
`measurement_location = "revenue_meter"` is recommended.

## Outputs

Under the selected scope's `delivery2/` directory:

- `site_phase_profile.parquet`: one row per observed site/phase.
- `site_profile.parquet`: one row per site with candidate DER phases and confidence.
- `structured_phase_intervals/`: canonical rows plus time, measurement and mapping flags.
- `structured_site_intervals/`: one row per site/timestamp with safe phase aggregation.

The validation report is `audit/delivery2_validation.json`. Site-level complete
power is null when an inferred DER phase is missing; missing values are never
silently treated as zero.

The local daytime/nighttime medians used for candidate phase ranking are
streaming approximate quantiles. They are diagnostics, not energy estimates.

## What Delivery 2 deliberately does not do

It does not decompose load/PV, separate battery power, estimate uncurtailed PV,
correct revenue-meter voltage to inverter-terminal voltage, or assess Volt-Var /
Volt-Watt conformance. Those constraints are recorded in `METHODOLOGY_GATES.md`.
