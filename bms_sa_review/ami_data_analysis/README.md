# Synthetic AMI dataset — from `ts` + `meta_up23c` to `ami_raw` / `ami_meter` / `ami_raw_phaseseparate`

This document describes the three tables the AMI dataset build (`lib/ami_build.py`,
Phase 5 step 4) produces, and exactly how each column is derived from the
underlying Solar Analytics `ts` (interval telemetry) and `meta_up23c`
(circuit/site metadata) tables. It assumes Phases 1–4 (data-lake inventory,
signal taxonomy, per-site circuit resolution) have already run and produced a
`resolution` frame — the per-circuit keep/drop verdict every one of these
tables is built from.

## Why three tables, not one

A synthetic AMI dataset needs to answer two different questions, and no
single table can answer both without either hiding information a real
disaggregation algorithm wouldn't have, or hiding information a validator
needs:

- **What would a real smart meter report?** → `ami_meter`
- **What is the true underlying answer, to grade an algorithm's output
  against?** → `ami_raw` (site-level) and `ami_raw_phaseseparate` (per-phase)

`ami_meter` is a lossy, per-phase, net-of-PV view — exactly what a
disaggregation algorithm is meant to work from. `ami_raw`/
`ami_raw_phaseseparate` reconstruct the PV-independent truth behind it,
using a PV signal (`pv_site_net`) a real meter never separately exposes.

## Unit and naming convention

All three tables follow the naming/unit style of the project's other
Athena-side table, `structured_data` (built by `build_structured_data.py`
from the same two source tables): `site_id`/`t_stamp` keys, explicit `year`/
`month` INT columns (UTC, mirroring `structured_data`'s own
`year(t_stamp)`/`month(t_stamp)`), `V` for voltage, kW/kvar/kWh rather than
raw Watts/var/Wh. Every `_w`/`_var`/`_wh`-suffixed quantity in `ts` is
divided by 1000 before it reaches any of these three tables.

PV generation is **normalized by site capacity** (`P_kw_norm`), matching
`structured_data`'s own `P_kw_norm`/`normalization_basis` pattern — PV output
scales directly with installed capacity, and `meta_up23c` has a real capacity
metric (`S_99`, the empirical 99th-percentile output, or `ac_capacity_kw`,
the nameplate rating) to normalize by. Load is **not** normalized — there is
no equivalent capacity metric for a load circuit, so `P_kw` for load stays an
absolute reading in kW.

## The three tables

### `ami_raw` — site-level ground truth

One row per `(site_id, t_stamp)`.

Columns are kept as two symmetric trios — house load and PV system — plus
a couple of convenience/audit columns:

| column | type | derivation | unit |
|---|---|---|---|
| `site_id` | BIGINT | | |
| `t_stamp` | TIMESTAMP | as landed, UTC | |
| `year`, `month` | INT | `t_stamp`'s UTC year/month | |
| `V_load` | DOUBLE | average `voltage` across the site's kept `ac_load_net` circuits | V |
| `V_pv` | DOUBLE | average `voltage` across the site's kept `pv_site_net` circuits | V |
| `P_load_kw` | DOUBLE | signed, `circuit_polarity`-corrected `ac_load_net` power, summed across load phases | kW |
| `P_pv_kw` | DOUBLE | signed, `circuit_polarity`-corrected `pv_site_net` power, summed across PV circuits | kW |
| `P_kw_norm` | DOUBLE | `P_pv_kw ÷ normalization_capacity` | fraction of capacity |
| `P_kw` | DOUBLE | `gross_load` = `P_load_kw + P_pv_kw` — convenience column, not independent | kW |
| `Q_load_kvar` | DOUBLE | signed, corrected `ac_load_net` reactive power, summed | kvar |
| `Q_pv_kvar` | DOUBLE | signed, corrected `pv_site_net` reactive power, summed | kvar |
| `Q_kvar_norm` | DOUBLE | `Q_pv_kvar ÷ normalization_capacity` | fraction of capacity |
| `Q_kvar` | DOUBLE | `gross_reactive_load` = `Q_load_kvar + Q_pv_kvar` — convenience column | kvar |
| `S_99`, `ac_capacity_kw` | DOUBLE | carried through from a `site_capacity` lookup (from `meta_up23c`), for audit | kVA, kW |
| `normalization_basis` | STRING | which of `S_99`/`ac_capacity_kw` was actually used | |

`P_load_kw`/`Q_load_kvar` and `P_pv_kw`/`Q_pv_kvar` both reuse
`ami_signal.reconstruct_gross_load`'s own `load_signed`/`pv_signed`
output — the SAME per-side sums the `gross_load` reconstruction
(validated against real plots in Phase 4) is built from, just exposed
directly instead of only being visible pre-summed into `P_kw`/`Q_kvar`.
Both sides are `circuit_polarity`-corrected and signed — this is a
**fixed sign convention** as of this schema version: an earlier build
summed PV power *unsigned* (raw, uncorrected), on the theory that Phase
3's night-time diagnostic (Section 9) meant no correction was needed for
PV. That reading was wrong — Section 9 confirmed `pv_site_net` carries no
netted-in load (a NET-of-load question), not that it needs no SIGN
correction. Raw PV power reads negative while generating, so the old,
unsigned `P_kw_norm` came out negative during generation; it's now
positive, matching `Q_kvar_norm`, which was always built the corrected
way. **If you're comparing against an export taken before this fix,
`P_kw_norm`/`P_pv_kw` will have flipped sign.**

`P_kw`/`Q_kvar` (`gross_load`/`gross_reactive_load`) are kept as
convenience columns — literally `P_load_kw + P_pv_kw` (and the reactive
equivalent) by construction, i.e. "what the house would have consumed had
there been no PV." Not a third independently-computed quantity.

Reactive power throughout this table (`Q_load_kvar`, `Q_pv_kvar`,
`Q_kvar_norm`, `Q_kvar`) carries a caveat active power doesn't: only the
active-power reconstruction has been checked against real plots (Phase
4). Treat the reactive-power columns as reasonable-by-construction, not
independently validated, until an equivalent check is done.

`V_load` and `V_pv` are genuinely different measurement points and can
differ meaningfully — the PV/inverter side can read a slightly higher
voltage than the load side due to local voltage rise from the site's own
export. Neither is "the" site voltage; pick whichever side answers your
question. (Note also: your colleague's `structured_data` table's own `V`
is averaged across *its* circuits, which are PV-side only — closer in
spirit to this table's `V_pv` than to `V_load`, if you're cross-checking
voltage between the two.)

A timestamp where only the load side or only the PV side reported a value
contributes no row here — a ground-truth row needs both signals to mean
anything.

### `ami_meter` — the synthetic smart meter

One row per `(site_id, device_id, circuit_id, t_stamp)` — every kept
`ac_load_net` circuit's own reading, **kept per phase**, not summed across a
site's circuits. This is what a disaggregation algorithm is meant to see.

| column | type | derivation | unit |
|---|---|---|---|
| `site_id`, `device_id`, `circuit_id` | BIGINT | | |
| `t_stamp` | TIMESTAMP | UTC | |
| `year`, `month` | INT | | |
| `V` | DOUBLE | raw per-circuit `voltage` | V |
| `P_kw` | DOUBLE | raw `power`, **not** polarity-corrected | kW |
| `Q_kvar` | DOUBLE | raw `reactive_power` (itself derived once, in the interval table, from `energy_reactive`) | kvar |
| `S_kva` | DOUBLE | `sqrt(P_kw² + Q_kvar²)` | kVA |
| `power_factor` | DOUBLE | raw pass-through from `ts` | dimensionless, [0,1] |
| `current_a` | DOUBLE | raw pass-through | A |
| `energy_import_kwh` | DOUBLE | raw `energy_import`, the real measured register | kWh |
| `energy_export_kwh` | DOUBLE | raw `energy_export`, the real measured register | kWh |

`ami_meter.P_kw` is deliberately **not** polarity-corrected: `ac_load_net`
is already net-of-PV as landed, and this table is meant to show what a real
meter would actually report, sign quirks included. `energy_import_kwh`/
`energy_export_kwh` use the real measured registers `ts` already carries
(`energy_import`/`energy_export`), not a value derived by clipping
instantaneous power at zero — more authentic to what a real smart meter's
accumulators would show.

Any of these source columns can be absent from the underlying interval
table (e.g. a circuit missing `current`); the corresponding output column
is simply omitted, not an error.

### `ami_raw_phaseseparate` — per-phase ground truth, PV-allocated

One row per `(site_id, device_id, circuit_id, t_stamp)`, on the load side
only — same grain as `ami_meter`, answering "what would the site-level
ground truth look like broken out by phase."

| column | type | derivation | unit |
|---|---|---|---|
| `site_id`, `device_id`, `circuit_id` | BIGINT | | |
| `t_stamp` | TIMESTAMP | UTC | |
| `year`, `month` | INT | | |
| `V` | DOUBLE | this circuit's raw `voltage` — no polarity correction (voltage has no sign-convention ambiguity) | V |
| `load_kw_signed` | DOUBLE | this circuit's `power`, **polarity-corrected** via `circuit_polarity` | kW |
| `pv_allocation_kw` | DOUBLE | this phase's PV share — see allocation rule below | kW |
| `gross_load_kw` | DOUBLE | `load_kw_signed + pv_allocation_kw` | kW |
| `Q_kvar_signed` | DOUBLE | this circuit's `reactive_power`, **polarity-corrected** via `circuit_polarity` | kvar |
| `pv_reactive_allocation_kvar` | DOUBLE | this phase's reactive PV share, same allocation rule as `pv_allocation_kw` | kvar |
| `gross_reactive_load_kvar` | DOUBLE | `Q_kvar_signed + pv_reactive_allocation_kvar` | kvar |
| `n_phases_at_site` | INT | number of kept load circuits at the site this month | |
| `pv_allocation_method` | STRING | `direct_matched_circuit` / `equal_split_across_load_phases` / `no_pv_present` | |

**`load_kw_signed`/`Q_kvar_signed` are deliberately not the same values as
`ami_meter.P_kw`/`Q_kvar`.** They use the same polarity correction `ami_raw`'s
own `gross_load`/`gross_reactive_load` reconstruction applies, specifically
so that `sum(gross_load_kw)`/`sum(gross_reactive_load_kvar)` across a site's
phases reconcile with `ami_raw.P_kw`/`ami_raw.Q_kvar` for that
site/timestamp. `ami_meter.P_kw`/`Q_kvar` are the raw sensor readings;
`ami_raw_phaseseparate` is a decomposition of the corrected ground truth.
They answer different questions and are not meant to share a sign
convention.

The reactive-power columns here reuse the SAME per-site allocation decision
(direct-match vs. equal-split, and which PV circuit pairs with which load
circuit) as the active-power columns — decided once from the active-power
side's circuit counts, not re-derived separately for reactive power — and
carry the same not-independently-validated caveat noted for `ami_raw.Q_kvar`
above.

**PV allocation rule**, decided once per site from that month's kept
circuit sets, then applied to every timestamp for that site:

- If the number of kept `pv_site_net` circuits equals the number of kept
  `ac_load_net` circuits, each load circuit is paired with one PV circuit
  (sorted `circuit_id` order) and uses that PV circuit's own signed reading
  directly — real per-circuit data, method `direct_matched_circuit`.
- Otherwise (most commonly: one `pv_site_net` circuit serving a 2- or
  3-phase load), the site's total signed PV is split evenly across load
  phases — method `equal_split_across_load_phases`.
- A site with no surviving PV circuit gets `no_pv_present` and a zero
  allocation.

**Why direct matching is a heuristic, not a proven pairing:** even when PV
circuit counts happen to match phase counts, `meta_up23c` carries no
verified circuit-to-phase (A/B/C) label for either PV or load circuits —
"3 circuits = 3 phases" is already an ordering assumption for load, and
pairing PV circuit *N* with load circuit *N* by sorted `circuit_id` is one
more assumption on top of that. It's physically plausible (checked against
real fleet metadata: every one of the 14,353 clean sites has exactly one
physical `inverter_count`, so multiple PV circuits at a site come from one
inverter's separate phase outputs, not independent PV sources) but not a
confirmed label. That's why every row carries its own `pv_allocation_method`
— filter on it if the pairing assumption ever needs auditing or excluding.

## How the three tables relate — a validation workflow

```sql
-- what a real meter shows, aggregated to site level (should match ami_raw.P_kw
-- only up to any polarity corrections that differ between the two tables)
SELECT site_id, t_stamp, sum(P_kw) AS net_load_kw
FROM ami_meter GROUP BY site_id, t_stamp;

-- ground truth, ready to use directly
SELECT site_id, t_stamp, P_kw AS gross_load_kw, Q_kvar AS gross_reactive_load_kvar,
       P_kw_norm AS pv_generation_norm
FROM ami_raw;

-- per-phase ground truth -- sums to ami_raw.P_kw / ami_raw.Q_kvar by construction
SELECT site_id, t_stamp, sum(gross_load_kw) AS gross_load_kw,
       sum(gross_reactive_load_kvar) AS gross_reactive_load_kvar
FROM ami_raw_phaseseparate GROUP BY site_id, t_stamp;
```

To grade a disaggregation algorithm: run it against `ami_meter` (the only
table it should ever see), then compare its site-level PV/load estimate
against `ami_raw`, or its per-phase estimate against
`ami_raw_phaseseparate`.

## The `apply_power_correction` toggle — matching `structured_data`

Some circuits — strongly correlated with `device_type == "CATCH Power"`
(73.6% of that model flagged vs. 0% of `Watt Watcher`, per the Phase 3
fleet-wide diagnostic) — report through a **whole-Watt-hour energy
register** whose true accumulation interval is close to but not exactly
5 minutes (typically ~4.9 min). Phase 3 found real evidence that these
circuits' raw `power` field is unreliable, so by default `ami_resolution
.build_interval_table` (and therefore every table built from it —
`ami_raw`, `ami_meter`, `ami_raw_phaseseparate`) substitutes a re-derived
value, `power = energy × 60 / implied_interval_minutes`, for any circuit
flagged `power_correction_applied`. Because the underlying energy register
only ticks in whole Wh, this re-derived power is visibly **stair-stepped**
rather than smooth — a real characteristic of the correction, not a bug.

`structured_data`'s own build (`Write_structured_data.ipynb`) does not
apply this correction — it uses raw `power` unconditionally for every
circuit, including flagged ones. That's a legitimate methodological
difference, not an error in either pipeline: one table treats the raw
field as untrustworthy and corrects it (at the cost of a blockier curve),
the other doesn't (at the cost of carrying a known-unreliable reading).

To match `structured_data`'s treatment, pass `apply_power_correction=False`
to `Build.run_build`/`Build.run_phase_split_build` (it threads straight
through to `build_interval_table`). The default (`True`, i.e. the
correction stays on) is unchanged for any existing call — this is
opt-in, not a behavior change:

```python
build_manifest = Build.run_build(
    Reval.iter_month_partitions(Config.STORE_DIR, FULL_YEAR_MONTHS),
    final_resolution, circuit_polarity_lookup,
    Config.store_path("ami_raw"), Config.store_path("ami_meter"),
    site_capacity=site_capacity_lookup,
    apply_power_correction=False,   # match structured_data's treatment
)
```

Reactive power is unaffected either way — `ts` has no raw instantaneous
reactive-power field at all, so `Q_kvar`/`reactive_power` is always
derived from `energy_reactive`, regardless of this setting.

**Identifying, and optionally excluding, the sites this affects** — rather
than changing how a flagged site's power is derived, you can drop it from
the dataset entirely with `Resolution.sites_with_power_correction`, a
per-site rollup (True if the site has *any* kept circuit flagged) of the
same underlying diagnostic:

```python
flagged = Resolution.sites_with_power_correction(final_resolution)
clean_site_ids = flagged[~flagged].index
dropped_site_ids = flagged[flagged].index
print(f"{len(dropped_site_ids):,} of {flagged.index.nunique():,} kept sites "
      f"would be excluded ({len(dropped_site_ids) / flagged.index.nunique():.1%}).")

clean_resolution = final_resolution[final_resolution.site_id.isin(clean_site_ids)]

# then build from clean_resolution instead of final_resolution -- the
# flagged sites never enter ami_raw/ami_meter/ami_raw_phaseseparate at all,
# rather than being filtered out of an already-built table afterwards:
build_manifest = Build.run_build(
    Reval.iter_month_partitions(Config.STORE_DIR, FULL_YEAR_MONTHS),
    clean_resolution, circuit_polarity_lookup,
    Config.store_path("ami_raw_clean"), Config.store_path("ami_meter_clean"),
    site_capacity=site_capacity_lookup,
)
phase_split_manifest = Build.run_phase_split_build(
    Reval.iter_month_partitions(Config.STORE_DIR, FULL_YEAR_MONTHS),
    clean_resolution, circuit_polarity_lookup,
    Config.store_path("ami_raw_phaseseparate_clean"),
)
```

`sites_with_power_correction` rolls up with `.any()`, not per side: a
ground-truth row needs both the load and PV side, so one untrustworthy
circuit (either side) taints the whole site's reconstruction, not just
that circuit's own columns. A site whose only circuit(s) were dropped
during resolution (`kept=False`) is excluded from the rollup entirely —
neither counted clean nor flagged, since it never reaches the AMI tables
regardless.

Note that `clean_resolution` reuses the SAME local Parquet store
(`Config.STORE_DIR`) — no re-extraction needed, since the store already
holds every surviving circuit's data regardless of the correction flag.
Point the two build calls at new `store_dir`s (as above) if you want the
`clean`-filtered tables to coexist alongside the full ones, rather than
overwriting them.

## Known limitations

- `P_kw_norm`/`S_99`/`ac_capacity_kw`/`normalization_basis` in `ami_raw` are
  entirely null unless a `site_capacity` lookup (from `meta_up23c`) is
  supplied to the build — this is optional so an existing build call keeps
  working without it.
- No table carries a genuine A/B/C phase label — only an internal
  `circuit_id`/`device_id`, consistent across `ami_meter` and
  `ami_raw_phaseseparate` for the same physical circuit, but not tied to any
  real-world phase designation.
- `ami_raw_phaseseparate`'s `direct_matched_circuit` pairing is the
  best-available heuristic given real fleet evidence (single inverter per
  site), not a verified ground-truth pairing.
- `Q_kvar`/`Q_kvar_norm` (`ami_raw`) and `Q_kvar_signed`/
  `pv_reactive_allocation_kvar`/`gross_reactive_load_kvar`
  (`ami_raw_phaseseparate`) reuse the same reconstruction formula and
  allocation logic validated for active power, but reactive power itself has
  no equivalent real-plot validation (see "The three tables" above) — treat
  it as reasonable-by-construction, not independently confirmed.
- By default, `CATCH Power`-model circuits get their active power
  re-derived from a whole-Wh energy register (`apply_power_correction`,
  see its own section above) rather than trusting a raw `power` field
  Phase 3 found unreliable for that device model — this makes those
  circuits' power visibly stair-stepped rather than smooth. Pass
  `apply_power_correction=False` to match `structured_data`'s (uncorrected)
  treatment instead, or use `Resolution.sites_with_power_correction` to
  exclude the affected sites from the dataset entirely.

## Storage layout

Each table is written one month at a time to a Hive-partitioned Parquet
directory: `<store_dir>/dt_month=YYYY-MM/part-0000.parquet`, where
`store_dir` is `ami_config.store_path("ami_raw")` /
`ami_config.store_path("ami_meter")` /
`ami_config.store_path("ami_raw_phaseseparate")` respectively. Query with
DuckDB's `read_parquet(..., hive_partitioning=1)` — see `05_ami_build.ipynb`
for worked examples.
