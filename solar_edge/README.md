# CICCADA — SolarEdge extension

Local-first reproduction of the `bms_sa_review` Volt-VAr / Volt-Watt conformance and
curtailment analysis on the SolarEdge fleet dataset (1,602 sites, NSW / SA / QLD,
5-minute resolution, calendar year 2025).

The Solar Analytics analysis runs on AWS Athena. This one runs entirely on your machine,
with DuckDB querying local Parquet. The only AWS touch is a single bounded extract of BOM
satellite irradiance (D12a), which lands a local Parquet file; everything after that is
local.

## Why the methods are ported rather than reinvented

The point of this work is to compare SolarEdge against Solar Analytics. That only holds
if the methods are the same. So:

* AS/NZS 4777.2:2020 set-points are **imported** from
  `bms_sa_review/shared/ciccada_config.py` and `as4777_curves.py`, never restated.
* The SQL fragments those modules emit (`vvar_required_q_sql`, `vw_max_p_sql`,
  `q_cap_absorbing_sql`, `q_impact_nearest_edge_sql`) run in DuckDB essentially
  unchanged, so the SolarEdge queries stay line-comparable with the Athena originals.
* Where the data forces a deviation — there is no nameplate capacity, so `s_99` is the
  sole capacity basis — the substitution is labelled, printed in `manifest()`, and swept
  in the sensitivity notebook.

## Layout

```
solar_edge/
├── config/se_config.py     paths, raw schema contract, sign/unit/time conventions,
│                           store registry. The only place these are written down.
├── lib/
│   ├── se_store.py         DuckDB connection factory and view registration
│   ├── se_diagnostics.py   inventory, schema contract, data-quality checks
│   └── ...                 (added deliverable by deliverable)
├── notebooks/              thin orchestrators — no logic
├── artefacts/              small, reviewable outputs that belong in git
└── tests/
```

The derived store lives **outside** the repository, beside the raw data, because it is
large and regenerable:

```
CICCADA - Data/solar edge/_store/
```

## Getting started

```python
from solar_edge.config import se_config as C
from solar_edge.lib import se_store, se_diagnostics as diag

con = se_store.connect()
diag.run_d1_checks(con)          # is the delivery what we think it is?
se_store.store_status(con)       # what has been built so far?
```

If your data is not at `~/OneDrive - UNSW/Documents/CICCADA - Data/solar edge`, set
`CICCADA_SE_DATA_ROOT` before importing.

Start with `notebooks/00_environment_check.ipynb`. It reads Parquet footers only and
completes in seconds.

## What the data is

| | |
|---|---|
| Files | 12 monthly Parquet, 2025-01 to 2025-12, one schema |
| Rows | 86,643,185 |
| Size | 1.5 GB compressed on disk (about 9–12 GB as float64 in memory) |
| Sites | 1,602 across 507 postcodes — SA 574, NSW 570, QLD 458 |
| Resolution | 5 min, but **not** aligned to a common grid; each site has its own offset |
| Columns | per-phase active power (W), reactive power (var), voltage (V), frequency (Hz), plus `derating_active_flag` |
| Absent | nameplate capacity, inverter model, DNSP, install date, **irradiance** |

Two things will bite if handled carelessly, both resolved once at ingest:

**Timestamps are per-site local civil time, including daylight saving.** Confirmed from
the power-weighted diurnal centroid: QLD is stable across seasons (11.79 → 11.96 h) while
NSW shifts 11.91 → 13.07 h and SA 12.30 → 13.48 h between June and January. April
overlaps an hour, October deletes one. See `se_config.STATE_TIMEZONE` and the DST policy
constants.

**Reactive power uses the load convention.** SolarEdge reports a *mixed* convention:
active power as a production magnitude (already generator-positive, never negative), but
reactive power with **positive = absorbing**. CICCADA and AS/NZS 4777.2 Fig 3.2 use the
generator convention, where negative = absorbing. So `Q` is multiplied by `-1` at ingest
(`se_config.REACTIVE_POWER_SIGN`). The evidence and the caveat about the three-phase
cohort are documented in full in `se_config`, section 3.

## The store

`se_interval` — one row per site per 5-minute interval, in the CICCADA convention.

| | |
|---|---|
| Rows | 86,640,968 (2,217 duplicate rows removed from 86,643,185) |
| Size | ~1.5 GB, 24 files, partitioned by AEST month, sorted by `(site_alias, ts_utc)` |
| Build | ~35 s per month; peak RSS ≈ DuckDB memory limit + 200 MB, flat across the run |
| Columns | `site_alias, ts_utc, ts_aest, state, postcode, P_kW, Q_kvar, V_max, V_mean, n_phases_reporting, freq_hz, derating_active` |

`ts_aest` is computed by the registered view rather than stored — it is exactly
`ts_utc + 10 h`, and materialising it cost 21 MB per month for nothing. Queries can use
`hour(ts_aest)` and `date(ts_aest)` directly, which removes the `+ interval '10' hour`
bug class that produced R3/R9 in the legacy conformance tables.

Single-site queries run ~14× faster than against the raw delivery, whose files have one
row group each and so cannot be pruned at all.

### Known data-quality finding

**20 sites carry mis-framed timestamps** — 22,124 rows, 0.026% of the store — most
consistent with a UTC-stamped subset landing at 21:00–03:00 AEST and showing several kW
of impossible night-time generation. Site list in
`artefacts/night_generation_anomaly_sites.csv`. These sites are **not** excluded at
ingest; that is an analysis-layer decision for D6/D7 where it can be swept. It matters
most for anything touching the overnight envelope.

## Deliverable status

| | Deliverable | Status |
|---|---|---|
| D0 | Package skeleton, config, DuckDB store layer | **done** |
| D1 | Raw inventory and schema contract | **done** |
| D2 | Timestamp and DST resolution, unit-tested | **done** — 22 tests |
| D3 | Tidy store builder | **done** — reconciles exactly |
| D4 | Site dimension, capacity proxies, BOM grid-point mapping | next |
| D5 | Params, contract, manifest | |
| D6–D7 | Fleet EDA and data-quality report | |
| D8 | Q sign convention locked | resolved early — see `se_config` §3 |
| D9–D10 | Volt-VAr and Volt-Watt conformance | |
| D11 | Method A symptom scan | |
| D12a–c | BOM extract, `se_structured`, GHI model and counterfactual | |
| D13 | Method B attribution and evidence tiers | |
| D14 | Method C derating-flag corroboration | |
| D15 | Sensitivity analysis | |

The full architecture and rationale are in the project's SolarEdge architecture proposal.
