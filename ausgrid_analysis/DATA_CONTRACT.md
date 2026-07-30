# Data contract

## Telemetry source

Configured source:

```text
C:\Users\{id}\{drive}\Documents\{project} - Data\{dnsp}\{dnsp}_pv_combined.parquet
```

Required columns:

| Provider column | Meaning | Canonical field |
|---|---|---|
| `serial` | anonymised site/customer identifier | `serial` |
| `MeasureTime` | naive representation of a UTC/GMT instant | `timestamp_utc` |
| `Vphase` | AMI phase label | `phase` |
| `Volts` | instantaneous voltage, V | `voltage_v` |
| `Curr` | instantaneous current, A | `current_a` |
| `ReactPow` | instantaneous reactive power, VAr | `reactive_power_raw_var` |
| `ActivePow` | instantaneous active power, W | `active_power_raw_w` |
| `month` | source-file month | `source_month` |
| `source_file` | source archive stem | `source_file` |

`serial` is stored as text in derived data so identifier formatting is never
affected by numeric operations.

## Metadata source

Configured workbook:

```text
C:\Users\{id}\{drive}\Documents\{project} - Data\{dnsp}\{dnsp}_meta.xlsx
```

Configured sheet:

```text
Cust_DER_Network Data
```

The provider metadata is normalised to snake_case names. Capacity fields remain
separate:

- `approved_capacity_kw`;
- `solar_capacity_kw`;
- `battery_inverter_capacity_kw`;
- `s_rated_kva`: null until a defensible source is available.

Neither approved capacity nor solar capacity is silently treated as inverter rated apparent power.

## Nulls

- Null physical measurements remain null.
- All-null phase or site aggregates must not become zero in later stages.
- Missing metadata does not remove a telemetry row from the canonical phase
  table.
- Magnitude analysis will remain unassessable where rating is unavailable.

## Power sample semantics

P and Q are documented as instantaneous values at the measurement interval.
Delivery 1 does not multiply them by `5/60` or claim measured interval energy.

## Quality flags

Delivery 1 stores:

- `row_has_null_measurement`;
- `voltage_physical_ok`;
- `duplicate_status`;
- `duplicate_count`;
- source provenance.

Future deliveries will add phase mapping, irradiance coverage, response
assessability and mechanism-specific flags.

