# Stage 1/Stage 2 methods and sensitivity runs

This document records the assumptions that must accompany every calculation run. 

## Capacity concepts

- `rating_basis`: scales the AS/NZS 4777.2 Volt-Watt and Volt-VAr curves, the 4% analytical band and the 20% assessability threshold.
- `empirical_limit_basis`: defines the apparent-power boundary used by the empirical Volt-VAr curtailment symptom.
- `normalization_basis`: scales Stage 1 P and Q before fitting the GHI model.
- `counterfactual_cap_basis`: caps the final Stage 1 counterfactual.

Allowed capacity labels are `ac_capacity_kw` and `s_99`.
`counterfactual_cap_basis` additionally accepts `none`, which reproduces original uncapped production INSERT. 
The `s_99` rating case is a sensitivity analysis, not a claim that the empirical percentile is the standard's rated apparent power.

## Original analysis (for milestone report 3)

| Calculation | Original basis/operation | Evidence |
| --- | --- | --- |
| Volt-Watt required P curve | `ac_capacity_kw` | `Volt-Watt.ipynb`: nameplate passed to the Volt-Watt UDF |
| Volt-Watt 4% band | `ac_capacity_kw` | `+ .04*df.ac_capacity_kw` |
| Volt-VAr required Q curve | `ac_capacity_kw` | `get_voltvar_Q_udf(df.voltage, df.ac_capacity_kw)` |
| Volt-VAr 4% band | `ac_capacity_kw` | `Q_voltvar +/- .04*df.ac_capacity_kw` |
| Volt-VAr 20% assessability | No separate non-assessable output; the capability UDF used `ac_capacity_kw` | `Volt-Var.ipynb` |
| Figure 2.1/capability | `ac_capacity_kw` | `Q_capability_absorbing_udf(df.P_kW, df.ac_capacity_kw)` |
| Empirical Volt-VAr apparent-limit symptom | `S_99` in the legacy curtailment work | `curtailment_voltvar.ipynb` |
| Stage 1 P/Q normalisation | `S_99` | `Write_structured_data.ipynb` cell 7 |
| Stage 1 floor | prediction floored at measured normalised P | `Write_All_uncartailedPV.ipynb` cell 14 |
| Stage 1 final cap | No explicit final nameplate/S_99 cap in the production INSERT | cell 14; later cells are diagnostics only |

Literal excerpts from the legacy cells remove any ambiguity. Volt-Watt cell 5:

```sql
avg(voltage) as V, max(ac_capacity_kw) as ac_capacity_kw
...
(case when V < v1 then ac_capacity_kw
 when V > v2 then .2 * ac_capacity_kw
 else (...) end) + 0.04*ac_capacity_kw as max_P_volt_watt
```

Volt-Var cell 7:

```sql
case when V < 207 then 0.44*ac_capacity_kw
...
else -0.6*ac_capacity_kw end as Q_voltvar,
case when abs(P_kW) < .2*ac_capacity_kw then 0
     when abs(P_kW) <= .6*ac_capacity_kw then -0.44*ac_capacity_kw
     when abs(P_kW) <= .8*ac_capacity_kw then ...
     else -sqrt(power(ac_capacity_kw,2)-power(abs(P_kW),2))
end as Q_cap_absorbing
...
Q_voltvar + .04*ac_capacity_kw as Q_voltvar_max
```

The same cell used the empirical limit only for the curtailment symptom:

```sql
case when V <= 253
  and sqrt(power(Q_kvar,2)+power(P_kW,2)) >= S_99
then uncurtailed_P-P_kW end as curtailment_voltvar
```

Stage 1 structured-data cell 7 and MAPE cell 8:

```sql
sum(power*circuit_polarity)/1000/max(S_99) as P_kw_norm,
sum(energy_reactive*circuit_polarity)/1000/max(S_99)*12 as Q_kvar_norm,
avg(voltage) as V
...
FILTER (WHERE abs(P_kw_norm) > 0.2 AND P_kw_norm_est IS NOT NULL)
```

Stage 1 all-uncurtailed cell 14 floored and rescaled, but did not cap:

```sql
case when P_kw_norm_cs*(a+b*x) >= P_kw_norm
     then P_kw_norm_cs*(a+b*x) else P_kw_norm end AS P_kw_norm_est
...
P_kw_norm_est*S_99 as uncurtailed_P
```

## Clear-sky heuristics

The constants live at the top of `build_structured_data.py`:

- three lowest-`cloud_sum` days per AEST month/BOM grid;
- `cloud_sum < 60`;
- daily `max_GHI > 200`;
- nearest qualifying day at the same grid;
- absolute separation `<45` days;
- centred seven-reading (`3` before/current/`3` after) 60th percentile;
- start a new smoothing segment after a gap over 30 minutes.

These values reproduce the legacy implementation. 
They are empirical model choices, not thresholds from AS/NZS 4777.2 or definitions supplied by BOM.
Any changed values require a new run label plus coverage/error comparison.

## MAPE gate

MAPE is unstable close to zero because every absolute error is divided by the
actual value. An error of 0.02 is 4% when actual is 0.50, but 40% when actual is
0.05. The revised builder therefore also writes WAPE, normalised MAE/RMSE,
bias, validation interval count and validation day count. The default minimum
coverage (30 intervals across 3 days) is a project heuristic and must be
reported with the 50% MAPE threshold.

## Flex-export flag

`build_flex_export_detected.py` reproduces the legacy plateau rule into an auditable side table. 
It does not append a field to raw `ts`. The metadata flag is joined to telemetry when the cohort is selected:

- `flex_selection="exclude"`: explicitly false only;
- `flex_selection="include"`: all flags, including NULL;
- `flex_selection="only"`: flagged sites only, for diagnostics.

Potential confounding includes fixed export limits, inverter/battery/load control, telemetry quantisation, flat weather, etc.
Potential false negatives include variable or noisy limits and events shorter than the five-minute-tempora-resolution. The detector is not programme-enrolment truth.

## BOM/NCI provenance found in this repository

`BOM_NCI/rclone.ipynb` downloads Himawari solar-product files from the NCI
remote. `BOM_NCI/process_bom.ipynb` processes them and writes local CSV files;
`process_SA.ipynb` and postcode notebooks also write local CSV products. No
notebook source in `BOM_NCI` creates, uploads or registers `bom_nci.solar` in
S3/Glue/Athena. The catalog/load step is therefore absent from this checkout.
Likely recovery sources are Athena query history, Glue table parameters and
location, S3 object metadata, scheduled Glue jobs/crawlers, or Hossein's shell
history/automation repository.

## Power-sample sums versus energy

Daily `P_kW_sum`, kvar sums and curtailment sums are sample sums. For complete
five-minute intervals:

```sql
SELECT
    sum(P_kW_sum) * (5.0 / 60.0) AS generated_kwh,
    sum(curtailment_voltvar_sum) * (5.0 / 60.0) AS curtailed_kwh
FROM conformance_voltvar_v3_s99_avg;
```

Use actual timestamp durations instead of `5/60` if irregular cadence or gaps
must contribute energy. Never relabel the raw `_sum` fields as kWh/kvarh.