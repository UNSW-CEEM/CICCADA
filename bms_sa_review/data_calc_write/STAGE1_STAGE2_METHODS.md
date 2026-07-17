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

Build and inspect the side table before changing metadata:

```python
import build_flex_export_detected as flex

FLEX_AUDIT = "flex_export_detection_v2"
flex.create_table(aq, SAI, target=FLEX_AUDIT)
flex.run(aq, SAI, years=(2024, 2025), target=FLEX_AUDIT)

comparison = aq(f"""
    SELECT
      coalesce(m.flex_export_detected, False) AS current_meta_flag,
      f.flex_export_detected AS reconstructed_flag,
      count(DISTINCT f.site_id) AS n_sites
    FROM {FLEX_AUDIT} f
    LEFT JOIN meta_up23c m ON f.site_id=m.site_id AND m.is_pv=True
    GROUP BY 1, 2 ORDER BY 1, 2
""", database=SAI)
display(comparison)
```

Only after reviewing disagreements, the following single atomic metadata
update makes the selectors use the reconstructed flag. It updates metadata,
not `ts`, and is deliberately guarded:

```python
flex.write_back_meta(
    aq, SAI, target=FLEX_AUDIT,
    confirmation="UPDATE_META_UP23C_FROM_AUDITED_FLEX_TABLE",
)
```

## Recommended all-S_99 sensitivity run

Run in a fresh kernel so imported module constants cannot retain an older
version. This creates new `_v3_s99_avg` tables and does not overwrite `_v2`.

```python
import pathlib, sys, importlib

ROOT = pathlib.Path.cwd().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "data_calc_write" / "stage1_ghi_pipeline"))
sys.path.insert(0, str(ROOT / "data_calc_write" / "stage2_conformance"))

from aws_config import aq
from ciccada_config import SAI
import build_structured_data as b1
import build_split_days as b2
import build_ghi_model as b3
import build_mape_quality_gate as gate
import build_all_uncurtailedpv as b4
import build_conformance_voltvar as vv
import build_conformance_voltwatt as vw

SD = "structured_data_v3_s99_avg"
SPLIT = "split_days_v3_s99_avg"
MODEL = "pv_ghi_norm_model_v3_s99_avg"
UNC = "all_uncurtailedpv_v3_s99cap"
VV = "conformance_voltvar_v3_s99_avg"
VW = "conformance_voltwatt_v3_s99_avg"
VWGHI = "conformance_voltwattghi_v3_s99_avg"
MAPE_IDS = "mape_under50_v3_s99_avg.csv"
MAPE_METRICS = "mape_metrics_v3_s99_avg.csv"
```

Stage 1:

```python
print(b1.create_table(aq, SAI, target=SD))
for year in (2024, 2025):
    b1.run_resilient(
        b1, aq, SAI, year, range(1, 13), n_parts=16,
        target=SD,
        normalization_basis="s_99",
        voltage_aggregation="avg",
        flex_selection="exclude",
    )
b1.validate(aq, SAI, target=SD)

print(b2.create_table(aq, SAI, target=SPLIT))
print(b2.run(aq, SAI, source=SD, target=SPLIT))
b2.validate(aq, SAI, target=SPLIT)

print(b3.create_table(aq, SAI, target=MODEL))
b3.run(aq, SAI, years=(2024, 2025), sd=SD, split=SPLIT, target=MODEL)
b3.validate(aq, SAI, target=MODEL)

mape_df, good_sites = gate.run(
    aq, SAI,
    sd=SD, model=MODEL, split=SPLIT,
    min_actual_norm=0.20,
    min_val_intervals=30,
    min_val_days=3,
    csv_path=MAPE_IDS,
    metrics_csv_path=MAPE_METRICS,
)

print(b4.create_table(aq, SAI, target=UNC))
for year in (2024, 2025):
    b4.run_year(
        aq, SAI, year, MAPE_IDS, n_parts=6,
        sd=SD, model=MODEL, target=UNC,
        normalization_basis="s_99",
        normalization_capacity_col="normalization_capacity",
        counterfactual_cap_basis="s_99",
    )
b4.validate(aq, SAI, target=UNC)
```

Stage 2 smoke test first, then the complete run:

```python
print(vv.create_table(aq, SAI, target=VV))
print(vw.create_table_basic(aq, SAI, target=VW))
print(vw.create_table_ghi(aq, SAI, target=VWGHI))

COMMON = dict(
    rating_basis="s_99",
    voltage_aggregation="avg",
    flex_selection="exclude",
)

# Smoke test: January, one of eight site slices.
vv.run_months_voltvar(
    aq, SAI, 2024, [1], n_parts=8, parts=[0], target=VV,
    uncurtailed=UNC, empirical_limit_basis="s_99", **COMMON,
)
vw.run_months_basic(
    aq, SAI, 2024, [1], n_parts=8, parts=[0], target=VW, **COMMON,
)
vw.run_months_ghi(
    aq, SAI, 2024, [1], n_parts=8, parts=[0], target=VWGHI,
    uncurtailed=UNC, **COMMON,
)

# Validate the smoke test, inspect SQL/results, then recreate the three empty
# targets before the full run so the smoke-test rows are not duplicated.
vv.validate(aq, SAI, target=VV)
vw.validate_basic(aq, SAI, target=VW)
vw.validate_ghi(aq, SAI, target=VWGHI)
```

After recreating the empty Stage 2 targets, run all months:

```python
for year in (2024, 2025):
    months = range(1, 13)
    vv.run_months_voltvar(
        aq, SAI, year, months, n_parts=8, target=VV,
        uncurtailed=UNC, empirical_limit_basis="s_99", **COMMON,
    )
    vw.run_months_basic(
        aq, SAI, year, months, n_parts=8, target=VW, **COMMON,
    )
    vw.run_months_ghi(
        aq, SAI, year, months, n_parts=8, target=VWGHI,
        uncurtailed=UNC, **COMMON,
    )

vv.validate(aq, SAI, target=VV)
vw.validate_basic(aq, SAI, target=VW)
vw.validate_ghi(aq, SAI, target=VWGHI)
vw.cross_check(aq, SAI, target_basic=VW, target_ghi=VWGHI)
```

## Sensitivity reconstruction rules

| Changed factor | Rebuild required |
| --- | --- |
| Stage 1 normalisation basis | structured data, split, model, MAPE gate, all-uncurtailed P |
| Clear-sky heuristic or Stage 1 voltage aggregation | entire Stage 1 |
| Counterfactual cap only | all-uncurtailed P; then Stage 2 GHI-aware Volt-Watt and Volt-VAr curtailment outputs |
| Stage 2 rating basis only | all three Stage 2 tables; Stage 1 can be reused |
| Empirical apparent-limit basis only | Volt-VAr Stage 2 table only |
| Stage 2 voltage aggregation | all three Stage 2 tables |
| Flex selection | every table whose cohort is being compared |

Do not begin with a full factorial. Start with:

1. legacy Stage 1: S_99 normalisation with no counterfactual cap;
2. provider-rating/provider-cap baseline;
3. S_99 rating/S_99 cap sensitivity;
4. provider rating with S_99 cap, to isolate counterfactual capping;
5. S_99 rating with provider cap, to isolate standards-curve scaling;
6. average-versus-maximum voltage as a separate topology sensitivity.

Always compare retained sites/intervals, counterfactual coverage, cap frequency,
conformance counts, curtailment energy and the distribution of
`S_99/ac_capacity_kw`.

## Reconstructing S_99 when provenance is unavailable

The repository contains consumers of `meta_up23c.S_99`, but no job that
creates it. `stage1_ghi_pipeline/build_s99_estimates.py` is therefore an
explicitly labelled reconstruction, not a recovery of missing provenance. It
calculates site-level `P_kw` and `Q_kvar`, then
`sqrt(P_kw^2 + Q_kvar^2)`, and stores the 99th percentile with interval counts
and date coverage in a separate table. It never updates `meta_up23c`.

This reconstruction assumes `power` is instantaneous W, `energy_reactive` is
five-minute kvarh and `circuit_polarity` puts PV generation on the positive
sign convention. The `*12` conversion is invalid if the provider field is
already instantaneous var/kvar. Confirm those source contracts before treating
the values as more than exploratory sensitivity outputs.

Run the zero, 5%, and 20% active-power populations as sensitivity variants in
different target tables. If results change materially, the estimate is
population-dependent and must not be treated as a stable nameplate.

```python
import build_s99_estimates as s99

s99.create_table(aq, DB, target="s99_estimates_v1_all")
s99.run(aq, DB, years=(2024, 2025), target="s99_estimates_v1_all",
        min_active_power_fraction=0.0)

s99.create_table(aq, DB, target="s99_estimates_v1_p05")
s99.run(aq, DB, years=(2024, 2025), target="s99_estimates_v1_p05",
        min_active_power_fraction=0.05)

s99.create_table(aq, DB, target="s99_estimates_v1_p20")
s99.run(aq, DB, years=(2024, 2025), target="s99_estimates_v1_p20",
        min_active_power_fraction=0.20)
```

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
