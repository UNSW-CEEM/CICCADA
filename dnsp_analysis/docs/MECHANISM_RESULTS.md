# Mechanism result tables

## What this stage builds

The mechanism stage writes three independent result tables:

| Table | Grain | Question |
|---|---|---|
| `voltvar_proxy_results.parquet` | site × UTC year/month × inferred-phase scope × voltage bin | Does net-meter Q fall inside the configured Volt-VAr band when all gates are assessable? |
| `voltwatt_proxy_results.parquet` | site × UTC year/month × inferred-phase scope × voltage bin | Does net export exceed the configured Volt-Watt ceiling when all gates are assessable? |
| `response_observability.parquet` | site × UTC year/month × telemetry phase | Is a response in the expected direction visible under sufficient voltage excitation? |

These questions are not blended. `formal_inverter_conformance_assessable` is
always false because P/Q are measured at the revenue meter and include load.

Counterfactual-supported curtailment is not implemented. Gate 7 requires a
validated, uncertainty-aware load–PV decomposition and uncurtailed-PV estimate.

## Method choices

- P sign and Q sign have separate review states.
- The current raw-to-derived signs remain hypotheses until the sign notebook
  stage is reviewed and Ausgrid confirms the provider convention.
- Curve Q uses generator convention: `+Q` supplying, `-Q` absorbing.
- Site Q is converted once from persisted positive-absorbing Q by
  `power_conventions.q_generator_from_absorbing*`.
- Default comparison voltage is the maximum valid voltage across inferred DER
  phases. This conservatively represents high-voltage exposure; it does not
  claim to reproduce a polyphase inverter's internal control voltage.
- Per-phase and site min/mean/max revenue-meter voltages remain available in
  the structured source and sign diagnostics.
- The only permitted magnitude basis is `s_rated_kva`. There is no silent
  fallback to approved or solar capacity.
- Because the current `s_rated_kva` is null, magnitude rows default to
  `capacity_unavailable`; this is expected, not a build failure.

## Interpretation guardrails

Volt-Watt uses positive net-meter export only. An export value above a valid
curve ceiling can be conservative proxy evidence. A value below the ceiling is
not proof of inverter conformance because household load suppresses net export.

Volt-VAr compares net-meter Q with the curve only when sign, inputs, activation,
active-power capability and verified rating gates all pass. Household reactive
load can still influence the proxy.

Response observability reports slopes/correlations only. It does not prove that
an inverter mode caused the observed relationship.

## Denominators and keys

Every source interval enters exactly one mutually exclusive denominator state.
Validation proves that those states sum back to structured-site rows for both
curve tables and to structured-phase rows for observability.

All grouping uses UTC-derived year/month. `timestamp_local` is descriptive and
is never a uniqueness key, so the repeated NSW daylight-saving fall-back hour
is retained rather than deduplicated.

## Output locations

Sample:

```text
derived/samples/<scope>/mechanism_results/
derived/samples/<scope>/audit/mechanism_results_validation.json
```

Full:

```text
derived/mechanism_results/
derived/audit/mechanism_results_validation.json
```

The bounded sign-review files are under:

```text
derived/mechanism_results/sign_diagnostics/
```
