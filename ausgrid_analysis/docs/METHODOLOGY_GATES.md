# Methodology gates for AMI-based inverter analysis

These gates prevent net-meter observations from being presented as direct inverter
measurements. A later delivery may change a gate only with recorded evidence.

| # | Question | Delivery 2 treatment | Required before formal conformance |
|---|---|---|---|
| 1 | How are household load and PV separated? | No decomposition. P and Q are labelled `net_meter`. | Estimate load, PV and uncertainty before curve comparison. |
| 2 | How are battery sites handled? | Identified from metadata and excluded from the primary solar-only cohort. They remain available for exploratory plots. | Develop and validate battery separation, or retain exclusion. |
| 3 | Is the P/Q sign convention verified? | Working configuration is retained, but labelled unverified. Raw P/Q are retained. | Obtain Ausgrid confirmation and run empirical day/night checks. |
| 4 | Which telemetry phase contains the inverter? | Candidate phases are inferred from power availability and a local day/night export signature, with a method and confidence. | Review mappings; low/unknown mappings are not assessable. |
| 5 | Are GMT timestamps converted before time logic? | UTC and `Australia/Sydney` local time are both retained. Local date/hour and UTC offset are materialised. | Validation must show only +10:00/+11:00 offsets. |
| 6 | Is voltage at the inverter terminals? | No. It is labelled `revenue_meter`; zero/invalid readings are excluded from voltage summaries. No line-drop correction is invented. | Treat results as meter-observed evidence, or validate a terminal-voltage model. |
| 7 | How is uncurtailed PV estimated for Volt-Watt? | Not estimated in Delivery 2. | Delivery 3 must estimate counterfactual PV from decomposed PV, irradiance and uncertainty—not raw net power. |
| 8 | What quantities are compared with droop curves? | None in Delivery 2. `formal_inverter_conformance_assessable=false`. | Use estimated inverter P/Q when defensible; otherwise label the result a net-meter proxy or `not_assessable`. |

## Interpretation rule

Delivery 2 is a structured observation layer, not a conformance result. A site can
be useful for exploration or decomposition while still being formally
`not_assessable`. This distinction must survive into every results table.

