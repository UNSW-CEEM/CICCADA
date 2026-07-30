# Questions for Ausgrid

1. What exactly is measured by `ActivePower` and `Reactive Power`: whole-customer
   net flow at the AMI meter, or DER/inverter output?

2. Please confirm the sign conventions:
   - Does negative `ActivePower` mean export?
   - Does negative `Reactive Power` mean absorbing reactive power?

3. Should `(Unique Number ID, MeasureTime, Vphase)` be unique? If duplicate
   records occur, are they exact duplicates, corrections/revisions, or separate
   measurements?

4. The telemetry contains 1,342 IDs, while the metadata contains 1,282. Can
   metadata or an explanation be provided for the 60 unmatched IDs?

5. Can inverter rated apparent power (`S_rated`, kVA) be supplied? Please also
   confirm the meanings of `Solar_kW (total capacity)` and
   `Approved Capacity (kW)`.

6. For one- and two-phase DER systems, can the connected phase label(s)—A, B
   and/or C—be supplied?

