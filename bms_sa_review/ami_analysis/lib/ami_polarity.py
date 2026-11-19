"""
`circuit_polarity` correction for `ami_meter_residential` -- Phase 7
(`bms_sa_review/ami_analysis`), notebook 05's bootstrap (Section 1).

WHY THIS EXISTS
--------------------------------------------------------------------------
`ami_meter_residential`'s `P_kw`/`Q_kvar` are, by design, the RAW reading a
real monitoring device would report -- `ami_build.py`'s own docstring is
explicit that no polarity correction is applied there ("this table is
meant to be what a real meter would show, polarity quirks and all"). That
is a deliberate choice for genuine AMI behaviour (net-of-solar as landed,
no correction attempted on THAT), but a per-circuit sign inversion
(`circuit_polarity == -1`) is a different thing: it is a CT/monitoring
-device wiring defect at that one circuit, not something a real AMI meter
does. It was discovered by an exact, airtight match: site 13683586's
circuit 448410 has `corr(ami_meter.P_kw, ami_raw_phaseseparate.P_kw_signed)
== -1.0000` and the same for Q -- every single interval, both columns,
perfectly negated -- while its sibling circuits 448408/448409 both show
`corr == +1.0000` (matching `ami_circuit_metadata.circuit_polarity == 1`
for those two).

Correcting for this does not compromise the "blind" (AMI-only) premise of
notebook 05: `circuit_polarity` is genuinely available metering
installation metadata (from `meta_up23c`, surfaced via
`ami_circuit_metadata`), not the true PV-separated ground truth. A real
analyst working from AMI + circuit metadata alone would have access to
exactly this. It stays out of `ami_meter`/`ami_meter_clean`/
`ami_meter_residential` themselves (Phase 5 artifacts, likely other
consumers, and the point of `ami_meter_residential` is to test whether a
blind pipeline can be BUILT to handle this on its own) -- it's applied
here, in the analysis layer, where the AMI reading and the circuit
metadata are both already in hand.

WHERE THIS GETS WIRED IN
--------------------------------------------------------------------------
Applied ONCE, as a single choke point: notebook 05's bootstrap cell
redefines the `ami_meter_residential` VIEW itself (not the underlying
parquet -- nothing on disk changes) to be
`build_polarity_corrected_sql("_ami_meter_residential_raw", ...)` instead
of a plain passthrough. Every other cell and every `ami_analysis/lib`
function that queries `ami_meter_residential` by name through the same
`con` -- Section 2's PV-phase inference, Section 4/6's fleet-wide
conformance SQL, `ami_conformance_plot.py`'s `fetch_site_day_conformance`/
`fetch_circuit_scope_conformance` -- sees the corrected values
automatically and needs no changes of its own, because a DuckDB view is
re-evaluated live on every query, not materialised once. This applies
uniformly to every circuit in scope: circuits with `circuit_polarity == 1`
(or no `ami_circuit_metadata` match at all) come out numerically
unchanged, since multiplying by 1 is a no-op.
"""

from __future__ import annotations

__all__ = ["build_polarity_corrected_sql"]


def build_polarity_corrected_sql(
    source_table: str,
    circuit_metadata_table: str = "ami_circuit_metadata",
    *,
    power_column: str = "P_kw",
    reactive_power_column: str = "Q_kvar",
    circuit_column: str = "circuit_id",
    polarity_column: str = "circuit_polarity",
    default_polarity: int = 1,
) -> str:
    """
    One row per row of `source_table`, with `power_column`/
    `reactive_power_column` multiplied by that circuit's
    `circuit_metadata_table.polarity_column` (LEFT JOINed on
    `circuit_column`). A circuit with no matching row in
    `circuit_metadata_table` -- or a NULL polarity there -- is assumed
    correctly wired (`default_polarity`, 1) rather than dropped or turned
    into NaN; every other column of `source_table` passes through
    untouched (in particular `S_kva`, which is `hypot(P, Q)` and therefore
    sign-invariant -- it is NOT recomputed here, since it is already
    correct regardless of polarity).

    Adds one extra column, `circuit_polarity_applied`, so the correction
    actually used for each row is auditable rather than invisible -- e.g.
    `SELECT DISTINCT circuit_id, circuit_polarity_applied FROM ...` shows
    exactly which circuits were flipped.
    """
    return f"""
        WITH polarity AS (
            SELECT
                s.*,
                COALESCE(c.{polarity_column}, {default_polarity}) AS circuit_polarity_applied
            FROM {source_table} s
            LEFT JOIN {circuit_metadata_table} c USING ({circuit_column})
        )
        SELECT
            * EXCLUDE ({power_column}, {reactive_power_column}, circuit_polarity_applied),
            {power_column} * circuit_polarity_applied AS {power_column},
            {reactive_power_column} * circuit_polarity_applied AS {reactive_power_column},
            circuit_polarity_applied
        FROM polarity
    """.strip()
