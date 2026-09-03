"""
Phase 2: which dataset do we build the synthetic AMI meter from?
==================================================================

The comparison logic behind `02_source_selection.ipynb` -- pure and testable,
and the ONE place "which source qualifies" is decided.

The question this module exists to answer, stated plainly: a real AMI meter
records net import/export, but disaggregation needs the two things that sum
to it -- `pv_generation` and `gross_load` -- as SEPARATE signals to serve as
ground truth. A candidate source only qualifies if it can supply both, at a
grain fine enough to actually tell them apart. Cost and cleanliness are real,
but secondary to that one hard constraint -- this module keeps them separate
so a cheap, clean, disqualified candidate cannot look like a contender.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bms_sa_review.synthetic_ami_creation.lib import ami_athena as Athena

__all__ = [
    "SourceCandidate", "verify_is_pv_only", "compare_circuit_and_row_shares",
    "build_comparison_table", "recommend",
]


@dataclass(frozen=True)
class SourceCandidate:
    """
    One row of the Phase 2 comparison.

    Evidence in, verdict out -- the notebook cell that constructs one of these
    should point at the specific evidence cell each field's value came from,
    so a reader can trace "has_load_signal=False" back to a real check rather
    than trusting a bare assertion.
    """
    name: str
    grain: str                    # e.g. "circuit, 5-min" | "site, 5-min"
    has_load_signal: bool
    has_pv_signal: bool
    decomposable: bool            # can load and PV be told apart in this table
    n_rows: int | None = None
    size_bytes: float | None = None
    cleanliness_notes: str = ""
    complexity_notes: str = ""


def verify_is_pv_only(schema: pd.DataFrame, column: str = "column_name") -> dict:
    """
    Does a table's schema even carry an `is_pv` (or similarly named) column? Pure.

    This is STRUCTURAL evidence, independent of any query result: if the schema
    has no such column, the table cannot distinguish PV from load no matter what
    its rows contain -- whatever exclusion happened, happened upstream of this
    table existing at all. Returns `{"is_pv_only": bool | None, "reason": str}`.

    `is_pv_only=True` means "structurally cannot carry both PV and load
    circuits" -- it does NOT by itself prove which one signal the table does
    carry; that requires reading the pipeline that built it (see the notebook).

    `is_pv_only=None` (with a reason) when no schema was supplied, so a missing
    check is visibly missing rather than silently read as "verified false".
    """
    if schema is None or not len(schema) or column not in schema.columns:
        return {"is_pv_only": None, "reason": "no schema provided -- cannot assess"}
    names = {str(c).strip().lower() for c in schema[column]}
    has_is_pv = "is_pv" in names
    return {
        "is_pv_only": not has_is_pv,
        "reason": (
            "no `is_pv` column in the schema -- the table cannot carry both signals"
            if not has_is_pv else
            "`is_pv` column present -- both signals may coexist here"
        ),
    }


def compare_circuit_and_row_shares(
    circuit_counts: pd.DataFrame,
    ts_rows_false: int,
    ts_rows_true: int,
    is_pv_column: str = "is_pv",
) -> pd.DataFrame:
    """
    Circuit-count share vs row-volume share, by is_pv. Pure.

    These are two different measures of the same split and need not agree: a
    circuit reporting for a shorter window, or at lower completeness, contributes
    fewer rows than its circuit-count share would suggest. A large divergence is
    worth noticing rather than assumed away -- this makes it visible instead of
    silently averaging over it.

    `circuit_counts` is expected from a live `meta_up23c` query: columns
    `is_pv` (bool-like), `n_circuits`, `n_sites`. A value that is not
    recognisably True/False (including null) gets its own "unknown" row rather
    than being dropped, so a data-quality issue surfaces instead of vanishing.
    """
    if circuit_counts is None or not len(circuit_counts):
        return pd.DataFrame()

    frame = circuit_counts.copy()

    def _label(value) -> str:
        if isinstance(value, bool):
            return "is_pv=true (pv)" if value else "is_pv=false (load)"
        text = str(value).strip().lower()
        if text in ("true", "1"):
            return "is_pv=true (pv)"
        if text in ("false", "0"):
            return "is_pv=false (load)"
        return f"is_pv=unknown ({value!r})"

    frame["label"] = frame[is_pv_column].map(_label)
    frame["share_of_circuits"] = frame.n_circuits / frame.n_circuits.sum()

    ts_rows_by_label = {
        "is_pv=false (load)": ts_rows_false,
        "is_pv=true (pv)": ts_rows_true,
    }
    frame["n_rows_in_ts"] = frame["label"].map(ts_rows_by_label)
    total_ts_rows = ts_rows_false + ts_rows_true
    frame["share_of_ts_rows"] = frame.n_rows_in_ts / total_ts_rows if total_ts_rows else None

    return frame[["label", "n_circuits", "share_of_circuits", "n_sites",
                  "n_rows_in_ts", "share_of_ts_rows"]]


def build_comparison_table(candidates: list[SourceCandidate]) -> pd.DataFrame:
    """
    The Phase 2 trade-off table: granularity, both signals, cost. Pure.

    `full_scan_cost_aud` is what ONE full unfiltered scan of the table would
    cost at Athena's per-TB rate -- not what Phase 4's actual extract will cost
    (that depends on the extraction mechanism, decided in Phase 4), just a
    same-units number that makes the size column concrete.
    """
    rows = []
    for c in candidates:
        cost = Athena.estimate_cost(c.size_bytes) if c.size_bytes else None
        rows.append({
            "candidate": c.name,
            "grain": c.grain,
            "has_load_signal": c.has_load_signal,
            "has_pv_signal": c.has_pv_signal,
            "decomposable": c.decomposable,
            "n_rows": c.n_rows,
            "size": Athena.fmt_bytes(c.size_bytes) if c.size_bytes else "",
            "full_scan_cost_aud": None if cost is None else round(cost, 2),
            "cleanliness": c.cleanliness_notes,
            "complexity": c.complexity_notes,
        })
    return pd.DataFrame(rows)


def recommend(candidates: list[SourceCandidate]) -> dict:
    """
    Apply the one hard constraint and return a verdict. Pure.

    A candidate qualifies iff `has_load_signal`, `has_pv_signal` and
    `decomposable` are all True -- cost and cleanliness never override this,
    by design, so a cheap disqualified candidate cannot look like a contender.

    Returns `{"qualifying": [names], "excluded": {name: reason}}`. Zero or more
    than one qualifying candidate is reported as-is rather than guessed at --
    a human reads `excluded` either way and decides.
    """
    qualifying = []
    excluded = {}
    for c in candidates:
        if c.has_load_signal and c.has_pv_signal and c.decomposable:
            qualifying.append(c.name)
        else:
            missing = []
            if not c.has_load_signal:
                missing.append("no load signal")
            if not c.has_pv_signal:
                missing.append("no PV signal")
            if not c.decomposable:
                missing.append("not decomposable at this grain")
            excluded[c.name] = ", ".join(missing)
    return {"qualifying": qualifying, "excluded": excluded}
