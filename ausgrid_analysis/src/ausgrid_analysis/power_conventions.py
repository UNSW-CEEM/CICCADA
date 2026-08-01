"""Named power-convention conversions used by mechanism analysis.

The canonical sign contract lives in ``schemas.normalize_reactive_power``:
``q_absorbing_var`` is positive while absorbing and generator-convention Q is
the exact opposite.  Delivery 4 calls the helpers here instead of introducing
independent negations in SQL or notebooks.
"""

from __future__ import annotations


def q_generator_from_absorbing(
    q_absorbing_var: float | None,
) -> float | None:
    """Convert positive-absorbing Q to generator convention.

    Generator convention is ``+Q`` supplying and ``-Q`` absorbing.
    """

    return None if q_absorbing_var is None else -q_absorbing_var


def q_generator_from_absorbing_sql(column_sql: str) -> str:
    """Return the SQL twin of :func:`q_generator_from_absorbing`.

    ``column_sql`` must be an internally selected SQL expression, never
    untrusted user input.
    """

    if not column_sql.strip():
        raise ValueError("column_sql cannot be blank")
    return f"(-({column_sql}))"
