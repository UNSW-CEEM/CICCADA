"""CSIP conformance data-quality checks."""

import polars as pl


def has_usable_csip_export_limit(site_telemetry: pl.DataFrame) -> bool:
    """Return whether the raw CSIP export-limit signal is usable.

    At least one finite, positive limit is required. Once a site passes this
    check, zero-valued limits remain valid CSIP commands.
    """
    export_limit_w = site_telemetry["csip_export_limit_w"]
    return bool(
        (export_limit_w.is_finite() & (export_limit_w > 0)).fill_null(False).any()
    )
