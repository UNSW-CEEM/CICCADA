"""
Per-site circuit resolution (Phase 4): keep/drop decisions, the two
duplicate-resolution rules (cross-type is deterministic, same-type is a
flagged tie-break), the device/meter-model power correction, and the two
output tables built from a resolution frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bms_sa_review.ami_data_analysis.lib import ami_resolution as R


def _meta(rows):
    """rows: list of (site_id, circuit_id, circuit_type, device_id, is_pv)."""
    return pd.DataFrame(rows, columns=[
        "site_id", "circuit_id", "circuit_type", "device_id", "is_pv",
    ])


def _series(circuit_id, site_id, n=48, power=1000.0, energy=None, energy_reactive=50.0, vary=True):
    """
    Fake ts rows for one circuit. `power` (and `energy`, proportionally)
    varies slightly across timesteps by default, seeded by `circuit_id` so
    two DIFFERENT circuits never spuriously correlate, while a `.copy()` of
    one circuit's frame onto another circuit_id (the standard way these
    tests build a genuine duplicate) stays exactly correlated regardless.
    `find_duplicate_circuits` cannot define a correlation against a
    zero-variance (constant) series, so a fixture needs real variation for
    duplicate detection to have anything to find. `vary=False` keeps
    `energy` exactly constant instead, for tests that need an exact
    whole-Wh value (the device/meter-model granularity check).
    """
    t_stamp = pd.date_range("2025-06-01", periods=n, freq="5min")
    if vary:
        rng = np.random.RandomState(int(circuit_id) % (2**31 - 1))
        wobble = 1.0 + 0.02 * rng.normal(size=n)
    else:
        wobble = np.ones(n)
    power_series = power * wobble
    if energy is None:
        energy_series = power_series / 12.0  # power(W) * (5/60)h -> Wh
    else:
        energy_series = energy * wobble
    return pd.DataFrame({
        "circuit_id": circuit_id,
        "site_id": site_id,
        "t_stamp": t_stamp,
        "power": power_series,
        "energy": energy_series,
        "energy_reactive": energy_reactive,
        "voltage": 240.0,
    })


# ── classify_circuit_counts ─────────────────────────────────────────────────

def test_classify_circuit_counts_splits_clean_from_other():
    counts = pd.DataFrame({"site_id": [1, 2, 3, 4], "n_circuits": [1, 3, 2, 5]})
    clean, other = R.classify_circuit_counts(counts)
    assert clean == [1, 2]
    assert other == [3, 4]


def test_classify_circuit_counts_empty_input():
    assert R.classify_circuit_counts(pd.DataFrame()) == ([], [])
    assert R.classify_circuit_counts(None) == ([], [])


# ── resolve_site_circuits: trivial / no-intervention case ──────────────────

def test_single_phase_site_needs_no_intervention():
    meta = _meta([
        (1, 101, "ac_load_net", 9001, False),
        (1, 102, "pv_site_net", 9002, True),
    ])
    sample = pd.concat([_series(101, 1, power=1500.0), _series(102, 1, power=-800.0)])
    out = R.resolve_site_circuits(meta, sample)
    assert out.kept.all()
    assert out.drop_reason.isna().all()
    assert not out.needs_manual_review.any()


# ── cross-type duplicates: deterministic drop of the load side ─────────────

def test_cross_type_duplicate_drops_load_side_keeps_pv_side():
    meta = _meta([
        (2, 201, "ac_load_net", 9101, False),   # mistagged duplicate of 202
        (2, 202, "pv_site_net", 9102, True),
    ])
    identical = _series(201, 2, power=-500.0)
    twin = identical.copy()
    twin["circuit_id"] = 202
    sample = pd.concat([identical, twin])

    out = R.resolve_site_circuits(meta, sample)
    kept = out.set_index("circuit_id")
    assert kept.loc[201, "kept"] == False
    assert kept.loc[201, "drop_reason"] == "duplicate_cross_type"
    assert kept.loc[201, "needs_manual_review"] == False
    assert kept.loc[202, "kept"] == True


# ── same-type duplicates: coverage tie-break, flagged for review ──────────

def test_same_type_duplicate_group_keeps_most_covered_and_flags_review():
    meta = _meta([
        (3, 301, "pv_site_net", 9201, True),
        (3, 302, "pv_site_net", 9202, True),
        (3, 303, "pv_site_net", 9203, True),
    ])
    base = _series(301, 3, n=48, power=-1000.0)
    full_twin = base.copy(); full_twin["circuit_id"] = 302
    partial_twin = base.copy(); partial_twin["circuit_id"] = 303
    partial_twin = partial_twin.iloc[:24]  # less coverage than 301/302

    sample = pd.concat([base, full_twin, partial_twin])
    out = R.resolve_site_circuits(meta, sample)
    by_id = out.set_index("circuit_id")

    n_kept = int(by_id.kept.sum())
    assert n_kept == 1
    # the least-covered circuit must not be the one kept
    assert by_id.loc[303, "kept"] == False
    assert by_id.loc[303, "drop_reason"] == "duplicate_same_type"
    # every circuit in the group is flagged, including whichever was kept
    assert out.needs_manual_review.all()


# ── inactive circuits ───────────────────────────────────────────────────────

def test_inactive_circuit_is_dropped():
    meta = _meta([
        (4, 401, "ac_load_net", 9301, False),
        (4, 402, "pv_site_net", 9302, True),
    ])
    sample = pd.concat([
        _series(401, 4, power=1200.0),
        _series(402, 4, power=1.0, energy=1.0 / 12.0),  # near-zero all day
    ])
    out = R.resolve_site_circuits(meta, sample)
    by_id = out.set_index("circuit_id")
    assert by_id.loc[402, "kept"] == False
    assert by_id.loc[402, "drop_reason"] == "inactive"
    assert by_id.loc[402, "needs_manual_review"] == False


def test_duplicate_takes_priority_over_inactive_reason():
    # A circuit that is both a duplicate AND inactive should be reported as
    # a duplicate -- the more specific, more diagnostic reason.
    meta = _meta([
        (5, 501, "ac_load_net", 9401, False),
        (5, 502, "pv_site_net", 9402, True),
    ])
    quiet = _series(501, 5, power=2.0, energy=2.0 / 12.0)
    # Same shape (guarantees correlation with `quiet`), but scaled up so the
    # kept PV-side circuit reads well above the inactive threshold --
    # otherwise dropping it as "inactive" would be the correct call, not a
    # priority-ordering bug.
    twin = quiet.copy()
    twin["circuit_id"] = 502
    twin["power"] = quiet["power"] * 300
    twin["energy"] = quiet["energy"] * 300
    sample = pd.concat([quiet, twin])

    out = R.resolve_site_circuits(meta, sample)
    dropped_reasons = set(out.loc[~out.kept, "drop_reason"])
    assert "duplicate_cross_type" in dropped_reasons
    assert "inactive" not in dropped_reasons


# ── device/meter-model correction ──────────────────────────────────────────

def test_flagged_circuit_gets_power_correction_applied():
    meta = _meta([(6, 601, "ac_load_net", 9501, False)])
    # A whole-Wh energy register with an implied ~4.9-minute accumulation
    # window instead of the nominal 5 -- chosen so `energy` lands on an
    # exact integer, matching the real device/meter-model signature
    # (share_integer_energy > 0.9) rather than just the interval mismatch.
    power = 600.0
    energy = 49.0  # power * 4.9/60 exactly
    sample = _series(601, 6, power=power, energy=energy, vary=False)
    out = R.resolve_site_circuits(meta, sample)
    row = out.iloc[0]
    assert row.kept
    assert row.power_correction_applied
    assert row.implied_interval_minutes == pytest.approx(4.9, abs=0.05)


def test_unflagged_circuit_has_no_correction():
    meta = _meta([(7, 701, "ac_load_net", 9601, False)])
    sample = _series(701, 7, power=1000.0, vary=False)  # exact 5-min ratio
    out = R.resolve_site_circuits(meta, sample)
    row = out.iloc[0]
    assert row.power_correction_applied == False


def test_empty_meta_returns_empty_frame_with_expected_columns():
    out = R.resolve_site_circuits(pd.DataFrame(columns=[
        "site_id", "circuit_id", "circuit_type", "device_id", "is_pv"
    ]), pd.DataFrame())
    assert out.empty
    assert "kept" in out.columns
    assert "needs_manual_review" in out.columns


# ── build_interval_table ────────────────────────────────────────────────────

def test_interval_table_excludes_dropped_circuits_and_keeps_device_id_tag():
    meta = _meta([
        (8, 801, "ac_load_net", 9701, False),
        (8, 802, "pv_site_net", 9702, True),
    ])
    identical = _series(801, 8, power=-400.0)
    twin = identical.copy(); twin["circuit_id"] = 802
    sample = pd.concat([identical, twin])

    resolution = R.resolve_site_circuits(meta, sample)
    interval = R.build_interval_table(sample, resolution)

    assert set(interval.circuit_id.unique()) == {802}
    assert "device_id" in interval.columns
    assert (interval.device_id == 9702).all()


def test_interval_table_applies_active_power_correction_when_flagged():
    # `implied_interval_minutes` is the MEDIAN ratio across the circuit's
    # whole sample, so a fixture needs more than one row to demonstrate the
    # correction actually overriding a bad raw `power` value -- with only
    # one row, "corrected" and "raw" are tautologically the same number
    # (the implied interval is derived FROM that row's own ratio). Nine
    # normal rows establish the circuit's real ~4.9-minute accumulation
    # window; one glitch row has a raw `power` spike NOT reflected in its
    # `energy` reading -- exactly the real-world failure mode (a bad
    # instantaneous reading on an otherwise-trustworthy energy register).
    meta = _meta([(9, 901, "ac_load_net", 9801, False)])
    n_normal = 9
    t_stamp = pd.date_range("2025-06-01", periods=n_normal + 1, freq="5min")
    power = [1000.0] * n_normal + [2000.0]     # last row: a raw-power glitch
    energy = [82.0] * (n_normal + 1)           # every row's energy is consistent
    sample = pd.DataFrame({
        "circuit_id": 901, "site_id": 9, "t_stamp": t_stamp,
        "power": power, "energy": energy, "energy_reactive": 10.0, "voltage": 240.0,
    })

    resolution = R.resolve_site_circuits(meta, sample)
    row = resolution.iloc[0]
    assert row.power_correction_applied
    implied = row.implied_interval_minutes

    interval = R.build_interval_table(sample, resolution)
    glitch_row = interval.sort_values("t_stamp").iloc[-1]
    expected_corrected = 82.0 * 60.0 / implied
    assert glitch_row.power == pytest.approx(expected_corrected, rel=1e-6)
    assert glitch_row.power != pytest.approx(2000.0)


def test_interval_table_reactive_power_always_derived():
    meta = _meta([(10, 1001, "ac_load_net", 9901, False)])
    sample = _series(1001, 10, power=1000.0, energy_reactive=25.0, vary=False)
    resolution = R.resolve_site_circuits(meta, sample)
    interval = R.build_interval_table(sample, resolution)

    # nominal 5-minute interval, unflagged circuit: 25 Wh * 60 / 5 = 300 W
    assert interval.reactive_power.iloc[0] == pytest.approx(300.0)


def test_interval_table_empty_resolution_returns_empty_with_columns():
    resolution = pd.DataFrame(columns=[
        "site_id", "circuit_id", "circuit_type", "device_id",
        "kept", "drop_reason", "needs_manual_review",
        "power_correction_applied", "implied_interval_minutes",
    ])
    out = R.build_interval_table(pd.DataFrame(), resolution)
    assert out.empty
    assert "reactive_power" in out.columns


# ── build_site_metadata ─────────────────────────────────────────────────────

def test_site_metadata_audit_trail_records_kept_and_dropped():
    meta = _meta([
        (11, 1101, "ac_load_net", 9911, False),
        (11, 1102, "pv_site_net", 9912, True),
    ])
    identical = _series(1101, 11, power=-300.0)
    twin = identical.copy(); twin["circuit_id"] = 1102
    sample = pd.concat([identical, twin])
    resolution = R.resolve_site_circuits(meta, sample)

    site_level = pd.DataFrame({"site_id": [11], "state": ["NSW"]})
    out = R.build_site_metadata(site_level, resolution)

    assert len(out) == 1
    row = out.iloc[0]
    assert row.kept_circuit_ids == [1102]
    assert row.dropped_circuits == [{"circuit_id": 1101, "reason": "duplicate_cross_type"}]
    assert row.device_id_groups == {9912: [1102]}
    assert row.needs_manual_review == False
    assert row.state == "NSW"


def test_site_metadata_inner_joins_on_site_id():
    resolution = pd.DataFrame({
        "site_id": [99], "circuit_id": [1], "device_id": [1],
        "kept": [True], "drop_reason": [None], "needs_manual_review": [False],
    })
    site_level = pd.DataFrame({"site_id": [1], "state": ["VIC"]})  # no overlap
    out = R.build_site_metadata(site_level, resolution)
    assert out.empty


# ── build_coverage_report ───────────────────────────────────────────────────

def test_coverage_report_tallies_each_outcome_once():
    resolution = pd.DataFrame([
        # site 1: no intervention
        {"site_id": 1, "circuit_id": 11, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": False},
        # site 2: cross-type duplicate auto-resolved
        {"site_id": 2, "circuit_id": 21, "kept": False, "drop_reason": "duplicate_cross_type",
         "needs_manual_review": False, "power_correction_applied": False},
        {"site_id": 2, "circuit_id": 22, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": False},
        # site 3: inactive auto-resolved, and the surviving circuit is
        # flagged for the power correction
        {"site_id": 3, "circuit_id": 31, "kept": False, "drop_reason": "inactive",
         "needs_manual_review": False, "power_correction_applied": False},
        {"site_id": 3, "circuit_id": 32, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": True},
        # site 4: same-type duplicate, flagged for manual review
        {"site_id": 4, "circuit_id": 41, "kept": False, "drop_reason": "duplicate_same_type",
         "needs_manual_review": True, "power_correction_applied": False},
        {"site_id": 4, "circuit_id": 42, "kept": True, "drop_reason": None,
         "needs_manual_review": True, "power_correction_applied": False},
    ])
    report = R.build_coverage_report(resolution)
    assert report["n_sites"] == 4
    assert report["n_no_intervention"] == 1
    assert report["n_auto_resolved_duplicate_cross_type"] == 1
    assert report["n_auto_resolved_inactive"] == 1
    assert report["n_flagged_manual_review"] == 1
    assert report["n_power_correction_applied"] == 1


def test_coverage_report_empty_input():
    report = R.build_coverage_report(pd.DataFrame())
    assert report["n_sites"] == 0


# ── exclude_flagged_sites ────────────────────────────────────────────────────

def test_exclude_flagged_sites_drops_surviving_circuits_at_flagged_site():
    resolution = pd.DataFrame([
        {"site_id": 1, "circuit_id": 11, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": False},
        {"site_id": 1, "circuit_id": 12, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": False},
        {"site_id": 2, "circuit_id": 21, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": False},
    ])
    out = R.exclude_flagged_sites(resolution, excluded_site_ids=[1])
    by_id = out.set_index("circuit_id")
    assert by_id.loc[11, "kept"] == False
    assert by_id.loc[11, "drop_reason"] == "storage_or_sign_issue"
    assert by_id.loc[12, "kept"] == False
    assert by_id.loc[21, "kept"] == True  # untouched, different site


def test_exclude_flagged_sites_preserves_existing_drop_reason():
    resolution = pd.DataFrame([
        {"site_id": 1, "circuit_id": 11, "kept": False, "drop_reason": "inactive",
         "needs_manual_review": False, "power_correction_applied": False},
        {"site_id": 1, "circuit_id": 12, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": False},
    ])
    out = R.exclude_flagged_sites(resolution, excluded_site_ids=[1])
    by_id = out.set_index("circuit_id")
    # already-dropped circuit keeps its ORIGINAL, more specific reason
    assert by_id.loc[11, "drop_reason"] == "inactive"
    assert by_id.loc[12, "drop_reason"] == "storage_or_sign_issue"


def test_exclude_flagged_sites_empty_exclusion_list_is_a_no_op():
    resolution = pd.DataFrame([
        {"site_id": 1, "circuit_id": 11, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": False},
    ])
    out = R.exclude_flagged_sites(resolution, excluded_site_ids=[])
    assert out.iloc[0]["kept"] == True


def test_exclude_flagged_sites_empty_resolution():
    assert R.exclude_flagged_sites(pd.DataFrame(), [1]).empty


# ── build_coverage_report: storage/sign exclusion bucket ────────────────────

def test_coverage_report_counts_excluded_sites_separately():
    resolution = pd.DataFrame([
        {"site_id": 1, "circuit_id": 11, "kept": False, "drop_reason": "storage_or_sign_issue",
         "needs_manual_review": False, "power_correction_applied": False},
        {"site_id": 2, "circuit_id": 21, "kept": True, "drop_reason": None,
         "needs_manual_review": False, "power_correction_applied": False},
    ])
    report = R.build_coverage_report(resolution)
    assert report["n_excluded_storage_or_sign_issue"] == 1
    assert report["n_no_intervention"] == 1  # site 2 only
