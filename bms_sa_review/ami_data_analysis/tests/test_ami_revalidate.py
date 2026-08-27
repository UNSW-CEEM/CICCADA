"""
Unit tests for `ami_revalidate` -- the full-year recheck of inactivity and
load-reconstruction/storage findings, month at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bms_sa_review.ami_data_analysis.lib import ami_revalidate as Reval


# --------------------------------------------------------------------------- #
# iter_month_partitions
# --------------------------------------------------------------------------- #

def test_iter_month_partitions_reads_and_concatenates_part_files(tmp_path):
    partition_dir = tmp_path / "ami_extract" / "dt_month=2025-06"
    partition_dir.mkdir(parents=True)
    pd.DataFrame({"circuit_id": [1], "power": [10.0]}).to_parquet(
        partition_dir / "part-load-0000.parquet"
    )
    pd.DataFrame({"circuit_id": [2], "power": [20.0]}).to_parquet(
        partition_dir / "part-pv-0000.parquet"
    )
    results = list(Reval.iter_month_partitions(tmp_path, [(2025, 6)]))
    assert len(results) == 1
    year, month, frame = results[0]
    assert (year, month) == (2025, 6)
    assert len(frame) == 2
    assert set(frame.circuit_id) == {1, 2}


def test_iter_month_partitions_missing_month_yields_empty_frame_not_skipped(tmp_path):
    results = list(Reval.iter_month_partitions(tmp_path, [(2025, 6), (2025, 7)]))
    assert len(results) == 2  # both months yielded, even with nothing landed
    for year, month, frame in results:
        assert len(frame) == 0


# --------------------------------------------------------------------------- #
# revalidate_inactive_circuits_over_history
# --------------------------------------------------------------------------- #

def test_revalidate_inactive_finds_circuit_that_only_dies_in_a_later_month():
    # Circuit 1 is alive in June, goes dead (near-zero) in July.
    june = pd.DataFrame({"circuit_id": [1, 2], "power": [500.0, 300.0]})
    july = pd.DataFrame({"circuit_id": [1, 2], "power": [1.0, 300.0]})
    month_frames = [(2025, 6, june), (2025, 7, july)]

    result = Reval.revalidate_inactive_circuits_over_history(month_frames)
    assert list(result.circuit_id) == [1]
    row = result.iloc[0]
    assert row.n_months_checked == 2
    assert row.months_inactive == ["2025-07"]


def test_revalidate_inactive_circuit_alive_every_month_is_absent():
    frames = [(2025, m, pd.DataFrame({"circuit_id": [1], "power": [500.0]})) for m in (6, 7)]
    result = Reval.revalidate_inactive_circuits_over_history(frames)
    assert len(result) == 0


def test_revalidate_inactive_skips_empty_months_but_still_counts_them():
    frames = [
        (2025, 6, pd.DataFrame({"circuit_id": [1], "power": [500.0]})),
        (2025, 7, pd.DataFrame(columns=["circuit_id", "power"])),
        (2025, 8, pd.DataFrame({"circuit_id": [1], "power": [1.0]})),
    ]
    result = Reval.revalidate_inactive_circuits_over_history(frames)
    assert result.iloc[0].n_months_checked == 3
    assert result.iloc[0].months_inactive == ["2025-08"]


def test_revalidate_inactive_no_months_at_all_returns_empty_with_columns():
    result = Reval.revalidate_inactive_circuits_over_history([])
    assert len(result) == 0
    assert list(result.columns) == ["circuit_id", "n_months_checked", "months_inactive"]


# --------------------------------------------------------------------------- #
# revalidate_reconstruction_over_history
# --------------------------------------------------------------------------- #

def _resolution(rows):
    """rows: list of (site_id, circuit_id, circuit_type, device_id, kept)."""
    frame = pd.DataFrame(rows, columns=[
        "site_id", "circuit_id", "circuit_type", "device_id", "kept",
    ])
    frame["drop_reason"] = None
    frame["needs_manual_review"] = False
    frame["power_correction_applied"] = False
    frame["implied_interval_minutes"] = np.nan
    return frame


def _month_frame(circuit_id, power_by_hour_utc, base_date="2025-06-01"):
    """One circuit's rows for one day, timestamps UTC, one per hour."""
    rows = []
    for hour, power in power_by_hour_utc.items():
        rows.append({
            "circuit_id": circuit_id,
            "t_stamp": pd.Timestamp(f"{base_date} {hour:02d}:00:00", tz="UTC"),
            "power": power,
            "energy": power / 12.0,
            "energy_reactive": 0.0,
            "voltage": 240.0,
        })
    return pd.DataFrame(rows)


def test_revalidate_reconstruction_flags_site_that_only_fails_in_one_month():
    resolution = _resolution([
        (1, 10, "ac_load_net", 100, True),
        (1, 11, "pv_site_net", 100, True),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})

    # AEST night hours (1-4am) are UTC 15-18 the previous day (UTC+10).
    # June: load reads sanely (small positive at night). July: load goes
    # sharply negative at night (a hidden battery only cycling in winter).
    june_load = _month_frame(10, {15: 200.0, 16: 200.0})
    june_pv = _month_frame(11, {15: 0.0, 16: 0.0})
    july_load = _month_frame(10, {15: -500.0, 16: -500.0})
    july_pv = _month_frame(11, {15: 0.0, 16: 0.0})

    june_frame = pd.concat([june_load, june_pv], ignore_index=True)
    july_frame = pd.concat([july_load, july_pv], ignore_index=True)
    month_frames = [(2025, 6, june_frame), (2025, 7, july_frame)]

    result = Reval.revalidate_reconstruction_over_history(
        month_frames, resolution, circuit_polarity,
    )
    assert list(result.site_id) == [1]
    row = result.iloc[0]
    assert row.months_flagged == ["2025-07"]
    assert row.worst_min_reconstructed_load < -100.0


def test_revalidate_reconstruction_site_clean_every_month_is_absent():
    resolution = _resolution([
        (1, 10, "ac_load_net", 100, True),
        (1, 11, "pv_site_net", 100, True),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    clean_frame = pd.concat([
        _month_frame(10, {15: 200.0, 16: 200.0}),
        _month_frame(11, {15: 0.0, 16: 0.0}),
    ], ignore_index=True)
    month_frames = [(2025, 6, clean_frame), (2025, 7, clean_frame)]

    result = Reval.revalidate_reconstruction_over_history(month_frames, resolution, circuit_polarity)
    assert len(result) == 0


def test_revalidate_reconstruction_empty_month_frame_is_skipped_safely():
    resolution = _resolution([(1, 10, "ac_load_net", 100, True)])
    circuit_polarity = pd.DataFrame({"circuit_id": [10], "circuit_polarity": [1]})
    month_frames = [(2025, 6, pd.DataFrame(columns=["circuit_id", "t_stamp", "power"]))]
    result = Reval.revalidate_reconstruction_over_history(month_frames, resolution, circuit_polarity)
    assert len(result) == 0


# --------------------------------------------------------------------------- #
# apply_full_year_findings
# --------------------------------------------------------------------------- #

def test_apply_full_year_findings_drops_circuit_and_excludes_site_independently():
    resolution = _resolution([
        (1, 10, "ac_load_net", 100, True),   # site 1: circuit goes inactive later
        (1, 11, "pv_site_net", 100, True),
        (2, 20, "ac_load_net", 200, True),   # site 2: reconstruction fails later
        (2, 21, "pv_site_net", 200, True),
        (3, 30, "ac_load_net", 300, True),   # site 3: untouched, stays kept
        (3, 31, "pv_site_net", 300, True),
    ])
    inactive = pd.DataFrame({
        "circuit_id": [10], "n_months_checked": [12], "months_inactive": [["2025-09"]],
    })
    reconstruction = pd.DataFrame({
        "site_id": [2], "n_months_checked": [12], "months_flagged": [["2025-11"]],
        "worst_min_reconstructed_load": [-800.0],
    })

    out = Reval.apply_full_year_findings(resolution, inactive, reconstruction)
    by_id = out.set_index("circuit_id")

    # Site 1: only circuit 10 dropped, its own reason; circuit 11 (PV) untouched.
    assert by_id.loc[10, "kept"] == False
    assert by_id.loc[10, "drop_reason"] == "inactive_full_year"
    assert by_id.loc[11, "kept"] == True

    # Site 2: BOTH circuits excluded (whole-site exclusion), same reason.
    assert by_id.loc[20, "kept"] == False
    assert by_id.loc[20, "drop_reason"] == "storage_or_sign_issue_full_year"
    assert by_id.loc[21, "kept"] == False
    assert by_id.loc[21, "drop_reason"] == "storage_or_sign_issue_full_year"

    # Site 3: untouched.
    assert by_id.loc[30, "kept"] == True
    assert by_id.loc[31, "kept"] == True


def test_apply_full_year_findings_no_findings_is_a_no_op():
    resolution = _resolution([(1, 10, "ac_load_net", 100, True)])
    empty_inactive = pd.DataFrame(columns=["circuit_id", "n_months_checked", "months_inactive"])
    empty_reconstruction = pd.DataFrame(columns=[
        "site_id", "n_months_checked", "months_flagged", "worst_min_reconstructed_load",
    ])
    out = Reval.apply_full_year_findings(resolution, empty_inactive, empty_reconstruction)
    assert out.set_index("circuit_id").loc[10, "kept"] == True


def test_apply_full_year_findings_empty_resolution_is_returned_as_is():
    empty = pd.DataFrame(columns=["site_id", "circuit_id", "kept", "drop_reason"])
    out = Reval.apply_full_year_findings(empty, pd.DataFrame(), pd.DataFrame())
    assert len(out) == 0


def test_apply_full_year_findings_does_not_resurrect_already_dropped_circuit():
    resolution = _resolution([(1, 10, "ac_load_net", 100, False)])
    resolution.loc[0, "drop_reason"] = "duplicate_cross_type"
    inactive = pd.DataFrame({
        "circuit_id": [10], "n_months_checked": [12], "months_inactive": [["2025-09"]],
    })
    out = Reval.apply_full_year_findings(resolution, inactive, pd.DataFrame())
    # already dropped -- inactive_full_year must NOT overwrite the original reason
    assert out.set_index("circuit_id").loc[10, "drop_reason"] == "duplicate_cross_type"
