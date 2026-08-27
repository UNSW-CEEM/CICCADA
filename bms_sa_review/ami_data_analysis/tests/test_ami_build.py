"""
Unit tests for `ami_build` -- deriving `ami_raw` (pv_generation, gross_load)
and `ami_meter` (per-phase net readings) from a resolved interval table,
and the month-at-a-time write/orchestration around it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bms_sa_review.ami_data_analysis.lib import ami_build as Build


def _interval_table(rows):
    """rows: (site_id, device_id, circuit_id, circuit_type, t_stamp, power)."""
    return pd.DataFrame(rows, columns=[
        "site_id", "device_id", "circuit_id", "circuit_type", "t_stamp", "power",
    ])


def _resolution(rows):
    """rows: (site_id, circuit_id, circuit_type, device_id, kept)."""
    frame = pd.DataFrame(rows, columns=[
        "site_id", "circuit_id", "circuit_type", "device_id", "kept",
    ])
    frame["drop_reason"] = None
    frame["needs_manual_review"] = False
    frame["power_correction_applied"] = False
    frame["implied_interval_minutes"] = np.nan
    return frame


# --------------------------------------------------------------------------- #
# build_ami_raw
# --------------------------------------------------------------------------- #

def test_build_ami_raw_combines_pv_and_reconstructed_load():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([
        (1, 100, 10, "ac_load_net", t, -300.0),   # exporting more than drawing
        (1, 100, 11, "pv_site_net", t, 1000.0),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})

    result = Build.build_ami_raw(interval_table, circuit_polarity)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.site_id == 1
    assert row.pv_generation == 1000.0
    # gross_load = load_signed + pv_signed = (-300*1) + (1000*-1) = -1300
    assert row.gross_load == pytest.approx(-1300.0)


def test_build_ami_raw_sums_multiple_pv_circuits_at_one_site():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([
        (1, 100, 10, "ac_load_net", t, 200.0),
        (1, 101, 11, "pv_site_net", t, 500.0),
        (1, 102, 12, "pv_site_net", t, 300.0),
    ])
    circuit_polarity = pd.DataFrame({
        "circuit_id": [10, 11, 12], "circuit_polarity": [1, -1, -1],
    })
    result = Build.build_ami_raw(interval_table, circuit_polarity)
    assert result.iloc[0].pv_generation == 800.0


def test_build_ami_raw_drops_timestamps_missing_either_side():
    t1 = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    t2 = pd.Timestamp("2025-06-01 13:00", tz="UTC")  # load-only, no PV row here
    interval_table = _interval_table([
        (1, 100, 10, "ac_load_net", t1, 200.0),
        (1, 100, 11, "pv_site_net", t1, 500.0),
        (1, 100, 10, "ac_load_net", t2, 200.0),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    result = Build.build_ami_raw(interval_table, circuit_polarity)
    assert len(result) == 1
    assert result.iloc[0].t_stamp == t1


def test_build_ami_raw_empty_interval_table_returns_empty_with_columns():
    result = Build.build_ami_raw(pd.DataFrame(), pd.DataFrame())
    assert len(result) == 0
    assert list(result.columns) == ["site_id", "t_stamp", "pv_generation", "gross_load"]


# --------------------------------------------------------------------------- #
# build_ami_meter
# --------------------------------------------------------------------------- #

def test_build_ami_meter_keeps_phases_separate_and_splits_import_export():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([
        (1, 100, 10, "ac_load_net", t, 500.0),   # phase A: importing
        (1, 100, 20, "ac_load_net", t, -200.0),  # phase B: exporting
        (1, 100, 30, "pv_site_net", t, 900.0),   # not a meter reading -- excluded
    ])
    result = Build.build_ami_meter(interval_table)
    assert set(result.circuit_id) == {10, 20}  # PV circuit excluded entirely

    phase_a = result.set_index("circuit_id").loc[10]
    assert phase_a.net_power_w == 500.0
    assert phase_a.net_import_w == 500.0
    assert phase_a.net_export_w == 0.0

    phase_b = result.set_index("circuit_id").loc[20]
    assert phase_b.net_power_w == -200.0
    assert phase_b.net_import_w == 0.0
    assert phase_b.net_export_w == 200.0


def test_build_ami_meter_no_load_circuits_returns_empty_with_columns():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([(1, 100, 30, "pv_site_net", t, 900.0)])
    result = Build.build_ami_meter(interval_table)
    assert len(result) == 0
    assert "net_import_w" in result.columns


def test_build_ami_meter_empty_interval_table():
    result = Build.build_ami_meter(pd.DataFrame())
    assert len(result) == 0


# --------------------------------------------------------------------------- #
# write_month_table
# --------------------------------------------------------------------------- #

def test_write_month_table_writes_hive_partitioned_parquet(tmp_path):
    frame = pd.DataFrame({"site_id": [1], "pv_generation": [500.0], "gross_load": [800.0]})
    path = Build.write_month_table(frame, tmp_path, 2025, 6, table_name="ami_raw")
    assert path is not None
    assert "ami_raw" in str(path) and "dt_month=2025-06" in str(path)
    assert len(pd.read_parquet(path)) == 1


def test_write_month_table_empty_frame_writes_nothing(tmp_path):
    empty = pd.DataFrame(columns=["site_id", "pv_generation"])
    path = Build.write_month_table(empty, tmp_path, 2025, 6, table_name="ami_raw")
    assert path is None
    assert not list(tmp_path.rglob("*.parquet"))


def test_write_month_table_rerun_overwrites(tmp_path):
    frame1 = pd.DataFrame({"site_id": [1]})
    frame2 = pd.DataFrame({"site_id": [1, 2]})
    Build.write_month_table(frame1, tmp_path, 2025, 6, table_name="ami_meter")
    path2 = Build.write_month_table(frame2, tmp_path, 2025, 6, table_name="ami_meter")
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1
    assert len(pd.read_parquet(path2)) == 2


# --------------------------------------------------------------------------- #
# run_build
# --------------------------------------------------------------------------- #

def test_run_build_writes_both_tables_for_every_month(tmp_path):
    resolution = _resolution([
        (1, 10, "ac_load_net", 100, True),
        (1, 11, "pv_site_net", 100, True),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})

    def _frame(month):
        t = pd.Timestamp(f"2025-{month:02d}-01 12:00", tz="UTC")
        return _interval_table([
            (1, 100, 10, "ac_load_net", t, 200.0),
            (1, 100, 11, "pv_site_net", t, 500.0),
        ])[["circuit_id", "t_stamp", "power"]].assign(
            energy=lambda d: d.power / 12.0, energy_reactive=0.0, voltage=240.0,
        )

    month_frames = [(2025, 6, _frame(6)), (2025, 7, _frame(7))]
    raw_dir, meter_dir = tmp_path / "ami_raw", tmp_path / "ami_meter"

    manifest = Build.run_build(month_frames, resolution, circuit_polarity, raw_dir, meter_dir)
    assert len(manifest) == 2
    assert (manifest.n_raw_rows > 0).all()
    assert (manifest.n_meter_rows > 0).all()
    for path in list(manifest.raw_path) + list(manifest.meter_path):
        assert path.exists()


def test_run_build_month_with_no_rows_still_gets_a_manifest_row(tmp_path):
    resolution = _resolution([(1, 10, "ac_load_net", 100, True)])
    circuit_polarity = pd.DataFrame({"circuit_id": [10], "circuit_polarity": [1]})
    empty_frame = pd.DataFrame(columns=["circuit_id", "t_stamp", "power", "energy", "energy_reactive", "voltage"])
    manifest = Build.run_build(
        [(2025, 6, empty_frame)], resolution, circuit_polarity,
        tmp_path / "ami_raw", tmp_path / "ami_meter",
    )
    assert len(manifest) == 1
    assert manifest.iloc[0].n_raw_rows == 0
    assert manifest.iloc[0].raw_path is None
