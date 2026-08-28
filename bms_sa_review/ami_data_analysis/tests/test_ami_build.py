"""
Unit tests for `ami_build` -- deriving `ami_raw` (site-level ground truth),
`ami_meter` (per-phase synthetic smart meter), and `ami_raw_phaseseparate`
(per-phase ground truth, PV-allocated) from a resolved interval table, and
the month-at-a-time write/orchestration around all three.
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

def test_build_ami_raw_combines_pv_and_reconstructed_load_in_kw():
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
    assert row.year == 2025 and row.month == 6
    # gross_load_w = (-300*1) + (1000*-1) = -1300 -> P_kw = -1.3
    assert row.P_kw == pytest.approx(-1.3)
    # P_kw = P_load_kw + P_pv_kw by construction:
    # load_signed = -300*1 = -300 -> P_load_kw = -0.3
    # pv_signed = 1000*-1 = -1000 -> P_pv_kw = -1.0
    assert row.P_load_kw == pytest.approx(-0.3)
    assert row.P_pv_kw == pytest.approx(-1.0)
    # no reactive_power in this fixture -> Q fields null, not an error
    assert pd.isna(row.Q_kvar)
    assert pd.isna(row.Q_load_kvar)
    assert pd.isna(row.Q_pv_kvar)
    # no site_capacity supplied -> normalized fields present but null
    assert pd.isna(row.P_kw_norm)
    assert pd.isna(row.Q_kvar_norm)
    assert pd.isna(row.S_99)


def test_build_ami_raw_reactive_power_reconstruction_mirrors_active():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([
        {"site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": -300.0, "reactive_power": 50.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 11, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 1000.0, "reactive_power": 20.0},
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    site_capacity = pd.DataFrame({"site_id": [1], "S_99": [10.0], "ac_capacity_kw": [8.0]})

    result = Build.build_ami_raw(interval_table, circuit_polarity, site_capacity=site_capacity)
    row = result.iloc[0]
    # gross_reactive_load_var = (50*1) + (20*-1) = 30 -> Q_kvar = 0.03
    assert row.Q_kvar == pytest.approx(0.03)
    # Q_load_kvar = 50*1 / 1000 = 0.05 ; Q_pv_kvar = 20*-1 / 1000 = -0.02
    assert row.Q_load_kvar == pytest.approx(0.05)
    assert row.Q_pv_kvar == pytest.approx(-0.02)
    # pv_reactive_generation_var = 20*-1 = -20 -> kvar = -0.02, / S_99(10kW) = -0.002
    assert row.Q_kvar_norm == pytest.approx(-0.002)


def test_build_ami_raw_normalizes_pv_by_site_capacity():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([
        (1, 100, 10, "ac_load_net", t, 200.0),
        (1, 100, 11, "pv_site_net", t, 5000.0),  # 5 kW instantaneous
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    site_capacity = pd.DataFrame({"site_id": [1], "S_99": [10.0], "ac_capacity_kw": [8.0]})

    result = Build.build_ami_raw(interval_table, circuit_polarity, site_capacity=site_capacity)
    row = result.iloc[0]
    assert row.S_99 == 10.0
    assert row.ac_capacity_kw == 8.0
    assert row.normalization_basis == "s_99"
    # P_pv_kw is circuit_polarity-corrected: raw power=5000W * polarity(-1) =
    # -5000W = -5kW -> P_kw_norm = -5kW / 10kW S_99 = -0.5.
    assert row.P_pv_kw == pytest.approx(-5.0)
    assert row.P_kw_norm == pytest.approx(-0.5)


def test_build_ami_raw_normalization_basis_ac_capacity_kw():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([
        (1, 100, 10, "ac_load_net", t, 200.0),
        (1, 100, 11, "pv_site_net", t, 4000.0),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    site_capacity = pd.DataFrame({"site_id": [1], "S_99": [10.0], "ac_capacity_kw": [5.0]})

    result = Build.build_ami_raw(
        interval_table, circuit_polarity, site_capacity=site_capacity,
        normalization_basis="ac_capacity_kw",
    )
    row = result.iloc[0]
    assert row.normalization_basis == "ac_capacity_kw"
    # raw power=4000W * polarity(-1) = -4000W = -4kW -> P_kw_norm = -4kW / 5kW
    assert row.P_pv_kw == pytest.approx(-4.0)
    assert row.P_kw_norm == pytest.approx(-0.8)


def test_build_ami_raw_zero_capacity_gives_null_norm_not_inf():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([
        (1, 100, 10, "ac_load_net", t, 200.0),
        (1, 100, 11, "pv_site_net", t, 4000.0),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    site_capacity = pd.DataFrame({"site_id": [1], "S_99": [0.0], "ac_capacity_kw": [0.0]})

    result = Build.build_ami_raw(interval_table, circuit_polarity, site_capacity=site_capacity)
    assert pd.isna(result.iloc[0].P_kw_norm)
    assert not np.isinf(result.iloc[0].P_kw_norm) if pd.notna(result.iloc[0].P_kw_norm) else True


def test_build_ami_raw_v_is_avg_voltage_of_load_circuits():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([
        {"site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 200.0, "voltage": 240.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 20, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 100.0, "voltage": 242.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 11, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 500.0, "voltage": 245.0},
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 20, 11], "circuit_polarity": [1, 1, -1]})

    result = Build.build_ami_raw(interval_table, circuit_polarity)
    # avg of the two LOAD circuits' voltage only (240, 242) -- PV's 245 excluded
    assert result.iloc[0].V_load == pytest.approx(241.0)
    # V_pv is the analogous average across PV circuits -- just the one here
    assert result.iloc[0].V_pv == pytest.approx(245.0)


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
    assert list(result.columns) == [
        "site_id", "t_stamp", "year", "month",
        "V_load", "V_pv",
        "P_load_kw", "P_pv_kw", "P_kw_norm", "P_kw",
        "Q_load_kvar", "Q_pv_kvar", "Q_kvar_norm", "Q_kvar",
        "S_99", "ac_capacity_kw", "normalization_basis",
    ]


# --------------------------------------------------------------------------- #
# build_ami_meter
# --------------------------------------------------------------------------- #

def test_build_ami_meter_keeps_phases_separate_in_kw():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([
        (1, 100, 10, "ac_load_net", t, 500.0),   # phase A: importing
        (1, 100, 20, "ac_load_net", t, -200.0),  # phase B: exporting
        (1, 100, 30, "pv_site_net", t, 900.0),   # not a meter reading -- excluded
    ])
    result = Build.build_ami_meter(interval_table)
    assert set(result.circuit_id) == {10, 20}

    phase_a = result.set_index("circuit_id").loc[10]
    assert phase_a.P_kw == pytest.approx(0.5)
    phase_b = result.set_index("circuit_id").loc[20]
    assert phase_b.P_kw == pytest.approx(-0.2)
    assert phase_a.year == 2025 and phase_a.month == 6


def test_build_ami_meter_carries_all_metrology_fields_through_in_correct_units():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([{
        "site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
        "t_stamp": t, "power": 3000.0, "reactive_power": 400.0, "voltage": 241.5,
        "current": 12.5, "power_factor": 0.98,
        "energy_import": 250.0, "energy_export": 50.0,
    }])
    result = Build.build_ami_meter(interval_table)
    row = result.iloc[0]
    assert row.P_kw == pytest.approx(3.0)
    assert row.Q_kvar == pytest.approx(0.4)
    assert row.S_kva == pytest.approx((3.0 ** 2 + 0.4 ** 2) ** 0.5)
    assert row.V == 241.5
    assert row.current_a == 12.5
    assert row.power_factor == 0.98
    assert row.energy_import_kwh == pytest.approx(0.25)
    assert row.energy_export_kwh == pytest.approx(0.05)


def test_build_ami_meter_missing_source_columns_omits_output_columns_not_errors():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([(1, 100, 10, "ac_load_net", t, 500.0)])
    result = Build.build_ami_meter(interval_table)
    assert "P_kw" in result.columns
    for col in ("Q_kvar", "S_kva", "V", "current_a", "power_factor",
                "energy_import_kwh", "energy_export_kwh"):
        assert col not in result.columns


def test_build_ami_meter_no_load_circuits_returns_empty_with_columns():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([(1, 100, 30, "pv_site_net", t, 900.0)])
    result = Build.build_ami_meter(interval_table)
    assert len(result) == 0
    assert "P_kw" in result.columns


def test_build_ami_meter_empty_interval_table():
    result = Build.build_ami_meter(pd.DataFrame())
    assert len(result) == 0


# --------------------------------------------------------------------------- #
# build_ami_raw_phaseseparate
# --------------------------------------------------------------------------- #

def test_phaseseparate_direct_matches_when_pv_count_equals_phase_count():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([
        {"site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 500.0, "reactive_power": 40.0, "voltage": 241.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 20, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 300.0, "reactive_power": 20.0, "voltage": 242.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 11, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 900.0, "reactive_power": 10.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 21, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 700.0, "reactive_power": 5.0},
    ])
    circuit_polarity = pd.DataFrame({
        "circuit_id": [10, 20, 11, 21], "circuit_polarity": [1, 1, -1, -1],
    })
    result = Build.build_ami_raw_phaseseparate(interval_table, circuit_polarity)
    assert (result.pv_allocation_method == "direct_matched_circuit").all()
    assert set(result.n_phases_at_site) == {2}

    row10 = result.set_index("circuit_id").loc[10]   # paired with lower-sorted PV circuit 11
    assert row10.load_kw_signed == pytest.approx(0.5)
    assert row10.pv_allocation_kw == pytest.approx(-0.9)
    assert row10.gross_load_kw == pytest.approx(-0.4)
    assert row10.V == 241.0
    assert row10.Q_kvar_signed == pytest.approx(0.04)
    assert row10.pv_reactive_allocation_kvar == pytest.approx(-0.01)
    assert row10.gross_reactive_load_kvar == pytest.approx(0.03)

    row20 = result.set_index("circuit_id").loc[20]   # paired with PV circuit 21
    assert row20.pv_allocation_kw == pytest.approx(-0.7)
    assert row20.pv_reactive_allocation_kvar == pytest.approx(-0.005)


def test_phaseseparate_equal_splits_when_counts_dont_match():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([
        {"site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 500.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 20, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 300.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 30, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 100.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 11, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 3000.0},
    ])
    circuit_polarity = pd.DataFrame({
        "circuit_id": [10, 20, 30, 11], "circuit_polarity": [1, 1, 1, -1],
    })
    result = Build.build_ami_raw_phaseseparate(interval_table, circuit_polarity)
    assert (result.pv_allocation_method == "equal_split_across_load_phases").all()
    assert set(result.n_phases_at_site) == {3}
    # signed pv total = 3000*-1 = -3000W = -3kW, split 3 ways = -1kW each
    assert result.pv_allocation_kw.apply(lambda v: v == pytest.approx(-1.0)).all()

    row10 = result.set_index("circuit_id").loc[10]
    assert row10.gross_load_kw == pytest.approx(0.5 - 1.0)


def test_phaseseparate_no_pv_present_gives_zero_allocation():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([(1, 100, 10, "ac_load_net", t, 500.0)])
    circuit_polarity = pd.DataFrame({"circuit_id": [10], "circuit_polarity": [1]})
    result = Build.build_ami_raw_phaseseparate(interval_table, circuit_polarity)
    row = result.iloc[0]
    assert row.pv_allocation_method == "no_pv_present"
    assert row.pv_allocation_kw == 0.0
    assert row.gross_load_kw == pytest.approx(0.5)


def test_phaseseparate_gross_load_reconciles_with_ami_raw_p_kw():
    # sum(gross_load_kw)/sum(gross_reactive_load_kvar) across a site's
    # phases should equal ami_raw.P_kw/Q_kvar for that site/timestamp.
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([
        {"site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 500.0, "reactive_power": 30.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 20, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": -300.0, "reactive_power": 15.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 30, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 100.0, "reactive_power": 5.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 11, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 1500.0, "reactive_power": 25.0},
    ])
    circuit_polarity = pd.DataFrame({
        "circuit_id": [10, 20, 30, 11], "circuit_polarity": [1, 1, 1, -1],
    })
    phase_split = Build.build_ami_raw_phaseseparate(interval_table, circuit_polarity)
    ami_raw = Build.build_ami_raw(interval_table, circuit_polarity)

    assert phase_split.gross_load_kw.sum() == pytest.approx(ami_raw.iloc[0].P_kw)
    assert phase_split.gross_reactive_load_kvar.sum() == pytest.approx(ami_raw.iloc[0].Q_kvar)


def test_phaseseparate_empty_interval_table():
    result = Build.build_ami_raw_phaseseparate(pd.DataFrame(), pd.DataFrame())
    assert len(result) == 0
    assert "pv_allocation_method" in result.columns


# --------------------------------------------------------------------------- #
# write_month_table
# --------------------------------------------------------------------------- #

def test_write_month_table_writes_hive_partitioned_parquet(tmp_path):
    table_dir = tmp_path / "ami_raw"
    frame = pd.DataFrame({"site_id": [1], "P_kw": [0.8]})
    path = Build.write_month_table(frame, table_dir, 2025, 6)
    assert path is not None
    assert "dt_month=2025-06" in str(path)
    assert str(path).count("ami_raw") == 1
    assert len(pd.read_parquet(path)) == 1


def test_write_month_table_empty_frame_writes_nothing(tmp_path):
    empty = pd.DataFrame(columns=["site_id", "P_kw"])
    path = Build.write_month_table(empty, tmp_path / "ami_raw", 2025, 6)
    assert path is None
    assert not list(tmp_path.rglob("*.parquet"))


def test_write_month_table_rerun_overwrites(tmp_path):
    table_dir = tmp_path / "ami_meter"
    frame1 = pd.DataFrame({"site_id": [1]})
    frame2 = pd.DataFrame({"site_id": [1, 2]})
    Build.write_month_table(frame1, table_dir, 2025, 6)
    path2 = Build.write_month_table(frame2, table_dir, 2025, 6)
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1
    assert len(pd.read_parquet(path2)) == 2


# --------------------------------------------------------------------------- #
# run_build
# --------------------------------------------------------------------------- #

def _full_month_frame(month, power_load=200.0, power_pv=500.0):
    t = pd.Timestamp(f"2025-{month:02d}-01 12:00", tz="UTC")
    return pd.DataFrame([
        {"circuit_id": 10, "t_stamp": t, "power": power_load,
         "energy": power_load / 12.0, "energy_reactive": 0.0, "voltage": 240.0,
         "current": 1.0, "power_factor": 0.99, "energy_import": power_load / 12.0,
         "energy_export": 0.0},
        {"circuit_id": 11, "t_stamp": t, "power": power_pv,
         "energy": power_pv / 12.0, "energy_reactive": 0.0, "voltage": 245.0,
         "current": 2.0, "power_factor": 1.0, "energy_import": 0.0,
         "energy_export": power_pv / 12.0},
    ])


def test_run_build_writes_both_tables_for_every_month(tmp_path):
    resolution = _resolution([
        (1, 10, "ac_load_net", 100, True),
        (1, 11, "pv_site_net", 100, True),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    site_capacity = pd.DataFrame({"site_id": [1], "S_99": [5.0], "ac_capacity_kw": [5.0]})

    month_frames = [(2025, 6, _full_month_frame(6)), (2025, 7, _full_month_frame(7))]
    raw_dir, meter_dir = tmp_path / "ami_raw", tmp_path / "ami_meter"

    manifest = Build.run_build(
        month_frames, resolution, circuit_polarity, raw_dir, meter_dir,
        site_capacity=site_capacity,
    )
    assert len(manifest) == 2
    assert (manifest.n_raw_rows > 0).all()
    assert (manifest.n_meter_rows > 0).all()
    for path in manifest.raw_path:
        assert path.exists()
        assert str(path).count("ami_raw") == 1
        assert pd.read_parquet(path).P_kw_norm.notna().all()
    for path in manifest.meter_path:
        assert path.exists()
        assert str(path).count("ami_meter") == 1


def test_run_build_without_site_capacity_still_works_but_norm_is_null(tmp_path):
    resolution = _resolution([
        (1, 10, "ac_load_net", 100, True),
        (1, 11, "pv_site_net", 100, True),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    month_frames = [(2025, 6, _full_month_frame(6))]
    raw_dir, meter_dir = tmp_path / "ami_raw", tmp_path / "ami_meter"

    manifest = Build.run_build(month_frames, resolution, circuit_polarity, raw_dir, meter_dir)
    raw = pd.read_parquet(manifest.iloc[0].raw_path)
    assert raw.P_kw_norm.isna().all()


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


# --------------------------------------------------------------------------- #
# run_phase_split_build
# --------------------------------------------------------------------------- #

def test_run_phase_split_build_writes_a_table_per_month(tmp_path):
    resolution = _resolution([
        (1, 10, "ac_load_net", 100, True),
        (1, 11, "pv_site_net", 100, True),
    ])
    circuit_polarity = pd.DataFrame({"circuit_id": [10, 11], "circuit_polarity": [1, -1]})
    month_frames = [(2025, 6, _full_month_frame(6)), (2025, 7, _full_month_frame(7))]
    out_dir = tmp_path / "ami_raw_phaseseparate"

    manifest = Build.run_phase_split_build(month_frames, resolution, circuit_polarity, out_dir)
    assert len(manifest) == 2
    assert (manifest.n_rows > 0).all()
    for path in manifest.path:
        assert path.exists()
        assert str(path).count("ami_raw_phaseseparate") == 1


def test_run_phase_split_build_month_with_no_rows_still_gets_a_manifest_row(tmp_path):
    resolution = _resolution([(1, 10, "ac_load_net", 100, True)])
    circuit_polarity = pd.DataFrame({"circuit_id": [10], "circuit_polarity": [1]})
    empty_frame = pd.DataFrame(columns=["circuit_id", "t_stamp", "power", "energy", "energy_reactive", "voltage"])
    manifest = Build.run_phase_split_build(
        [(2025, 6, empty_frame)], resolution, circuit_polarity, tmp_path / "ami_raw_phaseseparate",
    )
    assert len(manifest) == 1
    assert manifest.iloc[0].n_rows == 0
    assert manifest.iloc[0].path is None
