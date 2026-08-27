"""
Synthetic dry-run harness for notebook 4 (`04_site_resolution.ipynb`).

Executes every code cell of the real, live notebook file in-process, against
a monkeypatched `ami_athena.aq` backed by a small synthetic fixture covering
every resolution outcome the notebook needs to demonstrate: a clean
single-phase site, a clean 3-phase site (device_id grouping), a cross-type
duplicate, a same-type duplicate group, an inactive circuit, and a device
-model (`CATCH Power`) power-correction case -- plus one `OTHER_COUNT`
-style site that must NOT appear in the validation batch at all, since
Phase 4 defers that cohort.

This is the "exercise the whole notebook end-to-end before calling a change
done" step the repo's conventions call for -- distinct from
`test_ami_resolution.py`'s unit tests (which check the `lib` functions in
isolation) and `test_notebook_names.py`'s static check (which only proves
every name is bound somewhere, never that the notebook actually runs).
No real AWS credentials or network access are used or required.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from bms_sa_review.ami_data_analysis.config import ami_config as Config
from bms_sa_review.ami_data_analysis.lib import ami_athena as AthenaModule

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "04_site_resolution.ipynb"
)


# ── synthetic fixture -------------------------------------------------------

def _wobble(circuit_id, n):
    rng = np.random.RandomState(int(circuit_id) % (2**31 - 1))
    return 1.0 + 0.02 * rng.normal(size=n)


def _ts_rows(circuit_id, power, n=48, energy=None, vary=True, energy_reactive=20.0):
    t_stamp = pd.date_range("2025-06-01", periods=n, freq="5min")
    wobble = _wobble(circuit_id, n) if vary else np.ones(n)
    power_series = power * wobble
    energy_series = (power_series / 12.0) if energy is None else (energy * wobble)
    return pd.DataFrame({
        "circuit_id": circuit_id, "t_stamp": t_stamp,
        "power": power_series, "energy": energy_series,
        "energy_reactive": energy_reactive,
        "energy_import": np.clip(energy_series, 0, None),
        "energy_export": np.clip(-energy_series, 0, None),
        "energy_reactive_import": energy_reactive, "energy_reactive_export": 0.0,
        "power_factor": 0.98, "voltage": 240.0, "current": 5.0,
    })


#: (circuit_id, site_id, circuit_type, device_id, is_pv, device_type)
_META_ROWS = [
    # Site 1001: clean single-phase, no issues at all.
    (11, 1001, "ac_load_net", 9001, False, "Watt Watcher"),
    (12, 1001, "pv_site_net", 9002, True, "Watt Watcher"),
    # Site 1002: clean 3-phase (one device_id, three ac_load_net circuits).
    (21, 1002, "ac_load_net", 9101, False, "Watt Watcher"),
    (22, 1002, "ac_load_net", 9101, False, "Watt Watcher"),
    (23, 1002, "ac_load_net", 9101, False, "Watt Watcher"),
    (24, 1002, "pv_site_net", 9102, True, "Watt Watcher"),
    # Site 1003: cross-type duplicate -- 31 (load) mirrors 32 (PV) exactly.
    (31, 1003, "ac_load_net", 9201, False, "Watt Watcher"),
    (32, 1003, "pv_site_net", 9202, True, "Watt Watcher"),
    # Site 1004: same-type duplicate -- two independent PV registrations.
    (41, 1004, "ac_load_net", 9301, False, "Watt Watcher"),
    (42, 1004, "pv_site_net", 9302, True, "Watt Watcher"),
    (43, 1004, "pv_site_net", 9303, True, "Watt Watcher"),
    # Site 1005: an inactive (dead) PV circuit alongside a live load circuit.
    (51, 1005, "ac_load_net", 9401, False, "Watt Watcher"),
    (52, 1005, "pv_site_net", 9402, True, "Watt Watcher"),
    # Site 1006: CATCH Power device/meter-model correction case.
    (61, 1006, "ac_load_net", 9501, False, "CATCH Power"),
    (62, 1006, "pv_site_net", 9502, True, "Watt Watcher"),
    # Site 1007: OTHER_COUNT (2 ac_load_net circuits, 2 distinct devices) --
    # must be EXCLUDED from CLEAN_SITE_IDS, and so absent from the batch.
    (71, 1007, "ac_load_net", 9601, False, "Watt Watcher"),
    (72, 1007, "ac_load_net", 9602, False, "Watt Watcher"),
    (73, 1007, "pv_site_net", 9603, True, "Watt Watcher"),
    # Site 1008: otherwise-clean site with an EXPLICIT, separately-metered
    # battery circuit_type -- caught by name (sites_with_storage_circuits),
    # not by the reconstruction check (its load/PV behave normally).
    (81, 1008, "ac_load_net", 9701, False, "Watt Watcher"),
    (82, 1008, "pv_site_net", 9702, True, "Watt Watcher"),
    (83, 1008, "load_battery", 9703, False, "Watt Watcher"),
    # Site 1009: a HIDDEN anomaly -- ac_load_net reads meaningfully negative
    # even at night (no PV to explain it away, no separate storage
    # circuit_type either). Only the reconstruction check should catch this.
    (91, 1009, "ac_load_net", 9801, False, "Watt Watcher"),
    (92, 1009, "pv_site_net", 9802, True, "Watt Watcher"),
]


def _synthetic_meta() -> pd.DataFrame:
    frame = pd.DataFrame(_META_ROWS, columns=[
        "circuit_id", "site_id", "circuit_type", "device_id", "is_pv", "device_type",
    ])
    frame["circuit_polarity"] = np.where(frame.is_pv, -1, 1)
    for col in ("ac_capacity_kw", "dc_capacity_kw", "export_limit_kw", "inverter_count",
                "m_id", "voltage_class", "min_time", "max_time", "s_99", "postcode",
                "dnsp_name", "flex_export_detected", "manufacturer", "model",
                "monitoring_start", "pv_install_date"):
        frame[col] = None
    frame["state"] = "NSW"
    return frame


def _synthetic_ts() -> pd.DataFrame:
    parts = [
        _ts_rows(11, power=1500.0),
        _ts_rows(12, power=-900.0),
        _ts_rows(21, power=500.0), _ts_rows(22, power=520.0), _ts_rows(23, power=480.0),
        _ts_rows(24, power=-1800.0),
    ]
    # Site 1003: 31 is a raw duplicate of 32 (mistagged, opposite polarity
    # convention aside -- find_duplicate_circuits correlates RAW power).
    pv_1003 = _ts_rows(32, power=-700.0)
    dup_1003 = pv_1003.copy()
    dup_1003["circuit_id"] = 31
    parts += [dup_1003, pv_1003]

    # Site 1004: 42 and 43 are two independent registrations of the same
    # physical PV reading; 41 is an unrelated, genuinely separate load.
    base_1004 = _ts_rows(42, power=-1100.0)
    twin_1004 = base_1004.copy()
    twin_1004["circuit_id"] = 43
    parts += [_ts_rows(41, power=600.0), base_1004, twin_1004]

    # Site 1005: 52 is dead (near-zero all day).
    parts += [_ts_rows(51, power=1300.0), _ts_rows(52, power=1.0, energy=1.0 / 12.0)]

    # Site 1006: 61 is a CATCH Power circuit with a whole-Wh energy register
    # and an implied ~4.9-minute accumulation window, not the nominal 5.
    parts += [
        _ts_rows(61, power=600.0, energy=49.0, vary=False),
        _ts_rows(62, power=-2000.0),
    ]

    # Site 1007 (OTHER_COUNT): included in the ts fixture for completeness,
    # even though the notebook's CLEAN_SITE_IDS filter should never request it.
    parts += [_ts_rows(71, power=400.0), _ts_rows(72, power=410.0), _ts_rows(73, power=-1200.0)]

    # Site 1008: explicit-storage site, otherwise well-behaved -- no
    # reconstruction anomaly (this is the "caught by name only" case).
    parts += [
        _rows_at_aest_hour(81, "2025-06-01", 2, power=500.0),   # night: load only
        _rows_at_aest_hour(81, "2025-06-01", 12, power=500.0),  # day: load only
        _rows_at_aest_hour(82, "2025-06-01", 2, power=0.0),     # night: no PV
        _rows_at_aest_hour(82, "2025-06-01", 12, power=-1000.0),  # day: generating
    ]

    # Site 1009: hidden anomaly -- ac_load_net negative even at night, no PV
    # to explain it, no named storage circuit either (the "caught only by
    # reconstruction" case). PV reads a small but NON-zero trickle (2-10 W)
    # rather than exactly 0 -- an exactly-zero circuit would itself get
    # dropped as "inactive" before ever reaching the reconstruction check.
    parts += [
        _rows_at_aest_hour(91, "2025-06-01", 2, power=-400.0),
        _rows_at_aest_hour(91, "2025-06-01", 12, power=-400.0),
        _rows_at_aest_hour(92, "2025-06-01", 2, power=2.0),
        _rows_at_aest_hour(92, "2025-06-01", 12, power=10.0),
    ]

    return pd.concat(parts, ignore_index=True)


def _rows_at_aest_hour(circuit_id, base_date, aest_hour, power, n=6, step_minutes=5):
    """Fixed-power ts rows landing at a specific AEST hour once `Plots.to_aest`
    converts them (t_stamp is naive/UTC on the wire, matching real `ts`)."""
    utc_start = pd.Timestamp(base_date) + pd.Timedelta(hours=aest_hour - 10)
    t_stamp = pd.date_range(utc_start, periods=n, freq=f"{step_minutes}min")
    return pd.DataFrame({
        "circuit_id": circuit_id, "t_stamp": t_stamp, "power": power,
        "energy": power / 12.0, "energy_reactive": 10.0,
        "energy_import": max(power, 0.0), "energy_export": max(-power, 0.0),
        "energy_reactive_import": 10.0, "energy_reactive_export": 0.0,
        "power_factor": 0.98, "voltage": 240.0, "current": 5.0,
    })


_CIRCUIT_ID_LIST_RE = re.compile(r"circuit_id IN \(([^)]*)\)")
_IS_PV_RE = re.compile(r"is_pv\s*=\s*(true|false)", re.IGNORECASE)


def _fake_aq(meta: pd.DataFrame, ts: pd.DataFrame):
    def aq(sql: str, database=None, *, label=None, allow_full_scan=False, meter=True):
        if "GROUP BY circuit_type, is_pv" in sql:
            return (
                meta.groupby(["circuit_type", "is_pv"])
                .agg(n_circuits=("circuit_id", "count"), n_sites=("site_id", "nunique"))
                .reset_index()
            )
        if "FROM meta_up23c" in sql:
            return meta.copy()
        if "FROM ts" in sql:
            id_match = _CIRCUIT_ID_LIST_RE.search(sql)
            pv_match = _IS_PV_RE.search(sql)
            assert id_match and pv_match, f"unrecognised ts query shape: {sql[:200]}"
            ids = {int(x) for x in id_match.group(1).split(",")}
            is_pv_value = pv_match.group(1).lower() == "true"
            allowed = set(meta.loc[meta.is_pv == is_pv_value, "circuit_id"])
            ids &= allowed
            return ts[ts.circuit_id.isin(ids)].reset_index(drop=True)
        raise AssertionError(f"unrecognised query in dry run: {sql[:200]}")

    return aq


# ── harness -------------------------------------------------------------

def _run_notebook_cells(path: Path, namespace: dict) -> dict:
    nb = nbformat.read(path, as_version=4)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        lines = [ln for ln in cell.source.splitlines() if not ln.strip().startswith("%")]
        source = "\n".join(lines)
        if not source.strip():
            continue
        try:
            exec(compile(source, f"<cell {i}>", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - re-raised with cell context
            raise RuntimeError(f"notebook cell {i} raised {exc!r}:\n{source}") from exc
    return namespace


def test_notebook_04_runs_end_to_end_against_synthetic_fixture(monkeypatch, tmp_path):
    meta = _synthetic_meta()
    ts = _synthetic_ts()

    monkeypatch.setattr(AthenaModule, "aq", _fake_aq(meta, ts))
    monkeypatch.setattr(AthenaModule, "require_credentials", lambda: {"ok": True})
    monkeypatch.setattr(AthenaModule, "reset_scan_log", lambda: None)
    monkeypatch.setattr(Config, "ARTEFACT_DIR", tmp_path)

    namespace = {"__name__": "__dry_run__", "display": lambda *a, **k: None}
    result = _run_notebook_cells(NOTEBOOK_PATH, namespace)

    # --- cohort: OTHER_COUNT site excluded, everything else included ------
    assert 1007 not in result["CLEAN_SITE_IDS"]
    assert 1007 in result["OTHER_COUNT_SITE_IDS"]
    for site_id in (1001, 1002, 1003, 1004, 1005, 1006, 1008, 1009):
        assert site_id in result["CLEAN_SITE_IDS"]

    resolution = result["resolution"]
    by_id = resolution.set_index("circuit_id")

    # --- clean sites: no intervention -------------------------------------
    for cid in (11, 12, 21, 22, 23, 24):
        assert by_id.loc[cid, "kept"], f"circuit {cid} should not have been dropped"

    # --- cross-type duplicate: load side (31) dropped, PV side (32) kept --
    assert by_id.loc[31, "kept"] == False
    assert by_id.loc[31, "drop_reason"] == "duplicate_cross_type"
    assert by_id.loc[32, "kept"] == True

    # --- same-type duplicate: exactly one of 42/43 survives, both flagged -
    assert by_id.loc[41, "kept"] == True  # unrelated load circuit, untouched
    assert int(by_id.loc[[42, 43], "kept"].sum()) == 1
    assert by_id.loc[42, "needs_manual_review"] or by_id.loc[43, "needs_manual_review"]

    # --- inactive circuit dropped -------------------------------------------
    assert by_id.loc[52, "kept"] == False
    assert by_id.loc[52, "drop_reason"] == "inactive"
    assert by_id.loc[51, "kept"] == True

    # --- CATCH Power device/meter-model correction --------------------------
    assert by_id.loc[61, "kept"] == True
    assert by_id.loc[61, "power_correction_applied"] == True

    # --- OTHER_COUNT site never entered the batch at all ---------------------
    assert 1007 not in set(resolution.site_id)

    # --- Section 7: storage detection + load reconstruction sanity check ---
    storage_site_ids = result["storage_site_ids"]
    assert storage_site_ids == [1008]   # only the EXPLICIT battery circuit_type

    reconstruction_report = result["reconstruction_report"].set_index("site_id")
    # Site 1008: named storage detected, but load+PV reconstruction itself
    # looks fine (no night-time negative violation).
    assert reconstruction_report.loc[1008, "likely_storage_or_sign_issue"] == False
    # Site 1009: no named storage circuit, but the reconstruction is
    # negative even at night -- the hidden-anomaly case.
    assert reconstruction_report.loc[1009, "likely_storage_or_sign_issue"] == True
    assert reconstruction_report.loc[1009, "share_negative_night"] == pytest.approx(1.0)
    # A well-behaved site should show neither symptom.
    assert reconstruction_report.loc[1001, "likely_storage_or_sign_issue"] == False

    only_reconstruction_flagged = result["only_reconstruction_flagged"]
    assert only_reconstruction_flagged == {1009}

    # --- 7b: both flagged sites (1008 by name, 1009 by reconstruction) ------
    # get dropped entirely -- `interval_table`/`site_metadata`/
    # `coverage_report` are REBOUND by the exclusion cell to the final,
    # post-exclusion versions, since that's what the notebook actually saves.
    excluded_site_ids = result["excluded_site_ids"]
    assert excluded_site_ids == {1008, 1009}

    final_resolution = result["final_resolution"]
    final_by_id = final_resolution.set_index("circuit_id")
    for cid in (81, 82, 91, 92):
        assert final_by_id.loc[cid, "kept"] == False
        assert final_by_id.loc[cid, "drop_reason"] == "storage_or_sign_issue"
    # a circuit dropped for another reason before exclusion keeps ITS reason
    assert final_by_id.loc[52, "drop_reason"] == "inactive"

    # --- output tables are non-empty and internally consistent --------------
    interval_table = result["interval_table"]
    assert len(interval_table) > 0
    assert set(interval_table.circuit_id) == set(final_by_id.index[final_by_id.kept])
    assert interval_table.reactive_power.notna().all()
    assert 1008 not in set(interval_table.site_id)
    assert 1009 not in set(interval_table.site_id)

    site_metadata = result["site_metadata"]
    # All 8 validation sites still appear -- build_site_metadata's inner join
    # keeps every site site_level_meta lists, regardless of exclusion, so the
    # audit trail documents WHY a site was excluded rather than silently
    # vanishing it. Exclusion shows up as an empty kept_circuit_ids list.
    assert set(site_metadata.site_id) == {1001, 1002, 1003, 1004, 1005, 1006, 1008, 1009}
    site_meta_by_id = site_metadata.set_index("site_id")
    for sid in (1008, 1009):
        assert site_meta_by_id.loc[sid, "kept_circuit_ids"] == []
    for sid in (1001, 1002, 1003, 1004, 1005, 1006):
        assert len(site_meta_by_id.loc[sid, "kept_circuit_ids"]) > 0

    coverage_report = result["coverage_report"]
    assert coverage_report["n_sites"] == 8
    assert coverage_report["n_excluded_storage_or_sign_issue"] == 2   # 1008, 1009
    # "no intervention" means no circuit was DROPPED at that site -- site
    # 1006's only special treatment is the power correction (nothing there
    # was dropped), so it counts alongside 1001/1002, not against them.
    assert coverage_report["n_no_intervention"] == 3   # 1001, 1002, 1006
    assert coverage_report["n_auto_resolved_duplicate_cross_type"] == 1  # site 1003
    assert coverage_report["n_auto_resolved_inactive"] == 1              # site 1005
    assert coverage_report["n_flagged_manual_review"] == 1               # site 1004
    assert coverage_report["n_power_correction_applied"] == 1            # site 1006

    # --- artefacts were written to the (monkeypatched, tmp) artefact dir ----
    assert (tmp_path / "phase4_interval_table_validation_batch.csv").exists()
    assert (tmp_path / "phase4_site_metadata_validation_batch.csv").exists()
    assert (tmp_path / "phase4_coverage_report_validation_batch.csv").exists()
