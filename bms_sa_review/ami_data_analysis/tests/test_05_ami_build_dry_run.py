"""
Synthetic dry-run harness for notebook 5 (`05_ami_build.ipynb`).

Executes every code cell of the real, live notebook file in-process against
a monkeypatched `ami_athena.aq`, covering all four steps end to end:
fleet-wide day-1 resolution (including a manual-review site and a
day-1-visible reconstruction failure, both dropped before extraction ever
happens), chunked full-year extraction to a real local Parquet store (a
pytest `tmp_path`, standing in for `ami_config.STORE_DIR`), full-year
revalidation (a circuit that goes inactive mid-year; a site whose
reconstruction only fails starting in a later month), and the final
`ami_raw`/`ami_meter` build.

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
    Path(__file__).resolve().parents[1] / "notebooks" / "05_ami_build.ipynb"
)

# ── synthetic fixture -------------------------------------------------------

#: (circuit_id, site_id, circuit_type, device_id, is_pv, device_type)
_META_ROWS = [
    # Site 2001: clean single-phase, well-behaved every month all year.
    (201, 2001, "ac_load_net", 5001, False, "Watt Watcher"),
    (202, 2001, "pv_site_net", 5002, True, "Watt Watcher"),
    # Site 2002: same-type duplicate on day 1 -- flagged needs_manual_review,
    # dropped ENTIRELY at the day-1 conservative-exclusion step. Must never
    # reach extraction.
    (211, 2002, "ac_load_net", 5101, False, "Watt Watcher"),
    (212, 2002, "pv_site_net", 5102, True, "Watt Watcher"),
    (213, 2002, "pv_site_net", 5103, True, "Watt Watcher"),
    # Site 2003: reconstruction fails on day 1 itself -- excluded at the
    # Section 1d/1e stage, must never reach extraction.
    (221, 2003, "ac_load_net", 5201, False, "Watt Watcher"),
    (222, 2003, "pv_site_net", 5202, True, "Watt Watcher"),
    # Site 2004: clean on day 1, but its load circuit goes inactive from
    # month 9 onward -- full-year revalidation should drop that ONE
    # circuit, not the whole site.
    (231, 2004, "ac_load_net", 5301, False, "Watt Watcher"),
    (232, 2004, "pv_site_net", 5302, True, "Watt Watcher"),
    # Site 2005: clean on day 1, but its load circuit starts reading
    # meaningfully negative at night from month 11 onward (a battery
    # installed mid-year) -- full-year revalidation should exclude the
    # WHOLE site, not just that circuit.
    (241, 2005, "ac_load_net", 5401, False, "Watt Watcher"),
    (242, 2005, "pv_site_net", 5402, True, "Watt Watcher"),
]


def _synthetic_meta() -> pd.DataFrame:
    frame = pd.DataFrame(_META_ROWS, columns=[
        "circuit_id", "site_id", "circuit_type", "device_id", "is_pv", "device_type",
    ])
    frame["circuit_polarity"] = np.where(frame.is_pv, -1, 1)
    for col in ("dc_capacity_kw", "export_limit_kw", "inverter_count",
                "m_id", "voltage_class", "min_time", "max_time", "postcode",
                "dnsp_name", "flex_export_detected", "manufacturer", "model",
                "monitoring_start", "pv_install_date"):
        frame[col] = None
    # Real, non-null capacity figures -- exercised by the site_capacity_lookup
    # query shape (`GROUP BY site_id`) in _fake_aq below, so P_kw_norm/
    # Q_kvar_norm in the real ami_raw build actually get a denominator
    # instead of staying null throughout this dry run.
    frame["s_99"] = 5.0
    frame["ac_capacity_kw"] = 6.0
    frame["state"] = "NSW"
    return frame


def _wobble(circuit_id, n, scale=0.05):
    """Deterministic, per-circuit-seeded multiplicative noise. Two circuits
    fed the SAME circuit_id (via `.assign(circuit_id=...)` on a copied
    frame, not a fresh call) share the exact same noise -- a genuine
    duplicate. Two different circuit_ids get independent noise, so an
    otherwise-similar day/night shape does NOT correlate near +-1 just for
    sharing a shape (a real, previously-hit bug: any two-level day/night
    step function correlates at EXACTLY +-1 with any other, no matter the
    magnitudes -- this avoids that by varying every individual timestamp,
    not just two blocks)."""
    rng = np.random.RandomState(int(circuit_id) % (2**31 - 1))
    return 1.0 + scale * rng.normal(size=n)


def _day_series(circuit_id, base_date, *, day_power, night_power=None, n=48, freq_minutes=30):
    """
    One circuit's rows across a full local day, `freq_minutes` apart,
    independently noised per circuit (see `_wobble`). `t_stamp` is naive,
    matching the real wire format (UTC, no tz) that `ami_plots.to_aest`
    expects to localize itself.

    Every row defaults to `day_power` (independently noised); any row whose
    AEST hour falls in `[1, 4)` (the window `evaluate_load_reconstruction`
    checks) is overridden to `night_power` (also independently noised) when
    given -- this is what makes a site's night-time behaviour exactly
    controllable for the reconstruction-check tests, while every other hour
    still carries per-timestamp noise that breaks spurious correlation.
    """
    t_stamp = pd.date_range(base_date, periods=n, freq=f"{freq_minutes}min")
    aest_hour = (t_stamp + pd.Timedelta(hours=10)).hour
    is_night = (aest_hour >= 1) & (aest_hour < 4)
    baseline = np.where(is_night & (night_power is not None),
                         night_power if night_power is not None else 0.0, day_power)
    power_series = baseline * _wobble(circuit_id, n)
    return pd.DataFrame({
        "circuit_id": circuit_id, "t_stamp": t_stamp, "power": power_series,
        "energy": power_series / 12.0, "energy_reactive": 10.0,
        "energy_import": np.clip(power_series, 0, None),
        "energy_export": np.clip(-power_series, 0, None),
        "energy_reactive_import": 10.0, "energy_reactive_export": 0.0,
        "power_factor": 0.98, "voltage": 240.0, "current": 5.0,
    })


#: Rows per circuit for sites where a night/day distinction matters
#: (2001, 2003, 2004, 2005). Deliberately BELOW `find_duplicate_circuits`'s
#: `min_overlap=20` -- these sites each have exactly one load + one PV
#: circuit, and a real day/night cycle makes them genuinely, if mildly,
#: time-correlated (both tend to be "high" by day, "low" by night). At
#: n=48 that shared macro-structure alone pushes correlation past 0.99
#: (a real trap: two *different* physical circuits sharing a day/night
#: shape look identical to `find_duplicate_circuits` no matter their
#: independent per-timestamp noise). Below `min_overlap`, the check is
#: skipped entirely for that pair -- same reason notebook 4's dry-run
#: fixture keeps its Section-7 sites' row counts small (n=6 there).
_SPARSE_N, _SPARSE_FREQ_MINUTES = 16, 90

#: Rows for site 2002, where genuine duplicate detection must actually
#: fire -- needs to clear `min_overlap=20`.
_DENSE_N, _DENSE_FREQ_MINUTES = 48, 30


def _day1_ts() -> pd.DataFrame:
    """The single-day fixture used for the fleet-wide day-1 resolution pull."""
    # Site 2002's PV pair: dense enough for `find_duplicate_circuits` to
    # actually compare them. No night/day split needed at this site (it's
    # excluded for the duplicate, not for any reconstruction reason), so a
    # single noisy constant avoids the day/night correlation trap above.
    pv_212 = _day_series(212, "2025-06-01", day_power=-900.0,
                          n=_DENSE_N, freq_minutes=_DENSE_FREQ_MINUTES)
    pv_213 = pv_212.assign(circuit_id=213)  # exact duplicate registration
    load_211 = _day_series(211, "2025-06-01", day_power=250.0,
                            n=_DENSE_N, freq_minutes=_DENSE_FREQ_MINUTES)

    def sparse(circuit_id, day_power, night_power):
        return _day_series(circuit_id, "2025-06-01", day_power=day_power,
                            night_power=night_power, n=_SPARSE_N, freq_minutes=_SPARSE_FREQ_MINUTES)

    parts = [
        # Site 2001: normal -- positive load at night, PV ~0 at night.
        sparse(201, day_power=300.0, night_power=200.0),
        sparse(202, day_power=-1500.0, night_power=0.0),
        # Site 2002: 212 and 213 are two independent registrations of the
        # SAME physical PV reading -- a same-type duplicate. 211 is a
        # genuinely separate load circuit.
        load_211, pv_212, pv_213,
        # Site 2003: negative at night from day 1 -- reconstruction fails
        # immediately, no need to wait for a later month.
        sparse(221, day_power=-500.0, night_power=-500.0),
        sparse(222, day_power=-10.0, night_power=2.0),
        # Site 2004: normal on day 1 (degrades only from month 9).
        sparse(231, day_power=600.0, night_power=500.0),
        sparse(232, day_power=-1000.0, night_power=0.0),
        # Site 2005: normal on day 1 (degrades only from month 11).
        sparse(241, day_power=300.0, night_power=200.0),
        sparse(242, day_power=-1200.0, night_power=0.0),
    ]
    return pd.concat(parts, ignore_index=True)


def _month_ts(year: int, month: int) -> pd.DataFrame:
    """
    The full-year landed extract fixture -- only for circuits that survive
    the day-1 resolution (2001, 2004, 2005). 2002/2003's circuits never
    appear here; they were excluded before extraction and the notebook
    must never query for them. No duplicate check runs on this data (only
    the day-1 sample feeds `resolve_site_circuits`), so the sparse row
    count is just for a small fixture, not to dodge `min_overlap`.
    """
    base_date = f"{year}-{month:02d}-01"

    def sparse(circuit_id, day_power, night_power):
        return _day_series(circuit_id, base_date, day_power=day_power,
                            night_power=night_power, n=_SPARSE_N, freq_minutes=_SPARSE_FREQ_MINUTES)

    parts = [
        # Site 2001: well-behaved every month, all year.
        sparse(201, day_power=300.0, night_power=200.0),
        sparse(202, day_power=-1500.0, night_power=0.0),
        # Site 2004: circuit 231 goes inactive (near-zero) from month 9.
        sparse(231, day_power=(1.0 if month >= 9 else 600.0),
               night_power=(1.0 if month >= 9 else 500.0)),
        sparse(232, day_power=-1000.0, night_power=0.0),
        # Site 2005: circuit 241 starts reading negative at night from month 11.
        sparse(241, day_power=300.0, night_power=(-600.0 if month >= 11 else 200.0)),
        sparse(242, day_power=-1200.0, night_power=0.0),
    ]
    return pd.concat(parts, ignore_index=True)


_CIRCUIT_ID_LIST_RE = re.compile(r"circuit_id IN \(([^)]*)\)")
_IS_PV_RE = re.compile(r"is_pv\s*=\s*(true|false)", re.IGNORECASE)
_YEAR_MONTH_RE = re.compile(r"year = (\d+) AND month = (\d+)")


def _fake_aq(meta: pd.DataFrame):
    day1 = _day1_ts()

    def aq(sql: str, database=None, *, label=None, allow_full_scan=False, meter=True):
        if "GROUP BY circuit_type, is_pv" in sql:
            return (
                meta.groupby(["circuit_type", "is_pv"])
                .agg(n_circuits=("circuit_id", "count"), n_sites=("site_id", "nunique"))
                .reset_index()
            )
        if "GROUP BY site_id" in sql:
            # site_capacity_lookup's shape: SELECT site_id, max(S_99) AS S_99,
            # max(ac_capacity_kw) AS ac_capacity_kw FROM meta_up23c ...
            # GROUP BY site_id -- simulate the real aggregation, including the
            # uppercase `S_99` alias the real query (and build_ami_raw's own
            # column selection) expect, rather than returning the raw fixture
            # frame's lowercase `s_99` unmodified.
            return (
                meta.groupby("site_id")
                .agg(S_99=("s_99", "max"), ac_capacity_kw=("ac_capacity_kw", "max"))
                .reset_index()
            )
        if "FROM meta_up23c" in sql:
            return meta.copy()
        if "FROM ts" in sql:
            id_match = _CIRCUIT_ID_LIST_RE.search(sql)
            pv_match = _IS_PV_RE.search(sql)
            ym_match = _YEAR_MONTH_RE.search(sql)
            assert id_match and pv_match and ym_match, f"unrecognised ts query shape: {sql[:200]}"
            ids = {int(x) for x in id_match.group(1).split(",")}
            is_pv_value = pv_match.group(1).lower() == "true"
            allowed = set(meta.loc[meta.is_pv == is_pv_value, "circuit_id"])
            ids &= allowed

            is_day_sample = "t_stamp >=" in sql
            source = day1 if is_day_sample else _month_ts(
                int(ym_match.group(1)), int(ym_match.group(2))
            )
            return source[source.circuit_id.isin(ids)].reset_index(drop=True)
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


def test_notebook_05_runs_end_to_end_against_synthetic_fixture(monkeypatch, tmp_path):
    meta = _synthetic_meta()

    artefact_dir = tmp_path / "artefacts"
    store_dir = tmp_path / "store"

    monkeypatch.setattr(AthenaModule, "aq", _fake_aq(meta))
    monkeypatch.setattr(AthenaModule, "require_credentials", lambda: {"ok": True})
    monkeypatch.setattr(AthenaModule, "reset_scan_log", lambda: None)
    monkeypatch.setattr(AthenaModule, "scan_report", lambda *a, **k: None)
    monkeypatch.setattr(Config, "ARTEFACT_DIR", artefact_dir)
    monkeypatch.setattr(Config, "STORE_DIR", store_dir)
    monkeypatch.setattr(Config, "store_path", lambda name: store_dir / name)

    # The real notebook's extraction cell (cell ~19) has a
    # skip-and-reuse-the-prior-manifest fallback that reads this CSV back
    # from ARTEFACT_DIR rather than re-running Extract.run_extraction --
    # pre-seed a stub here so the dry run (which starts from a fresh
    # tmp_path, unlike a real restart where a prior run's manifest already
    # exists on disk) exercises that fallback path successfully too.
    artefact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"year": [], "month": [], "is_pv": [], "chunk_index": [],
                  "n_circuits": [], "n_rows": [], "path": []}).to_csv(
        artefact_dir / "phase5_extraction_manifest.csv", index=False
    )

    namespace = {"__name__": "__dry_run__", "display": lambda *a, **k: None}
    result = _run_notebook_cells(NOTEBOOK_PATH, namespace)

    # --- day-1 cohort: all 5 sites are CLEAN_SITE_IDS (1-circuit ac_load_net each)
    for site_id in (2001, 2002, 2003, 2004, 2005):
        assert site_id in result["CLEAN_SITE_IDS"]

    # --- day-1 resolution: same-type duplicate flagged at site 2002 --------
    fleet_resolution = result["fleet_resolution"]
    assert bool(
        fleet_resolution.loc[fleet_resolution.site_id == 2002, "needs_manual_review"].any()
    )

    # --- day-1 reconstruction: site 2003 flagged immediately ----------------
    assert 2003 in result["flagged_site_ids"]
    assert 2003 not in result.get("fleet_storage_site_ids", [])  # caught by reconstruction, not naming

    # --- 1e: manual-review (2002) and reconstruction-flagged (2003) sites
    # are BOTH excluded before extraction, each with its own reason --------
    fleet_resolution_final = result["fleet_resolution_final"]
    ffr_by_site = fleet_resolution_final.groupby("site_id")
    site_2002 = ffr_by_site.get_group(2002)
    assert not site_2002.kept.any()
    # one circuit was already dropped as a same-type duplicate by
    # resolve_site_circuits itself (keeps that original reason); the
    # circuit that dedup KEPT only gets excluded by the separate,
    # conservative manual-review drop -- both reasons should be visible.
    assert set(site_2002.drop_reason) == {
        "duplicate_same_type", "manual_review_conservatively_dropped",
    }
    assert not ffr_by_site.get_group(2003).kept.any()
    assert set(ffr_by_site.get_group(2003).drop_reason) == {"storage_or_sign_issue"}
    # 2001, 2004, 2005 all survive day-1 resolution untouched.
    for site_id in (2001, 2004, 2005):
        assert ffr_by_site.get_group(site_id).kept.all()

    # --- extraction: ONLY the day-1 survivors' circuits were ever queried --
    extraction_manifest = result["extraction_manifest"]
    assert len(extraction_manifest) > 0
    extracted_circuit_count = extraction_manifest.n_circuits.sum()
    # 3 surviving sites x 2 circuits each x 12 months = 72 "circuit slots"
    # queried across all (month, is_pv, chunk) rows.
    assert extracted_circuit_count == 3 * 2 * 12
    for path in extraction_manifest.path.dropna():
        assert path.exists()
    # excluded sites' circuits (211-213, 221-222) never landed anywhere.
    landed = pd.concat([pd.read_parquet(p) for p in extraction_manifest.path.dropna()])
    assert not set(landed.circuit_id) & {211, 212, 213, 221, 222}

    # --- revalidation: circuit 231 caught as inactive from month 9 on ------
    inactive_over_history = result["inactive_over_history"]
    assert set(inactive_over_history.circuit_id) == {231}
    assert inactive_over_history.set_index("circuit_id").loc[231, "months_inactive"] == [
        f"2025-{m:02d}" for m in range(9, 13)
    ]

    # --- revalidation: site 2005 caught failing reconstruction from month 11
    reconstruction_over_history = result["reconstruction_over_history"]
    assert set(reconstruction_over_history.site_id) == {2005}
    assert reconstruction_over_history.set_index("site_id").loc[2005, "months_flagged"] == [
        f"2025-{m:02d}" for m in (11, 12)
    ]

    # --- final_resolution reflects BOTH full-year findings correctly -------
    final_resolution = result["final_resolution"]
    final_by_id = final_resolution.set_index("circuit_id")
    # site 2004: only circuit 231 dropped, 232 (PV) untouched.
    assert final_by_id.loc[231, "kept"] == False
    assert final_by_id.loc[231, "drop_reason"] == "inactive_full_year"
    assert final_by_id.loc[232, "kept"] == True
    # site 2005: BOTH circuits excluded.
    assert final_by_id.loc[241, "kept"] == False
    assert final_by_id.loc[241, "drop_reason"] == "storage_or_sign_issue_full_year"
    assert final_by_id.loc[242, "kept"] == False
    assert final_by_id.loc[242, "drop_reason"] == "storage_or_sign_issue_full_year"
    # site 2001: fully untouched all year.
    assert final_by_id.loc[201, "kept"] == True
    assert final_by_id.loc[202, "kept"] == True

    # --- final tally ---------------------------------------------------------
    coverage_report = result["coverage_report"]
    assert coverage_report["n_sites"] == 5
    n_final_sites = int(final_resolution.groupby("site_id").kept.any().sum())
    assert n_final_sites == 2  # only 2001 and 2004 have any surviving circuit

    # --- build: ami_raw/ami_meter only contain the truly clean data --------
    build_manifest = result["build_manifest"]
    assert len(build_manifest) == 12
    raw_paths = build_manifest.raw_path.dropna()
    meter_paths = build_manifest.meter_path.dropna()
    assert len(raw_paths) > 0 and len(meter_paths) > 0

    ami_raw = pd.concat([pd.read_parquet(p) for p in raw_paths])
    ami_meter = pd.concat([pd.read_parquet(p) for p in meter_paths])

    # site 2005 excluded entirely -- never appears in either output table.
    assert 2005 not in set(ami_raw.site_id)
    assert 2005 not in set(ami_meter.site_id)
    # site 2004's dropped circuit (231) never appears in ami_meter; its PV
    # circuit alone can't produce an ami_raw row (needs both sides), so
    # site 2004 also contributes nothing to ami_raw specifically.
    assert 231 not in set(ami_meter.circuit_id)
    assert 2004 not in set(ami_raw.site_id)
    # site 2001 (fully clean, all year) appears in both.
    assert 2001 in set(ami_raw.site_id)
    assert 2001 in set(ami_meter.site_id)

    # --- artefacts were written to the (monkeypatched, tmp) artefact dir ----
    assert (artefact_dir / "phase5_site_metadata_full_fleet.csv").exists()
    assert (artefact_dir / "phase5_coverage_report_full_fleet.csv").exists()
    assert (artefact_dir / "phase5_extraction_manifest.csv").exists()
    assert (artefact_dir / "phase5_build_manifest.csv").exists()
