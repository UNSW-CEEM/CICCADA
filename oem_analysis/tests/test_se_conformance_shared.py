"""
OEM Volt-VAr metrics use the shared rules: se_conformance.voltvar_zone_counts
(DuckDB, OEM conventions) must give the same counts as
bms_sa_review/shared/vvar_metrics on the same intervals, and its legacy columns
must equal the legacy path (voltvar_site_day).

Runs on a tiny invented store built in DuckDB (no real data). Needs
CICCADA_SE_DATA_ROOT set (any value) because se_config reads it at import.

Run:  pytest oem_analysis/tests/test_se_conformance_shared.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _REPO_ROOT / "bms_sa_review"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("CICCADA_SE_DATA_ROOT", str(_REPO_ROOT / "_no_data"))

duckdb = pytest.importorskip("duckdb")

from oem_analysis.lib import se_conformance as SC  # noqa: E402
from oem_analysis.lib import se_params  # noqa: E402
from shared.vvar_metrics import score_intervals, zone_counts  # noqa: E402


def _fake_con(seed=3, n_sites=5, days=3):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=288 * days, freq="5min")
    hour = t.hour + t.minute / 60
    rows, sites, caps = [], [], []
    for k in range(n_sites):
        alias = f"SITE{k:03d}"
        S = [5.0, 8.0, 10.0, 6.0, 5.0][k]
        sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
        awake = sun > 0.02  # daylight-only telemetry, as the OEM data
        v = 236 + 12 * sun + rng.normal(0, 2.5, len(t)) - 18 * (k == 3) * (hour < 9)
        p = S * sun * rng.uniform(0.5, 1.0, len(t))
        gain = [-1.0, 0.0, 0.6, -0.8, 0.2][k]
        q = gain * np.clip((v - 240) / 18, 0, 1) * 0.6 * S + rng.normal(0, 0.03 * S, len(t))
        d = pd.DataFrame(
            {
                "site_alias": alias,
                "ts_aest": t,
                "dt_month": "2025-01",
                "P_kW": p,
                "Q_kvar": q,
                "V_mean": v,
                "V_max": v + 1,
                "derating_active": False,
            }
        )[awake]
        d.loc[d.sample(frac=0.01, random_state=k).index, "Q_kvar"] = np.nan
        rows.append(d)
        sites.append(
            {
                "site_alias": alias,
                "is_three_phase": k == 2,
                "state": "NSW",
                "n_days_observed": days,
                "has_night_generation_anomaly": False,
            }
        )
        caps.append({"site_alias": alias, "s_99": S, "s_95": S, "s_max": S})
    con = duckdb.connect()
    con.register("_i", pd.concat(rows, ignore_index=True))
    con.register("_s", pd.DataFrame(sites))
    con.register("_c", pd.DataFrame(caps))
    con.execute("CREATE VIEW se_interval AS SELECT * FROM _i")
    con.execute("CREATE VIEW se_site AS SELECT * FROM _s")
    con.execute("CREATE VIEW se_site_capacity AS SELECT * FROM _c")
    return con, pd.concat(rows, ignore_index=True), pd.DataFrame(caps)


CFG = se_params.CONFIG.with_changes(months=("2025-01",))


@pytest.mark.parametrize("tol", [0.0, 0.04])
def test_zone_counts_match_shared_python(tol):
    con, raw, caps = _fake_con()
    sql = SC.voltvar_zone_counts(con, CFG, tol=tol).set_index("site_alias").sort_index()
    df = raw.merge(caps[["site_alias", "s_99"]], on="site_alias").rename(
        columns={"V_mean": "V", "s_99": "s_rated"}
    )
    df["day"] = df.ts_aest.dt.date
    py = zone_counts(score_intervals(df, tol=tol), by=["site_alias"]).set_index("site_alias").sort_index()
    cols = [c for c in py.columns if c.startswith("n_") and c in sql.columns]
    assert {"n_A_adverse", "n_B_voltage_deadband_violation", "n_days_A", "n_legacy_fail"} <= set(cols)
    pd.testing.assert_frame_equal(sql[cols].astype("int64"), py[cols].astype("int64"))


def test_legacy_columns_equal_the_legacy_path():
    con, _, _ = _fake_con()
    zc = SC.voltvar_zone_counts(con, CFG).set_index("site_alias").sort_index()
    sd = SC.voltvar_site_day(con, CFG).groupby("site_alias").sum(numeric_only=True).sort_index()
    fails = sd[["Q_adverse_count", "Q_inactive_count", "Q_significant_shortfall_count"]].sum(axis=1)
    assert (zc.n_legacy_total == sd.total_count).all()
    assert (zc.n_legacy_fail == fails).all()


def test_metrics_report_coverage_and_tags():
    con, _, _ = _fake_con()
    m = SC.voltvar_metrics(con, CFG)
    assert m.coverage.between(0, 1).all() and m.coverage.median() < 0.8  # daylight only
    for metric in ("nonconformance_frequency", "remote_conformance_rate", "legacy_conformance_rate"):
        assert m[f"{metric}_verdict"].isin(["conformant", "non-conformant", "not assessable"]).all()
