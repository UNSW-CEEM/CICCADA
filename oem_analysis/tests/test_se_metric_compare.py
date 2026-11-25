"""
Notebook 03 sections 3c/3d helpers (se_metric_compare), on the tiny invented
store from test_se_conformance_shared: flip table, example picks, the two
side-by-side figures (smoke test) and hourly coverage.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from oem_analysis.tests.test_se_conformance_shared import CFG, _fake_con  # noqa: E402

pytest.importorskip("duckdb")
from oem_analysis.lib import se_conformance as SC  # noqa: E402
from oem_analysis.lib import se_metric_compare as cmp  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    con, _, _ = _fake_con()
    zc = SC.voltvar_zone_counts(con, CFG)
    m = SC.voltvar_metrics(con, CFG, counts=zc)
    return con, zc, m


def test_flip_table_classes_and_zone_shares(setup):
    _, _, m = setup
    f = cmp.flip_table(m, "legacy_conformance_rate", "nonconformance_frequency")
    assert len(f) == len(m)
    same = f.legacy_conformance_rate_verdict == f.nonconformance_frequency_verdict
    assert f.change[same].str.startswith("same:").all()
    assert f.change[~same].str.contains("->").all()
    shares = f[[f"failures_in_zone_{z}_%" for z in "ABCD"]].sum(axis=1)
    assert ((shares.round(0) == 100) | shares.isna() | (shares == 0)).all()


def test_pick_examples_adds_month(setup):
    _, zc, m = setup
    f = cmp.flip_table(m, "legacy_conformance_rate", "nonconformance_frequency")
    ex = cmp.pick_examples(f, zc, n_changed=1, n_same=1)
    assert len(ex) >= 1 and ex.month.notna().all()


def test_figures_and_coverage(setup):
    con, zc, m = setup
    alias = m.site_alias.iloc[0]
    day = cmp.best_day(con, CFG, alias, "2025-01")
    df, info = cmp.site_frame(con, CFG, alias, day, pd.Timestamp(day) + pd.Timedelta(days=1))
    assert set(df.columns) == {"t_aest", "V", "P_kw", "Q_kvar"} and info["rating_kva"] > 0
    assert cmp.plot_metric_comparison_day(df, info) is not None
    dfm, infom = cmp.site_frame(con, CFG, alias, "2025-01-01", "2025-02-01")
    assert cmp.plot_qv_metric_comparison(dfm, infom) is not None
    cov = cmp.oem_hourly_coverage(con, CFG)
    assert cov.rows_pct_of_expected.max() > cov.rows_pct_of_expected.min()  # daylight-only fake data
