from pathlib import Path
import sys

import pytest

# Add bms_sa_review to the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.as4777_curves import (
    q_impact_nearest_edge,
    q_impact_nearest_edge_sql,
)


def test_absorbing_shortfall_uses_upper_edge():
    impact = q_impact_nearest_edge(
        q_kvar=-1.831,
        q_min_final=-2.767,
        q_max_final=-1.967,
    )

    assert impact == pytest.approx(
        1.831 / 1.967,
        abs=1e-6,
    )
    assert 0.9 <= impact <= 1.1


def test_absorbing_surplus_uses_lower_edge():
    impact = q_impact_nearest_edge(
        q_kvar=-3.0,
        q_min_final=-2.767,
        q_max_final=-1.967,
    )

    assert impact == pytest.approx(
        3.0 / 2.767,
        abs=1e-6,
    )


def test_supplying_shortfall_uses_lower_edge():
    impact = q_impact_nearest_edge(
        q_kvar=1.831,
        q_min_final=1.967,
        q_max_final=2.767,
    )

    assert impact == pytest.approx(
        1.831 / 1.967,
        abs=1e-6,
    )


def test_wrong_direction_is_negative():
    impact = q_impact_nearest_edge(
        q_kvar=0.5,
        q_min_final=-2.767,
        q_max_final=-1.967,
    )

    assert impact < 0


def test_low_power_is_not_assessable():
    assert q_impact_nearest_edge(
        q_kvar=-1.831,
        q_min_final=-2.767,
        q_max_final=-1.967,
        capability_assessable=False,
    ) is None


def test_generated_sql_compares_distance_to_edges():
    sql = q_impact_nearest_edge_sql(
        "Q_kvar",
        "Q_min_final",
        "Q_max_final",
        "capability_assessable",
    )

    normalised = " ".join(sql.split())

    assert (
        "abs(Q_kvar - Q_max_final) "
        "<= abs(Q_kvar - Q_min_final)"
    ) in normalised