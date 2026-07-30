from __future__ import annotations

import duckdb

from ausgrid_analysis.duplicates import duplicate_audit_query


def test_duplicate_classification() -> None:
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE source (
            serial VARCHAR,
            measure_time TIMESTAMP,
            phase VARCHAR,
            voltage_v DOUBLE,
            current_a DOUBLE,
            reactive_power_raw_var DOUBLE,
            active_power_raw_w DOUBLE,
            source_month VARCHAR,
            source_file VARCHAR
        )
        """
    )
    connection.execute(
        """
        INSERT INTO source VALUES
            ('1', '2025-04-01 00:00:00', 'A', 240, 1, -100, -500, '2025-04', 'a'),
            ('1', '2025-04-01 00:05:00', 'A', 241, 2, -110, -600, '2025-04', 'a'),
            ('1', '2025-04-01 00:05:00', 'A', 241, 2, -110, -600, '2025-04', 'b'),
            ('2', '2025-04-01 00:00:00', 'B', 240, 1, -100, -500, '2025-04', 'a'),
            ('2', '2025-04-01 00:00:00', 'B', 250, 1, -100, -500, '2025-04', 'a')
        """
    )
    result = connection.execute(
        duplicate_audit_query("SELECT * FROM source", tolerance=1e-6)
    ).fetchdf()
    classes = {
        row.serial: row.duplicate_class
        for row in result.itertuples(index=False)
    }
    assert classes == {
        "1": "identical_duplicate",
        "2": "conflicting_duplicate",
    }
    assert int(result.loc[result["serial"].eq("1"), "source_file_count"].iloc[0]) == 2

