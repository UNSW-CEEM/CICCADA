"""
Data calc-write pipeline: Stage 1. Step 4 of 4
===================================================

WHAT IT DOES
------------
Applies the fitted GHI model to EVERY eligible interval to produce the
counterfactual "what this site would have generated without curtailment":

        P_norm_est = P_norm_cs * (a + b * GHI/GHI_cs)   [floored at actual]
        uncurtailed_P = P_norm_est * selected normalisation capacity

The estimate is first floored at observed P, matching Hossein's production
INSERT, then optionally capped at a separately selected capacity basis. Both
bases and whether each row was floored/capped are written to the output.

READS
-----
`structured_data{SUFFIX}` + `pv_ghi_norm_model{SUFFIX}`, restricted to sites in
the local `mape_under50_sites.csv` quality gate (MAPE < 50% plus the documented
minimum validation coverage).

SAFE-BY-DEFAULT
---------------
Writes to `all_uncurtailedpv{SUFFIX}` (default "_v2").
"""

import pandas as pd
from build_structured_data import TABLE_SUFFIX
from pipeline_options import capacity_column

SD    = f"structured_data{TABLE_SUFFIX}"
MODEL = f"pv_ghi_norm_model{TABLE_SUFFIX}"
TARGET = f"all_uncurtailedpv{TABLE_SUFFIX}"
WAREHOUSE = "Trino-Warehouse/solar_analytics"

TIME_BIN_MIN = 5


def create_table(aq, database, target=TARGET):
    aq(f"DROP TABLE IF EXISTS {target}", database=database)
    aq(f"""
        CREATE TABLE {target} (
            site_id       BIGINT,
            t_stamp       TIMESTAMP,
            year          INT,
            month         INT,
            uncurtailed_P DOUBLE,
            P_kw          DOUBLE,
            GHI           DOUBLE,
            n_train       BIGINT,
            model_prediction_raw DOUBLE,
            uncurtailed_P_floored DOUBLE,
            capacity_limit DOUBLE,
            normalization_basis STRING,
            counterfactual_cap_basis STRING,
            floor_applied BOOLEAN,
            capped        BOOLEAN,
            year_p        INT,
            month_p       INT
        )
        PARTITIONED BY (year_p, month_p)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{target}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {target}"


def _acceptable_sites_csv(csv_path):
    """Read the MAPE<50% site list into a comma-separated SQL fragment."""
    ids = pd.read_csv(csv_path)["site_id"].tolist()
    return ", ".join(map(str, ids))


def run_year(
    aq, 
    database, 
    year, 
    mape_csv_path, 
    n_parts=3, 
    parts=None, *,
    sd=SD, 
    model=MODEL, 
    target=TARGET,
    normalization_basis="s_99", 
    normalization_capacity_col="S_99",
    counterfactual_cap_basis="ac_capacity_kw",
    min_model_bin_n=5,
):
    """
    Apply the model for one year. `mape_csv_path` points at your local
    mape<50_sites.csv quality-gate file.
    """
    model_bin_predicate = (
        ""
        if min_model_bin_n is None
        else f"AND mo.n >= {int(min_model_bin_n)}"
    )
    if parts is None:
        parts = list(range(n_parts))

    acceptable = _acceptable_sites_csv(mape_csv_path)
    if counterfactual_cap_basis == "none":
        cap_select = "CAST(NULL AS DOUBLE)"
        final_counterfactual = "uncurtailed_P_floored"
        capped_flag = "False"
    else:
        cap_col = capacity_column(counterfactual_cap_basis)
        cap_select = f"m.{cap_col}"
        # If measured P itself exceeds the chosen cap, preserve telemetry rather
        # than manufacture a counterfactual below the observation.
        final_counterfactual = (
            "greatest(least(uncurtailed_P_floored, capacity_limit), P_kw)"
        )
        capped_flag = "(uncurtailed_P_floored > capacity_limit)"
    results = []
    for part in parts:
        part_filter = f"sd.site_id % {n_parts} = {part}"
        aq(f"""
            INSERT INTO {target}
            WITH eligible AS (
                SELECT
                    sd.site_id, sd.actual_day, sd.t_stamp,
                    CAST(CAST(date_trunc('minute', t_stamp + interval '10' hour)
                        - interval '1' minute * (minute(t_stamp + interval '10' hour) % {TIME_BIN_MIN})
                        AS TIME) AS VARCHAR) AS tod_bin,
                    sd.GHI / sd.GHI_cs AS x,
                    sd.P_kw_norm, sd.P_kw_norm_cs, sd.S_99,
                    sd.{normalization_capacity_col} AS normalization_capacity,
                    m.ac_capacity_kw,
                    m.S_99 AS metadata_s_99,
                    {cap_select} AS capacity_limit
                FROM {sd} sd
                JOIN (SELECT site_id,
                             max(ac_capacity_kw) AS ac_capacity_kw,
                             max(S_99) AS S_99
                    FROM meta_up23c
                    WHERE is_pv = True
                    GROUP BY site_id) m
                ON sd.site_id = m.site_id
                WHERE sd.P_kw_norm_cs > 0.2 AND sd.GHI > 50 AND sd.P_kw_norm > 0.05
                  AND sd.P_kw_norm <= sd.P_kw_norm_cs
                  AND sd.year = {year} AND {part_filter}
                  AND sd.site_id IN ({acceptable})
            ),
            applied AS (
                SELECT
                    e.site_id, e.t_stamp, e.x AS GHI,
                    e.P_kw_norm * e.normalization_capacity AS P_kw,
                    e.capacity_limit,
                    e.P_kw_norm_cs * (mo.a + mo.b * e.x)
                        * e.normalization_capacity AS model_prediction_raw,
                    mo.n AS n_train
                FROM eligible e
                JOIN {model} mo
                ON e.site_id = mo.site_id
                AND e.tod_bin = mo.tod_bin
                {model_bin_predicate}
            ),
            adjusted AS (
                SELECT *,
                       greatest(model_prediction_raw, P_kw)
                           AS uncurtailed_P_floored
                FROM applied
            )
            SELECT
                site_id, t_stamp,
                CAST(year(t_stamp) AS INT)  AS year,
                CAST(month(t_stamp) AS INT) AS month,
                {final_counterfactual} AS uncurtailed_P,
                P_kw, GHI, n_train,
                model_prediction_raw,
                uncurtailed_P_floored,
                capacity_limit,
                '{normalization_basis}' AS normalization_basis,
                '{counterfactual_cap_basis}' AS counterfactual_cap_basis,
                (model_prediction_raw < P_kw) AS floor_applied,
                {capped_flag} AS capped,
                CAST(year(t_stamp) AS INT)  AS year_p,
                CAST(month(t_stamp) AS INT) AS month_p
            FROM adjusted
            WHERE model_prediction_raw IS NOT NULL
        """, database=database)
        results.append(f"applied year={year} part={part}/{n_parts}")
        print(results[-1])
    return results


def validate(aq, database, target=TARGET):
    # Sanity: uncurtailed_P should never be below measured P_kw
    below = aq(f"SELECT count(*) AS n FROM {target} WHERE uncurtailed_P < P_kw",
               database=database)
    # How often did the R6 cap bite?
    cap = aq(f"""
        SELECT count(*) AS n_rows,
               sum(CASE WHEN capped THEN 1 ELSE 0 END) AS n_capped,
               round(100.0 * sum(CASE WHEN capped THEN 1 ELSE 0 END) / count(*), 3) AS pct_capped
        FROM {target}
    """, database=database)
    print(f"Rows with uncurtailed_P < P_kw (should be 0): {int(below['n'].iloc[0])}")
    print("Counterfactual cap impact:")
    print(cap.to_string(index=False))
    return below, cap
