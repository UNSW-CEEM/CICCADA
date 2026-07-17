"""
Data calc-write pipeline: Stage 1. Step 2 of 4
============================================

WHAT IT DOES
------------
Assigns each site-day to 'train' (80%) or 'val' (20%), randomly WITHIN each
site (so no site leaks across the split). 

The GHI model in step 3 trains only on 'train' days and can be validated on 'val' days.

READS
-----
`structured_data{SUFFIX}` (built in step 1), NOT raw ts. 

This is a simplification. Reading the already-built structured table is cheaper and guarantees the split is over
exactly the same rows the model will see.

SAFE-BY-DEFAULT
---------------
Writes to `split_days{SUFFIX}` (default "_v2").
"""

from build_structured_data import TABLE_SUFFIX  # reuse the same suffix

SOURCE = f"structured_data{TABLE_SUFFIX}"
TARGET = f"split_days{TABLE_SUFFIX}"
WAREHOUSE = "Trino-Warehouse/solar_analytics"

TRAIN_FRAC = 0.8



def create_table(aq, database, target=TARGET):
    aq(f"DROP TABLE IF EXISTS {target}", database=database)
    aq(f"""
        CREATE TABLE {target} (
            site_id    BIGINT,
            actual_day DATE,
            day_type   STRING
        )
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{target}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {target}"


def run(aq, database, source=SOURCE, target=TARGET):
    """
    Build the split over ALL sites/days present in structured_data{SUFFIX}.
    The eligibility filter mirrors the model's training filter so the split is
    over the model-eligible population:
        P_kw_norm_cs > 0.2, GHI > 50, P_kw_norm > 0.05,
        V <= 253, (P_kw_norm >= 1 OR S_norm < 1.001)
    """
    aq(f"""
        INSERT INTO {target}
        WITH eligible AS (
            SELECT DISTINCT site_id, actual_day
            FROM {source}
            WHERE P_kw_norm_cs > 0.2 AND GHI > 50 AND P_kw_norm > 0.05
              AND V <= 253 AND (P_kw_norm >= 1 OR S_norm < 1.001)
        ),
        ranked AS (
            SELECT site_id, actual_day,
                   row_number() OVER (PARTITION BY site_id ORDER BY random()) AS rn,
                   count(*)    OVER (PARTITION BY site_id)                     AS total_days
            FROM eligible
        )
        SELECT site_id, actual_day,
               CASE WHEN rn <= {TRAIN_FRAC} * total_days THEN 'train' ELSE 'val' END AS day_type
        FROM ranked
    """, database=database)
    return f"Populated {TARGET}"


def validate(aq, database, target=TARGET):
    counts = aq(f"""
        SELECT day_type, count(*) AS n_site_days, count(DISTINCT site_id) AS n_sites
        FROM {target}
        GROUP BY day_type
        ORDER BY day_type
    """, database=database)
    print("Train / val split:")
    print(counts.to_string(index=False))
    return counts
