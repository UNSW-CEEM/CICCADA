"""Create the cleaned EBM parquet required by conformance."""
# needs to be imported

import sys
from pathlib import Path

import polars as pl

CONFORMANCE_DIR = Path(__file__).resolve().parents[1]
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

# from ebm_workflow.loading import (
#     load_sapn_circuit_details,
#     load_sapn_cleaned_data,
#     load_sapn_site_details,
# )
from ebm_workflow.preprocessing import write_cleaned_site_data
from ebm_workflow.ebm_all_paths.ebm_nsw_path4 import (
    # CAPACITY_DERIVED_PATH,
    CIRCUIT_DETAILS_PATH,
    CLEANED_SITE_DATA_PATH,
    LOCAL_TIMEZONE,
    RAW_SITE_DATA_CSV_GLOB,
    RAW_SITE_DATA_PATH,
)

print(
    "Building deduplicated SAPN site data in 128 circuit buckets "
    "(the 4 GB source may take several minutes)...\n"
    f"Output: {CLEANED_SITE_DATA_PATH}",
    flush=True,
)
# Clean the raw measurements in 128 memory-bounded partitions and write all
# cleaned partitions to one Parquet file. This returns the output file path.
cleaned_data_path = write_cleaned_site_data(
    raw_path=RAW_SITE_DATA_PATH,
    raw_csv_glob=RAW_SITE_DATA_CSV_GLOB,
    circuit_details_path=CIRCUIT_DETAILS_PATH,
    cleaned_path=CLEANED_SITE_DATA_PATH,
    local_timezone=LOCAL_TIMEZONE,
    deduplicate=True,
    num_buckets=128,
)
print(f"Saved cleaned site data to {cleaned_data_path}")

# site_details = load_sapn_site_details()
# circuit_details = load_sapn_circuit_details()
# # Lazily scan the entire cleaned Parquet dataset for downstream calculations.
# all_data = load_sapn_cleaned_data(cleaned_data_path)
# pv_site_net_counts = {
#     row["site_id"]: int(row["pv_site_net_count"])
#     for row in (
#         circuit_details.filter(pl.col("con_type") == "pv_site_net")
#         .group_by("site_id")
#         .len()
#         .rename({"len": "pv_site_net_count"})
#         .to_dicts()
#     )
# }
# candidate_site_ids = (
#     all_data.select("c_id")
#     .unique()
#     .join(
#         circuit_details.select(["c_id", "site_id"]).unique().lazy(),
#         on="c_id",
#         how="inner",
#     )
#     .select("site_id")
#     .unique()
#     .collect()["site_id"]
#     .to_list()
# )

# print(f"Generating SAPN capacity CSV at {CAPACITY_DERIVED_PATH}.", flush=True)
# generate_rated_capacity(
#     site_details,
#     circuit_details,
#     all_data,
#     candidate_site_ids,
#     pv_site_net_counts,
#     CAPACITY_DERIVED_PATH,
# )
# print(f"Saved SAPN capacity CSV to {CAPACITY_DERIVED_PATH}")
