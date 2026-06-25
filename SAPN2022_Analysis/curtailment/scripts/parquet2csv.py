from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# all data - build_structured_high_resolution.py
folderName = "all_structured_data_test"
fileName   = "structured_data"

# all data - build_structured_5m.py
folderName = "all_structured_data_5m"
fileName   = "structured_data_5m"

# trained model - fit_ghi_model.py
# folderName = "local_model"
# fileName   = "pv_ghi_norm_model_new"

# 5m resolution for fit_ghi_model.py
# folderName = "model"
# fileName   = "pv_ghi_norm_model_5m"

# # # all uncurtailed power - write_all_uncurtailedPV.py
# # folderName = "local_scored"
# # fileName   = "all_uncurtailedPV"

# 5m resolution for write_all_uncurtailedPV.py
folderName = "prediction"
fileName   = "all_uncurtailedPV_5m"

# # eligible timestamps for phase b - run_sapn2022_metrics.py
# folderName = "curtailed_estimates"
# fileName   = "curtailment_sapn2022"

# eligible timestamps for phase b - run_sapn2022_metrics.py
# folderName = "curtailed_estimates_5m"
# fileName   = "curtailment_sapn2022_5m"

input_path = OUTPUTS_ROOT / folderName / f"{fileName}.parquet"
output_path = OUTPUTS_ROOT / folderName / f"{fileName}.csv"

df = pd.read_parquet(input_path)

# Save as a CSV file
df.to_csv(output_path, index=False)
