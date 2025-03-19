from pathlib import Path

import pandas as pd

from ciccada.clear_sky_days import detect_clear_sky_day

test_data_filepaths = Path("D:/ciccada/ghi/5007_ghi_csvs").glob("*.csv")

algorithm_classification = {}
mean_change = {}
for file_path in test_data_filepaths:
    test_data = pd.read_csv(file_path)
    test_data = test_data.rename(columns={"surface_global_irradiance": "ghi_mean"})
    date = test_data["time"].iloc[0][:10]
    algorithm_classification[date], mean_change[date] = detect_clear_sky_day(
        test_data, 500.0
    )

    is_cloudy_algo = pd.DataFrame(
        {
            "date": algorithm_classification.keys(),
            "is_clear_algo": algorithm_classification.values(),
            "mean_change": mean_change.values(),
        }
    )

    is_cloudy_algo.to_csv("5007_detect_clearsky_results.csv")
