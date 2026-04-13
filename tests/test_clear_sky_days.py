from pathlib import Path

import pandas as pd
from ciccada.clear_sky_days import detect_clear_sky_day


def test_average_rate_of_ghi_change_too_high():
    ghi_data = pd.DataFrame({"ghi_mean": [501.0, 502.0, 603.0]})

    assert not detect_clear_sky_day(ghi_data, min_max_ghi=500.0)


def test_max_ghi_not_high_enough():
    ghi_data = pd.DataFrame({"ghi_mean": [501.0, 502.0, 603.0]})

    assert not detect_clear_sky_day(ghi_data, min_max_ghi=700.0)


def test_slow_rate_of_change_and_max_ghi_high_enough():
    ghi_data = pd.DataFrame({"ghi_mean": [501.0, 502.0, 503.0]})

    assert detect_clear_sky_day(ghi_data, min_max_ghi=500.0)


def test_against_manually_classified_data():
    """
    Daily GHI CSV has been manually classified as clear or cloudy. This
    test runs each day of data through the detect_clear_sky_day algorithm and then
    compares the results to the manual classification. The test fails if there are
    any difference between the manual classification and the algorithm results.

    The test data is Creative Commons Attribution 4.0:
    https://creativecommons.org/licenses/by/4.0/

    Data sourced from energydata.info:
    https://energydata.info/dataset/lebanon-solar-radiation-measurements

    A number of changes where made to the data:
    - Broken into daily CSVs
    - Column "JulianTime" relabeled to "time"
    - Column "GHI_ThPyra1_Wm-2_avg" relabeled to "mean_ghi"
    """
    test_data_filepaths = Path("tests/data/ghi_csvs").glob("*.csv")
    manual_classification_df = pd.read_csv(Path("tests/data/manual_classification.csv"))

    algorithm_classification = {}
    for file_path in test_data_filepaths:
        test_data = pd.read_csv(file_path)
        date = test_data["time"].iloc[0][:10]
        algorithm_classification[date] = detect_clear_sky_day(test_data, 500.0)

    is_cloudy_algo = pd.DataFrame(
        {
            "date": algorithm_classification.keys(),
            "is_clear_algo": algorithm_classification.values(),
        }
    )

    manual_classification_df = pd.merge(
        manual_classification_df, is_cloudy_algo, on="date"
    )
    manual_classification_df = manual_classification_df.loc[
        :, ["date", "unsure", "is_clear", "is_clear_algo"]
    ]

    manual_classification_df["diff"] = (
        manual_classification_df["is_clear"]
        != manual_classification_df["is_clear_algo"]
    )

    manual_classification_df.to_csv(Path("tests/data/comparison.csv"))
    assert (~manual_classification_df["diff"]).all()
