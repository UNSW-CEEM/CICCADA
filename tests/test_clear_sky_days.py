from pathlib import Path

import pandas as pd

from ciccada.clear_sky_days import detect_clear_sky_day


def test_average_rate_of_ghi_change_too_high():
    ghi_data = pd.DataFrame({"mean_ghi": [501.0, 502.0, 603.0]})

    assert not detect_clear_sky_day(ghi_data, min_max_ghi=500.0)


def test_max_ghi_not_high_enough():
    ghi_data = pd.DataFrame({"mean_ghi": [501.0, 502.0, 603.0]})

    assert not detect_clear_sky_day(ghi_data, min_max_ghi=700.0)


def test_slow_rate_of_change_and_max_ghi_high_enough():
    ghi_data = pd.DataFrame({"mean_ghi": [501.0, 502.0, 503.0]})

    assert detect_clear_sky_day(ghi_data, min_max_ghi=500.0)


def test_against_manually_classified_data():
    test_data_filepaths = Path("tests/data/ghi_csvs").glob("*.csv")
    manual_classification = pd.read_csv(Path("tests/data/is_cloudy.csv"))
    manual_classification["is_clear"] = ~manual_classification["is_cloudy"]
    manual_classification = dict(
        zip(manual_classification["date"], manual_classification["is_clear"])
    )
    for file_path in test_data_filepaths:
        test_data = pd.read_csv(file_path)
        date = test_data["time"].iloc[0][:10]
        assert manual_classification[date] == detect_clear_sky_day(test_data, 500.0)
