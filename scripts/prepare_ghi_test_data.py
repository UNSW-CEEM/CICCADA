import random
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

data_folder = Path("D:/ciccada/ghi")

ghi_data_by_postcode_file = data_folder / Path(
    "ghi_postcode/NCI_processed_Adelaide_grouped.csv"
)

ghi_data_by_postcode = pd.read_csv(ghi_data_by_postcode_file)

ghi_data_by_postcode["postcode"] = ghi_data_by_postcode["postcode"].astype(int)

single_postcode_data = ghi_data_by_postcode.loc[
    ghi_data_by_postcode["postcode"] == 5007, :
]

single_postcode_data["time"] = pd.to_datetime(
    single_postcode_data["time"], format="%Y-%m-%d %H:%M:%S"
)

single_postcode_data["time"] += timedelta(hours=10, minutes=30)

unique_dates = single_postcode_data["time"].dt.date.unique()

# Randomly select 30 dates
selected_dates = random.sample(list(unique_dates), 30)

for date in unique_dates:
    # Filter data for the current date
    daily_data = single_postcode_data[single_postcode_data["time"].dt.date == date]

    csv_filename = data_folder / Path(f"5007_ghi_csvs/ghi_{date}.csv")
    daily_data.to_csv(csv_filename, index=False)

    # Create and save plot
    plt.figure(figsize=(10, 6))
    plt.plot(daily_data["time"], daily_data["surface_global_irradiance"])
    plt.title(f"GHI Data for {date}")
    plt.xlabel("Time")
    plt.ylabel("GHI")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(data_folder / Path(f"5007_ghi_plots/ghi_{date}.png"))
    plt.close()
