import glob
import os
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series(input_dir, output_dir):
    """
    Iterate through all CSV files in input_dir, plot the time series data,
    and save plots as PNG files in output_dir.

    Parameters:
    -----------
    input_dir : str
        Directory containing the CSV data files
    output_dir : str
        Directory where PNG plot files will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each CSV file
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        file_stem = os.path.splitext(filename)[0]

        print(f"Processing: {filename}")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Convert time column to datetime
        df["time"] = pd.to_datetime(df["time"])

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df["time"], df["ghi_mean"], "-", linewidth=1)

        # Format the plot
        plt.title(f"Global Horizontal Irradiance (GHI) - {file_stem}")
        plt.xlabel("Time")
        plt.ylabel("Mean GHI")
        plt.grid(True, alpha=0.3)

        # Format x-axis to show date nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.gcf().autofmt_xdate()

        # Save the plot as PNG
        output_path = os.path.join(output_dir, f"{file_stem}_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Saved plot to: {output_path}")

    print("Processing complete!")


if __name__ == "__main__":
    # Set your input and output directories here
    input_directory = Path("tests/data/ghi_csvs")
    output_directory = Path("tests/data/ghi_data_plotted")

    plot_time_series(input_directory, output_directory)
