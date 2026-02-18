import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import natsort


def analyze_tif(file_path, fill_value=65535):
    """Return missing percentage for one tif."""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
        total_pixels = data.size
        missing_pixels = np.sum(data == fill_value)
        return (missing_pixels / total_pixels) * 100
    except Exception:
        return 100.0

def _get_files(directory, valid_days=None):
    """Retrieve .tif files from the nested day/hour/quad structure."""
    files_by_day_hour = defaultdict(list)

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return files_by_day_hour

    for day_folder in natsort.natsorted(os.listdir(directory)):
        full_day_path = os.path.join(directory, day_folder)
        if not os.path.isdir(full_day_path):
            continue
        if valid_days and day_folder not in valid_days:
            continue

        for hour_folder in natsort.natsorted(os.listdir(full_day_path)):
            full_hour_path = os.path.join(full_day_path, hour_folder)
            if not os.path.isdir(full_hour_path):
                continue

            for quad_folder in natsort.natsorted(os.listdir(full_hour_path)):
                full_quad_path = os.path.join(full_hour_path, quad_folder)
                if not os.path.isdir(full_quad_path):
                    continue

                tif_files = [
                    f for f in os.listdir(full_quad_path) if f.lower().endswith(".tif")
                ]

                for tif_file in tif_files:
                    key = (day_folder, hour_folder)
                    files_by_day_hour[key].append(os.path.join(full_quad_path, tif_file))

    return files_by_day_hour


def analyze_year(base_path):
    """Analyzes missing data by averaging per hour per day."""
    files_by_day_hour = _get_files(base_path)
    if not files_by_day_hour:
        print("No files found.")
        return [], [], []

    # Dictionaries to store the average missing percentage for each day and hour
    avg_daily_missing_data = defaultdict(list)
    avg_hourly_missing_data = defaultdict(list)

    # Process files and calculate averages per day and hour
    total_keys = len(files_by_day_hour)
    processed_count = 0
    with tqdm(total=total_keys, desc="Processing day/hour combinations") as pbar:
        for (day, hour), file_list in files_by_day_hour.items():
            missing_pct_list = [analyze_tif(f) for f in file_list]
            if missing_pct_list:
                # Average all quad-hash files for a given day and hour
                avg_missing_for_day_hour = np.mean(missing_pct_list)
                
                # Store the result for daily and hourly analysis
                avg_daily_missing_data[int(day)].append(avg_missing_for_day_hour)
                avg_hourly_missing_data[int(hour)].append(avg_missing_for_day_hour)
            
            pbar.update(1)

    # Compute final averages from the collected data
    all_daily_averages = [np.mean(avg_daily_missing_data[d]) for d in sorted(avg_daily_missing_data)]
    avg_hourly_missing_final = [np.mean(avg_hourly_missing_data[h]) for h in range(24)]

    return [], all_daily_averages, avg_hourly_missing_final

# ------------------- Main Script -------------------

if __name__ == "__main__":
    base_path = "/s/chopin/e/proj/hyperspec/diurnalModel/LSTCTargetTest/"
    # The first list in the return is now empty as we are not collecting all individual missing percentages
    _, avg_daily_missing, avg_hourly_missing = analyze_year(base_path)

    print("Daily averaged missing data (approx. 365 values):", len(avg_daily_missing))
    print("Hourly averaged missing data (24 values):", len(avg_hourly_missing))
    # You can now use these variables to plot as you did before
    # For example:
    # ---- Hourly bar plot ----
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), avg_hourly_missing, color="salmon", edgecolor="black")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Missing Data (%)")
    plt.title("Average Missing Data Per Hour (2022)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("/s/chopin/k/grad/varsh/diurnalModel/diurnal_curve_results/hourly_missing.png", dpi=300)
    plt.close()
    
    # ---- Daily bar plot ----
    plt.figure(figsize=(12, 5))
    days = range(1, len(avg_daily_missing) + 1)
    plt.bar(days, avg_daily_missing, color="steelblue", edgecolor="black")
    plt.xlabel("Day of Year")
    plt.ylabel("Average Missing Data (%)")
    plt.title("Daily Average Missing Data (2022)")
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, len(days), 30), rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("/s/chopin/k/grad/varsh/diurnalModel/diurnal_curve_results/daily_missing.png", dpi=300)
    plt.close()

    print("All plots saved: histogram, daily bar plot, and hourly bar plot.")