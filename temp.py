import os
import random
import pandas as pd

# Define dataset path (replace this with the actual path to your dataset)
dataset_path = r"D:\Fish Freshness Detection\dataset\train"  # Replace this with your dataset path

# Define temperature ranges
fresh_temperature_range = (0, 4)  # Fresh fish temperature range (0°C - 4°C)
non_fresh_temperature_range = (5, 10)  # Non-fresh fish temperature range (5°C - 10°C)

# Define folder names for freshness labels
fresh_folders = ["eye-fresh", "gill-fresh"]
non_fresh_folders = ["eye-non-fresh", "gill-non-fresh"]

# List to store metadata (image, freshness, temperature)
metadata = []

# Process fresh folders
for folder in fresh_folders:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(folder, file_name)
                # Assign a random temperature in the fresh range
                temperature = round(random.uniform(*fresh_temperature_range), 2)
                metadata.append({"image": file_path, "freshness": "fresh", "temperature": temperature})

# Process non-fresh folders
for folder in non_fresh_folders:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(folder, file_name)
                # Assign a random temperature in the non-fresh range
                temperature = round(random.uniform(*non_fresh_temperature_range), 2)
                metadata.append({"image": file_path, "freshness": "non-fresh", "temperature": temperature})

# Create a DataFrame
df = pd.DataFrame(metadata)

# Save to CSV (you can change the name and path of the output CSV)
output_csv = "fish_freshness_with_temperature.csv"
df.to_csv(output_csv, index=False)

print(f"CSV file created: {output_csv}")
