import os
import pandas as pd

def load_vrptw_dataset(local_path="data/raw/solomon_dataset"):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"❌ Folder not found: {local_path}")

    # Load all CSV files in subdirectories
    data_dict = {}
    for root, _, files in os.walk(local_path):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))
                data_dict[file] = df

    print(f"Loaded {len(data_dict)} CSV files from {local_path}.")
    return data_dict