import os
import pandas as pd

def load_vrptw_dataset(local_path="data/raw/solomon_dataset"):
    """
    Load the raw Solomon VRPTW dataset from the specified local directory.

    This function expects that the dataset has already been manually
    downloaded and organized into subfolders (C1, C2, R1, R2, RC1, RC2).
    It loads each CSV file into a dictionary of DataFrames.

    Args:
        local_path (str): Path to the local dataset directory.

    Returns:
        dict[str, pd.DataFrame]: Dictionary where each key is a filename
        and each value is its corresponding DataFrame.

    Example:
        >>> data = load_vrptw_dataset()
        >>> list(data.keys())
        ['C101.csv', 'C102.csv', ...]
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Folder not found: {local_path}")

    # Load all CSV files in subdirectories
    data_dict = {}
    for root, _, files in os.walk(local_path):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))
                data_dict[file] = df

    print(f"Loaded {len(data_dict)} CSV files from {local_path}.")
    return data_dict