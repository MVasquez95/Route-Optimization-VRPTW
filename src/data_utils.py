import os
import sys
import pandas as pd
from pathlib import Path
from IPython.display import Markdown, display

# Ensure we're at the project root by checking for specific marker directories
def ensure_project_root(markers=("src", "data")):
    """
    Ensure that the working directory is set to the project root.

    This function checks if the current working directory contains key
    project folders (e.g., 'src', 'data'). If not, it moves one level up
    in the directory hierarchy until it reaches the project root.
    This prevents file path errors when running notebooks from subfolders.

    Example:
        >>> ensure_project_root()
        Changed working directory to: /path/to/project

    Raises:
        FileNotFoundError: If the project structure markers ('src', 'data')
        are not found in any parent directory.
    """
    current = Path.cwd()
    sys.path.append(str(current))

    if not all((current / marker).exists() for marker in markers):
        os.chdir("..")
        current = Path.cwd()
        sys.path.append(str(current))
        display(Markdown(f"Changed working directory to project root: `{current}`"))
    else:
        display(Markdown(f"Already at project root: `{current}`"))

    return current

# Clean and concatenate VRPTW data into a single DataFrame.
def load_and_concat_vrptw(data_dict):
    """
    Combine all VRPTW instance files into a single standardized DataFrame.

    Each dataset instance (e.g., C101.csv) is cleaned, standardized,
    and augmented with an 'INSTANCE' column for tracking its origin.

    Args:
        data_dict (dict[str, pd.DataFrame]): Dictionary of DataFrames
            loaded by `load_vrptw_dataset()`.

    Returns:
        pd.DataFrame: Combined DataFrame with normalized columns.

    Example:
        >>> vrptw_df = load_and_concat_vrptw(data)
        >>> vrptw_df.head()
    """
    EXPECTED_COLS = ["CUST_NO", "XCOORD", "YCOORD", "DEMAND", "READY_TIME", "DUE_DATE", "SERVICE_TIME"]
    all_data = []

    for name, df in data_dict.items():
        df = df.copy()
        df.columns = [c.strip().replace('.', '').replace(' ', '_').upper() for c in df.columns]
        assert list(df.columns) == EXPECTED_COLS, f"Unexpected columns in {name}: {df.columns}"
        df["INSTANCE"] = name.split(".")[0]
        all_data.append(df)

    vrptw_df = pd.concat(all_data, ignore_index=True)
    vrptw_df["TYPE"] = vrptw_df["INSTANCE"].str.extract(r"([A-Z]+)", expand=False)
    print(f"Loaded and concatenated {len(vrptw_df)} rows, {len(vrptw_df.columns)} columns from {len(data_dict)} instances.")
    return vrptw_df

# Call the function to ensure correct working directory
def load_cleaned_vrptw(processed_path="data/processed/vrptw_cleaned.csv"):
    """
    Load the preprocessed Solomon VRPTW dataset from disk.

    This function assumes the cleaned dataset was saved during
    the EDA or preprocessing step. It serves as a quick access point
    for downstream modeling or feature engineering stages.

    Args:
        clean_path (str): Path to the cleaned CSV file.

    Returns:
        pd.DataFrame: Cleaned VRPTW DataFrame.

    Raises:
        FileNotFoundError: If the cleaned dataset file does not exist.

    Example:
        >>> df = load_cleaned_vrptw()
        >>> df.info()
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_path}. "
            "Please run 01_eda.ipynb to generate it first."
        )
    
    print(f"Loading processed VRPTW dataset from: {processed_path}")
    df = pd.read_csv(processed_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def save_processed_data(df, filename: str, folder: str = "data/processed"):
    """
    Saves a processed DataFrame to CSV in a standardized location.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Output CSV filename.
        folder (str): Directory where to save the file (default: 'data/processed').

    Returns:
        str: Path where the file was saved.
    """
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, filename)
    df.to_csv(output_path, index=False)

    print(f"✅ Data saved successfully at: {output_path}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    return output_path