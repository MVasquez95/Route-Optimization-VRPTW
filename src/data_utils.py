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

def load_processed_data(processed_path: str) -> pd.DataFrame:
    """
    Load a processed dataset from the 'data/processed' directory.

    Automatically detects if the first column represents an index
    (e.g., for square matrices like distance/time) and reloads 
    accordingly. This allows consistent loading across all pipeline stages.

    Args:
        processed_path (str): Full path to the processed CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame from the specified path.

    Raises:
        FileNotFoundError: If the specified dataset does not exist.

    Example:
        >>> df = load_processed_data("data/processed/vrptw_distance_matrix.csv")
        >>> df.shape
        (5656, 5655)
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_path}. "
            "Please ensure the previous pipeline step has been executed successfully."
        )

    print(f"Loading processed dataset from: {processed_path}")

    # Load normally first
    df = pd.read_csv(processed_path)

    # If first column looks like an index (e.g., "Unnamed: 0"), reload properly
    if df.columns[0].startswith("Unnamed"):
        df = pd.read_csv(processed_path, index_col=0)

    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def save_processed_data(df: pd.DataFrame, filename: str, folder: str = "data/processed", enforce_index_columns: bool = False) -> str:
    """
    Save a processed DataFrame to CSV in a standardized location.

    This utility ensures consistency in saving intermediate datasets 
    (e.g., cleaned data, normalized data, or VRPTW matrices). 
    Optionally enforces that index and column labels are aligned, 
    which is useful for square matrices (e.g., distance/time).

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): Output CSV filename (e.g., "vrptw_distance_matrix.csv").
        folder (str, optional): Directory where the file will be saved. 
            Defaults to "data/processed".
        enforce_index_columns (bool, optional): 
            If True, checks whether index labels match column names 
            (common for symmetric matrices). Raises ValueError if not. 
            Defaults to False.

    Returns:
        str: The full path where the file was saved.

    Raises:
        ValueError: If `enforce_index_columns=True` and index/columns mismatch.
    """
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, filename)

    # Optional integrity check for square matrices
    if enforce_index_columns:
        if df.shape[0] != df.shape[1]:
            raise ValueError(
                f"Matrix must be square when enforce_index_columns=True. "
                f"Got {df.shape[0]}x{df.shape[1]}."
            )
        if not all(df.index.astype(str) == df.columns.astype(str)):
            raise ValueError(
                "Index and column labels must match exactly when enforcing harmonization."
            )

    # Save including the index for reproducibility
    df.to_csv(output_path, index=True)

    print(f"✅ Data saved successfully at: {output_path}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    return output_path