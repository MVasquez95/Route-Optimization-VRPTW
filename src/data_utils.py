import os
import sys
import pandas as pd
from pathlib import Path
from IPython.display import Markdown, display

def ensure_project_root(markers=("src", "data")):
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

def load_cleaned_vrptw(processed_path="data/processed/vrptw_cleaned.csv"):
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"❌ Processed dataset not found at {processed_path}. "
            "Please run 01_eda.ipynb to generate it first."
        )
    
    print(f"Loading processed VRPTW dataset from: {processed_path}")
    df = pd.read_csv(processed_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df