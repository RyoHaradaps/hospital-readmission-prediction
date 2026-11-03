# src/data/loader.py
"""
Loads the raw UCI Diabetes 130-US Hospitals dataset.
Run: python -m src.data.loader
"""
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/diabetic_data.csv")

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load CSV, treating '?' and 'None' as missing."""
    df = pd.read_csv(path, na_values=["?", "None", ""], low_memory=False)
    return df

if __name__ == "__main__":
    df = load_raw()
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):\n{df.columns.tolist()}")
    print(f"\nTarget distribution:\n{df['readmitted'].value_counts(dropna=False)}")
    print(f"\nUnique patients: {df['patient_nbr'].nunique()}")
    print(f"Encounters: {len(df)}")
    print(f"\nMissing per column (top 15):\n{df.isna().sum().sort_values(ascending=False).head(15)}")