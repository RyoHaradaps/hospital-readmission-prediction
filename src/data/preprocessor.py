# src/data/preprocessor.py
"""
Leakage-safe preprocessing pipeline for UCI Diabetes 130-US Hospitals dataset.

Usage:
    python -m src.data.preprocessor

Produces (in data/processed/):
    - train.parquet, val.parquet, test.parquet   (cleaned, patient-grouped splits)
    - patient_groups.parquet                     (patient_nbr → split mapping)
    - preprocessing_report.json                  (audit trail of rows dropped)

Design decisions:
    1. Exclude hospice/death discharges — these patients cannot be readmitted.
    2. Binary target: <30d = positive, else negative.
    3. Group-split by patient_nbr (GroupShuffleSplit, 85/15/15).
    4. Drop high-sparsity columns (weight) and ID columns.
    5. Missing clinical values encoded as "Unknown" category (not median-imputed)
       because "test not ordered" is clinically meaningful.
    6. Only FIRST encounter per patient appears in the TEST set (mimics deployment).
    7. encounter_id is KEPT through splitting (needed for chronological sort),
       then dropped in save_splits().
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ─── Paths ─────────────────────────────────────────────────────────────────
RAW_PATH       = Path("data/raw/diabetic_data.csv")
PROCESSED_DIR  = Path("data/processed")
REPORT_PATH    = PROCESSED_DIR / "preprocessing_report.json"

# ─── Constants ─────────────────────────────────────────────────────────────
HOSPICE_DEATH_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

# NOTE: encounter_id and patient_nbr are NOT dropped here — they are needed
# during the split (chronological sort + grouping). They are dropped in
# save_splits() right before writing parquet files.
DROP_COLUMNS = [
    "weight",       # 96.9% missing
    "examide",      # constant (all "No")
    "citoglipton",  # constant (all "No")
]

# ─── Pipeline stages ──────────────────────────────────────────────────────

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load CSV, treating '?' / 'None' / empty as missing."""
    return pd.read_csv(path, na_values=["?", "None", ""], low_memory=False)


def filter_non_readmittable(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove encounters where readmission is impossible (hospice/death)."""
    mask = df["discharge_disposition_id"].isin(HOSPICE_DEATH_DISPOSITIONS)
    n_dropped = int(mask.sum())
    return df.loc[~mask].copy(), n_dropped


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: <30 days = 1, >30 or NO = 0."""
    df = df.copy()
    df["target_30d"] = (df["readmitted"] == "<30").astype("int8")
    df = df.drop(columns=["readmitted"])
    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constants + ultra-sparse columns. IDs dropped later, in save_splits."""
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def impute_categorical_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical values with explicit 'Unknown' string."""
    df = df.copy()
    # Explicit 'string' selection silences the Pandas4 deprecation warning.
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
    return df


def group_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-grouped split: no patient appears in >1 split.
    Test set contains ONLY the first (index) encounter per patient.
    """
    if "patient_nbr" not in df.columns:
        raise ValueError("patient_nbr must exist at split time.")
    if "encounter_id" not in df.columns:
        raise ValueError("encounter_id must exist at split time (needed for sort).")

    # Split 1: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["patient_nbr"]))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df      = df.iloc[test_idx].copy()

    # Split 2: train vs val
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size / (1 - test_size),
        random_state=seed + 1,
    )
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df["patient_nbr"]))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df   = train_val_df.iloc[val_idx].copy()

    # TEST: only first encounter per patient (index admission)
    test_df = (
        test_df.sort_values("encounter_id")
               .groupby("patient_nbr", as_index=False)
               .first()
    )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path = PROCESSED_DIR,
) -> dict:
    """Persist splits as parquet + return summary stats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save patient → split mapping (before dropping patient_nbr)
    groups = pd.concat([
        train_df[["patient_nbr"]].assign(split="train"),
        val_df[["patient_nbr"]].assign(split="val"),
        test_df[["patient_nbr"]].assign(split="test"),
    ]).drop_duplicates()
    groups.to_parquet(out_dir / "patient_groups.parquet", index=False)

    id_cols = [c for c in ["patient_nbr", "encounter_id"] if c in train_df.columns]

    stats = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df_out = df.drop(columns=id_cols)
        df_out.to_parquet(out_dir / f"{name}.parquet", index=False)

        stats[name] = {
            "n_rows":            int(len(df_out)),
            "n_unique_patients": int(df["patient_nbr"].nunique()),
            "n_positive":        int(df["target_30d"].sum()),
            "positive_rate":     round(float(df["target_30d"].mean()), 4),
            "n_features":        int(df_out.shape[1]),
        }
    return stats


def run() -> None:
    """End-to-end preprocessing pipeline."""
    print("Loading raw data …")
    df = load_raw()
    n_raw = len(df)
    print(f"  Raw shape: {df.shape}  (target <30d rate: "
          f"{(df['readmitted'] == '<30').mean():.4f})")

    print("Filtering non-readmittable discharges (hospice / expired) …")
    df, n_hospice = filter_non_readmittable(df)
    print(f"  Removed {n_hospice} encounters ({n_hospice / n_raw:.2%})")

    print("Creating binary target (readmitted < 30 days) …")
    df = create_target(df)

    print("Dropping constant / ultra-sparse columns …")
    df = drop_unwanted_columns(df)

    print("Filling categorical unknowns …")
    df = impute_categorical_unknowns(df)

    print("Patient-grouped train / val / test split …")
    train_df, val_df, test_df = group_split(df)

    print("Saving parquet splits …")
    stats = save_splits(train_df, val_df, test_df)

    report = {
        "n_raw": n_raw,
        "n_after_hospice_filter": int(n_raw - n_hospice),
        "n_hospice_death_removed": n_hospice,
        "positive_rate_overall": round(float(df["target_30d"].mean()), 4),
        "splits": stats,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n─── Preprocessing complete ───")
    print(json.dumps(report, indent=2))
    print(f"\nArtifacts saved to: {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    run()