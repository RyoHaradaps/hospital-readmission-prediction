# src/features/build_features.py
"""
Clinical feature engineering for readmission prediction.

Adds domain-informed features on top of the cleaned splits:
    - Charlson Comorbidity Index (from ICD-9 diag_1/2/3)
    - Prior Utilization Score (weighted historical admissions)
    - Polypharmacy flags (≥5, ≥10 concurrent meds)
    - Diagnostic Complexity (diagnoses per day)
    - Glycemic Fluctuation interaction term
    - ICD-9 → CCS-style broad disease category mapping
    - Medication change signals (change, diabetesMed, insulin)

Usage:
    python -m src.features.build_features

Reads:  data/processed/{train,val,test}.parquet
Writes: data/processed/{train,val,test}_features.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "val", "test"]

# ─── ICD-9 → Charlson Comorbidity Index mapping (weights from Charlson 1987) ─
# Each entry: (regex/prefix match, Charlson weight)
# Reference: Quan et al. 2005 ICD-9 adaptation
CHARLSON_RULES = [
    (r"^410|^412|^414",            1),  # Myocardial infarction, CAD
    (r"^428",                      1),  # Congestive heart failure
    (r"^440|^441|^443|^444|^447",  1),  # Peripheral vascular disease
    (r"^430|^431|^432|^433|^434|^436|^437|^438", 1),  # Cerebrovascular
    (r"^290",                      1),  # Dementia
    (r"^49[0-3]|^49[4-6]",         1),  # COPD
    (r"^710|^714|^715|^716|^718",  1),  # Rheumatic / connective tissue
    (r"^531|^532|^533|^534",       1),  # Peptic ulcer
    (r"^571|^572|^573|^574|^575",  1),  # Mild liver disease
    (r"^250",                      1),  # Diabetes (uncomplicated)
    (r"^250[4-7]",                 2),  # Diabetes with complications
    (r"^342|^434|^436|^437|^438",  2),  # Hemiplegia
    (r"^582|^583|^585|^586|^588",  2),  # Renal disease
    (r"^140|^141|^142|^143|^144|^145|^146|^147|^148|^149|^150|^151|^152|^153|^154|^155|^156|^157|^158|^159|^160|^161|^162|^163|^164|^165|^170|^171|^172|^174|^175|^176|^179|^180|^181|^182|^183|^184|^185|^186|^187|^188|^189|^190|^191|^192|^193|^194|^195|^200|^201|^202|^203|^204|^205|^206|^207|^208", 2),  # Malignancy
    (r"^456|^457|^458",            3),  # Moderate/severe liver disease
    (r"^196|^197|^198|^199",       6),  # Metastatic solid tumor
    (r"^042|^043|^044",            6),  # HIV/AIDS
]

# ─── ICD-9 → broad disease category (CCS-style, simplified) ──────────────
# Purpose: reduce ~700 raw codes to ~15 meaningful groups for tree models.
ICD9_CATEGORY_MAP = {
    "circulatory":   r"^39[0-9]|^40[0-9]|^41[0-9]|^42[0-9]|^43[0-9]|^44[0-9]|^45[0-9]",
    "respiratory":   r"^46[0-9]|^47[0-9]|^48[0-9]|^49[0-9]|^50[0-9]|^51[0-9]",
    "digestive":     r"^52[0-9]|^53[0-9]|^54[0-9]|^55[0-9]|^56[0-9]|^57[0-9]",
    "genitourinary": r"^58[0-9]|^59[0-9]|^60[0-9]|^61[0-9]|^62[0-9]|^63[0-9]",
    "complications": r"^96[0-9]|^97[0-9]|^98[0-9]|^99[0-9]",
    "musculoskeletal": r"^71[0-9]|^72[0-9]|^73[0-9]",
    "neoplasms":     r"^1[4-9][0-9]|^20[0-9]|^2[1-3][0-9]",
    "endocrine":     r"^24[0-9]|^25[0-9]|^26[0-9]|^27[0-9]",
    "mental":        r"^29[0-9]|^30[0-9]|^31[0-9]",
    "nervous":       r"^32[0-9]|^33[0-9]|^34[0-9]|^35[0-9]|^36[0-9]",
    "injury":        r"^8[0-9][0-9]|^9[0-4][0-9]",
    "external":      r"^E",           # external causes
    "supplementary": r"^V",           # V-codes (factors influencing health)
}


def _icd9_category(code: str) -> str:
    """Map a raw ICD-9 code string to a broad category."""
    if code == "Unknown" or not isinstance(code, str):
        return "Unknown"
    for category, pattern in ICD9_CATEGORY_MAP.items():
        if pd.Series([code]).str.match(pattern).iloc[0]:
            return category
    return "other"


def _charlson_weight(code: str) -> int:
    """Return Charlson weight for a single ICD-9 code (0 if none)."""
    if code == "Unknown" or not isinstance(code, str):
        return 0
    for pattern, weight in CHARLSON_RULES:
        if pd.Series([code]).str.match(pattern).iloc[0]:
            return weight
    return 0


def add_charlson(df: pd.DataFrame) -> pd.DataFrame:
    """Add Charlson Comorbidity Index from diag_1/2/3 (max weight per dx, sum)."""
    df = df.copy()
    weights = []
    for _, row in df[["diag_1", "diag_2", "diag_3"]].iterrows():
        # Per-patient: take max weight across their three codes, then sum
        # (Standard Charlson practice: each condition counted once, use max severity)
        per_code = [_charlson_weight(row[c]) for c in ["diag_1", "diag_2", "diag_3"]]
        # Sum of distinct weights (approximation — full impl would dedupe by condition)
        weights.append(sum(per_code))
    df["charlson_index"] = weights
    return df


def add_icd_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add broad disease category for primary diagnosis only (diag_1)."""
    df = df.copy()
    df["diag_1_category"] = df["diag_1"].apply(_icd9_category)
    return df


def add_prior_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted prior utilization score.
    Inpatient > emergency > outpatient (based on cost / clinical severity).
    """
    df = df.copy()
    df["prior_utilization_score"] = (
        3.0 * df["number_inpatient"]
        + 2.0 * df["number_emergency"]
        + 1.0 * df["number_outpatient"]
    )
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype("int8")
    df["has_prior_emergency"] = (df["number_emergency"] > 0).astype("int8")
    return df


def add_polypharmacy(df: pd.DataFrame) -> pd.DataFrame:
    """Polypharmacy tier flags (≥5 meds, ≥10 meds)."""
    df = df.copy()
    df["polypharmacy_5"]  = (df["num_medications"] >= 5).astype("int8")
    df["polypharmacy_10"] = (df["num_medications"] >= 10).astype("int8")
    return df


def add_diagnostic_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct diagnoses per day of stay (comorbidity density)."""
    df = df.copy()
    df["diagnostic_complexity"] = df["number_diagnoses"] / df["time_in_hospital"].clip(lower=1)
    return df


def add_glycemic_fluctuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction: HbA1c status × insulin adjustment × length of stay.
    Encodes "poorly controlled diabetic whose insulin was changed mid-stay"
    — a known 30-day readmission risk pattern.
    """
    df = df.copy()
    # Ordinal encode A1C result: None/Unknown = 0, Norm = 1, >7 = 2, >8 = 3
    a1c_map = {"Unknown": 0, "None": 0, "Norm": 1, ">7": 2, ">8": 3}
    a1c_sev = df["A1Cresult"].map(a1c_map).fillna(0).astype("int8")

    # Insulin change: No=0, Steady=0, Up=1, Down=1
    insulin_changed = df["insulin"].isin(["Up", "Down"]).astype("int8")

    df["glycemic_fluctuation"] = a1c_sev * insulin_changed * df["time_in_hospital"]
    df["a1c_severity"] = a1c_sev
    return df


def add_medication_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidated medication change flags."""
    df = df.copy()
    df["any_med_change"] = (df["change"] == "Ch").astype("int8")
    df["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype("int8")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature builders in sequence."""
    df = add_charlson(df)
    df = add_icd_categories(df)
    df = add_prior_utilization(df)
    df = add_polypharmacy(df)
    df = add_diagnostic_complexity(df)
    df = add_glycemic_fluctuation(df)
    df = add_medication_signals(df)
    return df


def run() -> None:
    """Process all three splits."""
    print("Building clinical features on all splits …\n")
    for split in SPLITS:
        in_path  = PROCESSED_DIR / f"{split}.parquet"
        out_path = PROCESSED_DIR / f"{split}_features.parquet"
        df = pd.read_parquet(in_path)
        n_before = df.shape[1]
        df = build_features(df)
        df.to_parquet(out_path, index=False)
        print(f"  {split:5s}: {df.shape[0]:>6d} rows | "
              f"{n_before} → {df.shape[1]} features | "
              f"saved to {out_path.name}")

    # Quick sanity check on train
    train = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    print("\n─── New feature sanity check (train) ───")
    print(f"  charlson_index:      mean={train['charlson_index'].mean():.2f}, "
          f"max={train['charlson_index'].max()}")
    print(f"  prior_utilization:   mean={train['prior_utilization_score'].mean():.2f}, "
          f"max={train['prior_utilization_score'].max()}")
    print(f"  polypharmacy_5 rate: {train['polypharmacy_5'].mean():.2%}")
    print(f"  polypharmacy_10 rate:{train['polypharmacy_10'].mean():.2%}")
    print(f"\n  diag_1_category distribution (top 8):")
    print(train["diag_1_category"].value_counts().head(8).to_string())


if __name__ == "__main__":
    run()