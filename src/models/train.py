# src/models/train.py
"""
Model training, evaluation, and explainability.
Usage: python -m src.models.train
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
CONFIG_PATH = Path("config/model_config.yaml")
MODELS_DIR = Path("models")

def load_data():
    print("Loading data …")
    train = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val_features.parquet")
    
    X_train = train.drop(columns=["target_30d"])
    y_train = train["target_30d"]
    X_val = val.drop(columns=["target_30d"])
    y_val = val["target_30d"]
    
    return X_train, y_train, X_val, y_val

def get_preprocessor(X_train):
    """Define transformations for numeric vs categorical columns."""
    numeric_cols = X_train.select_dtypes(include=["int64", "float64", "int8", "int32"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    # We apply scaling to numeric, and One-Hot encoding to categories
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        verbose_feature_names_out=False
    )
    return preprocessor

def evaluate_model(model, X_val, y_val, model_name, threshold=0.5):
    """Calculate clinical evaluation metrics."""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Primary metric: PR-AUC
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)
    
    # Find threshold for 80% recall
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    # Find the first index where recall >= 0.80
    valid_idx = np.where(recalls >= 0.80)[0]
    if len(valid_idx) > 0:
        # We want the highest precision among those achieving 80% recall
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
        thresh_80 = thresholds[best_idx] if best_idx < len(thresholds) else threshold
        prec_at_80 = precisions[best_idx]
    else:
        thresh_80 = 1.0
        prec_at_80 = 0.0

    metrics = {
        "model": model_name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "recall_at_0.5": float(round(np.sum((y_pred == 1) & (y_val == 1)) / np.sum(y_val == 1), 4)),
        "precision_at_80_recall": float(round(prec_at_80, 4)),
        "threshold_for_80_recall": float(round(thresh_80, 4))
    }
    return metrics

def main():
    X_train, y_train, X_val, y_val = load_data()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    
    # 1. Fit Preprocessor on Train ONLY
    print("Fitting preprocessor …")
    preprocessor = get_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Save the preprocessor for inference
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    print(f"Saved preprocessor to {MODELS_DIR}/preprocessor.joblib")

    # 2. Define Models
    models = {
        "LogisticRegression": LogisticRegression(**config["logistic_regression"]),
        "XGBoost": XGBClassifier(**config["xgboost"]),
        "LightGBM": LGBMClassifier(**config["lightgbm"])
    }
    
    results = []
    best_model = None
    best_pr_auc = 0

    # 3. Train and Evaluate
    for name, model in models.items():
        print(f"\nTraining {name} …")
        model.fit(X_train_processed, y_train)
        
        metrics = evaluate_model(model, X_val_processed, y_val, name)
        results.append(metrics)
        
        print(f"  PR-AUC: {metrics['pr_auc']} | ROC-AUC: {metrics['roc_auc']}")
        print(f"  Precision at 80% Recall: {metrics['precision_at_80_recall']} (threshold {metrics['threshold_for_80_recall']})")
        
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_model = model
            best_model_name = name

    print(f"\n🏆 Best Model: {best_model_name} (PR-AUC: {best_pr_auc})")
    
    # 4. Save the best model
    joblib.dump(best_model, MODELS_DIR / "model.joblib")
    print(f"Saved model to {MODELS_DIR}/model.joblib")
    
    # Save metrics report
    report = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "metrics": results
    }
    (MODELS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2))

    # 5. Explainability (SHAP) - we use TreeSHAP for XGBoost/LightGBM
    if best_model_name in ["XGBoost", "LightGBM"]:
        print("\nGenerating SHAP summary plot …")
        # Get feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()
        
        # Create TreeExplainer
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_val_processed)
        
        # Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_val_processed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
        print(f"Saved SHAP plot to {MODELS_DIR}/shap_summary.png")

if __name__ == "__main__":
    main()