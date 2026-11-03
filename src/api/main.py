# src/api/main.py
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PatientRecord, PredictionResponse, ExplanationResponse
from src.features.build_features import build_features

MODELS_DIR = Path("models")
MODEL_VERSION = "v0.1.0"

# Global variables to hold our artifacts
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor on startup."""
    print("Loading artifacts …")
    try:
        ml_models["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.joblib")
        ml_models["model"] = joblib.load(MODELS_DIR / "model.joblib")
        # Initialize SHAP explainer (TreeExplainer is fast)
        ml_models["explainer"] = shap.TreeExplainer(ml_models["model"])
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e
    yield
    ml_models.clear()

app = FastAPI(
    title="Hospital Readmission Prediction API",
    version=MODEL_VERSION,
    lifespan=lifespan
)

# Allow cross-origin requests (so your future React/Streamlit frontend can talk to it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _prepare_patient_data(patient: PatientRecord) -> pd.DataFrame:
    """Convert a Pydantic model into the DataFrame expected by build_features."""
    # 1. Convert to dictionary
    data = patient.model_dump()
    
    # 2. Add missing columns (the ones we dropped from schema to keep payload small)
    # Default all unmentioned medication columns to "No"
    med_cols = [
        "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    for col in med_cols:
        if col not in data:
            data[col] = "No"

    # 3. Create DataFrame (1 row)
    df = pd.DataFrame([data])
    
    # 4. Apply the exact same feature engineering pipeline
    # (This adds charlson_index, prior_utilization_score, diag_1_category, etc.)
    df = build_features(df)
    
    # 5. Drop raw ID columns to match training data
    id_cols = ["encounter_id", "patient_nbr", "readmitted", "target_30d"]
    df = df.drop(columns=[c for c in id_cols if c in df.columns])
    
    return df

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": MODEL_VERSION}

@app.post("/v1/predict", response_model=PredictionResponse)
def predict(patient: PatientRecord):
    try:
        df = _prepare_patient_data(patient)
        
        # Preprocess and predict
        X_processed = ml_models["preprocessor"].transform(df)
        risk_prob = float(ml_models["model"].predict_proba(X_processed)[0, 1])
        
        # Determine risk tier
        if risk_prob >= 0.6:
            tier = "High"
        elif risk_prob >= 0.3:
            tier = "Medium"
        else:
            tier = "Low"
            
        return PredictionResponse(
            risk_score=round(risk_prob, 4),
            risk_tier=tier,
            model_version=MODEL_VERSION,
            clinical_summary=f"Patient has a {risk_prob:.1%} risk of readmission. Risk Tier: {tier}."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/explain", response_model=ExplanationResponse)
def explain(patient: PatientRecord):
    try:
        df = _prepare_patient_data(patient)
        X_processed = ml_models["preprocessor"].transform(df)
        
        risk_prob = float(ml_models["model"].predict_proba(X_processed)[0, 1])
        
        # Get SHAP values for this single patient
        # (shap_values shape: [1, num_features])
        shap_values = ml_models["explainer"].shap_values(X_processed)[0]
        feature_names = ml_models["preprocessor"].get_feature_names_out()
        
        # Pair features with their SHAP impact, sort by magnitude
        impacts = {name: float(val) for name, val in zip(feature_names, shap_values)}
        # Filter to only top 5 most impactful features
        top_impacts = dict(sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
                # Split factors into risk-increasing and risk-decreasing
        risk_increasers = [k.replace("_", " ") for k, v in top_impacts.items() if v > 0]
        risk_decreasers = [k.replace("_", " ") for k, v in top_impacts.items() if v < 0]

        if risk_increasers:
            summary = f"Top factors increasing risk: {', '.join(risk_increasers)}."
        else:
            summary = f"Top protective factors keeping risk low: {', '.join(risk_decreasers)}."

        return ExplanationResponse(
            risk_score=round(risk_prob, 4),
            top_risk_factors=top_impacts,
            clinical_summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))