# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class PatientRecord(BaseModel):
    """Represents a single patient encounter at discharge."""
    
    # Demographics
    race: str = "Unknown"
    gender: Literal["Male", "Female", "Unknown/Invalid"] = "Unknown/Invalid"
    age: str = Field(..., description="Age bracket, e.g., '[50-60)'")
    weight: str = "Unknown"

    # Admission details
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = Field(..., ge=1, le=14, description="Length of stay in days")
    payer_code: str = "Unknown"
    medical_specialty: str = "Unknown"

    # Clinical
    num_lab_procedures: int = Field(..., ge=0)
    num_procedures: int = Field(..., ge=0)
    num_medications: int = Field(..., ge=0)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    diag_1: str = Field(..., description="Primary ICD-9 code")
    diag_2: str = "Unknown"
    diag_3: str = "Unknown"
    number_diagnoses: int = Field(..., ge=1)

    # Labs & Meds
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    insulin: Literal["No", "Steady", "Up", "Down"] = "No"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["Yes", "No"] = "No"

    # We only ask for a few of the medication columns to keep the API payload small
    # (defaults to "No" for the rest in main.py)
    metformin: Literal["No", "Steady", "Up", "Down"] = "No"


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Probability of readmission < 30 days")
    risk_tier: Literal["Low", "Medium", "High"]
    model_version: str
    clinical_summary: str


class ExplanationResponse(BaseModel):
    risk_score: float
    top_risk_factors: dict[str, float] = Field(..., description="Feature name -> SHAP value")
    clinical_summary: str