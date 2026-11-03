# Hospital Readmission Prediction System

A production-ready, end-to-end Machine Learning and Healthcare Data Science project to predict 30-day hospital readmissions, understand patient risk trajectories, and empower clinical teams with actionable insights.

---

## 📋 Table of Contents
1. [Implementation Status](#-implementation-status)
2. [Project Overview & Problem Statement](#-project-overview--problem-statement)
3. [Domain & Clinical Significance](#-domain--clinical-significance)
4. [Data Collection & Dataset Attributes](#-data-collection--dataset-attributes)
5. [Data Preprocessing & Cleaning](#-data-preprocessing--cleaning)
6. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
7. [Feature Engineering & Domain Indicators](#-feature-engineering--domain-indicators)
8. [Model Training, Evaluation & Interpretability](#-model-training-evaluation--interpretability)
9. [System Design & API Architecture](#-system-design--api-architecture)
10. [LLM & Generative AI Integration](#-llm--generative-ai-integration)
11. [Cloud Deployment Strategy (AWS & GCP)](#-cloud-deployment-strategy-aws--gcp)
12. [Repository Structure](#-repository-structure)
13. [Installation & Quickstart Guide](#-installation--quickstart-guide)
14. [Compliance & Clinical Disclaimer](#-compliance--clinical-disclaimer)

---

## 🚧 Implementation Status

This project is built incrementally. The table below tracks what is **production-ready**, what is **in progress**, and what is **specified but not yet implemented**. When in doubt, trust this table over the rest of the README — the sections below describe the *target* architecture, not necessarily the *current* state.

| Component | Status | Location |
|---|---|---|
| Data loader | ✅ Implemented | `src/data/loader.py` |
| Preprocessing pipeline (cleaning, hospice filter, patient-grouped split) | ✅ Implemented | `src/data/preprocessor.py` |
| Feature engineering (Charlson, prior utilization, polypharmacy, glycemic fluctuation) | ⏳ In progress | `src/features/build_features.py` |
| Model training (LogReg → XGBoost → LightGBM, calibration, SHAP) | 🕐 Planned | `src/models/train.py` |
| Inference logic | 🕐 Planned | `src/models/predict.py` |
| Explainability utilities | 🕐 Planned | `src/models/explain.py` |
| FastAPI service | 🕐 Planned | `src/api/main.py` |
| LLM summarizer | 🕐 Planned | `src/llm/summarizer.py` |
| Docker + CI/CD | 🕐 Planned | `Dockerfile`, `.github/workflows/ci_cd.yaml` |
| Cloud deployment | 🕐 Planned | — |

**Legend:** ✅ Implemented · ⏳ In progress · 🕐 Planned

---

## 🏥 Project Overview & Problem Statement

Hospital readmission within 30 days of discharge is a critical quality and financial indicator in modern healthcare. Unplanned readmissions often point to incomplete recovery, post-discharge complications, or gaps in transition of care.

### Core Objectives:
- **Risk Identification**: Predict the likelihood of an inpatient being readmitted within 30 days post-discharge.
- **Root Cause & Factor Analysis**: Understand why a patient who appeared well at discharge deteriorates and requires hospital re-entry.
- **Decision Support**: Provide healthcare providers with interpretable risk scores and automated clinical summaries to formulate timely intervention plans.

---

## 🩺 Domain & Clinical Significance

- **Patient Outcomes**: Early identification allows targeted discharge planning, timely medication reconciliation, and scheduled home health visits.
- **Healthcare Economics**: Under programs like the Hospital Readmissions Reduction Program (HRRP), hospitals face significant penalties for excess readmissions.
- **Care Coordination**: Bridges the critical window between inpatient discharge and primary care follow-up.

---

## 📊 Data Collection & Dataset Attributes

### Data Sources
- **Kaggle**: *Hospital Readmissions Dataset* / *Diabetes 130-US Hospitals Dataset* (1999–2008).
- **UCI Machine Learning Repository**: Comprehensive inpatient records representing clinical encounters across multiple facilities.

### Key Clinical & Demographic Features
- **Demographics**: Patient ID, Age group, Gender, Race/Ethnicity.
- **Admission & Encounter Details**: Admission type (Emergency, Urgent, Elective), Admission source, Length of Stay (LOS in days).
- **Clinical History**: Number of outpatient visits, emergency visits, and inpatient admissions in the preceding 12 months.
- **Diagnoses & Comorbidities**: Primary, secondary, and additional ICD-9/ICD-10 diagnosis codes (e.g., Diabetes, Circulatory, Respiratory, Heart Failure).
- **Diagnostics & Laboratory Tests**: Number of lab procedures, glucose tests, HbA1c test results (`Norm`, `>7`, `>8`, `None`).
- **Medications & Treatment**: Number of medications administered, changes in medications (`No`, `Ch`, `Up`, `Down`), insulin management, and specific drug classes.
- **Target Variable**: Readmission status:
  - `<30` days (Primary positive class for high-risk early readmission)
  - `>30` days
  - `NO` (No readmission recorded)

---

## 🧹 Data Preprocessing & Cleaning

1. **Missing Data Handling**:
   - Identify placeholder symbols (e.g., `?`, `None`, empty strings).
   - Impute clinical variables using domain-aware strategies (median for skewed metrics, mode or dedicated `"Unknown"` category for categorical fields).
   - Drop features with excessive sparsity (>50% missing, such as `weight` or `payer_code`).
2. **Data De-duplication & Leakage Prevention**:
   - Filter encounters to handle multiple admissions per patient (e.g., analyzing initial index admissions vs. recurring stays).
   - Ensure strict separation of training, validation, and test splits grouped by `patient_id` to prevent data leakage.
3. **Hospice & Death Filtering**:
   - Encounters ending in death or discharge to hospice (`discharge_disposition_id` ∈ {11, 13, 14, 19, 20, 21}) are excluded — these patients cannot be readmitted, and including them teaches the model a medically meaningless pattern. This removes **2,423 encounters (2.38%)** from the raw 101,766 records.
4. **Class Imbalance Mitigation**:
   - 30-day readmissions typically represent a minority class (~8–12%).
   - Strategies: Synthetic Minority Over-sampling Technique (SMOTE), ADASYN, and algorithm-level cost-sensitive learning (`scale_pos_weight` / class weights). Resampling is applied **only to the training fold**, never before splitting.
5. **Encoding & Scaling**:
   - One-Hot Encoding for nominal variables with low cardinality.
   - Frequency/Target Encoding or grouping for high-cardinality ICD diagnosis codes.
   - Standard / Robust Scaler applied to numerical attributes (e.g., length of stay, medication count).
   - **Encoders and scalers are fit on the training split only** and applied to val/test, to prevent encoding leakage.

### Test Set Construction

The test split contains **only the first (index) encounter** for each patient, while train/val splits contain all encounters (grouped by `patient_nbr` so no patient appears in more than one split). This mimics real deployment, where each new patient arrives without prior encounter history at the facility.

Because readmission risk correlates with prior encounter count, the test set's positive rate (**~9.0%**) is naturally lower than the full dataset's (**~11.4%**). Train/val retain the expected ~11.4% positive rate. Reported metrics therefore measure **cold-start performance** — the operationally relevant scenario — and are not directly comparable to papers reporting random-split results.

Reproducibility note: splits are deterministic (`random_state=42`), so reruns produce identical train/val/test partitions.

---

## 🔍 Exploratory Data Analysis (EDA)

- **Target Distribution**: Quantifying class imbalance across readmission timeframes.
- **Demographic Disparities**: Evaluating readmission risk across age brackets and chronic disease profiles.
- **Length of Stay vs. Readmission**: Analyzing non-linear relationships between inpatient duration and post-discharge vulnerability.
- **Utilization Patterns**: Visualizing how previous emergency and inpatient visits correlate with recurrent admissions.
- **Medication Dynamics**: Correlation between acute medication adjustments (e.g., insulin dosage changes during admission) and readmission probability.

---

## ⚙️ Feature Engineering & Domain Indicators

- **Prior Utilization Index**: Cumulative count and weighted score of historical emergency and inpatient admissions.
- **Comorbidity Scores**: Grouping ICD codes into established clinical indices (Charlson Comorbidity Index / Elixhauser Comorbidity categories).
- **Polypharmacy Indicator**: Binary and tiered flags for patients prescribed ≥5 or ≥10 concurrent medications.
- **Diagnostic Complexity**: Total distinct diagnoses per encounter divided by length of stay.
- **Glycemic Control Fluctuation**: Interaction term between HbA1c status, insulin adjustments, and length of stay.

---

## 🤖 Model Training, Evaluation & Interpretability

### Model Candidates
- **Baseline**: Logistic Regression (L1/L2 regularized) for linear benchmarking.
- **Ensemble Tree Models**: Random Forest, XGBoost, LightGBM, and CatBoost.
- **Deep Learning Baseline**: Multi-Layer Perceptron (MLP) with dropout and batch normalization.

### Clinical Evaluation Metrics
- **ROC-AUC**: Overall discriminatory power across classification thresholds.
- **Precision-Recall AUC (PR-AUC)**: Primary benchmark due to heavy class imbalance.
- **Sensitivity / Recall (Target ≥80%)**: Prioritized to avoid false negatives (missing an at-risk patient carries high clinical cost).
- **Calibration (Brier Score / Calibration Curves)**: Ensuring predicted probabilities reflect true empirical risk.

### Model Explainability
- **Global Explanations**: Tree SHAP (SHapley Additive exPlanations) to rank overall feature importance across the patient cohort.
- **Local Explanations**: Patient-level SHAP force plots and LIME to pinpoint why an individual patient is flagged as high-risk.

---

## 🏗️ System Design & API Architecture

```
[ Electronic Health Record (EHR) / Client ]
                     │
                     ▼
             [ API Gateway ]
                     │
                     ▼
          [ FastAPI Microservice ]
         ┌───────────┴───────────┐
         ▼                       ▼
 [ Preprocessing Pipeline ]  [ Authentication & Rate Limiting ]
         │
         ▼
  [ Model Server (ONNX / Scikit-learn / XGBoost) ]
         │
         ├───► [ Prediction & Risk Score Engine ]
         │
         ▼
[ LLM Explainer Service (LangChain / Claude / GPT) ]
         │
         ▼
[ Audit & Prediction Log DB (PostgreSQL / DynamoDB) ]
```


### REST API Endpoints
- `GET /health`: Service health and model version check.
- `POST /v1/predict`: Single patient real-time readmission risk scoring.
- `POST /v1/predict/batch`: High-throughput batch inference for discharge queues.
- `POST /v1/explain`: Returns SHAP attribution values and generated clinical summary.

---

## 🧠 LLM & Generative AI Integration

1. **Unstructured Clinical Notes Mining**:
   - Extract clinical risk factors, social determinants of health (SDOH), and lifestyle notes from unstructured discharge summaries and physician progress notes using LLMs.
2. **Automated Discharge Risk Summaries**:
   - Translate numerical risk scores and top SHAP features into clear, clinical natural-language narratives for attending physicians.
3. **Patient-Facing Transition Plans**:
   - Generate tailored post-discharge checklists and medication instructions in plain language suited to patient literacy levels.

---

## ☁️ Cloud Deployment Strategy (AWS & GCP)

### AWS Deployment Path
- **Containerization**: Docker multi-stage build.
- **Inference Hosting**: AWS SageMaker Real-Time Endpoint or AWS ECS (Fargate) behind an Application Load Balancer (ALB).
- **Data & Model Registry**: Amazon S3 for model artifacts and training sets, AWS ECR for container images.
- **Monitoring**: AWS CloudWatch + Evidently AI for model drift and input data schema drift.

### GCP Deployment Path
- **Inference Hosting**: Google Cloud Run (serverless container) or Vertex AI Endpoints.
- **Storage & Registry**: Google Cloud Storage (GCS) and Artifact Registry.
- **Pipeline Orchestration**: Vertex AI Pipelines / Kubeflow for automated model retraining.
- **Monitoring**: Google Cloud Operations Suite with custom drift metrics.

---

## 📁 Repository Structure

```
hospital-readmission-prediction/
├── README.md # Project documentation and specifications
├── requirements.txt # Production and training dependencies
├── Dockerfile # Container definition for API deployment
├── config/
│ ├── model_config.yaml # Hyperparameters and feature definitions
│ └── api_config.yaml # Server and runtime configuration
├── data/
│ ├── raw/ # Raw datasets from Kaggle / UCI
│ └── processed/ # Cleaned, encoded, and transformed splits
├── notebooks/
│ ├── 01_data_exploration_eda.ipynb # Comprehensive EDA and visualizations
│ ├── 02_feature_engineering.ipynb # Feature generation and selection
│ └── 03_model_benchmarking.ipynb # Model training, tuning, and evaluation
├── src/
│ ├── init.py
│ ├── data/
│ │ ├── loader.py # Ingestion scripts
│ │ └── preprocessor.py # Cleaning and transformation pipelines
│ ├── features/
│ │ └── build_features.py # Clinical indices and polypharmacy logic
│ ├── models/
│ │ ├── train.py # Model training and cross-validation
│ │ ├── predict.py # Inference logic
│ │ └── explain.py # SHAP and explainability utilities
│ ├── llm/
│ │ └── summarizer.py # LLM integration for clinical explanations
│ └── api/
│ ├── main.py # FastAPI application entrypoint
│ └── schemas.py # Pydantic request/response schemas
├── tests/
│ ├── test_preprocessor.py # Unit tests for preprocessing
│ └── test_api.py # API integration tests
└── .github/
└── workflows/
└── ci_cd.yaml # Automated testing, linting, and deployment
```

---

## 🚀 Installation & Quickstart Guide

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/hospital-readmission-prediction.git
cd hospital-readmission-prediction
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2. Download the Dataset
Download the UCI Diabetes 130-US Hospitals dataset from:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Extract and place diabetic_data.csv into data/raw/.

### 2. Run Data Pipeline & Model Training
```bash
python -m src.data.loader
python -m src.data.preprocessor
python -m src.features.build_features
python -m src.models.train --model xgboost --evaluate
```

### 3. Launch the Prediction API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API docs available at: `http://localhost:8000/docs`

### 4. Build and Run via Docker
```bash
docker build -t hospital-readmission-api:v1 .
docker run -p 8000:8000 hospital-readmission-api:v1
```

---

## 📜 Compliance & Clinical Disclaimer
*This system is intended for research, decision support, and operational workflow enhancement. Predictions must be verified by licensed medical professionals and are not intended to replace clinical judgment.*

*Model outputs do not constitute medical advice. Any deployment in a clinical setting requires IRB review, HIPAA compliance validation, prospective validation on local patient cohorts, and continuous monitoring for demographic bias and calibration drift.*
