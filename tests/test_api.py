import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import xgboost as xgb
import numpy as np

from src.api.main import app

@pytest.fixture
def client():
    # 1. Create a dummy XGBoost model so SHAP doesn't crash
    X_dummy = np.random.rand(10, 5)
    y_dummy = np.random.randint(0, 2, 10)
    dummy_model = xgb.XGBClassifier(n_estimators=1, max_depth=1)
    dummy_model.fit(X_dummy, y_dummy)
    
    # 2. Create dummy preprocessor and explainer
    dummy_preprocessor = MagicMock()
    dummy_preprocessor.get_feature_names_out.return_value = ["f1", "f2", "f3", "f4", "f5"]
    
    # 3. Patch the lifespan so it injects our dummies instead of loading from disk
    mock_models = {
        "model": dummy_model,
        "preprocessor": dummy_preprocessor,
        "explainer": MagicMock()
    }
    
    with patch.dict("src.api.main.ml_models", mock_models):
        with TestClient(app) as c:
            yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_version"] == "v0.1.0"