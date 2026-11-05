import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import xgboost as xgb
import numpy as np

from src.api.main import app

@pytest.fixture
def client():
    # 1. Create a dummy XGBoost model
    X_dummy = np.random.rand(10, 5)
    y_dummy = np.random.randint(0, 2, 10)
    dummy_model = xgb.XGBClassifier(n_estimators=1, max_depth=1)
    dummy_model.fit(X_dummy, y_dummy)
    
    # 2. Inject the dummy models directly into the app's global state
    app.ml_models = {
        "model": dummy_model,
        "preprocessor": MagicMock(),
        "explainer": MagicMock()
    }
    
    # 3. We DO NOT use `with TestClient(app)` because that triggers the lifespan.
    # We just instantiate it.
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"