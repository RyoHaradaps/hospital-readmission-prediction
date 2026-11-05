import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import src.api.main as main_module

# We use TestClient without the lifespan manager to avoid loading the actual models
@pytest.fixture
def client():
    # Patch the model loading part so it doesn't crash without joblib files
    with patch("joblib.load") as mock_load:
        # Return a fake preprocessor and model for the test
        mock_load.return_value = None
        # We bypass the lifespan by directly using TestClient
        with TestClient(main_module.app) as c:
            yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_version"] == "v0.1.0"