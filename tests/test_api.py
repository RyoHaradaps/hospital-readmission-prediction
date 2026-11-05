import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app

@pytest.fixture
def client():
    # Patch joblib.load so the tests don't need the actual model files
    with patch("joblib.load") as mock_load:
        mock_load.return_value = None
        with TestClient(app) as c:
            yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"