# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

# Note: We don't load the model in tests, so we mock the lifespan
# For a full integration test, you would need the models/ directory present.
client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_version" in response.json()