import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    # This tells TestClient NOT to run the startup lifespan
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_version"] == "v0.1.0"