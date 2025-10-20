import os
from fastapi.testclient import TestClient

# Ensure we run in test environment
os.environ.setdefault("ENVIRONMENT", "development")

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "version" in body
    assert "environment" in body
