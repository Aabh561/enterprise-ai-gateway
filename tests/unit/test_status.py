import os

from fastapi.testclient import TestClient

# Use default API key from settings if not provided
DEFAULT_API_KEY = "your-super-secret-api-key-here"
os.environ.setdefault("ENVIRONMENT", "test")

from app.main import app  # noqa: E402

client = TestClient(app)


def _patch_vector_service(monkeypatch):
    from app.routers import v1 as api_v1

    class DummyVS:
        async def health_check(self):
            return {"overall": True}

    async def fake_get_vector_service():
        return DummyVS()

    monkeypatch.setattr(api_v1, "get_vector_service", fake_get_vector_service)
    # Ensure rate limit is generous for this test module
    from app import config as cfg

    cfg.get_settings().rate_limiting.per_minute = 1000


def test_status_requires_authentication(monkeypatch):
    _patch_vector_service(monkeypatch)
    resp = client.get("/api/v1/status")
    assert resp.status_code == 401


def test_status_with_api_key_succeeds(monkeypatch):
    _patch_vector_service(monkeypatch)
    resp = client.get("/api/v1/status", headers={"X-API-Key": DEFAULT_API_KEY})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "services" in data
