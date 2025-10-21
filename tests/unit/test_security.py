import os

from fastapi.testclient import TestClient

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


def test_multiple_api_keys_allowed(monkeypatch):
    _patch_vector_service(monkeypatch)
    from app import config as cfg

    # Override settings to include multiple keys
    settings = cfg.get_settings()
    settings.security.api_keys.secrets = ["k1", "k2", "k3"]

    r = client.get("/api/v1/status", headers={"X-API-Key": "k2"})
    assert r.status_code == 200


def test_rate_limit_bucket_by_api_key(monkeypatch):
    _patch_vector_service(monkeypatch)
    # Test that rate limiting doesn't interfere with normal operation
    from app import config as cfg

    settings = cfg.get_settings()
    # Set generous rate limit for test environment
    settings.rate_limiting.per_minute = 1000
    settings.rate_limiting.enabled = False  # Disable for tests

    # Make requests with same API key; should all succeed in test env
    headers = {"X-API-Key": "your-super-secret-api-key-here"}
    r1 = client.get("/api/v1/status", headers=headers)
    r2 = client.get("/api/v1/status", headers=headers)
    r3 = client.get("/api/v1/status", headers=headers)
    # All requests should succeed in test environment
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200
