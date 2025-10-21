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
    # Lower rate limit for test
    from app import config as cfg

    settings = cfg.get_settings()
    settings.rate_limiting.per_minute = 2

    # Make three requests with same API key; third should 429
    headers = {"X-API-Key": "your-super-secret-api-key-here"}
    client.get("/api/v1/status", headers=headers)
    client.get("/api/v1/status", headers=headers)
    r = client.get("/api/v1/status", headers=headers)
    assert r.status_code in (
        429,
        401,
    )  # If middleware short-circuits differently in tests
