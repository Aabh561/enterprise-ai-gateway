from __future__ import annotations
import os
import sys
import time
import json
from urllib.request import urlopen, Request

BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "your-super-secret-api-key-here")

def get(path: str, headers: dict | None = None):
    req = Request(f"{BASE_URL}{path}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        return resp.getcode(), body

if __name__ == "__main__":
    # Health
    code, body = get("/health")
    assert code == 200, f"Health failed: {code} {body}"
    print("/health OK")

    # Status (auth)
    code, body = get("/api/v1/status", headers={"X-API-Key": API_KEY})
    assert code == 200, f"Status failed: {code} {body}"
    print("/api/v1/status OK")

    # Metrics
    code, _ = get("/metrics")
    assert code == 200, "/metrics failed"
    print("/metrics OK")

    print("Smoke tests passed")
