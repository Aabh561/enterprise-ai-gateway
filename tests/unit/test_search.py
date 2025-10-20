import os
from fastapi.testclient import TestClient

os.environ.setdefault("ENVIRONMENT", "development")

from app.main import app  # noqa: E402

client = TestClient(app)


def test_search_pagination_and_filters(monkeypatch):
    # Monkeypatch vector_service search to return predictable results
    from app.routers import v1 as api_v1

    class DummyChunk:
        def __init__(self, idx):
            self.content = f"doc {idx}"
            self.metadata = type("M", (), {"title": f"t{idx}", "document_type": None, "file_path": None})
            self.chunk_index = idx

    class DummyResult:
        def __init__(self, idx):
            self.chunk = DummyChunk(idx)
            self.similarity_score = 0.9
            self.rank = idx

    async def fake_get_vector_service():
        class Svc:
            async def search(self, query, k, collection_name=None, filters=None):
                return [DummyResult(i) for i in range(20)]
        return Svc()

    monkeypatch.setattr(api_v1, "get_vector_service", fake_get_vector_service)

    headers = {"X-API-Key": "your-super-secret-api-key-here"}
    payload = {"query": "q", "limit": 5, "offset": 10}
    r = client.post("/api/v1/search", json=payload, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["total_results"] == 20
    assert len(data["results"]) == 5
    # First item in this page should be doc 10
    assert data["results"][0]["metadata"]["chunk_index"] == 10
