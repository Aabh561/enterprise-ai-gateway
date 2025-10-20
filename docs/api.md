# API Overview

This application uses FastAPI and exposes an OpenAPI schema.

Development authentication: include header `X-API-Key: your-super-secret-api-key-here` on secured endpoints.

Key endpoints:
- GET /health — health check
- GET /metrics — Prometheus metrics
- GET /api/v1/status — requires API key (X-API-Key)
- POST /api/v1/chat/generate — chat generation
- POST /api/v1/chat/stream — server-sent streaming
- POST /api/v1/documents/upload — document ingestion
- POST /api/v1/search — vector search with pagination and filters

- Live docs (development): /docs
- ReDoc (development): /redoc
- Raw OpenAPI JSON: /openapi.json

To export the OpenAPI schema for static documentation, run:

```bash
python scripts/export_openapi.py
```

This writes docs/openapi.json for reference.

Search API usage:
- Pagination: limit (default 5), offset (default 0)
- Filtering: filters object to filter by metadata (implementation depends on vector DB)
- Threshold: similarity threshold (0.0 - 1.0)

Example:

```bash
curl -X POST "$BASE_URL/api/v1/search" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "enterprise ai",
    "collection_name": "company_docs",
    "limit": 10,
    "offset": 0,
    "threshold": 0.7,
    "filters": {"type": "policy"}
  }'
```
