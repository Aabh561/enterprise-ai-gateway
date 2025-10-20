# Enterprise AI Gateway Documentation

Welcome to the Enterprise AI Gateway docs.

- What it is: A secure, intelligent, and observable platform for LLM interactions.
- Key features: Multi-provider LLMs, RAG, strong security, observability, and plugins.

Getting started:
- Run locally with `docker compose up -d` or `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
- Use dev header: `X-API-Key: your-super-secret-api-key-here`
- API docs at /docs (enabled in development)
- Health at /health, metrics at /metrics
- On Windows auto-reload: set `WATCHFILES_FORCE_POLLING=true` or run without `--reload`

## 🛠️ Troubleshooting

- 401 Unauthorized: Include header `X-API-Key: your-super-secret-api-key-here`.
- Uvicorn exits immediately with --reload on Windows: set `WATCHFILES_FORCE_POLLING=true` or run without `--reload`.
- docker compose not found: install Docker Desktop or run via Python `uvicorn`.
- 429 Too Many Requests: slow down or adjust `rate_limiting.per_minute` in configs/<env>.yaml.
- CORS errors in browser: set `api.cors.origins` in configs/<env>.yaml to your frontend origin(s).
- Readiness 503 or dependency errors: start backing services (Redis, vector DB) or disable features in config for local runs.
- Pydantic V2 warning about schema_extra: harmless; will be addressed in a future update.
