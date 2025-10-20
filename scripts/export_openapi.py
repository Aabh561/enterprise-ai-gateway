#!/usr/bin/env python3
import json
from pathlib import Path

from fastapi.openapi.utils import get_openapi

from app.main import app

if __name__ == "__main__":
    schema = get_openapi(
        title=app.title,
        version=app.version if hasattr(app, "version") else "0.1.0",
        routes=app.routes,
        description=app.description if hasattr(app, "description") else None,
    )
    out_path = Path("docs/openapi.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote OpenAPI schema to {out_path}")
