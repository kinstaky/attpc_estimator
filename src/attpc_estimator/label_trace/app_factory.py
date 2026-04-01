from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import build_api_router
from .services import TraceLabelService


def create_app(service: TraceLabelService, frontend_dist: Path) -> FastAPI:
    app = FastAPI(title="Trace Label")
    app.include_router(build_api_router(service))

    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_load_index_html(frontend_dist))

    @app.get("/{path:path}", response_class=HTMLResponse)
    async def spa_fallback(path: str) -> HTMLResponse:
        if path.startswith("api/"):
            return HTMLResponse("Not Found", status_code=404)
        return HTMLResponse(_load_index_html(frontend_dist))

    return app


def _load_index_html(frontend_dist: Path) -> str:
    index_path = frontend_dist / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Trace Label</title>
    <style>
      body {
        margin: 0;
        font-family: sans-serif;
        background: #f6f5f0;
        color: #1f1f1f;
        display: grid;
        place-items: center;
        min-height: 100vh;
      }
      main {
        max-width: 32rem;
        padding: 2rem;
        background: white;
        border: 1px solid #d4d0c8;
        border-radius: 16px;
      }
      code { background: #f0ede5; padding: 0.1rem 0.3rem; }
    </style>
  </head>
  <body>
    <main>
      <h1>Frontend build missing</h1>
      <p>The backend is running, but <code>frontend/dist</code> was not found.</p>
      <p>Build the Vue frontend and restart the app.</p>
    </main>
  </body>
</html>
"""
