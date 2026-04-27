from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .service.estimator import EstimatorService


class LabelAssignRequest(BaseModel):
    eventId: int
    traceId: int
    family: str
    label: str


class PointcloudLabelAssignRequest(BaseModel):
    eventId: int
    label: str


class CreateStrangeLabelRequest(BaseModel):
    name: str
    shortcutKey: str


class SessionRequest(BaseModel):
    mode: str
    run: int | None = None
    source: str | None = None
    family: str | None = None
    label: str | None = None
    filterFile: str | None = None
    eventId: int | None = None
    traceId: int | None = None


class HistogramJobRequest(BaseModel):
    metric: str
    mode: str
    run: int
    variant: str | None = None
    filterFile: str | None = None
    veto: bool = False


class PointcloudTraceRequest(BaseModel):
    run: int
    eventId: int
    traceIds: list[int]


class UiStateRequest(BaseModel):
    route: str | None = None
    shell: dict | None = None
    label: dict | None = None
    review: dict | None = None
    histograms: dict | None = None
    mapping: dict | None = None
    pointcloud: dict | None = None
    pointcloudLabel: dict | None = None


def create_app(
    service: EstimatorService,
    frontend_dist: Path,
    detector_dir: Path | None = None,
) -> FastAPI:
    detector_dir = detector_dir or Path(__file__).resolve().parent / "detector"
    app = FastAPI(title="Estimator WebUI", lifespan=_build_lifespan(service.close))
    app.include_router(build_api_router(service, detector_dir))
    _mount_frontend(app, frontend_dist, "AT-TPC WebUI")
    return app


def build_api_router(service: EstimatorService, detector_dir: Path) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/bootstrap")
    def bootstrap() -> dict:
        return service.bootstrap_state()

    @router.get("/mapping/pads")
    def mapping_pads() -> list[dict]:
        pads_path = detector_dir / "pads.json"
        if not pads_path.exists():
            raise HTTPException(status_code=404, detail=f"mapping pads file not found: {pads_path}")
        try:
            return json.loads(pads_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"invalid mapping pads file: {pads_path}") from exc

    @router.post("/session")
    def set_session(request: SessionRequest) -> dict:
        try:
            return service.set_session(
                mode=request.mode,
                run=request.run,
                source=request.source,
                family=request.family,
                label=request.label,
                filter_file=request.filterFile,
                event_id=request.eventId,
                trace_id=request.traceId,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/traces/current")
    def current_trace() -> dict:
        try:
            return service.current_trace()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/traces/next")
    def next_trace() -> dict:
        try:
            return service.next_trace()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/traces/previous")
    def previous_trace() -> dict:
        try:
            return service.previous_trace()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/traces/next-event")
    def next_event() -> dict:
        try:
            return service.next_event()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/traces/previous-event")
    def previous_event() -> dict:
        try:
            return service.previous_event()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/labels/assign")
    def assign_label(request: LabelAssignRequest) -> dict:
        try:
            return service.assign_label(
                event_id=request.eventId,
                trace_id=request.traceId,
                family=request.family,
                label=request.label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/labels/strange")
    def get_strange_labels() -> dict:
        return service.get_strange_labels()

    @router.post("/labels/strange")
    def create_strange_label(request: CreateStrangeLabelRequest) -> dict:
        try:
            return service.create_strange_label(request.name, request.shortcutKey)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/labels/strange/{label}")
    def delete_strange_label(label: str) -> list[dict]:
        try:
            return service.delete_strange_label(label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/histograms")
    def histogram(
        metric: str,
        mode: str,
        run: int,
        variant: str | None = None,
        filterFile: str | None = None,
        veto: bool = False,
    ) -> dict:
        try:
            return service.get_histogram(
                metric=metric,
                mode=mode,
                run=run,
                variant=variant,
                filter_file=filterFile,
                veto=veto,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/histograms/jobs")
    def create_histogram_job(request: HistogramJobRequest) -> dict[str, str]:
        try:
            return service.create_histogram_job(
                metric=request.metric,
                mode=request.mode,
                run=request.run,
                variant=request.variant,
                filter_file=request.filterFile,
                veto=request.veto,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.websocket("/histograms/jobs/{job_id}")
    async def histogram_job_socket(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        message_index = 0
        try:
            while True:
                try:
                    next_message = await asyncio.to_thread(
                        service.next_histogram_job_message,
                        job_id=job_id,
                        after_index=message_index,
                    )
                except LookupError:
                    await websocket.send_json(
                        {"type": "error", "detail": f"histogram job not found: {job_id}"}
                    )
                    return
                if next_message is None:
                    return
                message_index, payload = next_message
                await websocket.send_json(payload)
                if payload["type"] in {"complete", "error"}:
                    return
        except WebSocketDisconnect:
            return

    @router.get("/pointcloud/event")
    def pointcloud_event(run: int, eventId: int) -> dict:
        try:
            return service.get_pointcloud_event(run=run, event_id=eventId)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/pointcloud/current")
    def current_pointcloud() -> dict:
        try:
            return service.current_pointcloud_event()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/pointcloud/next")
    def next_pointcloud() -> dict:
        try:
            return service.next_pointcloud_event()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/pointcloud/previous")
    def previous_pointcloud() -> dict:
        try:
            return service.previous_pointcloud_event()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/pointcloud/traces")
    def pointcloud_traces(request: PointcloudTraceRequest) -> dict:
        try:
            return service.get_pointcloud_traces(
                run=request.run,
                event_id=request.eventId,
                trace_ids=request.traceIds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/pointcloud-label/current")
    def current_pointcloud_label() -> dict:
        try:
            return service.current_pointcloud_label_event()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/pointcloud-label/next")
    def next_pointcloud_label() -> dict:
        try:
            return service.next_pointcloud_label_event()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/pointcloud-label/previous")
    def previous_pointcloud_label() -> dict:
        try:
            return service.previous_pointcloud_label_event()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/pointcloud-label/assign")
    def assign_pointcloud_label(request: PointcloudLabelAssignRequest) -> dict:
        try:
            return service.assign_pointcloud_label(
                event_id=request.eventId,
                label=request.label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/ui-state")
    def update_ui_state(request: UiStateRequest) -> dict:
        return {"uiState": service.update_ui_state(request.model_dump())}

    return router


def _build_lifespan(close_func):
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            close_func()

    return lifespan


def _mount_frontend(app: FastAPI, frontend_dist: Path, title: str) -> None:
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_load_index_html(frontend_dist, title))

    @app.get("/{path:path}", response_class=HTMLResponse)
    async def spa_fallback(path: str) -> HTMLResponse:
        if path.startswith("api/"):
            return HTMLResponse("Not Found", status_code=404)
        return HTMLResponse(_load_index_html(frontend_dist, title))


def _load_index_html(frontend_dist: Path, title: str) -> str:
    index_path = frontend_dist / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      body {{
        margin: 0;
        font-family: sans-serif;
        background: #f6f5f0;
        color: #1f1f1f;
        display: grid;
        place-items: center;
        min-height: 100vh;
      }}
      main {{
        max-width: 32rem;
        padding: 2rem;
        background: white;
        border: 1px solid #d4d0c8;
        border-radius: 16px;
      }}
      code {{ background: #f0ede5; padding: 0.1rem 0.3rem; }}
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
