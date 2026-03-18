from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .services import TraceLabelService

class LabelRequest(BaseModel):
    eventId: int
    traceId: int
    family: str
    label: str


class CreateStrangeLabelRequest(BaseModel):
    name: str
    shortcutKey: str


class TraceModeRequest(BaseModel):
    mode: str
    family: str | None = None
    label: str | None = None


def build_api_router(service: TraceLabelService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/bootstrap")
    def bootstrap() -> dict:
        return service.bootstrap_state()

    @router.post("/trace/next")
    def next_trace() -> dict:
        try:
            return service.next_trace()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/trace/previous")
    def previous_trace() -> dict:
        try:
            return service.previous_trace()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/trace/label")
    def label_trace(request: LabelRequest) -> dict:
        try:
            return service.save_label(
                event_id=request.eventId,
                trace_id=request.traceId,
                family=request.family,
                label=request.label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/trace/mode")
    def set_trace_mode(request: TraceModeRequest) -> dict:
        try:
            return service.set_trace_mode(
                mode=request.mode,
                family=request.family,
                label=request.label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/label/strange")
    def get_strange_labels() -> dict:
        return service.get_strange_labels()

    @router.post("/label/strange")
    def create_strange_label(request: CreateStrangeLabelRequest) -> dict:
        try:
            return service.create_strange_label(request.name, request.shortcutKey)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/label/strange/{label}")
    def delete_strange_label(label: str) -> dict:
        try:
            return service.delete_strange_label(label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
