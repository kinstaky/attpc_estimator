from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .services import ViewerService


class FilterSelectionRequest(BaseModel):
    name: str


def build_api_router(service: ViewerService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/bootstrap")
    def bootstrap() -> dict:
        return service.bootstrap_state()

    @router.get("/histogram")
    def histogram(metric: str, mode: str, run: int, filterFile: str | None = None) -> dict:
        try:
            return service.get_histogram(
                metric=metric,
                mode=mode,
                run=run,
                filter_file=filterFile,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/review/filter")
    def select_filter(request: FilterSelectionRequest) -> dict:
        try:
            return service.select_filter_file(request.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/review/next")
    def next_trace() -> dict:
        try:
            return service.next_trace()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/review/previous")
    def previous_trace() -> dict:
        try:
            return service.previous_trace()
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
