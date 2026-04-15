from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from attpc_estimator.server import create_app


class DummyMergedService:
    def __init__(self) -> None:
        self.closed = False
        self.histogram_job_messages = [
            {
                "type": "progress",
                "current": 1,
                "total": 4,
                "percent": 25,
                "unit": "trace",
                "message": "event=1",
            },
            {
                "type": "complete",
                "payload": {
                    "metric": "cdf",
                    "mode": "filtered",
                    "run": 8,
                    "filterFile": "filter.npy",
                    "veto": True,
                },
            },
        ]

    def close(self) -> None:
        self.closed = True

    def bootstrap_state(self) -> dict:
        return {"appType": "merged", "session": {"mode": "label", "run": 8}}

    def set_session(
        self,
        *,
        mode: str,
        run: int | None = None,
        source: str | None = None,
        family: str | None = None,
        label: str | None = None,
        filter_file: str | None = None,
        event_id: int | None = None,
        trace_id: int | None = None,
    ) -> dict:
        return {
            "session": {
                "mode": mode,
                "run": run,
                "source": source,
                "family": family,
                "label": label,
                "filterFile": filter_file,
                "eventId": event_id,
                "traceId": trace_id,
            }
        }

    def next_trace(self) -> dict:
        return {"run": 8, "eventId": 1, "traceId": 2}

    def previous_trace(self) -> dict:
        raise LookupError("no previous trace")

    def next_event(self) -> dict:
        return {"run": 8, "eventId": 2, "traceId": 0}

    def previous_event(self) -> dict:
        raise LookupError("no previous event")

    def assign_label(self, *, event_id: int, trace_id: int, family: str, label: str) -> dict:
        return {
            "eventId": event_id,
            "traceId": trace_id,
            "family": family,
            "label": label,
        }

    def get_strange_labels(self) -> dict:
        return {"strangeLabels": []}

    def create_strange_label(self, name: str, shortcut_key: str) -> dict:
        return {"name": name, "shortcutKey": shortcut_key}

    def delete_strange_label(self, label: str) -> dict:
        return {"deleted": label}

    def get_histogram(
        self,
        *,
        metric: str,
        mode: str,
        run: int,
        variant: str | None = None,
        filter_file: str | None = None,
        veto: bool = False,
    ) -> dict:
        payload = {
            "metric": metric,
            "mode": mode,
            "run": run,
            "filterFile": filter_file,
            "veto": veto,
        }
        if variant is not None:
            payload["variant"] = variant
        return payload

    def create_histogram_job(
        self,
        *,
        metric: str,
        mode: str,
        run: int,
        variant: str | None = None,
        filter_file: str | None = None,
        veto: bool = False,
    ) -> dict:
        return {"jobId": "job-1"}

    def next_histogram_job_message(
        self,
        *,
        job_id: str,
        after_index: int,
    ) -> tuple[int, dict] | None:
        if job_id != "job-1":
            raise LookupError(f"histogram job not found: {job_id}")
        if after_index >= len(self.histogram_job_messages):
            return None
        return after_index + 1, self.histogram_job_messages[after_index]


def test_create_app_routes_and_fallback(tmp_path: Path) -> None:
    app = create_app(DummyMergedService(), tmp_path / "missing-dist")

    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/bootstrap").json() == {
            "appType": "merged",
            "session": {"mode": "label", "run": 8},
        }

        session = client.post(
            "/api/session",
            json={"mode": "review", "run": 8, "source": "label_set", "family": "normal"},
        )
        assert session.status_code == 200
        assert session.json() == {
            "session": {
                "mode": "review",
                "run": 8,
                "source": "label_set",
                "family": "normal",
                "label": None,
                "filterFile": None,
                "eventId": None,
                "traceId": None,
            }
        }

        assert client.post("/api/traces/next").json() == {"run": 8, "eventId": 1, "traceId": 2}
        previous = client.post("/api/traces/previous")
        assert previous.status_code == 404
        assert previous.json() == {"detail": "no previous trace"}
        assert client.post("/api/traces/next-event").json() == {
            "run": 8,
            "eventId": 2,
            "traceId": 0,
        }
        previous_event = client.post("/api/traces/previous-event")
        assert previous_event.status_code == 404
        assert previous_event.json() == {"detail": "no previous event"}

        assign = client.post(
            "/api/labels/assign",
            json={"eventId": 1, "traceId": 2, "family": "normal", "label": "0"},
        )
        assert assign.status_code == 200
        assert assign.json() == {"eventId": 1, "traceId": 2, "family": "normal", "label": "0"}

        histogram = client.get(
            "/api/histograms",
            params={"metric": "bitflip", "variant": "value", "mode": "all", "run": 8},
        )
        assert histogram.status_code == 200
        assert histogram.json() == {
            "metric": "bitflip",
            "mode": "all",
            "run": 8,
            "variant": "value",
            "filterFile": None,
            "veto": False,
        }

        histogram_job = client.post(
            "/api/histograms/jobs",
            json={
                "metric": "bitflip",
                "variant": "length",
                "mode": "filtered",
                "run": 8,
                "filterFile": "filter.npy",
                "veto": True,
            },
        )
        assert histogram_job.status_code == 200
        assert histogram_job.json() == {"jobId": "job-1"}

        with client.websocket_connect("/api/histograms/jobs/job-1") as websocket:
            assert websocket.receive_json() == {
                "type": "progress",
                "current": 1,
                "total": 4,
                "percent": 25,
                "unit": "trace",
                "message": "event=1",
            }
            assert websocket.receive_json() == {
                "type": "complete",
                "payload": {
                    "metric": "cdf",
                    "mode": "filtered",
                    "run": 8,
                    "filterFile": "filter.npy",
                    "veto": True,
                },
            }

        fallback = client.get("/some/client/route")
        assert fallback.status_code == 200
        assert "Frontend build missing" in fallback.text
        missing_api = client.get("/api/missing")
        assert missing_api.status_code == 404
        assert missing_api.text == "Not Found"
