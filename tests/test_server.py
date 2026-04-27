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
        return {
            "appType": "merged",
            "session": {"mode": "label", "run": 8},
            "uiState": {"route": "/label"},
        }

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

    def current_trace(self) -> dict:
        return {"run": 8, "eventId": 4, "traceId": 5}

    def current_pointcloud_label_event(self) -> dict:
        return {"run": 8, "eventId": 4, "currentLabel": None}

    def current_pointcloud_event(self) -> dict:
        return {"run": 8, "eventId": 7}

    def previous_trace(self) -> dict:
        raise LookupError("no previous trace")

    def next_pointcloud_label_event(self) -> dict:
        return {"run": 8, "eventId": 5, "currentLabel": "2"}

    def next_pointcloud_event(self) -> dict:
        return {"run": 8, "eventId": 8}

    def previous_pointcloud_label_event(self) -> dict:
        raise LookupError("no previous pointcloud event")

    def previous_pointcloud_event(self) -> dict:
        raise LookupError("no previous browse pointcloud event")

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

    def assign_pointcloud_label(self, *, event_id: int, label: str) -> dict:
        return {
            "eventId": event_id,
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

    def update_ui_state(self, payload: dict) -> dict:
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
    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()
    (detector_dir / "pads.json").write_text('[{"pad": 1, "cx": 1.0, "cy": 2.0}]', encoding="utf-8")
    app = create_app(DummyMergedService(), tmp_path / "missing-dist", detector_dir)

    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/bootstrap").json() == {
            "appType": "merged",
            "session": {"mode": "label", "run": 8},
            "uiState": {"route": "/label"},
        }
        assert client.get("/api/mapping/pads").json() == [{"pad": 1, "cx": 1.0, "cy": 2.0}]

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

        assert client.get("/api/pointcloud-label/current").json() == {
            "run": 8,
            "eventId": 4,
            "currentLabel": None,
        }
        assert client.get("/api/pointcloud/current").json() == {
            "run": 8,
            "eventId": 7,
        }
        assert client.post("/api/pointcloud/next").json() == {
            "run": 8,
            "eventId": 8,
        }
        previous_browse_pointcloud = client.post("/api/pointcloud/previous")
        assert previous_browse_pointcloud.status_code == 404
        assert previous_browse_pointcloud.json() == {"detail": "no previous browse pointcloud event"}
        assert client.post("/api/pointcloud-label/next").json() == {
            "run": 8,
            "eventId": 5,
            "currentLabel": "2",
        }
        previous_pointcloud = client.post("/api/pointcloud-label/previous")
        assert previous_pointcloud.status_code == 404
        assert previous_pointcloud.json() == {"detail": "no previous pointcloud event"}
        pointcloud_assign = client.post(
            "/api/pointcloud-label/assign",
            json={"eventId": 5, "label": "2"},
        )
        assert pointcloud_assign.status_code == 200
        assert pointcloud_assign.json() == {"eventId": 5, "label": "2"}

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

        ui_state = client.post(
            "/api/ui-state",
            json={"route": "/browse/trace", "shell": {"selectedRun": 8}},
        )
        assert ui_state.status_code == 200
        assert ui_state.json() == {"uiState": {"route": "/browse/trace", "shell": {"selectedRun": 8}}}

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
        assert client.get("/api/traces/current").json() == {"run": 8, "eventId": 4, "traceId": 5}
