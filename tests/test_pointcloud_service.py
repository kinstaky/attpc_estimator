from __future__ import annotations

import threading
from pathlib import Path

import h5py
import numpy as np

from attpc_estimator.service.pointcloud import PointcloudService


def _pad_row(trace_values: list[float], *, pad_id: int) -> np.ndarray:
    return np.asarray([[1, 2, 3, 4, pad_id, *trace_values]], dtype=np.float32)


def _write_trace_file(path: Path, event_ids: list[int]) -> None:
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["version"] = "libattpc_merger:2.0"
        events.attrs["min_event"] = min(event_ids)
        events.attrs["max_event"] = max(event_ids)
        events.attrs["bad_events"] = np.asarray([], dtype=np.int64)
        for event_id in event_ids:
            event = events.create_group(f"event_{event_id}")
            event.attrs["orig_run"] = 70
            event.attrs["orig_event"] = 700 + event_id
            get_group = event.create_group("get")
            rows = np.vstack(
                [
                    _pad_row([10.0 + event_id, 20.0 + event_id, 30.0 + event_id], pad_id=100 + event_id),
                    _pad_row([40.0 + event_id, 50.0 + event_id, 60.0 + event_id], pad_id=200 + event_id),
                    _pad_row([70.0 + event_id, 80.0 + event_id, 90.0 + event_id], pad_id=300 + event_id),
                ]
            )
            get_group.create_dataset("pads", data=rows)


def _write_pointcloud_file(path: Path, event_ids: list[int]) -> None:
    with h5py.File(path, "w") as handle:
        cloud = handle.create_group("cloud")
        cloud.attrs["min_event"] = min(event_ids)
        cloud.attrs["max_event"] = max(event_ids)
        cloud.attrs["fft_window_scale"] = 20.0
        cloud.attrs["micromegas_time_bucket"] = 10.0
        cloud.attrs["window_time_bucket"] = 560.0
        cloud.attrs["detector_length"] = 1000.0
        for event_id in event_ids:
            cloud.create_dataset(
                f"cloud_{event_id}",
                data=np.asarray(
                    [
                        [0.0, 1.0, 2.0, 100.0 + event_id, 10.0, 20.0, 10.0 + event_id, 11.0, 1.0],
                        [2.0, 3.0, 4.0, 200.0 + event_id, 30.0, 40.0, 20.0 + event_id, 22.0, 1.5],
                    ],
                    dtype=np.float64,
                ),
            )


def test_pointcloud_service_loads_unsorted_selected_traces(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    workspace = tmp_path / "workspace"
    pointcloud_root = workspace / "pointcloud"
    pointcloud_root.mkdir(parents=True)
    _write_trace_file(trace_root / "run_0007.h5", [1])
    _write_pointcloud_file(pointcloud_root / "run_0007.h5", [1])

    service = PointcloudService(trace_path=trace_root, workspace=workspace, baseline_window_scale=10.0)
    try:
        payload = service.get_traces(run=7, event_id=1, trace_ids=[2, 0, 2])
        assert [trace["traceId"] for trace in payload["traces"]] == [2, 0]
        assert payload["traces"][0]["raw"] == [71.0, 81.0, 91.0]
        assert payload["traces"][1]["raw"] == [11.0, 21.0, 31.0]
        assert [peak["padId"] for peak in payload["traces"][0]["peaks"]] == [21]
        assert [peak["padId"] for peak in payload["traces"][1]["peaks"]] == [11]
    finally:
        service.close()


def test_pointcloud_service_prefetches_neighbor_events(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    workspace = tmp_path / "workspace"
    pointcloud_root = workspace / "pointcloud"
    pointcloud_root.mkdir(parents=True)
    event_ids = list(range(1, 11))
    _write_trace_file(trace_root / "run_0007.h5", event_ids)
    _write_pointcloud_file(pointcloud_root / "run_0007.h5", event_ids)

    service = PointcloudService(trace_path=trace_root, workspace=workspace, baseline_window_scale=10.0)
    try:
        started = threading.Event()
        allow_prefetch = threading.Event()
        original_prefetch = service._prefetcher._prefetch_window

        def blocked_prefetch(generation, keys):
            started.set()
            allow_prefetch.wait(timeout=1.0)
            return original_prefetch(generation, keys)

        service._prefetcher._prefetch_window = blocked_prefetch

        payload = service.get_event(run=7, event_id=3)
        assert payload["eventIdRange"] == {"min": 1, "max": 10}
        assert started.wait(timeout=1.0)
        assert service._prefetcher.get_cached((7, 3)) is not None
        assert service._prefetcher.get_cached((7, 4)) is None

        allow_prefetch.set()
        assert service._wait_for_prefetch(timeout=1.0)
        assert service._prefetcher.get_cached((7, 1)) is not None
        assert service._prefetcher.get_cached((7, 8)) is not None
        assert service._prefetcher.get_cached((7, 9)) is None
    finally:
        service.close()
