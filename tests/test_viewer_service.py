from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from attpc_estimator.db import TraceLabelRepository
from attpc_estimator.viewer.cdf import CDF_THRESHOLDS
from attpc_estimator.viewer.filter import build_filter_rows
from attpc_estimator.viewer.services import ViewerService


def _gaussian_trace(
    height: float, center: float = 120.0, width: float = 8.0
) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return (height * np.exp(-0.5 * ((x - center) / width) ** 2)).astype(np.float32)


def _sine_trace(period: float, amplitude: float = 30.0) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return (amplitude * np.sin(2.0 * np.pi * x / period)).astype(np.float32)


def _pad_rows(traces: list[np.ndarray]) -> np.ndarray:
    rows = []
    for trace_id, trace in enumerate(traces):
        hardware = np.asarray(
            [10 + trace_id, 20 + trace_id, 30 + trace_id, 40 + trace_id, 50 + trace_id],
            dtype=np.float32,
        )
        rows.append(np.concatenate([hardware, trace]).astype(np.float32))
    return np.asarray(rows, dtype=np.float32)


def write_run_file(path: Path, events: dict[int, list[np.ndarray]]) -> None:
    with h5py.File(path, "w") as handle:
        event_ids = sorted(events)
        group = handle.create_group("events")
        group.attrs["min_event"] = min(event_ids)
        group.attrs["max_event"] = max(event_ids)
        group.attrs["bad_events"] = np.asarray([], dtype=np.int64)
        for event_id in event_ids:
            event_group = group.create_group(f"event_{event_id}")
            get_group = event_group.create_group("get")
            get_group.create_dataset("pads", data=_pad_rows(events[event_id]))


def seed_workspace(workspace: Path) -> None:
    repository = TraceLabelRepository(workspace / "trace_label.db")
    repository.initialize()
    repository.create_strange_label("Noise", "n")
    repository.save_label(8, 1, 0, "pad", 1, 1, 1, 1, 1, "normal", "0")
    repository.save_label(8, 1, 1, "pad", 2, 2, 2, 2, 2, "strange", "Noise")
    repository.connection.close()


def make_trace_root(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {
            1: [_gaussian_trace(20.0), _gaussian_trace(50.0), _gaussian_trace(80.0)],
            2: [_gaussian_trace(40.0)],
        },
    )
    seed_workspace(workspace)
    return workspace, trace_root


def test_build_filter_rows_amplitude_is_inclusive_and_ordered(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    _ = workspace

    rows = build_filter_rows(
        trace_path=trace_root,
        run=8,
        amplitude_range=(20.0, 50.0),
        peak_separation=20.0,
        peak_prominence=5.0,
        peak_width=40.0,
    )

    assert rows.dtype == np.int64
    assert rows.shape == (2, 3)
    assert rows.tolist() == [
        [8, 1, 1],
        [8, 2, 0],
    ]


def test_build_filter_rows_oscillation_uses_f60_cutoff(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _ = workspace
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {
            1: [
                _sine_trace(period=4.0),
                _sine_trace(period=64.0),
                _gaussian_trace(40.0),
            ],
        },
    )

    rows = build_filter_rows(
        trace_path=trace_root,
        run=8,
        oscillation=True,
        baseline_window_scale=10.0,
    )

    assert rows.dtype == np.int64
    assert rows.tolist() == [[8, 1, 0]]


def test_viewer_service_bootstrap_and_histograms(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    np.save(
        workspace / "filter_amp_20_50.npy",
        np.asarray([[8, 1, 0], [8, 1, 1]], dtype=np.int64),
    )
    np.save(
        workspace / "run_0008_cdf.npy",
        np.ones((len(CDF_THRESHOLDS), 100), dtype=np.int64),
    )
    np.save(
        workspace / "run_0008_labeled_cdf.npy",
        {
            "run_id": np.int64(8),
            "label_keys": np.asarray(["normal:0", "strange:Noise"], dtype=object),
            "label_titles": np.asarray(["0 peak", "Noise"], dtype=object),
            "trace_counts": np.asarray([1, 1], dtype=np.int64),
            "histograms": np.ones((2, len(CDF_THRESHOLDS), 100), dtype=np.int64),
        },
    )
    np.save(workspace / "run_0008_amp.npy", np.arange(16, dtype=np.int64))
    np.save(
        workspace / "run_0008_labeled_amp.npy",
        {
            "run_id": np.int64(8),
            "label_keys": np.asarray(["normal:0", "normal:4", "normal:9", "strange:Noise"], dtype=object),
            "trace_counts": np.asarray([1, 2, 3, 1], dtype=np.int64),
            "normal:0": np.arange(16, dtype=np.int64),
            "normal:4": np.ones(16, dtype=np.int64),
            "normal:9": np.full(16, 2, dtype=np.int64),
            "strange:Noise": np.arange(16, dtype=np.int64),
        },
    )

    service = ViewerService(trace_path=trace_root, workspace=workspace)
    try:
        bootstrap = service.bootstrap_state()
        assert bootstrap["appType"] == "viewer"
        assert bootstrap["runs"] == [8]
        assert bootstrap["filterFiles"] == [{"name": "filter_amp_20_50.npy"}]
        assert bootstrap["histogramAvailability"]["8"]["cdf"]["all"] is True
        assert bootstrap["histogramAvailability"]["8"]["amplitude"]["labeled"] is True
        assert bootstrap["histogramAvailability"]["8"]["cdf"]["filtered"] is True

        cdf_payload = service.get_histogram(metric="cdf", mode="labeled", run=8)
        assert cdf_payload["metric"] == "cdf"
        assert [series["title"] for series in cdf_payload["series"]] == [
            "0 peak",
            "Noise",
        ]

        amp_payload = service.get_histogram(metric="amplitude", mode="labeled", run=8)
        assert amp_payload["metric"] == "amplitude"
        assert [series["title"] for series in amp_payload["series"]] == [
            "0 peak",
            "4+ peaks",
            "Noise",
        ]
        assert amp_payload["series"][1]["traceCount"] == 5
        assert amp_payload["series"][1]["histogram"][0] == 3

        filtered_cdf = service.get_histogram(
            metric="cdf",
            mode="filtered",
            run=8,
            filter_file="filter_amp_20_50.npy",
        )
        assert filtered_cdf["mode"] == "filtered"
        assert filtered_cdf["filterFile"] == "filter_amp_20_50.npy"
        assert [series["traceCount"] for series in filtered_cdf["series"]] == [2]
    finally:
        service.close()


def test_viewer_service_selects_filter_and_navigates(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    np.save(
        workspace / "filter_amp_20_80.npy",
        np.asarray([[8, 1, 0], [8, 1, 1]], dtype=np.int64),
    )

    service = ViewerService(trace_path=trace_root, workspace=workspace)
    try:
        selected = service.select_filter_file("filter_amp_20_80.npy")
        assert selected["filterFile"] == "filter_amp_20_80.npy"
        assert selected["trace"]["run"] == 8
        assert selected["trace"]["currentLabel"] == {"family": "normal", "label": "0"}

        next_trace = service.next_trace()
        assert (next_trace["run"], next_trace["eventId"], next_trace["traceId"]) == (
            8,
            1,
            1,
        )
        assert next_trace["currentLabel"] == {"family": "strange", "label": "Noise"}

        previous_trace = service.previous_trace()
        assert (
            previous_trace["run"],
            previous_trace["eventId"],
            previous_trace["traceId"],
        ) == (8, 1, 0)
    finally:
        service.close()
