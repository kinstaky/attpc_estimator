from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from attpc_estimator.cli.line_property import main as line_property_main
from attpc_estimator.process.line_pipeline import RansacConfig
from attpc_estimator.process.line_property import (
    LinePropertyHistogramConfig,
    build_line_property_histograms,
    serialize_line_property_payload,
)
from attpc_estimator.service.histograms import HistogramService


def _make_hit(trace_id: int, x: float, y: float, z: float, amplitude: float) -> list[float]:
    return [x, y, z, amplitude, amplitude, float(trace_id), z, 1.0, float(trace_id)]


def _write_pointcloud_run(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        cloud = handle.create_group("cloud")
        cloud.attrs["min_event"] = 1
        cloud.attrs["max_event"] = 2
        cloud.attrs["bad_events"] = np.asarray([], dtype=np.int64)
        cloud.create_dataset(
            "cloud_1",
            data=np.asarray(
                [
                    _make_hit(0, 5.0, 0.0, 10.0, 4.0),
                    _make_hit(1, 10.0, 0.2, 20.0, 5.0),
                    _make_hit(2, 15.0, 0.1, 30.0, 8.0),
                    _make_hit(3, 25.0, 0.0, 50.0, 10.0),
                    _make_hit(4, 35.0, -0.2, 70.0, 12.0),
                    _make_hit(5, 0.0, 5.0, 10.0, 9.0),
                    _make_hit(6, 0.1, 10.0, 20.0, 8.0),
                    _make_hit(7, 0.0, 15.0, 30.0, 6.0),
                    _make_hit(8, -0.2, 25.0, 50.0, 5.0),
                    _make_hit(9, 0.0, 35.0, 70.0, 4.0),
                    _make_hit(10, -4.0, -3.0, 12.0, 3.0),
                    _make_hit(11, -8.0, -6.0, 24.0, 4.0),
                    _make_hit(12, -12.0, -9.0, 36.0, 8.0),
                    _make_hit(13, -20.0, -15.0, 60.0, 9.0),
                    _make_hit(14, -24.0, -18.0, 72.0, 11.0),
                    _make_hit(15, 40.0, 40.0, 5.0, 1.0),
                    _make_hit(16, -35.0, 25.0, 6.0, 1.0),
                ],
                dtype=np.float64,
            ),
        )
        cloud.create_dataset(
            "cloud_2",
            data=np.asarray(
                [
                    _make_hit(0, 0.0, 0.0, 0.0, 5.0),
                    _make_hit(1, 1.0, 1.0, 1.0, 5.0),
                ],
                dtype=np.float64,
            ),
        )


def test_build_line_property_histograms_accumulates_expected_counts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)

    payload = build_line_property_histograms(
        pointcloud_file_path=pointcloud_path,
        run=8,
        ransac_config=RansacConfig(
            residual_threshold=1.5,
            max_trials=100,
            max_iterations=8,
            target_labeled_ratio=1.0,
            min_inliers=3,
            max_start_radius=20.0,
        ),
        histogram_config=LinePropertyHistogramConfig(),
    )

    assert int(np.asarray(payload["processed_events"]).item()) == 2
    assert int(np.asarray(payload["accepted_line_total"]).item()) >= 3
    assert int(np.asarray(payload["line_length_histogram"]).sum()) >= 3
    assert int(np.asarray(payload["total_q_histogram"]).sum()) >= 3
    assert int(np.asarray(payload["half_q_ratio_histogram"]).sum()) >= 3
    distance_histograms = np.asarray(payload["distance_histograms"], dtype=np.int64)
    assert distance_histograms.shape == (4, 2, 120)
    assert int(distance_histograms[0].sum()) > 0
    assert int(distance_histograms[1].sum()) > 0
    serialized = serialize_line_property_payload(8, payload)
    assert serialized["metric"] == "line_property"
    assert len(serialized["plots"]) == 7
    assert serialized["plots"][0]["key"] == "line_length"
    assert serialized["plots"][3]["key"] == "distance_total"
    assert len(serialized["plots"][3]["series"]) == 2


def test_line_property_main_writes_histogram_artifact(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "line_property",
            "-w",
            str(workspace),
            "-r",
            "8",
            "--residual-threshold",
            "1.5",
            "--max-trials",
            "100",
            "--max-iterations",
            "8",
            "--target-labeled-ratio",
            "1.0",
            "--min-inliers",
            "3",
            "--max-start-radius",
            "20.0",
            "--merge-distance-threshold",
            "30.0",
            "--merge-angle-threshold",
            "3.0",
        ],
    )
    line_property_main()

    output_path = workspace / "histograms" / "run_0008_line_property.npz"
    payload = np.load(output_path)
    assert output_path.is_file()
    assert int(payload["run_id"]) == 8
    assert int(payload["processed_events"]) == 2


def test_histogram_service_loads_line_property_payload(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    (trace_root / "run_0008.h5").touch()
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)
    payload = build_line_property_histograms(
        pointcloud_file_path=pointcloud_path,
        run=8,
        ransac_config=RansacConfig(
            residual_threshold=1.5,
            target_labeled_ratio=1.0,
            min_inliers=3,
            max_start_radius=20.0,
        ),
    )
    histogram_root = workspace / "histograms"
    histogram_root.mkdir(parents=True, exist_ok=True)
    np.savez(histogram_root / "run_0008_line_property.npz", **payload)

    service = HistogramService(trace_path=trace_root, workspace=workspace)
    bootstrap = service.bootstrap_state()
    assert bootstrap["histogramAvailability"]["8"]["line_property"] == {
        "all": True,
        "labeled": False,
        "filtered": False,
    }

    result = service.get_histogram(metric="line_property", mode="all", run=8)
    assert result["metric"] == "line_property"
    assert result["mode"] == "all"
    assert len(result["plots"]) == 7
    assert result["plots"][4]["key"] == "distance_longest"
