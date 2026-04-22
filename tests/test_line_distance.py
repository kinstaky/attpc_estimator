from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from attpc_estimator.cli.line_distance import main as line_distance_main
from attpc_estimator.process.line_distance import (
    LineDistanceHistogramConfig,
    RansacConfig,
    build_line_distance_histograms,
)
from attpc_estimator.service.histograms import HistogramService


def _make_hit(trace_id: int, x: float, y: float, z: float) -> list[float]:
    return [x, y, z, 10.0, 1.0, float(trace_id), z, 1.0, float(trace_id)]


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
                    _make_hit(0, 1.0, 0.0, 10.0),
                    _make_hit(1, 2.0, 0.0, 20.0),
                    _make_hit(2, 3.0, 0.0, 30.0),
                    _make_hit(3, 4.0, 0.0, 40.0),
                    _make_hit(4, 20.0, 1.2, 12.0),
                    _make_hit(5, 20.0, 2.2, 22.0),
                    _make_hit(6, 20.0, 3.2, 32.0),
                    _make_hit(7, 20.0, 4.2, 42.0),
                    _make_hit(8, 25.0, 25.0, 10.0),
                ],
                dtype=np.float64,
            ),
        )
        cloud.create_dataset(
            "cloud_2",
            data=np.asarray(
                [
                    _make_hit(0, 1.5, 0.0, 15.0),
                    _make_hit(1, 2.5, 0.0, 25.0),
                    _make_hit(2, 3.5, 0.0, 35.0),
                    _make_hit(3, 4.5, 0.0, 45.0),
                    _make_hit(4, 30.0, -20.0, 10.0),
                ],
                dtype=np.float64,
            ),
        )


def test_build_line_distance_histograms_accumulates_expected_counts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)

    payload = build_line_distance_histograms(
        pointcloud_file_path=pointcloud_path,
        run=8,
        ransac_config=RansacConfig(
            min_samples=3,
            residual_threshold=1.0,
            max_trials=100,
            max_iterations=6,
            target_labeled_ratio=0.8,
            min_inliers=3,
            max_start_radius=40.0,
        ),
        histogram_config=LineDistanceHistogramConfig(),
    )

    assert int(np.asarray(payload["processed_events"]).item()) == 2
    assert int(np.asarray(payload["accepted_line_total"]).item()) >= 2
    assert int(np.asarray(payload["accepted_pair_total"]).item()) >= 1
    assert int(np.asarray(payload["line_count_histogram"]).sum()) == 2
    assert int(np.asarray(payload["labeled_ratio_histogram"]).sum()) == 2
    assert int(np.asarray(payload["distances1_histogram"]).sum()) >= 1
    assert int(np.asarray(payload["distances2_histogram"]).sum()) >= 1
    assert int(np.asarray(payload["joint_histogram"]).sum()) >= 1
    assert int(np.asarray(payload["point_line_distance_histograms"]).sum()) > 0


def test_line_distance_main_writes_histogram_artifact(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "line_distance",
            "-w",
            str(workspace),
            "-r",
            "8",
            "--residual-threshold",
            "1.0",
            "--min-inliers",
            "3",
        ],
    )
    line_distance_main()

    output_path = workspace / "histograms" / "run_0008_line_distance.npz"
    payload = np.load(output_path)
    assert output_path.is_file()
    assert int(payload["run_id"]) == 8
    assert int(payload["processed_events"]) == 2


def test_histogram_service_loads_line_distance_payload(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    (trace_root / "run_0008.h5").touch()
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)
    payload = build_line_distance_histograms(
        pointcloud_file_path=pointcloud_path,
        run=8,
        ransac_config=RansacConfig(
            residual_threshold=1.0,
            min_inliers=3,
        ),
    )
    histogram_root = workspace / "histograms"
    histogram_root.mkdir(parents=True, exist_ok=True)
    np.savez(histogram_root / "run_0008_line_distance.npz", **payload)

    service = HistogramService(trace_path=trace_root, workspace=workspace)
    bootstrap = service.bootstrap_state()
    assert bootstrap["histogramAvailability"]["8"]["line_distance"] == {
        "all": True,
        "labeled": False,
        "filtered": False,
    }

    result = service.get_histogram(metric="line_distance", mode="all", run=8)
    assert result["metric"] == "line_distance"
    assert result["mode"] == "all"
    assert len(result["plots"]) == 6
    assert result["plots"][0]["key"] == "line_count"
