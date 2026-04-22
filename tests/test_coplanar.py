from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from attpc_estimator.cli.coplanar import main as coplanar_main
from attpc_estimator.process.coplanar import (
    COPLANAR_RANGE,
    DEFAULT_COPLANAR_BINS,
    _accumulate_histogram,
    build_coplanar_histogram,
    coplanar_ratio,
)
from attpc_estimator.service.histograms import HistogramService


def _make_hit(trace_id: int, x: float, y: float, z: float) -> list[float]:
    return [x, y, z, 10.0, 1.0, float(trace_id), z, 1.0, float(trace_id)]


def _write_pointcloud_run(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        cloud = handle.create_group("cloud")
        cloud.attrs["min_event"] = 1
        cloud.attrs["max_event"] = 3
        cloud.attrs["bad_events"] = np.asarray([], dtype=np.int64)
        cloud.create_dataset(
            "cloud_1",
            data=np.asarray(
                [
                    _make_hit(0, 0.0, 0.0, 0.0),
                    _make_hit(1, 1.0, 0.0, 0.0),
                    _make_hit(2, 0.0, 1.0, 0.0),
                    _make_hit(3, 1.0, 1.0, 0.0),
                    _make_hit(4, 0.5, 0.5, 0.0),
                ],
                dtype=np.float64,
            ),
        )
        cloud.create_dataset(
            "cloud_2",
            data=np.asarray(
                [
                    _make_hit(0, 0.0, 0.0, 0.0),
                    _make_hit(1, 1.0, 0.0, 0.4),
                    _make_hit(2, 0.0, 1.0, 0.8),
                    _make_hit(3, 1.0, 1.0, 1.2),
                    _make_hit(4, 0.5, 0.2, 1.6),
                    _make_hit(5, 1.3, 0.4, 2.0),
                ],
                dtype=np.float64,
            ),
        )
        cloud.create_dataset(
            "cloud_3",
            data=np.asarray(
                [
                    _make_hit(0, 0.0, 0.0, 0.0),
                    _make_hit(1, 1.0, 1.0, 1.0),
                ],
                dtype=np.float64,
            ),
        )


def test_coplanar_ratio_is_lower_for_planar_events() -> None:
    planar = np.asarray(
        [
            _make_hit(0, 0.0, 0.0, 0.0),
            _make_hit(1, 1.0, 0.0, 0.0),
            _make_hit(2, 0.0, 1.0, 0.0),
            _make_hit(3, 1.0, 1.0, 0.0),
        ],
        dtype=np.float64,
    )
    volumetric = np.asarray(
        [
            _make_hit(0, 0.0, 0.0, 0.0),
            _make_hit(1, 1.0, 0.0, 0.4),
            _make_hit(2, 0.0, 1.0, 0.8),
            _make_hit(3, 1.0, 1.0, 1.2),
            _make_hit(4, 0.5, 0.2, 1.6),
            _make_hit(5, 1.3, 0.4, 2.0),
        ],
        dtype=np.float64,
    )

    planar_ratio = coplanar_ratio(planar)
    volumetric_ratio = coplanar_ratio(volumetric)

    assert planar_ratio is not None
    assert volumetric_ratio is not None
    assert planar_ratio < 1.0e-8
    assert volumetric_ratio > planar_ratio


def test_build_coplanar_histogram_accumulates_expected_counts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)

    payload = build_coplanar_histogram(
        pointcloud_file_path=pointcloud_path,
        run=8,
    )

    assert int(np.asarray(payload["processed_events"]).item()) == 3
    assert int(np.asarray(payload["accepted_events"]).item()) == 1
    assert int(np.asarray(payload["skipped_events"]).item()) == 2
    assert int(np.asarray(payload["valid_events"]).item()) == 2
    assert int(np.asarray(payload["trace_count"]).item()) == 1
    assert int(np.asarray(payload["histogram"]).sum()) == 1
    assert np.asarray(payload["bin_edges"]).shape == (DEFAULT_COPLANAR_BINS + 1,)
    assert np.asarray(payload["ratio_thresholds"]).tolist() == [0.001, 0.01, 0.02, 0.05]
    assert np.asarray(payload["ratio_threshold_counts"]).tolist() == [1, 1, 1, 1]


def test_coplanar_histogram_skips_out_of_range_values() -> None:
    histogram = np.zeros(DEFAULT_COPLANAR_BINS, dtype=np.int64)
    bin_edges = np.linspace(COPLANAR_RANGE[0], COPLANAR_RANGE[1], DEFAULT_COPLANAR_BINS + 1)

    accepted = _accumulate_histogram(histogram, COPLANAR_RANGE[1] + 0.01, bin_edges)

    assert accepted is False
    assert int(histogram.sum()) == 0


def test_coplanar_main_writes_histogram_artifact(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coplanar",
            "-w",
            str(workspace),
            "-r",
            "8",
        ],
    )
    coplanar_main()

    output_path = workspace / "histograms" / "run_0008_coplanar.npz"
    payload = np.load(output_path)
    stdout = capsys.readouterr().out
    assert output_path.is_file()
    assert int(payload["run_id"]) == 8
    assert int(payload["processed_events"]) == 3
    assert int(payload["accepted_events"]) == 1
    assert int(payload["skipped_events"]) == 2
    assert int(payload["valid_events"]) == 2
    assert "ratio under 0.001: 1/2 (0.500000)" in stdout
    assert "ratio under 0.01: 1/2 (0.500000)" in stdout
    assert "ratio under 0.02: 1/2 (0.500000)" in stdout
    assert "ratio under 0.05: 1/2 (0.500000)" in stdout


def test_histogram_service_loads_coplanar_payload(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    (trace_root / "run_0008.h5").touch()
    pointcloud_path = workspace / "pointcloud" / "run_0008.h5"
    _write_pointcloud_run(pointcloud_path)
    payload = build_coplanar_histogram(
        pointcloud_file_path=pointcloud_path,
        run=8,
    )
    histogram_root = workspace / "histograms"
    histogram_root.mkdir(parents=True, exist_ok=True)
    np.savez(histogram_root / "run_0008_coplanar.npz", **payload)

    service = HistogramService(trace_path=trace_root, workspace=workspace)
    bootstrap = service.bootstrap_state()
    assert bootstrap["histogramAvailability"]["8"]["coplanar"] == {
        "all": True,
        "labeled": False,
        "filtered": False,
    }

    result = service.get_histogram(metric="coplanar", mode="all", run=8)
    assert result["metric"] == "coplanar"
    assert result["mode"] == "all"
    assert result["binCount"] == DEFAULT_COPLANAR_BINS
    assert result["binLabel"] == "λ₃/λ₁"
    assert result["countLabel"] == "Event count"
    expected_first_center = COPLANAR_RANGE[0] + ((COPLANAR_RANGE[1] - COPLANAR_RANGE[0]) / DEFAULT_COPLANAR_BINS) / 2.0
    expected_last_center = COPLANAR_RANGE[1] - ((COPLANAR_RANGE[1] - COPLANAR_RANGE[0]) / DEFAULT_COPLANAR_BINS) / 2.0
    assert np.isclose(result["binCenters"][0], expected_first_center)
    assert np.isclose(result["binCenters"][-1], expected_last_center)
    assert result["series"][0]["traceCount"] == 1
    assert sum(result["series"][0]["histogram"]) == 1
