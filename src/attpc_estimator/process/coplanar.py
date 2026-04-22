from __future__ import annotations

from pathlib import Path

import numpy as np

from attpc_storage.hdf5 import PointcloudReader

from ..process.progress import ProgressReporter, emit_progress
from ..storage.run_paths import pointcloud_run_path

DEFAULT_COPLANAR_BINS = 500
COPLANAR_RANGE = (0.0, 0.05)
COPLANAR_RATIO_THRESHOLDS = (0.001, 0.01, 0.02, 0.05)


def build_coplanar_histogram(
    *,
    pointcloud_file_path: Path,
    run: int,
    bin_count: int = DEFAULT_COPLANAR_BINS,
    progress: ProgressReporter | None = None,
) -> dict[str, object]:
    bin_edges = np.linspace(COPLANAR_RANGE[0], COPLANAR_RANGE[1], int(bin_count) + 1)
    histogram = np.zeros(int(bin_count), dtype=np.int64)
    processed_events = 0
    accepted_events = 0
    skipped_events = 0
    valid_events = 0
    threshold_counts = np.zeros(len(COPLANAR_RATIO_THRESHOLDS), dtype=np.int64)

    reader = PointcloudReader(
        workspace=str(pointcloud_file_path.parent.parent),
        run=run,
        path=str(pointcloud_file_path),
    )
    try:
        event_range = reader.get_range()
        total_events = max(0, event_range[1] - event_range[0] + 1)
        emit_progress(progress, current=0, total=total_events, unit="event")
        for event_id in range(event_range[0], event_range[1] + 1):
            try:
                _, event = reader.read_event(event_id)
            except (KeyError, ValueError):
                continue

            ratio = coplanar_ratio(event)
            processed_events += 1
            if ratio is None:
                skipped_events += 1
            else:
                valid_events += 1
                _accumulate_threshold_counts(threshold_counts, float(ratio))
                if _accumulate_histogram(histogram, float(ratio), bin_edges):
                    accepted_events += 1
                else:
                    skipped_events += 1

            emit_progress(
                progress,
                current=processed_events,
                total=total_events,
                unit="event",
                message=f"event={event_id}",
            )
    finally:
        reader.close()

    return {
        "run_id": np.int64(run),
        "processed_events": np.int64(processed_events),
        "accepted_events": np.int64(accepted_events),
        "skipped_events": np.int64(skipped_events),
        "valid_events": np.int64(valid_events),
        "trace_count": np.int64(accepted_events),
        "histogram": histogram,
        "bin_edges": bin_edges,
        "ratio_thresholds": np.asarray(COPLANAR_RATIO_THRESHOLDS, dtype=np.float64),
        "ratio_threshold_counts": threshold_counts,
    }


def default_pointcloud_file_path(workspace: Path, run: int) -> Path:
    return pointcloud_run_path(workspace, run)


def coplanar_ratio(event: np.ndarray) -> float | None:
    points = np.asarray(event, dtype=np.float64)
    xyz = points[:, :3] if points.ndim == 2 and points.shape[1] >= 3 else np.empty((0, 3))
    if xyz.shape[0] < 3:
        return None

    centroid = np.mean(xyz, axis=0)
    centered = xyz - centroid
    covariance = np.dot(centered.T, centered) / float(xyz.shape[0])
    eigenvalues = np.linalg.eigvalsh(covariance)
    lambda_1 = float(eigenvalues[-1])
    lambda_3 = float(eigenvalues[0])
    if lambda_1 <= 0.0:
        return None
    return float(lambda_3 / lambda_1)


def _accumulate_histogram(histogram: np.ndarray, value: float, bin_edges: np.ndarray) -> bool:
    if value < float(bin_edges[0]) or value > float(bin_edges[-1]):
        return False
    index = int(np.searchsorted(bin_edges, value, side="right") - 1)
    index = max(0, min(index, int(histogram.shape[0]) - 1))
    histogram[index] += 1
    return True


def _accumulate_threshold_counts(threshold_counts: np.ndarray, value: float) -> None:
    for index, threshold in enumerate(COPLANAR_RATIO_THRESHOLDS):
        if value < threshold:
            threshold_counts[index] += 1
