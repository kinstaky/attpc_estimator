from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from attpc_storage.hdf5 import PointcloudReader

from ..process.progress import ProgressReporter, emit_progress
from ..storage.run_paths import pointcloud_run_path
from .line_pipeline import (
    LineCluster,
    RansacConfig,
    extract_line_clusters,
    ordered_centroid_to_line_distance,
)

DEFAULT_LINE_COUNT_BINS = 10
DEFAULT_LABELED_RATIO_BINS = 100
DEFAULT_PAIR_DISTANCE_BINS = 120
DEFAULT_ANGLE_BINS = 180

LINE_COUNT_RANGE = (0.0, 10.0)
LABELED_RATIO_RANGE = (0.0, 1.0)
PAIR_DISTANCE_RANGE = (0.0, 600.0)
ANGLE_RANGE = (0.0, 180.0)


@dataclass(frozen=True, slots=True)
class LineDistanceHistogramConfig:
    line_count_bins: int = DEFAULT_LINE_COUNT_BINS
    labeled_ratio_bins: int = DEFAULT_LABELED_RATIO_BINS
    pair_distance_bins: int = DEFAULT_PAIR_DISTANCE_BINS
    angle_bins: int = DEFAULT_ANGLE_BINS


def build_line_distance_histograms(
    *,
    pointcloud_file_path: Path,
    run: int,
    ransac_config: RansacConfig = RansacConfig(),
    histogram_config: LineDistanceHistogramConfig = LineDistanceHistogramConfig(),
    progress: ProgressReporter | None = None,
) -> dict[str, object]:
    line_count_edges = _bin_edges(histogram_config.line_count_bins, LINE_COUNT_RANGE)
    labeled_ratio_edges = _bin_edges(histogram_config.labeled_ratio_bins, LABELED_RATIO_RANGE)
    pair_distance_edges = _bin_edges(histogram_config.pair_distance_bins, PAIR_DISTANCE_RANGE)
    angle_edges = _bin_edges(histogram_config.angle_bins, ANGLE_RANGE)

    line_count_histogram = np.zeros(histogram_config.line_count_bins, dtype=np.int64)
    labeled_ratio_histogram = np.zeros(histogram_config.labeled_ratio_bins, dtype=np.int64)
    distances1_histogram = np.zeros(histogram_config.pair_distance_bins, dtype=np.int64)
    distances2_histogram = np.zeros(histogram_config.angle_bins, dtype=np.int64)
    joint_histogram = np.zeros(
        (histogram_config.pair_distance_bins, histogram_config.angle_bins),
        dtype=np.int64,
    )

    processed_events = 0
    accepted_line_total = 0
    accepted_pair_total = 0
    accepted_point_total = 0

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
            points = np.asarray(event, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] < 4:
                continue
            clusters, _ = extract_line_clusters(points, ransac_config=ransac_config)
            labeled_point_count = int(sum(cluster.inlier_indices.size for cluster in clusters))
            labeled_ratio = (
                float(labeled_point_count) / float(points.shape[0])
                if points.shape[0] > 0
                else 0.0
            )

            _accumulate_histogram(line_count_histogram, [float(len(clusters))], line_count_edges)
            _accumulate_histogram(labeled_ratio_histogram, [labeled_ratio], labeled_ratio_edges)

            if clusters:
                accepted_line_total += len(clusters)
                accepted_point_total += labeled_point_count
                accepted_pair_total += _accumulate_pair_metrics(
                    clusters=clusters,
                    distance_edges=pair_distance_edges,
                    angle_edges=angle_edges,
                    distance_histogram=distances1_histogram,
                    angle_histogram=distances2_histogram,
                    joint_histogram=joint_histogram,
                )

            processed_events += 1
            emit_progress(
                progress,
                current=processed_events,
                total=total_events,
                unit="event",
                message=f"event={event_id},lines={len(clusters)}",
            )
    finally:
        reader.close()

    return {
        "run_id": np.int64(run),
        "processed_events": np.int64(processed_events),
        "accepted_line_total": np.int64(accepted_line_total),
        "accepted_pair_total": np.int64(accepted_pair_total),
        "accepted_point_total": np.int64(accepted_point_total),
        "line_count_histogram": line_count_histogram,
        "line_count_bin_edges": line_count_edges,
        "labeled_ratio_histogram": labeled_ratio_histogram,
        "labeled_ratio_bin_edges": labeled_ratio_edges,
        "distances1_histogram": distances1_histogram,
        "distances1_bin_edges": pair_distance_edges,
        "distances2_histogram": distances2_histogram,
        "distances2_bin_edges": angle_edges,
        "joint_histogram": joint_histogram,
    }


def default_pointcloud_file_path(workspace: Path, run: int) -> Path:
    return pointcloud_run_path(workspace, run)


def _accumulate_pair_metrics(
    *,
    clusters: list[LineCluster],
    distance_edges: np.ndarray,
    angle_edges: np.ndarray,
    distance_histogram: np.ndarray,
    angle_histogram: np.ndarray,
    joint_histogram: np.ndarray,
) -> int:
    pair_count = 0
    for left_index in range(len(clusters)):
        for right_index in range(left_index + 1, len(clusters)):
            left = clusters[left_index]
            right = clusters[right_index]
            distance = ordered_centroid_to_line_distance(left, right)
            dot_product = float(np.clip(np.dot(left.direction, right.direction), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot_product)))
            _accumulate_histogram(distance_histogram, [distance], distance_edges)
            _accumulate_histogram(angle_histogram, [angle], angle_edges)
            x_index = _bin_index(distance, distance_edges)
            y_index = _bin_index(angle, angle_edges)
            if x_index is not None and y_index is not None:
                joint_histogram[x_index, y_index] += 1
            pair_count += 1
    return pair_count


def _bin_edges(count: int, value_range: tuple[float, float]) -> np.ndarray:
    return np.linspace(float(value_range[0]), float(value_range[1]), int(count) + 1, dtype=np.float64)


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


def _accumulate_histogram(histogram: np.ndarray, values: np.ndarray | list[float], edges: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return
    histogram += np.histogram(array, bins=edges)[0].astype(np.int64)


def _bin_index(value: float, edges: np.ndarray) -> int | None:
    if value < float(edges[0]) or value > float(edges[-1]):
        return None
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if index == len(edges) - 1:
        index -= 1
    if index < 0 or index >= len(edges) - 1:
        return None
    return index


def serialize_line_distance_payload(run: int, payload: dict[str, object]) -> dict[str, object]:
    return {
        "metric": "line_distance",
        "mode": "all",
        "run": int(run),
        "plots": [
            _serialize_histogram_plot(
                key="line_count",
                title="Accepted lines per event",
                histogram=np.asarray(payload["line_count_histogram"], dtype=np.int64),
                edges=np.asarray(payload["line_count_bin_edges"], dtype=np.float64),
                bin_label="Accepted lines",
                count_label="Event count",
            ),
            _serialize_histogram_plot(
                key="labeled_ratio",
                title="Labeled point ratio per event",
                histogram=np.asarray(payload["labeled_ratio_histogram"], dtype=np.int64),
                edges=np.asarray(payload["labeled_ratio_bin_edges"], dtype=np.float64),
                bin_label="Labeled ratio",
                count_label="Event count",
            ),
            _serialize_histogram_plot(
                key="distances1",
                title="Second-centroid to first-line distance",
                histogram=np.asarray(payload["distances1_histogram"], dtype=np.int64),
                edges=np.asarray(payload["distances1_bin_edges"], dtype=np.float64),
                bin_label="Second centroid to first line (mm)",
                count_label="Line-pair count",
            ),
            _serialize_histogram_plot(
                key="distances2",
                title="Line direction angle",
                histogram=np.asarray(payload["distances2_histogram"], dtype=np.int64),
                edges=np.asarray(payload["distances2_bin_edges"], dtype=np.float64),
                bin_label="Angle (deg)",
                count_label="Line-pair count",
            ),
            {
                "key": "joint",
                "render": "heatmap",
                "title": "Ordered centroid-to-line distance vs direction angle",
                "histogram": np.asarray(payload["joint_histogram"], dtype=np.int64).T.tolist(),
                "xBinCenters": _bin_centers(np.asarray(payload["distances1_bin_edges"], dtype=np.float64)).tolist(),
                "yBinCenters": _bin_centers(np.asarray(payload["distances2_bin_edges"], dtype=np.float64)).tolist(),
                "xLabel": "Second centroid to first line (mm)",
                "yLabel": "Angle (deg)",
                "countLabel": "Line-pair count",
            },
        ],
        "summary": {
            "processedEvents": int(np.asarray(payload["processed_events"]).item()),
            "acceptedLineTotal": int(np.asarray(payload["accepted_line_total"]).item()),
            "acceptedPairTotal": int(np.asarray(payload["accepted_pair_total"]).item()),
            "acceptedPointTotal": int(np.asarray(payload["accepted_point_total"]).item()),
        },
        "series": [],
    }


def _serialize_histogram_plot(
    *,
    key: str,
    title: str,
    histogram: np.ndarray,
    edges: np.ndarray,
    bin_label: str,
    count_label: str,
) -> dict[str, object]:
    return {
        "key": key,
        "render": "bar",
        "title": title,
        "histogram": np.asarray(histogram, dtype=np.int64).tolist(),
        "binCenters": _bin_centers(np.asarray(edges, dtype=np.float64)).tolist(),
        "binLabel": bin_label,
        "countLabel": count_label,
    }
