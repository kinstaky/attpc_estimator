from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from attpc_storage.hdf5 import PointcloudReader

from ..process.progress import ProgressReporter, emit_progress
from ..storage.run_paths import pointcloud_run_path
from .line_distance import _bin_centers
from .line_pipeline import (
    LineCluster,
    MergeConfig,
    RansacConfig,
    extract_line_clusters,
    merge_line_clusters,
    point_line_distance,
    refit_cluster_weighted,
)

DEFAULT_LINE_LENGTH_BINS = 120
DEFAULT_TOTAL_Q_BINS = 200
DEFAULT_HALF_Q_RATIO_BINS = 100
DEFAULT_DISTANCE_BINS = 120

LINE_LENGTH_RANGE = (0.0, 1000.0)
TOTAL_Q_RANGE = (0.0, 20000.0)
HALF_Q_RATIO_RANGE = (0.0, 5.0)
DISTANCE_RANGE = (0.0, 300.0)

DISTANCE_GROUP_KEYS = (
    "distance_total",
    "distance_longest",
    "distance_second_longest",
    "distance_third_longest",
)
DISTANCE_GROUP_TITLES = (
    "Point-to-line distance for all lines",
    "Point-to-line distance for the longest line",
    "Point-to-line distance for the second-longest line",
    "Point-to-line distance for the third-longest line",
)
DISTANCE_SERIES_KEYS = ("in_line", "out_line")
DISTANCE_SERIES_TITLES = ("In line", "Out of line")


@dataclass(frozen=True, slots=True)
class LinePropertyHistogramConfig:
    line_length_bins: int = DEFAULT_LINE_LENGTH_BINS
    total_q_bins: int = DEFAULT_TOTAL_Q_BINS
    half_q_ratio_bins: int = DEFAULT_HALF_Q_RATIO_BINS
    distance_bins: int = DEFAULT_DISTANCE_BINS


@dataclass(frozen=True, slots=True)
class LineMeasurement:
    line_length: float
    total_q: float
    half_q_ratio: float | None


def build_line_property_histograms(
    *,
    pointcloud_file_path: Path,
    run: int,
    ransac_config: RansacConfig = RansacConfig(),
    merge_config: MergeConfig = MergeConfig(),
    histogram_config: LinePropertyHistogramConfig = LinePropertyHistogramConfig(),
    progress: ProgressReporter | None = None,
) -> dict[str, object]:
    line_length_edges = _bin_edges(histogram_config.line_length_bins, LINE_LENGTH_RANGE)
    total_q_edges = _bin_edges(histogram_config.total_q_bins, TOTAL_Q_RANGE)
    half_q_ratio_edges = _bin_edges(histogram_config.half_q_ratio_bins, HALF_Q_RATIO_RANGE)
    distance_edges = _bin_edges(histogram_config.distance_bins, DISTANCE_RANGE)

    line_length_histogram = np.zeros(histogram_config.line_length_bins, dtype=np.int64)
    total_q_histogram = np.zeros(histogram_config.total_q_bins, dtype=np.int64)
    half_q_ratio_histogram = np.zeros(histogram_config.half_q_ratio_bins, dtype=np.int64)
    distance_histograms = np.zeros((4, 2, histogram_config.distance_bins), dtype=np.int64)

    processed_events = 0
    accepted_line_total = 0
    valid_half_ratio_total = 0

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
            rows = np.asarray(event, dtype=np.float64)
            if rows.ndim != 2 or rows.shape[1] < 4:
                continue

            extracted_clusters, _ = extract_line_clusters(rows, ransac_config=ransac_config)
            clusters = [
                refit_cluster_weighted(cluster)
                for cluster in merge_line_clusters(extracted_clusters, merge_config=merge_config)
            ]
            measurements = [
                _measure_line(cluster)
                for cluster in clusters
            ]

            _accumulate_histogram(
                line_length_histogram,
                [measurement.line_length for measurement in measurements],
                line_length_edges,
            )
            _accumulate_histogram(
                total_q_histogram,
                [measurement.total_q for measurement in measurements],
                total_q_edges,
            )
            half_q_values = [
                measurement.half_q_ratio
                for measurement in measurements
                if measurement.half_q_ratio is not None
            ]
            _accumulate_histogram(half_q_ratio_histogram, half_q_values, half_q_ratio_edges)
            valid_half_ratio_total += len(half_q_values)
            accepted_line_total += len(clusters)

            if clusters:
                _accumulate_distance_groups(
                    rows=rows,
                    clusters=clusters,
                    measurements=measurements,
                    distance_edges=distance_edges,
                    distance_histograms=distance_histograms,
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
        "valid_half_ratio_total": np.int64(valid_half_ratio_total),
        "line_length_histogram": line_length_histogram,
        "line_length_bin_edges": line_length_edges,
        "total_q_histogram": total_q_histogram,
        "total_q_bin_edges": total_q_edges,
        "half_q_ratio_histogram": half_q_ratio_histogram,
        "half_q_ratio_bin_edges": half_q_ratio_edges,
        "distance_histograms": distance_histograms,
        "distance_bin_edges": distance_edges,
        "distance_group_keys": np.asarray(DISTANCE_GROUP_KEYS, dtype=np.str_),
        "distance_group_titles": np.asarray(DISTANCE_GROUP_TITLES, dtype=np.str_),
        "distance_series_keys": np.asarray(DISTANCE_SERIES_KEYS, dtype=np.str_),
        "distance_series_titles": np.asarray(DISTANCE_SERIES_TITLES, dtype=np.str_),
    }


def default_pointcloud_file_path(workspace: Path, run: int) -> Path:
    return pointcloud_run_path(workspace, run)


def serialize_line_property_payload(run: int, payload: dict[str, object]) -> dict[str, object]:
    plots: list[dict[str, object]] = [
        _serialize_histogram_plot(
            key="line_length",
            title="Line length",
            histogram=np.asarray(payload["line_length_histogram"], dtype=np.int64),
            edges=np.asarray(payload["line_length_bin_edges"], dtype=np.float64),
            bin_label="Line length (mm)",
            count_label="Line count",
        ),
        _serialize_histogram_plot(
            key="total_q",
            title="Total Q on line",
            histogram=np.asarray(payload["total_q_histogram"], dtype=np.int64),
            edges=np.asarray(payload["total_q_bin_edges"], dtype=np.float64),
            bin_label="Total Q",
            count_label="Line count",
        ),
        _serialize_histogram_plot(
            key="half_q_ratio",
            title="Half-line Q ratio",
            histogram=np.asarray(payload["half_q_ratio_histogram"], dtype=np.int64),
            edges=np.asarray(payload["half_q_ratio_bin_edges"], dtype=np.float64),
            bin_label="Q large-z / Q small-z",
            count_label="Line count",
        ),
    ]

    group_keys = np.asarray(payload["distance_group_keys"], dtype=np.str_).tolist()
    group_titles = np.asarray(payload["distance_group_titles"], dtype=np.str_).tolist()
    series_keys = np.asarray(payload["distance_series_keys"], dtype=np.str_).tolist()
    series_titles = np.asarray(payload["distance_series_titles"], dtype=np.str_).tolist()
    distance_histograms = np.asarray(payload["distance_histograms"], dtype=np.int64)
    distance_centers = _bin_centers(np.asarray(payload["distance_bin_edges"], dtype=np.float64)).tolist()
    for group_index, (group_key, group_title) in enumerate(zip(group_keys, group_titles, strict=True)):
        plots.append(
            {
                "key": str(group_key),
                "render": "grouped_bar",
                "title": str(group_title),
                "binCenters": distance_centers,
                "binLabel": "Point-to-line distance (mm)",
                "countLabel": "Point count",
                "series": [
                    {
                        "labelKey": str(series_key),
                        "title": str(series_title),
                        "histogram": distance_histograms[group_index, series_index].tolist(),
                    }
                    for series_index, (series_key, series_title) in enumerate(
                        zip(series_keys, series_titles, strict=True)
                    )
                ],
            }
        )

    return {
        "metric": "line_property",
        "mode": "all",
        "run": int(run),
        "plots": plots,
        "summary": {
            "processedEvents": int(np.asarray(payload["processed_events"]).item()),
            "acceptedLineTotal": int(np.asarray(payload["accepted_line_total"]).item()),
            "validHalfRatioTotal": int(np.asarray(payload["valid_half_ratio_total"]).item()),
        },
        "series": [],
    }


def _measure_line(cluster: LineCluster) -> LineMeasurement:
    points = np.asarray(cluster.point_rows[:, :3], dtype=np.float64)
    amplitudes = np.asarray(cluster.point_rows[:, 3], dtype=np.float64)
    projections = (points - cluster.centroid) @ cluster.direction
    if projections.size == 0:
        return LineMeasurement(line_length=0.0, total_q=0.0, half_q_ratio=None)

    line_length = float(projections.max() - projections.min())
    total_q = float(np.clip(amplitudes, 0.0, None).sum())

    midpoint = 0.5 * (float(projections.min()) + float(projections.max()))
    first_mask = projections <= midpoint
    second_mask = ~first_mask
    first_q = float(np.clip(amplitudes[first_mask], 0.0, None).sum())
    second_q = float(np.clip(amplitudes[second_mask], 0.0, None).sum())
    half_q_ratio = None if first_q <= 0.0 else second_q / first_q
    return LineMeasurement(line_length=line_length, total_q=total_q, half_q_ratio=half_q_ratio)


def _accumulate_distance_groups(
    *,
    rows: np.ndarray,
    clusters: list[LineCluster],
    measurements: list[LineMeasurement],
    distance_edges: np.ndarray,
    distance_histograms: np.ndarray,
) -> None:
    all_indices = np.arange(rows.shape[0], dtype=np.int64)
    sorted_indices = sorted(
        range(len(clusters)),
        key=lambda index: measurements[index].line_length,
        reverse=True,
    )
    group_to_line = [None, *sorted_indices[:3]]

    for line_index, cluster in enumerate(clusters):
        distances = point_line_distance(rows[:, :3], cluster.centroid, cluster.direction)
        in_mask = np.isin(all_indices, cluster.inlier_indices)
        out_mask = ~in_mask
        _accumulate_histogram(distance_histograms[0, 0], distances[in_mask], distance_edges)
        _accumulate_histogram(distance_histograms[0, 1], distances[out_mask], distance_edges)

        for group_index, selected_line_index in enumerate(group_to_line[1:], start=1):
            if selected_line_index != line_index:
                continue
            _accumulate_histogram(distance_histograms[group_index, 0], distances[in_mask], distance_edges)
            _accumulate_histogram(distance_histograms[group_index, 1], distances[out_mask], distance_edges)


def _bin_edges(count: int, value_range: tuple[float, float]) -> np.ndarray:
    return np.linspace(float(value_range[0]), float(value_range[1]), int(count) + 1, dtype=np.float64)


def _accumulate_histogram(histogram: np.ndarray, values: np.ndarray | list[float], edges: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return
    histogram += np.histogram(array, bins=edges)[0].astype(np.int64)


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
