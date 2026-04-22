from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from attpc_storage.hdf5 import PointcloudReader

from ..process.progress import ProgressReporter, emit_progress
from ..storage.run_paths import pointcloud_run_path

DEFAULT_LINE_COUNT_BINS = 10
DEFAULT_LABELED_RATIO_BINS = 100
DEFAULT_PAIR_DISTANCE_BINS = 120
DEFAULT_ANGLE_BINS = 180
DEFAULT_POINT_LINE_DISTANCE_BINS = 120

LINE_COUNT_RANGE = (0.0, 10.0)
LABELED_RATIO_RANGE = (0.0, 1.0)
PAIR_DISTANCE_RANGE = (0.0, 600.0)
ANGLE_RANGE = (0.0, 180.0)
POINT_LINE_DISTANCE_RANGE = (0.0, 300.0)


@dataclass(frozen=True, slots=True)
class RansacConfig:
    min_samples: int = 3
    residual_threshold: float = 20.0
    max_trials: int = 200
    max_iterations: int = 10
    target_labeled_ratio: float = 0.8
    min_inliers: int = 20
    max_start_radius: float = 40.0


@dataclass(frozen=True, slots=True)
class LineDistanceHistogramConfig:
    line_count_bins: int = DEFAULT_LINE_COUNT_BINS
    labeled_ratio_bins: int = DEFAULT_LABELED_RATIO_BINS
    pair_distance_bins: int = DEFAULT_PAIR_DISTANCE_BINS
    angle_bins: int = DEFAULT_ANGLE_BINS
    point_line_distance_bins: int = DEFAULT_POINT_LINE_DISTANCE_BINS


@dataclass(frozen=True, slots=True)
class LineCluster:
    inlier_indices: np.ndarray
    centroid: np.ndarray
    direction: np.ndarray


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
    point_line_distance_edges = _bin_edges(
        histogram_config.point_line_distance_bins,
        POINT_LINE_DISTANCE_RANGE,
    )

    line_count_histogram = np.zeros(histogram_config.line_count_bins, dtype=np.int64)
    labeled_ratio_histogram = np.zeros(histogram_config.labeled_ratio_bins, dtype=np.int64)
    distances1_histogram = np.zeros(histogram_config.pair_distance_bins, dtype=np.int64)
    distances2_histogram = np.zeros(histogram_config.angle_bins, dtype=np.int64)
    joint_histogram = np.zeros(
        (histogram_config.pair_distance_bins, histogram_config.angle_bins),
        dtype=np.int64,
    )
    same_line_histogram = np.zeros(histogram_config.point_line_distance_bins, dtype=np.int64)
    other_line_histogram = np.zeros(histogram_config.point_line_distance_bins, dtype=np.int64)
    unlabeled_histogram = np.zeros(histogram_config.point_line_distance_bins, dtype=np.int64)

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
            xyz = points[:, :3] if points.ndim == 2 and points.shape[1] >= 3 else np.empty((0, 3))
            clusters, unlabeled_indices = extract_line_clusters(xyz, ransac_config=ransac_config)
            labeled_point_count = int(sum(cluster.inlier_indices.size for cluster in clusters))
            labeled_ratio = (
                float(labeled_point_count) / float(xyz.shape[0])
                if xyz.shape[0] > 0
                else 0.0
            )

            _accumulate_histogram(line_count_histogram, [float(len(clusters))], line_count_edges)
            _accumulate_histogram(labeled_ratio_histogram, [labeled_ratio], labeled_ratio_edges)

            if clusters:
                accepted_line_total += len(clusters)
                accepted_point_total += labeled_point_count
                _accumulate_point_line_distances(
                    xyz=xyz,
                    clusters=clusters,
                    unlabeled_indices=unlabeled_indices,
                    bin_edges=point_line_distance_edges,
                    same_line_histogram=same_line_histogram,
                    other_line_histogram=other_line_histogram,
                    unlabeled_histogram=unlabeled_histogram,
                )
                pair_count = _accumulate_pair_metrics(
                    clusters=clusters,
                    distance_edges=pair_distance_edges,
                    angle_edges=angle_edges,
                    distance_histogram=distances1_histogram,
                    angle_histogram=distances2_histogram,
                    joint_histogram=joint_histogram,
                )
                accepted_pair_total += pair_count

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
        "point_line_distance_histograms": np.asarray(
            [same_line_histogram, other_line_histogram, unlabeled_histogram],
            dtype=np.int64,
        ),
        "point_line_distance_series_keys": np.asarray(
            ["same_line", "other_line", "unlabeled"],
            dtype=np.str_,
        ),
        "point_line_distance_series_titles": np.asarray(
            ["Same line", "Other line", "Unlabeled"],
            dtype=np.str_,
        ),
        "point_line_distance_bin_edges": point_line_distance_edges,
    }


def default_pointcloud_file_path(workspace: Path, run: int) -> Path:
    return pointcloud_run_path(workspace, run)


def extract_line_clusters(
    xyz: np.ndarray,
    *,
    ransac_config: RansacConfig,
) -> tuple[list[LineCluster], np.ndarray]:
    data = np.asarray(xyz, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        return [], np.empty(0, dtype=np.int64)
    if data.shape[1] != 3:
        raise ValueError(f"expected pointcloud xyz data with shape (N, 3), got {data.shape}")

    unlabeled_data = data.copy()
    unlabeled_indices = np.arange(data.shape[0], dtype=np.int64)
    clusters: list[LineCluster] = []
    labeled_size = 0
    rng = np.random.default_rng(0)

    for _ in range(ransac_config.max_iterations):
        if unlabeled_data.shape[0] < max(ransac_config.min_samples, ransac_config.min_inliers):
            break
        if data.shape[0] > 0 and labeled_size / float(data.shape[0]) >= ransac_config.target_labeled_ratio:
            break
        inlier_mask = _ransac_inlier_mask(
            unlabeled_data,
            min_samples=ransac_config.min_samples,
            residual_threshold=ransac_config.residual_threshold,
            max_trials=ransac_config.max_trials,
            rng=rng,
        )
        if inlier_mask is None:
            break
        if int(inlier_mask.sum()) < ransac_config.min_inliers:
            continue

        cluster_points = unlabeled_data[inlier_mask]
        if np.min(np.linalg.norm(cluster_points[:, :2], axis=1)) >= ransac_config.max_start_radius:
            continue

        centroid, direction = fit_line(cluster_points)
        if abs(float(direction[2])) < 1.0e-6:
            continue
        if float(direction[2]) < 0.0:
            direction = -direction

        clusters.append(
            LineCluster(
                inlier_indices=np.asarray(unlabeled_indices[inlier_mask], dtype=np.int64),
                centroid=np.asarray(centroid, dtype=np.float64),
                direction=np.asarray(direction, dtype=np.float64),
            )
        )
        labeled_size += int(inlier_mask.sum())
        unlabeled_data = unlabeled_data[~inlier_mask]
        unlabeled_indices = unlabeled_indices[~inlier_mask]

    return clusters, np.asarray(unlabeled_indices, dtype=np.int64)


def fit_line(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(data, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = np.dot(centered.T, centered) / len(points)
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    direction = direction / np.linalg.norm(direction)
    return centroid, direction


def _ransac_inlier_mask(
    data: np.ndarray,
    *,
    min_samples: int,
    residual_threshold: float,
    max_trials: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    points = np.asarray(data, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected point matrix shaped (N, 3), got {points.shape}")
    sample_size = min(points.shape[0], max(3, int(min_samples)))
    if points.shape[0] < sample_size:
        return None

    design = np.empty((points.shape[0], 3), dtype=np.float64)
    design[:, :2] = points[:, :2]
    design[:, 2] = 1.0
    target = points[:, 2]

    best_mask: np.ndarray | None = None
    best_count = 0
    best_error = np.inf

    for _ in range(max(1, int(max_trials))):
        sample_indices = rng.choice(points.shape[0], size=sample_size, replace=False)
        sample_design = design[sample_indices]
        sample_target = target[sample_indices]
        coefficients, _, rank, _ = np.linalg.lstsq(sample_design, sample_target, rcond=None)
        if rank < 3:
            continue

        residuals = np.abs(target - design @ coefficients)
        mask = residuals <= float(residual_threshold)
        count = int(mask.sum())
        if count == 0:
            continue
        error = float(residuals[mask].sum())
        if count > best_count or (count == best_count and error < best_error):
            best_mask = mask
            best_count = count
            best_error = error
            if best_count == points.shape[0]:
                break

    if best_mask is None:
        return None

    refined_design = design[best_mask]
    refined_target = target[best_mask]
    coefficients, _, rank, _ = np.linalg.lstsq(refined_design, refined_target, rcond=None)
    if rank < 3:
        return best_mask
    refined_residuals = np.abs(target - design @ coefficients)
    return refined_residuals <= float(residual_threshold)


def point_line_distance(points: np.ndarray, centroid: np.ndarray, direction: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float64)
    diff = points_array - np.asarray(centroid, dtype=np.float64)
    distances = np.linalg.norm(np.cross(diff, np.asarray(direction, dtype=np.float64)), axis=1)
    return np.asarray(distances, dtype=np.float64)


def z0_xy_intersection(centroid: np.ndarray, direction: np.ndarray) -> np.ndarray:
    alpha = -float(centroid[2]) / float(direction[2])
    return np.asarray(centroid[:2] + alpha * direction[:2], dtype=np.float64)


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
            distance = float(
                np.linalg.norm(
                    z0_xy_intersection(left.centroid, left.direction)
                    - z0_xy_intersection(right.centroid, right.direction)
                )
            )
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


def _accumulate_point_line_distances(
    *,
    xyz: np.ndarray,
    clusters: list[LineCluster],
    unlabeled_indices: np.ndarray,
    bin_edges: np.ndarray,
    same_line_histogram: np.ndarray,
    other_line_histogram: np.ndarray,
    unlabeled_histogram: np.ndarray,
) -> None:
    unlabeled_set = {int(index) for index in np.asarray(unlabeled_indices, dtype=np.int64).tolist()}
    all_indices = np.arange(xyz.shape[0], dtype=np.int64)
    for cluster in clusters:
        distances = point_line_distance(xyz, cluster.centroid, cluster.direction)
        inlier_set = {int(index) for index in cluster.inlier_indices.tolist()}
        same_mask = np.asarray([int(index) in inlier_set for index in all_indices], dtype=bool)
        unlabeled_mask = np.asarray([int(index) in unlabeled_set for index in all_indices], dtype=bool)
        other_mask = ~(same_mask | unlabeled_mask)
        _accumulate_histogram(same_line_histogram, distances[same_mask], bin_edges)
        _accumulate_histogram(other_line_histogram, distances[other_mask], bin_edges)
        _accumulate_histogram(unlabeled_histogram, distances[unlabeled_mask], bin_edges)


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
                title="Z=0 line distance",
                histogram=np.asarray(payload["distances1_histogram"], dtype=np.int64),
                edges=np.asarray(payload["distances1_bin_edges"], dtype=np.float64),
                bin_label="Distance at z=0 (mm)",
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
                "title": "Z=0 distance vs direction angle",
                "histogram": np.asarray(payload["joint_histogram"], dtype=np.int64).T.tolist(),
                "xBinCenters": _bin_centers(np.asarray(payload["distances1_bin_edges"], dtype=np.float64)).tolist(),
                "yBinCenters": _bin_centers(np.asarray(payload["distances2_bin_edges"], dtype=np.float64)).tolist(),
                "xLabel": "Distance at z=0 (mm)",
                "yLabel": "Angle (deg)",
                "countLabel": "Line-pair count",
            },
            {
                "key": "point_line_distance",
                "render": "grouped_bar",
                "title": "Point-to-line distance by point label",
                "binCenters": _bin_centers(
                    np.asarray(payload["point_line_distance_bin_edges"], dtype=np.float64)
                ).tolist(),
                "binLabel": "Point-to-line distance (mm)",
                "countLabel": "Point count",
                "series": [
                    {
                        "labelKey": str(label_key),
                        "title": str(title),
                        "histogram": np.asarray(histogram, dtype=np.int64).tolist(),
                    }
                    for label_key, title, histogram in zip(
                        np.asarray(payload["point_line_distance_series_keys"], dtype=np.str_).tolist(),
                        np.asarray(payload["point_line_distance_series_titles"], dtype=np.str_).tolist(),
                        np.asarray(payload["point_line_distance_histograms"], dtype=np.int64),
                        strict=True,
                    )
                ],
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
