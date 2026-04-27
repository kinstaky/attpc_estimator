from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class RansacConfig:
    residual_threshold: float = 20.0
    max_trials: int = 200
    max_iterations: int = 10
    target_labeled_ratio: float = 0.8
    min_inliers: int = 20
    max_start_radius: float = 40.0


@dataclass(frozen=True, slots=True)
class MergeConfig:
    distance_threshold: float = 30.0
    angle_threshold: float = 3.0


@dataclass(frozen=True, slots=True)
class LineCluster:
    inlier_indices: np.ndarray
    point_rows: np.ndarray
    centroid: np.ndarray
    direction: np.ndarray


def extract_line_clusters(
    rows: np.ndarray,
    *,
    ransac_config: RansacConfig,
) -> tuple[list[LineCluster], np.ndarray]:
    data = np.asarray(rows, dtype=np.float64)
    xyz = data[:, :3] if data.ndim == 2 and data.shape[1] >= 3 else np.empty((0, 3))
    if xyz.ndim != 2 or xyz.shape[0] == 0:
        return [], np.empty(0, dtype=np.int64)
    if xyz.shape[1] != 3:
        raise ValueError(f"expected pointcloud xyz data with shape (N, 3), got {xyz.shape}")

    unlabeled_rows = data.copy()
    unlabeled_xyz = xyz.copy()
    unlabeled_indices = np.arange(data.shape[0], dtype=np.int64)
    clusters: list[LineCluster] = []
    labeled_size = 0
    rng = np.random.default_rng(0)

    for _ in range(ransac_config.max_iterations):
        if unlabeled_xyz.shape[0] < ransac_config.min_inliers:
            break
        if data.shape[0] > 0 and labeled_size / float(data.shape[0]) >= ransac_config.target_labeled_ratio:
            break
        inlier_mask = ransac_inlier_mask(
            unlabeled_xyz,
            residual_threshold=ransac_config.residual_threshold,
            max_trials=ransac_config.max_trials,
            start_radius=ransac_config.max_start_radius,
            rng=rng,
        )
        if inlier_mask is None:
            break
        if int(inlier_mask.sum()) < ransac_config.min_inliers:
            continue

        cluster_points = unlabeled_xyz[inlier_mask]
        if np.min(np.linalg.norm(cluster_points[:, :2], axis=1)) >= ransac_config.max_start_radius:
            continue

        centroid, direction = fit_line(cluster_points)
        clusters.append(
            LineCluster(
                inlier_indices=np.asarray(unlabeled_indices[inlier_mask], dtype=np.int64),
                point_rows=np.asarray(unlabeled_rows[inlier_mask], dtype=np.float64),
                centroid=np.asarray(centroid, dtype=np.float64),
                direction=np.asarray(direction, dtype=np.float64),
            )
        )
        labeled_size += int(inlier_mask.sum())
        unlabeled_rows = unlabeled_rows[~inlier_mask]
        unlabeled_xyz = unlabeled_xyz[~inlier_mask]
        unlabeled_indices = unlabeled_indices[~inlier_mask]

    return clusters, np.asarray(unlabeled_indices, dtype=np.int64)


def merge_line_clusters(
    clusters: list[LineCluster],
    *,
    merge_config: MergeConfig,
) -> list[LineCluster]:
    if len(clusters) < 2:
        return clusters

    centroids = np.asarray([cluster.centroid for cluster in clusters], dtype=np.float64)
    directions = np.asarray([cluster.direction for cluster in clusters], dtype=np.float64)
    diff = centroids[:, None, :] - centroids[None, :, :]
    cross = np.cross(diff, directions)
    distances = np.linalg.norm(cross, axis=-1)
    dot_product = np.clip(directions @ directions.T, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    mask = np.triu(np.ones((len(clusters), len(clusters)), dtype=bool), k=1)
    pair_distances = distances[mask]
    pair_angles = angle[mask]
    merge_mask = (
        pair_distances < float(merge_config.distance_threshold)
    ) & (
        pair_angles < float(merge_config.angle_threshold)
    )

    merged: list[LineCluster] = []
    labels = [-1] * len(clusters)
    pair_index = 0
    for left_index in range(len(clusters)):
        if labels[left_index] == -1:
            labels[left_index] = len(merged)
            merged.append(clusters[left_index])
        for right_index in range(left_index + 1, len(clusters)):
            if merge_mask[pair_index] and labels[right_index] == -1:
                label = labels[left_index]
                merged[label] = LineCluster(
                    inlier_indices=np.concatenate(
                        [merged[label].inlier_indices, clusters[right_index].inlier_indices]
                    ).astype(np.int64, copy=False),
                    point_rows=np.vstack([merged[label].point_rows, clusters[right_index].point_rows]),
                    centroid=merged[label].centroid,
                    direction=merged[label].direction,
                )
                labels[right_index] = label
            pair_index += 1

    return merged


def refit_cluster_weighted(cluster: LineCluster) -> LineCluster:
    centroid, direction = fit_line_weighted(cluster.point_rows[:, :3], cluster.point_rows[:, 3])
    return replace(
        cluster,
        centroid=np.asarray(centroid, dtype=np.float64),
        direction=np.asarray(direction, dtype=np.float64),
    )


def fit_line(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(data, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = np.dot(centered.T, centered) / len(points)
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    if float(direction[2]) < 0.0:
        direction = -direction
    return centroid, direction


def fit_line_weighted(data: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(data, dtype=np.float64)
    resolved_weights = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"expected point matrix shaped (N, 3), got {points.shape}")
    if resolved_weights.shape[0] != points.shape[0]:
        raise ValueError("weights length must match point count")
    positive_weights = np.clip(resolved_weights, 0.0, None)
    weight_sum = float(positive_weights.sum())
    if weight_sum <= 0.0:
        return fit_line(points)

    centroid = np.average(points, axis=0, weights=positive_weights)
    centered = points - centroid
    covariance = ((centered * positive_weights[:, None]).T @ centered) / weight_sum
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    if float(direction[2]) < 0.0:
        direction = -direction
    return np.asarray(centroid, dtype=np.float64), direction


def ransac_inlier_mask(
    data: np.ndarray,
    *,
    residual_threshold: float,
    max_trials: int,
    start_radius: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    points = np.asarray(data, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected point matrix shaped (N, 3), got {points.shape}")
    if points.shape[0] < 2:
        return None

    design = np.empty((points.shape[0], 3), dtype=np.float64)
    design[:, :2] = points[:, :2]
    design[:, 2] = 1.0
    target = points[:, 2]

    best_mask: np.ndarray | None = None
    best_count = 0
    best_error = np.inf

    start_points = points[np.linalg.norm(points[:, :2], axis=1) < start_radius, :]
    other_points = points[np.linalg.norm(points[:, :2], axis=1) >= start_radius, :]
    if start_points.shape[0] == 0 or other_points.shape[0] == 0:
        start_points = points
        other_points = points

    for _ in range(int(max_trials)):
        start_sample = start_points[rng.choice(start_points.shape[0])]
        other_sample = other_points[rng.choice(other_points.shape[0])]

        direction = other_sample - start_sample
        norm = np.linalg.norm(direction)
        if norm <= 0.0:
            continue
        direction /= norm
        residuals = np.linalg.norm(np.cross(points - start_sample, direction), axis=1)

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


def ordered_centroid_to_line_distance(first: LineCluster, second: LineCluster) -> float:
    return float(
        point_line_distance(
            np.asarray([second.centroid], dtype=np.float64),
            first.centroid,
            first.direction,
        )[0]
    )
