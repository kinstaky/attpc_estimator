from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from attpc_storage.hdf5 import PointcloudReader, RawTraceReader

from ..process.line_pipeline import (
    MergeConfig,
    RansacConfig,
    extract_line_clusters,
    merge_line_clusters,
)
from ..storage.run_paths import collect_run_files, pointcloud_dir, resolve_run_file
from ..utils.trace_data import preprocess_traces

EVENT_PREFETCH_RADIUS = 5
EventKey = tuple[int, int]


class _PointcloudPrefetcher:
    def __init__(self, loader: Callable[[EventKey], np.ndarray | None]) -> None:
        self._loader = loader
        self._cache: dict[EventKey, np.ndarray] = {}
        self._desired_keys: tuple[EventKey, ...] = ()
        self._scheduled_keys: tuple[EventKey, ...] = ()
        self._generation = 0
        self._completed_generation = 0
        self._active_generation: int | None = None
        self._cache_lock = threading.Lock()
        self._condition = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="pointcloud-prefetch",
            daemon=True,
        )
        self._thread.start()

    def get_cached(self, key: EventKey) -> np.ndarray | None:
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            return np.asarray(cached, dtype=np.float64)

    def schedule(self, window_keys: list[EventKey]) -> None:
        keys = tuple(window_keys)
        desired = set(keys)
        with self._cache_lock:
            self._desired_keys = keys
            self._cache = {
                key: value
                for key, value in self._cache.items()
                if key in desired
            }
        with self._condition:
            self._generation += 1
            self._scheduled_keys = keys
            self._condition.notify_all()

    def store_current(self, key: EventKey, hits: np.ndarray) -> None:
        with self._cache_lock:
            self._cache[key] = np.asarray(hits, dtype=np.float64)

    def wait(self, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while time.monotonic() < deadline:
                if self._active_generation is None and self._completed_generation >= self._generation:
                    return True
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
            return self._active_generation is None and self._completed_generation >= self._generation

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        with self._cache_lock:
            self._cache.clear()
            self._desired_keys = ()

    def _is_stale(self, generation: int) -> bool:
        with self._condition:
            return self._closed or generation != self._generation

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._closed
                    and self._generation == self._completed_generation
                    and self._active_generation is None
                ):
                    self._condition.wait()
                if self._closed:
                    return
                generation = self._generation
                keys = self._scheduled_keys
                self._active_generation = generation

            self._prefetch_window(generation, keys)

            with self._condition:
                if self._active_generation == generation:
                    self._active_generation = None
                if generation == self._generation:
                    self._completed_generation = generation
                if self._active_generation is None and self._completed_generation >= self._generation:
                    self._condition.notify_all()

    def _prefetch_window(self, generation: int, keys: tuple[EventKey, ...]) -> None:
        if not keys:
            return

        for key in keys:
            if self._is_stale(generation):
                return
            with self._cache_lock:
                if key in self._cache or key not in self._desired_keys:
                    continue
            hits = self._loader(key)
            if hits is None:
                continue
            if self._is_stale(generation):
                return
            with self._cache_lock:
                if key not in self._desired_keys:
                    continue
                self._cache[key] = np.asarray(hits, dtype=np.float64)
                desired = set(self._desired_keys)
                self._cache = {
                    cached_key: cached_hits
                    for cached_key, cached_hits in self._cache.items()
                    if cached_key in desired
                }


class PointcloudService:
    def __init__(
        self,
        *,
        trace_path: Path,
        workspace: Path,
        baseline_window_scale: float,
        micromegas_time_bucket: float | None = None,
        window_time_bucket: float | None = None,
        detector_length: float | None = None,
        event_prefetch_radius: int = EVENT_PREFETCH_RADIUS,
    ) -> None:
        self.trace_path = trace_path
        self.workspace = workspace
        self.default_baseline_window_scale = float(baseline_window_scale)
        self.default_micromegas_time_bucket = (
            None if micromegas_time_bucket is None else float(micromegas_time_bucket)
        )
        self.default_window_time_bucket = (
            None if window_time_bucket is None else float(window_time_bucket)
        )
        self.default_detector_length = (
            None if detector_length is None else float(detector_length)
        )
        self.event_prefetch_radius = max(0, int(event_prefetch_radius))
        self.trace_files = collect_run_files(trace_path)
        self.pointcloud_files = collect_run_files(pointcloud_dir(workspace))
        self._trace_readers: dict[int, RawTraceReader] = {}
        self._pointcloud_readers: dict[int, PointcloudReader] = {}
        self._processing_cache: dict[int, dict[str, float]] = {}
        self._event_ranges = self._collect_event_ranges()
        self._prefetcher = _PointcloudPrefetcher(self._load_event_hits)

    def close(self) -> None:
        self._prefetcher.close()
        for reader in self._trace_readers.values():
            reader.close()
        for reader in self._pointcloud_readers.values():
            reader.close()
        self._trace_readers.clear()
        self._pointcloud_readers.clear()
        self._processing_cache.clear()

    def bootstrap_state(self) -> dict[str, Any]:
        return {
            "runs": sorted(self.pointcloud_files),
            "eventRanges": {
                str(run): {"min": event_range[0], "max": event_range[1]}
                for run, event_range in self._event_ranges.items()
            },
        }

    def validate_processing_configs(self) -> None:
        for run in sorted(self.pointcloud_files):
            self._processing_config(int(run))

    def get_event(self, *, run: int, event_id: int) -> dict[str, Any]:
        event_range = self._require_event_range(run, event_id)
        hits = self._require_event_hits(run, event_id)
        projected_hits = _project_hit_coordinates(hits)
        self._schedule_prefetch(run, event_id)
        return {
            "run": int(run),
            "eventId": int(event_id),
            "eventIdRange": {"min": int(event_range[0]), "max": int(event_range[1])},
            "hits": [
                _serialize_hit_row(row, projected=projected)
                for row, projected in zip(hits, projected_hits, strict=True)
            ],
            "processing": self._processing_config(run),
        }

    def get_label_event(
        self,
        *,
        run: int,
        event_id: int,
        ransac_config: RansacConfig,
        merge_config: MergeConfig,
    ) -> dict[str, Any]:
        event_range = self._require_event_range(run, event_id)
        hits = self._require_event_hits(run, event_id)
        projected_hits = _project_hit_coordinates(hits)
        merged_labels, merged_line_count = _merged_cluster_labels(
            hits,
            ransac_config=ransac_config,
            merge_config=merge_config,
        )
        self._schedule_prefetch(run, event_id)
        return {
            "run": int(run),
            "eventId": int(event_id),
            "eventIdRange": {"min": int(event_range[0]), "max": int(event_range[1])},
            "hits": [
                _serialize_hit_row(row, projected=projected, merged_label=merged_label)
                for row, projected, merged_label in zip(
                    hits,
                    projected_hits,
                    merged_labels,
                    strict=True,
                )
            ],
            "processing": self._processing_config(run),
            "mergedLineCount": int(merged_line_count),
            "suggestedLabel": _pointcloud_bucket_from_count(merged_line_count),
        }

    def get_traces(
        self,
        *,
        run: int,
        event_id: int,
        trace_ids: list[int],
    ) -> dict[str, Any]:
        self._require_event_range(run, event_id)
        hits = self._require_event_hits(run, event_id)
        unique_trace_ids = [int(trace_id) for trace_id in dict.fromkeys(trace_ids)]
        baseline_window_scale = self._processing_config(run)["fftWindowScale"]
        if not unique_trace_ids:
            return {
                "run": int(run),
                "eventId": int(event_id),
                "baselineWindowScale": float(baseline_window_scale),
                "traces": [],
            }

        trace_reader = self._trace_reader(run)
        rows = trace_reader.load_pad_rows(
            event_id,
            trace_ids=np.asarray(unique_trace_ids, dtype=np.int64),
        )
        cleaned = preprocess_traces(rows[:, 5:], baseline_window_scale=baseline_window_scale)
        self._schedule_prefetch(run, event_id)

        traces_payload: list[dict[str, Any]] = []
        for position, trace_id in enumerate(unique_trace_ids):
            trace_hits = hits[hits[:, 8].astype(np.int64) == int(trace_id)]
            traces_payload.append(
                {
                    "traceId": int(trace_id),
                    "raw": np.asarray(rows[position, 5:], dtype=np.float32).tolist(),
                    "trace": np.asarray(cleaned[position], dtype=np.float32).tolist(),
                    "peaks": [
                        {
                            "timeBucket": float(row[6]),
                            "amplitude": float(row[3]),
                            "integral": float(row[4]),
                            "z": float(row[2]),
                            "padId": int(row[5]),
                        }
                        for row in trace_hits
                    ],
                }
            )
        return {
            "run": int(run),
            "eventId": int(event_id),
            "baselineWindowScale": float(baseline_window_scale),
            "traces": traces_payload,
        }

    def _collect_event_ranges(self) -> dict[int, tuple[int, int]]:
        ranges: dict[int, tuple[int, int]] = {}
        for run in self.pointcloud_files:
            try:
                ranges[int(run)] = self._pointcloud_reader(run).get_range()
            except (KeyError, ValueError):
                continue
        return ranges

    def _require_event_range(self, run: int, event_id: int) -> tuple[int, int]:
        if run not in self.pointcloud_files:
            raise LookupError(f"pointcloud run not found: {run}")
        event_range = self._event_ranges.get(run)
        if event_range is None:
            raise LookupError(f"pointcloud run not found: {run}")
        if event_id < event_range[0] or event_id > event_range[1]:
            raise LookupError(f"pointcloud event not found: {run}/{event_id}")
        return event_range

    def _pointcloud_reader(self, run: int) -> PointcloudReader:
        reader = self._pointcloud_readers.get(run)
        if reader is None:
            reader = PointcloudReader(
                workspace=str(self.workspace),
                run=run,
                path=str(self.pointcloud_files[run]),
            )
            self._pointcloud_readers[run] = reader
        return reader

    def _trace_reader(self, run: int) -> RawTraceReader:
        reader = self._trace_readers.get(run)
        if reader is None:
            trace_file = self.trace_files.get(run)
            if trace_file is None:
                try:
                    trace_file = resolve_run_file(self.trace_path, run)
                except ValueError as exc:
                    raise LookupError(f"trace run not found: {run}") from exc
                self.trace_files[run] = trace_file
            reader = RawTraceReader(
                workspace=str(self.workspace),
                run=run,
                path=str(trace_file),
            )
            self._trace_readers[run] = reader
        return reader

    def _processing_config(self, run: int) -> dict[str, float]:
        cached = self._processing_cache.get(run)
        if cached is not None:
            return cached
        attrs = self._pointcloud_reader(run).read_processing_attrs()
        config = {
            "fftWindowScale": self._resolve_processing_value(
                run,
                attrs,
                attr_key="fft_window_scale",
                fallback=self.default_baseline_window_scale,
            ),
            "micromegasTimeBucket": self._resolve_processing_value(
                run,
                attrs,
                attr_key="micromegas_time_bucket",
                fallback=self.default_micromegas_time_bucket,
            ),
            "windowTimeBucket": self._resolve_processing_value(
                run,
                attrs,
                attr_key="window_time_bucket",
                fallback=self.default_window_time_bucket,
            ),
            "detectorLength": self._resolve_processing_value(
                run,
                attrs,
                attr_key="detector_length",
                fallback=self.default_detector_length,
            ),
        }
        self._processing_cache[run] = config
        return config

    def _resolve_processing_value(
        self,
        run: int,
        attrs: dict[str, object],
        *,
        attr_key: str,
        fallback: float | None,
    ) -> float:
        value = attrs.get(attr_key)
        if value is not None:
            return float(value)
        if fallback is not None:
            return float(fallback)
        raise ValueError(
            f"pointcloud processing config is missing '{attr_key}' for run {run}; "
            "set it in the pointcloud HDF5 attrs or in the WebUI config file"
        )

    def _load_event_hits(self, key: EventKey) -> np.ndarray | None:
        run, event_id = key
        if run not in self.pointcloud_files:
            return None
        try:
            _, hits = self._pointcloud_reader(run).read_event(event_id)
        except (KeyError, ValueError):
            return None
        return self._filter_hits_by_z(run, np.asarray(hits, dtype=np.float64))

    def _require_event_hits(self, run: int, event_id: int) -> np.ndarray:
        key = (int(run), int(event_id))
        cached = self._prefetcher.get_cached(key)
        if cached is not None:
            return cached
        hits = self._load_event_hits(key)
        if hits is None:
            raise LookupError(f"pointcloud event not found: {run}/{event_id}")
        self._prefetcher.store_current(key, hits)
        return hits

    def _filter_hits_by_z(self, run: int, hits: np.ndarray) -> np.ndarray:
        rows = np.asarray(hits, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] == 0:
            return rows
        detector_length = float(self._processing_config(run)["detectorLength"])
        mask = np.isfinite(rows[:, 2]) & (rows[:, 2] >= 0.0) & (rows[:, 2] <= detector_length)
        return rows[mask]

    def _event_window(self, run: int, event_id: int) -> list[EventKey]:
        event_range = self._event_ranges.get(run)
        if event_range is None:
            return []
        start = max(event_range[0], event_id - self.event_prefetch_radius)
        stop = min(event_range[1], event_id + self.event_prefetch_radius)
        return [(int(run), candidate) for candidate in range(start, stop + 1)]

    def _schedule_prefetch(self, run: int, event_id: int) -> None:
        self._prefetcher.schedule(self._event_window(run, event_id))

    def _wait_for_prefetch(self, timeout: float = 1.0) -> bool:
        return self._prefetcher.wait(timeout=timeout)


def _serialize_hit_row(
    row: np.ndarray,
    *,
    projected: tuple[float | None, float | None] = (None, None),
    merged_label: int = -1,
) -> dict[str, Any]:
    return {
        "traceId": int(row[8]),
        "x": float(row[0]),
        "y": float(row[1]),
        "z": float(row[2]),
        "xPrime": projected[0],
        "yPrime": projected[1],
        "amplitude": float(row[3]),
        "integral": float(row[4]),
        "padId": int(row[5]),
        "timeBucket": float(row[6]),
        "scale": float(row[7]),
        "mergedLabel": int(merged_label),
    }


def _project_hit_coordinates(hits: np.ndarray) -> list[tuple[float | None, float | None]]:
    rows = np.asarray(hits, dtype=np.float64)
    xyz = rows[:, :3] if rows.ndim == 2 and rows.shape[1] >= 3 else np.empty((0, 3))
    if xyz.shape[0] < 3:
        return [(None, None) for _ in range(int(rows.shape[0]))]

    centered = xyz - np.mean(xyz, axis=0)
    try:
        covariance = np.dot(centered.T, centered) / float(xyz.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return [(None, None) for _ in range(int(rows.shape[0]))]

    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[:2]]
    if basis.shape != (3, 2) or not np.all(np.isfinite(basis)):
        return [(None, None) for _ in range(int(rows.shape[0]))]

    projected = np.dot(centered, basis)
    return [
        (float(values[0]), float(values[1]))
        if np.all(np.isfinite(values))
        else (None, None)
        for values in projected
    ]


def _merged_cluster_labels(
    hits: np.ndarray,
    *,
    ransac_config: RansacConfig,
    merge_config: MergeConfig,
) -> tuple[list[int], int]:
    rows = np.asarray(hits, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return [], 0
    clusters, _ = extract_line_clusters(rows, ransac_config=ransac_config)
    merged_clusters = merge_line_clusters(clusters, merge_config=merge_config)
    labels = np.full(rows.shape[0], -1, dtype=np.int64)
    for cluster_index, cluster in enumerate(merged_clusters):
        labels[np.asarray(cluster.inlier_indices, dtype=np.int64)] = cluster_index
    return labels.astype(int).tolist(), len(merged_clusters)


def _pointcloud_bucket_from_count(count: int) -> str:
    if count <= 0:
        return "0"
    if count >= 6:
        return "6+"
    return str(int(count))
