from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from attpc_storage.hdf5 import PointcloudReader, RawTraceReader

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
        event_prefetch_radius: int = EVENT_PREFETCH_RADIUS,
    ) -> None:
        self.trace_path = trace_path
        self.workspace = workspace
        self.default_baseline_window_scale = float(baseline_window_scale)
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

    def get_event(self, *, run: int, event_id: int) -> dict[str, Any]:
        event_range = self._require_event_range(run, event_id)
        hits = self._require_event_hits(run, event_id)
        self._schedule_prefetch(run, event_id)
        return {
            "run": int(run),
            "eventId": int(event_id),
            "eventIdRange": {"min": int(event_range[0]), "max": int(event_range[1])},
            "hits": [_serialize_hit_row(row) for row in hits],
            "processing": self._processing_config(run),
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
            trace_hits = hits[hits[:, 0].astype(np.int64) == int(trace_id)]
            traces_payload.append(
                {
                    "traceId": int(trace_id),
                    "raw": np.asarray(rows[position, 5:], dtype=np.float32).tolist(),
                    "trace": np.asarray(cleaned[position], dtype=np.float32).tolist(),
                    "peaks": [
                        {
                            "timeBucket": float(row[7]),
                            "amplitude": float(row[4]),
                            "integral": float(row[5]),
                            "z": float(row[3]),
                            "padId": int(row[6]),
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
            "fftWindowScale": float(attrs.get("fft_window_scale", self.default_baseline_window_scale)),
            "micromegasTimeBucket": float(attrs.get("micromegas_time_bucket", 10.0)),
            "windowTimeBucket": float(attrs.get("window_time_bucket", 560.0)),
            "detectorLength": float(attrs.get("detector_length", 1000.0)),
        }
        self._processing_cache[run] = config
        return config

    def _load_event_hits(self, key: EventKey) -> np.ndarray | None:
        run, event_id = key
        if run not in self.pointcloud_files:
            return None
        try:
            _, hits = self._pointcloud_reader(run).read_event(event_id)
        except (KeyError, ValueError):
            return None
        return np.asarray(hits, dtype=np.float64)

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


def _serialize_hit_row(row: np.ndarray) -> dict[str, Any]:
    return {
        "traceId": int(row[0]),
        "x": float(row[1]),
        "y": float(row[2]),
        "z": float(row[3]),
        "amplitude": float(row[4]),
        "integral": float(row[5]),
        "padId": int(row[6]),
        "timeBucket": float(row[7]),
        "scale": float(row[8]),
    }
