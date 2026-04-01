from __future__ import annotations

import h5py
import numpy as np
from pathlib import Path
import random
import threading
import time
from typing import Literal

from ..models import TraceRecord

TraceKey = tuple[int, int]
CACHE_RADIUS = 5
MAX_RANDOM_TRACE_ATTEMPTS = 1024


class TraceSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.run = int(path.stem.split("_")[1])
        self.file = h5py.File(path, "r")
        self.min_event = self.file["events"].attrs["min_event"]
        self.max_event = self.file["events"].attrs["max_event"]
        self.bad_events = {int(event_id) for event_id in self.file["events"].attrs["bad_events"]}
        self.label_event_stack: list[TraceKey] = []
        self.label_stack_pointer = 0
        self.review_event_stack: list[TraceKey] = []
        self.review_stack_pointer = 0
        self.trace_mode: Literal["label", "review"] = "label"
        self.review_filter: dict[str, str] | None = None
        self.labeled_traces: dict[tuple[int, int], tuple[str, str]] = {}
        self.trace_cache: dict[TraceKey, TraceRecord] = {}
        self._cache_lock = threading.Lock()
        self._prefetch_condition = threading.Condition()
        self._prefetch_requested = False
        self._prefetch_active = False
        self._prefetch_closed = False
        self._desired_cache_keys: tuple[TraceKey, ...] = ()
        self._scheduled_window_keys: tuple[TraceKey, ...] = ()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            name="trace-source-prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

    def get_run(self) -> int:
        return self.run

    def get_review_progress(self) -> dict[str, int] | None:
        if self.trace_mode != "review" or not self.review_event_stack or self.review_stack_pointer == 0:
            return None
        return {
            "current": min(self.review_stack_pointer, len(self.review_event_stack)),
            "total": len(self.review_event_stack),
        }

    def set_trace_mode(self, mode: Literal["label", "review"], family: str | None = None, label: str | None = None) -> int:
        self.trace_mode = mode
        if mode == "label":
            self.review_filter = None
            self._schedule_cache_refresh()
            return len(self.label_event_stack)
        self.review_filter = {"family": family, "label": label or ""}
        self.review_event_stack = self._build_review_stack(family=family, label=label)
        self.review_stack_pointer = 0
        self._schedule_cache_refresh()
        return len(self.review_event_stack)

    def get_trace(self, event_id: int, trace_id: int) -> TraceRecord | None:
        key = (event_id, trace_id)
        with self._cache_lock:
            cached = self.trace_cache.get(key)
        if cached is not None:
            return cached
        record = self._load_trace_record(event_id, trace_id)
        if record is not None:
            with self._cache_lock:
                self.trace_cache[key] = record
        return record

    def _load_trace_record(self, event_id: int, trace_id: int) -> TraceRecord | None:
        return self._load_trace_record_from_file(self.file, event_id, trace_id)

    def _load_trace_record_from_file(
        self,
        file_handle: h5py.File,
        event_id: int,
        trace_id: int,
    ) -> TraceRecord | None:
        if (
            event_id < self.min_event
            or event_id > self.max_event
            or event_id in self.bad_events
        ):
            return None
        pads = file_handle["events"][f"event_{event_id}"]["get"]["pads"]
        if trace_id >= pads.shape[0]:
            return None
        if (event_id, trace_id) in self.labeled_traces:
            family, label = self.labeled_traces[(event_id, trace_id)]
        else:
            family = None
            label = None
        hardware, raw_trace = self._extract_trace_parts(file_handle, event_id, trace_id)
        trace = self.preprocess_traces(raw_trace, baseline_window_scale=10.0)
        transformed = self.transform_trace(trace)
        return TraceRecord(
            run=self.run,
            event_id=event_id,
            trace_id=trace_id,
            detector="pad",
            hardware_id=hardware,
            raw=raw_trace,
            trace=trace,
            transformed=transformed,
            family=family,
            label=label,
        )

    def preprocess_traces(self, traces: np.ndarray, baseline_window_scale: float) -> np.ndarray:
        """JIT-ed Method for pre-cleaning the trace data in bulk before doing trace analysis

        These methods are more suited to operating on the entire dataset rather than on a trace by trace basis
        It includes

        - Removal of edge effects in traces (first and last time buckets can be noisy)
        - Baseline removal via fourier transform method (see J. Bradt thesis, pytpc library)

        Parameters
        ----------
        traces: ndarray
            A single trace or a (n, samples) matrix where each row corresponds to a trace.
        baseline_window_scale: float
            The scale of the baseline filter used to perform a moving average over the basline

        Returns
        -------
        ndarray
            A new trace or matrix which contains the traces with their baselines removed and
            edges smoothed
        """
        traces_array = np.array(traces, dtype=np.float32, copy=True)
        single_trace = traces_array.ndim == 1
        trace_matrix = np.atleast_2d(traces_array)
        sample_count = trace_matrix.shape[1]

        if sample_count < 2:
            return traces_array

        # Smooth out the edges of the traces
        trace_matrix[:, 0] = trace_matrix[:, 1]
        trace_matrix[:, -1] = trace_matrix[:, -2]

        # Remove peaks from baselines and replace with average
        bases = trace_matrix.copy()
        for row in bases:
            mean = np.mean(row)
            sigma = np.std(row)
            mask = np.abs(row - mean) > sigma * 1.5
            if np.any(~mask):
                row[mask] = np.mean(row[~mask])
            else:
                row.fill(mean)

        # Create the filter
        window = np.arange(sample_count, dtype=np.float32) - (sample_count // 2)
        fil = np.fft.ifftshift(np.sinc(window / baseline_window_scale))
        transformed = np.fft.fft(bases, axis=1)
        result = np.real(
            np.fft.ifft(transformed * fil[np.newaxis, :], axis=1)
        )  # Apply the filter -> multiply in Fourier = convolve in normal

        processed = trace_matrix - result
        if single_trace:
            return processed[0]
        return processed

    def transform_trace(self, trace: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(trace))

    def _extract_trace_parts(self, file_handle: h5py.File, event_id: int, trace_id: int) -> tuple:
        pads = file_handle["events"][f"event_{event_id}"]["get"]["pads"]
        row = pads[trace_id]
        hardware = row[:5].copy()
        trace = row[5:].copy()
        return hardware, trace

    def _random_trace(self) -> TraceKey:
        for _ in range(MAX_RANDOM_TRACE_ATTEMPTS):
            event_id = random.randint(int(self.min_event), int(self.max_event))
            if event_id in self.bad_events:
                continue
            pads = self.file["events"][f"event_{event_id}"]["get"]["pads"]
            if pads.shape[0] == 0:
                continue
            trace_id = random.randrange(int(pads.shape[0]))
            key = (event_id, trace_id)
            if key in self.labeled_traces or key in self.label_event_stack:
                continue
            return key
        raise LookupError("no unlabeled trace was found after repeated random sampling")

    def _build_review_stack(self, family: str, label: str | None) -> list[TraceKey]:
        return sorted(
            key
            for key, stored_label in self.labeled_traces.items()
            if stored_label[0] == family and self._label_matches_review_filter(family, stored_label[1], label)
        )

    @staticmethod
    def _label_matches_review_filter(family: str, stored_label: str, requested_label: str | None) -> bool:
        if requested_label is None:
            return True
        if family == "normal" and requested_label == "4+":
            return stored_label in {"4", "5", "6", "7", "8", "9"}
        return stored_label == requested_label

    def _refresh_review_stack_after_relabel(self, current_key: TraceKey) -> None:
        if self.review_filter is None:
            return
        previous_pointer = self.review_stack_pointer
        family = self.review_filter["family"]
        label = self.review_filter["label"] or None
        self.review_event_stack = self._build_review_stack(family=family, label=label)
        if not self.review_event_stack:
            self.review_stack_pointer = 0
            return
        if current_key in self.review_event_stack:
            self.review_stack_pointer = self.review_event_stack.index(current_key) + 1
            return
        self.review_stack_pointer = min(previous_pointer, len(self.review_event_stack))

    def _next_label_trace(self) -> TraceRecord:
        if self.label_stack_pointer < len(self.label_event_stack):
            event_id, trace_id = self.label_event_stack[self.label_stack_pointer]
            self.label_stack_pointer += 1
            return self.get_trace(event_id, trace_id)
        event_id, trace_id = self._random_trace()
        self.label_event_stack.append((event_id, trace_id))
        self.label_stack_pointer += 1
        return self.get_trace(event_id, trace_id)

    def _previous_label_trace(self) -> TraceRecord:
        if not self.label_event_stack:
            raise LookupError("no labeled-mode trace history is available")
        if self.label_stack_pointer > 1:
            self.label_stack_pointer -= 1
        event_id, trace_id = self.label_event_stack[self.label_stack_pointer - 1]
        return self.get_trace(event_id, trace_id)

    def _next_review_trace(self) -> TraceRecord:
        if not self.review_event_stack:
            raise LookupError("no traces match the selected review filter")
        if self.review_stack_pointer >= len(self.review_event_stack):
            event_id, trace_id = self.review_event_stack[-1]
            return self.get_trace(event_id, trace_id)
        event_id, trace_id = self.review_event_stack[self.review_stack_pointer]
        self.review_stack_pointer += 1
        return self.get_trace(event_id, trace_id)

    def _previous_review_trace(self) -> TraceRecord:
        if not self.review_event_stack:
            raise LookupError("no traces match the selected review filter")
        if self.review_stack_pointer <= 1:
            event_id, trace_id = self.review_event_stack[0]
            self.review_stack_pointer = min(1, len(self.review_event_stack))
            return self.get_trace(event_id, trace_id)
        self.review_stack_pointer -= 1
        event_id, trace_id = self.review_event_stack[self.review_stack_pointer - 1]
        return self.get_trace(event_id, trace_id)

    def next_trace(self) -> TraceRecord:
        if self.trace_mode == "review":
            record = self._next_review_trace()
        else:
            record = self._next_label_trace()
        self._schedule_cache_refresh()
        return record

    def previous_trace(self) -> TraceRecord:
        if self.trace_mode == "review":
            record = self._previous_review_trace()
        else:
            record = self._previous_label_trace()
        self._schedule_cache_refresh()
        return record

    def label_trace(self, event_id: int, trace_id: int, family: str, label: str) -> None:
        self.labeled_traces[(event_id, trace_id)] = (family, label)
        with self._cache_lock:
            cached = self.trace_cache.get((event_id, trace_id))
        if cached is not None:
            cached.family = family
            cached.label = label
        if self.trace_mode == "review":
            self._refresh_review_stack_after_relabel((event_id, trace_id))
        else:
            self._prune_prefetched_labeled_traces()
        self._schedule_cache_refresh()

    def set_labeled(self, labeled: dict[tuple[int, int], tuple[str, str]]) -> None:
        self.labeled_traces = labeled.copy()

    def _prune_prefetched_labeled_traces(self) -> None:
        if self.label_stack_pointer >= len(self.label_event_stack):
            return
        history = self.label_event_stack[:self.label_stack_pointer]
        prefetched = [
            key
            for key in self.label_event_stack[self.label_stack_pointer:]
            if key not in self.labeled_traces
        ]
        self.label_event_stack = history + prefetched

    def _current_trace_key(self) -> TraceKey | None:
        if self.trace_mode == "review":
            stack = self.review_event_stack
            pointer = self.review_stack_pointer
        else:
            stack = self.label_event_stack
            pointer = self.label_stack_pointer
        if not stack or pointer <= 0:
            return None
        current_index = min(pointer, len(stack)) - 1
        return stack[current_index]

    def _ensure_label_prefetch(self) -> None:
        current_key = self._current_trace_key()
        if current_key is None or self.trace_mode != "label":
            return
        current_index = self.label_stack_pointer - 1
        required_size = current_index + CACHE_RADIUS + 1
        while len(self.label_event_stack) < required_size:
            try:
                self.label_event_stack.append(self._random_trace())
            except LookupError:
                break

    def _cache_window_keys(self) -> list[TraceKey]:
        current_key = self._current_trace_key()
        if current_key is None:
            return []
        if self.trace_mode == "review":
            stack = self.review_event_stack
            current_index = self.review_stack_pointer - 1
        else:
            self._ensure_label_prefetch()
            stack = self.label_event_stack
            current_index = self.label_stack_pointer - 1
        start = max(0, current_index - CACHE_RADIUS)
        stop = min(len(stack), current_index + CACHE_RADIUS + 1)
        return stack[start:stop]

    def _schedule_cache_refresh(self) -> None:
        window_keys = tuple(self._cache_window_keys())
        desired_keys = set(window_keys)
        with self._cache_lock:
            self._desired_cache_keys = window_keys
            self.trace_cache = {
                key: record
                for key, record in self.trace_cache.items()
                if key in desired_keys
            }
        with self._prefetch_condition:
            self._scheduled_window_keys = window_keys
            self._prefetch_requested = True
            self._prefetch_condition.notify()

    def _prefetch_loop(self) -> None:
        with h5py.File(self.path, "r") as prefetch_file:
            while True:
                with self._prefetch_condition:
                    while not self._prefetch_requested and not self._prefetch_closed:
                        self._prefetch_condition.wait()
                    if self._prefetch_closed:
                        return
                    window_keys = self._scheduled_window_keys
                    self._prefetch_requested = False
                    self._prefetch_active = True

                self._prefetch_window(prefetch_file, window_keys)

                with self._prefetch_condition:
                    self._prefetch_active = False
                    if not self._prefetch_requested:
                        self._prefetch_condition.notify_all()

    def _prefetch_window(
        self,
        file_handle: h5py.File,
        window_keys: tuple[TraceKey, ...],
    ) -> None:
        desired_keys = set(window_keys)
        if not desired_keys:
            with self._cache_lock:
                self.trace_cache.clear()
            return

        for key in window_keys:
            with self._prefetch_condition:
                if self._prefetch_requested or self._prefetch_closed:
                    return

            with self._cache_lock:
                if key in self.trace_cache or key not in self._desired_cache_keys:
                    continue

            record = self._load_trace_record_from_file(file_handle, *key)
            if record is None:
                continue

            current_label = self.labeled_traces.get(key)
            if current_label is None:
                record.family = None
                record.label = None
            else:
                record.family, record.label = current_label

            with self._cache_lock:
                if key not in self._desired_cache_keys:
                    continue
                self.trace_cache[key] = record
                self.trace_cache = {
                    cached_key: cached_record
                    for cached_key, cached_record in self.trace_cache.items()
                    if cached_key in self._desired_cache_keys
                }

    def _wait_for_prefetch(self, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._prefetch_condition:
            while (self._prefetch_requested or self._prefetch_active) and time.monotonic() < deadline:
                self._prefetch_condition.wait(timeout=max(0.0, deadline - time.monotonic()))
            return not self._prefetch_requested and not self._prefetch_active

    def close(self) -> None:
        with self._prefetch_condition:
            self._prefetch_closed = True
            self._prefetch_condition.notify_all()
        self._prefetch_thread.join(timeout=1.0)
        self.file.close()
