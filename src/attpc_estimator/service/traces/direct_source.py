from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np

from ...model.label import StoredLabel
from ...model.trace import TraceRecord, TraceRef
from ...utils.trace_data import (
    describe_trace_events,
    event_trace_count,
    load_pad_rows,
    trace_record_from_pad_row,
)

EVENT_PREFETCH_RADIUS = 2


class DirectTraceSource:
    def __init__(
        self,
        trace_file: Path,
        *,
        run: int,
        labels: Mapping[TraceRef, StoredLabel] | None = None,
        baseline_window_scale: float = 10.0,
        event_prefetch_radius: int = EVENT_PREFETCH_RADIUS,
    ) -> None:
        self.trace_file = trace_file.resolve()
        self.run = int(run)
        self.baseline_window_scale = baseline_window_scale
        self.event_prefetch_radius = max(0, int(event_prefetch_radius))
        self._labels = dict(labels or {})
        self._handle = h5py.File(self.trace_file, "r")
        metadata = describe_trace_events(self._handle)
        self._min_event = metadata.min_event
        self._max_event = metadata.max_event
        self._bad_events = set(metadata.bad_events)
        self._event_count_cache: dict[int, int] = {}
        self._event_rows_cache: dict[int, np.ndarray] = {}
        self._current_event_id: int | None = None
        self._current_trace_id: int | None = None

    def replace_labels(self, labels: Mapping[TraceRef, StoredLabel]) -> None:
        self._labels = dict(labels)

    def close(self) -> None:
        self._event_rows_cache.clear()
        self._event_count_cache.clear()
        self._handle.close()

    def get_progress(self) -> None:
        return None

    def current_trace(self) -> TraceRecord | None:
        if self._current_event_id is None or self._current_trace_id is None:
            return None
        return self._record_for(self._current_event_id, self._current_trace_id)

    def set_position(self, *, event_id: int, trace_id: int) -> TraceRecord:
        trace_count = self._require_event_trace_count(int(event_id))
        normalized_trace_id = int(trace_id)
        if normalized_trace_id < 0 or normalized_trace_id >= trace_count:
            raise LookupError(
                f"trace {self.run}/{int(event_id)}/{normalized_trace_id} is not available"
            )
        self._current_event_id = int(event_id)
        self._current_trace_id = normalized_trace_id
        self._warm_cache(self._current_event_id)
        return self._record_for(self._current_event_id, self._current_trace_id)

    def next_trace(self) -> TraceRecord:
        self._require_current_position()
        assert self._current_event_id is not None
        assert self._current_trace_id is not None
        trace_count = self._require_event_trace_count(self._current_event_id)
        if self._current_trace_id + 1 < trace_count:
            self._current_trace_id += 1
        return self._record_for(self._current_event_id, self._current_trace_id)

    def previous_trace(self) -> TraceRecord:
        self._require_current_position()
        assert self._current_event_id is not None
        assert self._current_trace_id is not None
        if self._current_trace_id > 0:
            self._current_trace_id -= 1
        return self._record_for(self._current_event_id, self._current_trace_id)

    def next_event(self) -> TraceRecord:
        self._require_current_position()
        assert self._current_event_id is not None
        next_event_id = self._find_valid_event(self._current_event_id + 1, step=1)
        if next_event_id is None:
            return self.current_trace_or_raise()
        self._move_to_event(next_event_id)
        return self.current_trace_or_raise()

    def previous_event(self) -> TraceRecord:
        self._require_current_position()
        assert self._current_event_id is not None
        previous_event_id = self._find_valid_event(self._current_event_id - 1, step=-1)
        if previous_event_id is None:
            return self.current_trace_or_raise()
        self._move_to_event(previous_event_id)
        return self.current_trace_or_raise()

    def current_event_trace_count(self) -> int | None:
        if self._current_event_id is None:
            return None
        return self._require_event_trace_count(self._current_event_id)

    def event_id_range(self) -> dict[str, int]:
        return {"min": self._min_event, "max": self._max_event}

    def current_trace_or_raise(self) -> TraceRecord:
        record = self.current_trace()
        if record is None:
            raise LookupError("no direct trace is selected")
        return record

    def snapshot_state(self) -> dict[str, int | None]:
        return {
            "eventId": self._current_event_id,
            "traceId": self._current_trace_id,
        }

    def restore_state(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        event_id = payload.get("eventId")
        trace_id = payload.get("traceId")
        if not isinstance(event_id, int) or not isinstance(trace_id, int):
            return
        self.set_position(event_id=event_id, trace_id=trace_id)

    def _require_current_position(self) -> None:
        if self._current_event_id is None or self._current_trace_id is None:
            raise LookupError("no direct trace is selected")

    def _move_to_event(self, event_id: int) -> None:
        trace_count = self._require_event_trace_count(event_id)
        assert self._current_trace_id is not None
        self._current_event_id = event_id
        self._current_trace_id = min(self._current_trace_id, trace_count - 1)
        self._warm_cache(event_id)

    def _record_for(self, event_id: int, trace_id: int) -> TraceRecord:
        rows = self._get_event_rows(event_id)
        ref = TraceRef(run=self.run, event_id=event_id, trace_id=trace_id)
        record = trace_record_from_pad_row(
            run=self.run,
            event_id=event_id,
            trace_id=trace_id,
            row=rows[trace_id],
            baseline_window_scale=self.baseline_window_scale,
        )
        stored_label = self._labels.get(ref)
        if stored_label is None:
            record.family = None
            record.label = None
        else:
            record.family = stored_label.family
            record.label = stored_label.label
        return record

    def _warm_cache(self, center_event_id: int) -> None:
        desired_event_ids = [center_event_id]
        desired_event_ids.extend(
            self._neighbor_events(center_event_id, step=1, limit=self.event_prefetch_radius)
        )
        desired_event_ids.extend(
            self._neighbor_events(center_event_id, step=-1, limit=self.event_prefetch_radius)
        )
        for event_id in desired_event_ids:
            self._get_event_rows(event_id)
        desired = set(desired_event_ids)
        self._event_rows_cache = {
            event_id: rows
            for event_id, rows in self._event_rows_cache.items()
            if event_id in desired
        }

    def _neighbor_events(self, start_event_id: int, *, step: int, limit: int) -> list[int]:
        event_ids: list[int] = []
        event_id = start_event_id + step
        while len(event_ids) < limit and self._min_event <= event_id <= self._max_event:
            if self._require_event_trace_count(event_id, allow_missing=True) > 0:
                event_ids.append(event_id)
            event_id += step
        return event_ids

    def _find_valid_event(self, start_event_id: int, *, step: int) -> int | None:
        event_id = start_event_id
        while self._min_event <= event_id <= self._max_event:
            if self._require_event_trace_count(event_id, allow_missing=True) > 0:
                return event_id
            event_id += step
        return None

    def _get_event_rows(self, event_id: int) -> np.ndarray:
        cached = self._event_rows_cache.get(event_id)
        if cached is not None:
            return cached
        rows = load_pad_rows(
            self._handle,
            run=self.run,
            event_id=event_id,
            trace_ids=None,
        )
        self._event_rows_cache[event_id] = rows
        self._event_count_cache[event_id] = int(rows.shape[0])
        return rows

    def _require_event_trace_count(
        self,
        event_id: int,
        *,
        allow_missing: bool = False,
    ) -> int:
        if event_id in self._event_count_cache:
            return self._event_count_cache[event_id]
        if event_id < self._min_event or event_id > self._max_event or event_id in self._bad_events:
            if allow_missing:
                self._event_count_cache[event_id] = 0
                return 0
            raise LookupError(f"event {self.run}/{event_id} is not available")
        trace_count = event_trace_count(self._handle, event_id)
        self._event_count_cache[event_id] = int(trace_count)
        if trace_count <= 0 and not allow_missing:
            raise LookupError(f"event {self.run}/{event_id} is not available")
        return int(trace_count)
