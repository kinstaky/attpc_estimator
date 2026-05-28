from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ...model.label import StoredLabel
from ...model.trace import TraceRecord, TraceRef
from ...utils.trace_data import (
    DETECTOR_ATTPC,
    DETECTOR_GAGG,
    DETECTOR_IC,
    DETECTOR_SI,
    PAD_TRACE_OFFSET,
    gagg_event_selector,
    gagg_layer_counts,
    gagg_trace_id_from_selector,
    load_reader_trace_record,
    normalize_detector,
    open_storage_trace_reader,
    reader_event_trace_count,
    reader_event_rows,
    si_side_counts,
    si_trace_id_from_selector,
    si_trace_selector,
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
        detector: str = DETECTOR_ATTPC,
        filter_item: str = "none",
        filter_value: float | None = None,
    ) -> None:
        self.trace_file = trace_file.resolve()
        self.run = int(run)
        self.baseline_window_scale = baseline_window_scale
        self.event_prefetch_radius = max(0, int(event_prefetch_radius))
        self.detector = normalize_detector(detector)
        self.filter_item = self._normalize_filter_item(filter_item)
        self.filter_value = None if filter_value is None else float(filter_value)
        self._labels = dict(labels or {})
        self._reader = open_storage_trace_reader(
            run=self.run,
            path=str(self.trace_file),
            detector=self.detector,
        )
        self._min_event = int(self._reader.min_event)
        self._max_event = int(self._reader.max_event)
        self._bad_events = set(getattr(self._reader, "bad_events", set()))
        self._event_count_cache: dict[int, int] = {}
        self._event_rows_cache: dict[int, np.ndarray] = {}
        self._filtered_trace_cache: dict[int, list[int]] = {}
        self._current_event_id: int | None = None
        self._current_trace_id: int | None = None
        layout = getattr(self._reader, "layout", None)
        if str(getattr(layout, "value", "")) == "legacy_merger" and self.detector in {
            DETECTOR_SI,
            DETECTOR_GAGG,
        }:
            raise ValueError(
                f"{self.detector} direct trace review is not available for legacy trace files"
            )

    def replace_labels(self, labels: Mapping[TraceRef, StoredLabel]) -> None:
        self._labels = dict(labels)

    def close(self) -> None:
        self._event_rows_cache.clear()
        self._event_count_cache.clear()
        self._filtered_trace_cache.clear()
        self._reader.close()

    def get_progress(self) -> None:
        return None

    def current_trace(self) -> TraceRecord | None:
        if self._current_event_id is None or self._current_trace_id is None:
            return None
        return self._record_for(self._current_event_id, self._current_trace_id)

    def set_position(
        self,
        *,
        event_id: int,
        trace_id: int | None = None,
        si_side: str | None = None,
        si_index: int | None = None,
        gagg_layer: int | None = None,
        gagg_index: int | None = None,
    ) -> TraceRecord:
        normalized_trace_id = self.resolve_trace_id(
            event_id=int(event_id),
            trace_id=trace_id,
            si_side=si_side,
            si_index=si_index,
            gagg_layer=gagg_layer,
            gagg_index=gagg_index,
        )
        trace_count = self._require_event_trace_count(int(event_id))
        if normalized_trace_id < 0 or normalized_trace_id >= trace_count:
            trace_label = "IC trace" if self.detector == DETECTOR_IC else "trace"
            raise LookupError(
                f"{trace_label} {self.run}/{int(event_id)}/{normalized_trace_id} is not available"
            )
        normalized_trace_id = self._resolve_filtered_trace_id(
            int(event_id),
            normalized_trace_id,
            allow_fallback=True,
        )
        self._current_event_id = int(event_id)
        self._current_trace_id = normalized_trace_id
        self._warm_cache(self._current_event_id)
        return self._record_for(self._current_event_id, self._current_trace_id)

    def next_trace(self) -> TraceRecord:
        self._require_current_position()
        assert self._current_event_id is not None
        assert self._current_trace_id is not None
        matching_ids = self._matching_trace_ids(self._current_event_id)
        try:
            current_index = matching_ids.index(self._current_trace_id)
        except ValueError:
            self._current_trace_id = matching_ids[0]
            return self._record_for(self._current_event_id, self._current_trace_id)
        if current_index + 1 < len(matching_ids):
            self._current_trace_id = matching_ids[current_index + 1]
        return self._record_for(self._current_event_id, self._current_trace_id)

    def previous_trace(self) -> TraceRecord:
        self._require_current_position()
        assert self._current_event_id is not None
        assert self._current_trace_id is not None
        matching_ids = self._matching_trace_ids(self._current_event_id)
        try:
            current_index = matching_ids.index(self._current_trace_id)
        except ValueError:
            self._current_trace_id = matching_ids[0]
            return self._record_for(self._current_event_id, self._current_trace_id)
        if current_index > 0:
            self._current_trace_id = matching_ids[current_index - 1]
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

    def current_trace_selector(self) -> dict[str, object] | None:
        if self._current_event_id is None or self._current_trace_id is None:
            return None
        event_rows = self._get_event_rows_or_none(self._current_event_id)
        if self.detector == DETECTOR_SI and event_rows is not None:
            return si_trace_selector(event_rows, self._current_trace_id)
        if self.detector == DETECTOR_GAGG and event_rows is not None:
            return gagg_event_selector(event_rows, self._current_trace_id)
        return None

    def event_context(self) -> dict[str, object] | None:
        if self._current_event_id is None:
            return None
        previous_event_id = self._find_valid_event(self._current_event_id - 1, step=-1)
        next_event_id = self._find_valid_event(self._current_event_id + 1, step=1)
        return {
            "current": self._describe_event(self._current_event_id),
            "previous": (
                self._describe_event(previous_event_id)
                if previous_event_id is not None
                else None
            ),
            "next": self._describe_event(next_event_id) if next_event_id is not None else None,
        }

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
        if not isinstance(event_id, int):
            return
        if self.detector != DETECTOR_IC and not isinstance(trace_id, int):
            return
        self.set_position(
            event_id=event_id,
            trace_id=0 if self.detector == DETECTOR_IC else trace_id,
        )

    def resolve_trace_id(
        self,
        *,
        event_id: int,
        trace_id: int | None = None,
        si_side: str | None = None,
        si_index: int | None = None,
        gagg_layer: int | None = None,
        gagg_index: int | None = None,
    ) -> int:
        if self.detector == DETECTOR_IC:
            return 0
        if self.detector == DETECTOR_SI and si_side is not None and si_index is not None:
            rows = self._get_event_rows(event_id)
            return si_trace_id_from_selector(rows, side=si_side, index=int(si_index))
        if self.detector == DETECTOR_GAGG and gagg_layer is not None and gagg_index is not None:
            rows = self._get_event_rows(event_id)
            _ = rows
            return gagg_trace_id_from_selector(layer=int(gagg_layer), index=int(gagg_index))
        if trace_id is None:
            return 0
        return int(trace_id)

    def _require_current_position(self) -> None:
        if self._current_event_id is None or self._current_trace_id is None:
            raise LookupError("no direct trace is selected")

    def _move_to_event(self, event_id: int) -> None:
        assert self._current_trace_id is not None
        self._current_event_id = event_id
        self._current_trace_id = self._resolve_filtered_trace_id(
            event_id,
            self._current_trace_id,
            allow_fallback=True,
        )
        self._warm_cache(event_id)

    def _record_for(self, event_id: int, trace_id: int) -> TraceRecord:
        ref = TraceRef(run=self.run, event_id=event_id, trace_id=trace_id)
        record = load_reader_trace_record(
            self._reader,
            run=self.run,
            event_id=event_id,
            trace_id=trace_id,
            baseline_window_scale=self.baseline_window_scale,
            detector=self.detector,
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
            self._get_event_rows_or_none(event_id)
        desired = set(desired_event_ids)
        self._event_rows_cache = {
            event_id: rows
            for event_id, rows in self._event_rows_cache.items()
            if event_id in desired
        }
        self._filtered_trace_cache = {
            event_id: trace_ids
            for event_id, trace_ids in self._filtered_trace_cache.items()
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
            if self._matching_trace_count(event_id) > 0:
                return event_id
            event_id += step
        return None

    def _matching_trace_count(self, event_id: int) -> int:
        try:
            return len(self._matching_trace_ids(event_id))
        except LookupError:
            return 0

    def _get_event_rows(self, event_id: int) -> np.ndarray:
        cached = self._event_rows_cache.get(event_id)
        if cached is not None:
            return cached
        if self.detector == DETECTOR_IC:
            raise LookupError("IC direct review does not expose stacked event rows")
        rows = reader_event_rows(self._reader, event_id, detector=self.detector)
        self._event_rows_cache[event_id] = rows
        self._event_count_cache[event_id] = int(rows.shape[0])
        return rows

    def _get_event_rows_or_none(self, event_id: int) -> np.ndarray | None:
        if self.detector == DETECTOR_IC:
            return None
        try:
            return self._get_event_rows(event_id)
        except LookupError:
            return None

    def _matching_trace_ids(self, event_id: int) -> list[int]:
        cached = self._filtered_trace_cache.get(event_id)
        if cached is not None:
            return cached
        trace_count = self._require_event_trace_count(event_id, allow_missing=True)
        if trace_count <= 0:
            self._filtered_trace_cache[event_id] = []
            return []
        if self.filter_item == "none":
            trace_ids = list(range(trace_count))
            self._filtered_trace_cache[event_id] = trace_ids
            return trace_ids
        threshold = float(self.filter_value or 0.0)
        trace_ids = [
            trace_id
            for trace_id in range(trace_count)
            if self._trace_max_value(event_id, trace_id) > threshold
        ]
        self._filtered_trace_cache[event_id] = trace_ids
        return trace_ids

    def _trace_max_value(self, event_id: int, trace_id: int) -> float:
        if self.detector == DETECTOR_IC:
            record = self._record_for(event_id, 0)
            return float(np.max(record.raw)) if record.raw.size else 0.0
        rows = self._get_event_rows(event_id)
        row = np.asarray(rows[int(trace_id)], dtype=np.float32)
        if self.detector == DETECTOR_ATTPC:
            samples = row[PAD_TRACE_OFFSET:]
        elif self.detector == DETECTOR_SI:
            samples = row[2:]
        else:
            samples = row
        return float(np.max(samples)) if samples.size else 0.0

    def _resolve_filtered_trace_id(
        self,
        event_id: int,
        trace_id: int,
        *,
        allow_fallback: bool,
    ) -> int:
        matching_ids = self._matching_trace_ids(event_id)
        if not matching_ids:
            raise LookupError(
                f"no {self.detector} traces match the current filter in event {self.run}/{event_id}"
            )
        normalized_trace_id = int(trace_id)
        if normalized_trace_id in matching_ids:
            return normalized_trace_id
        if not allow_fallback:
            raise LookupError(
                f"{self.detector} trace {self.run}/{event_id}/{normalized_trace_id} does not match the current filter"
            )
        for candidate in matching_ids:
            if candidate >= normalized_trace_id:
                return candidate
        return matching_ids[-1]

    @staticmethod
    def _normalize_filter_item(filter_item: str | None) -> str:
        token = str(filter_item or "none").strip().lower()
        if token not in {"none", "max"}:
            raise ValueError("filterItem must be 'none' or 'max'")
        return token
        try:
            return self._get_event_rows(event_id)
        except LookupError:
            return None

    def _describe_event(self, event_id: int) -> dict[str, object]:
        trace_count = self._require_event_trace_count(event_id)
        selector: dict[str, object]
        if self.detector == DETECTOR_IC:
            selector = {"kind": "single_trace"}
        elif self.detector == DETECTOR_SI:
            rows = self._get_event_rows(event_id)
            selector = {"kind": "si", "sideCounts": si_side_counts(rows)}
        elif self.detector == DETECTOR_GAGG:
            rows = self._get_event_rows(event_id)
            selector = {"kind": "gagg", "layerCounts": gagg_layer_counts(rows)}
        else:
            selector = {"kind": "trace_id"}
        return {
            "eventId": int(event_id),
            "traceCount": int(trace_count),
            "traceIdMin": 0,
            "traceIdMax": max(0, int(trace_count) - 1),
            "selector": selector,
        }

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
        trace_count = reader_event_trace_count(
            self._reader,
            event_id,
            detector=self.detector,
        )
        self._event_count_cache[event_id] = int(trace_count)
        if trace_count <= 0 and not allow_missing:
            detector_label = "IC event" if self.detector == DETECTOR_IC else "event"
            raise LookupError(f"{detector_label} {self.run}/{event_id} is not available")
        return int(trace_count)
