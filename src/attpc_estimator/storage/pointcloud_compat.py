from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from attpc_storage import open_trace_reader


PROCESSING_ATTR_KEYS = (
    "fft_window_scale",
    "bitflip_baseline",
    "bitflip_min_count",
    "peak_separation",
    "peak_prominence",
    "peak_max_width",
    "peak_threshold",
    "peak_rel_height",
    "micromegas_time_bucket",
    "window_time_bucket",
    "detector_length",
)


@dataclass(frozen=True, slots=True)
class TraceEventDescription:
    min_event: int
    max_event: int
    bad_events: frozenset[int]

    @property
    def valid_event_span(self) -> int:
        span = max(0, self.max_event - self.min_event + 1)
        return max(0, span - len(self.bad_events))


class CompatRawTraceReader:
    def __init__(self, *, workspace: str, run: int, path: str) -> None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(workspace) / resolved
        self._handle = h5py.File(resolved, "r")
        self._run = int(run)
        self._reader = open_trace_reader(
            workspace=workspace,
            run=run,
            path=str(resolved),
            read_pad=True,
            read_si=False,
            read_gagg=False,
            read_ic=False,
        )
        self.min_event = int(self._reader.min_event)
        self.max_event = int(self._reader.max_event)
        self.bad_events = frozenset(int(value) for value in getattr(self._reader, "bad_events", ()))

    def describe_events(self) -> TraceEventDescription:
        return TraceEventDescription(
            min_event=self.min_event,
            max_event=self.max_event,
            bad_events=self.bad_events,
        )

    def read_event(self, event_id: int) -> tuple[Any, np.ndarray]:
        event = self._reader.read_event(int(event_id))
        if not isinstance(event, dict):
            raise LookupError(f"trace event {event_id} is not available")
        payload = event.get("pads")
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise LookupError(f"trace event {event_id} is not available")
        meta, rows = payload
        if "events" in self._handle:
            event_group = self._handle["events"].get(f"event_{int(event_id)}")
            if event_group is not None:
                orig_run = int(event_group.attrs.get("orig_run", self._run))
                orig_event = int(event_group.attrs.get("orig_event", int(event_id)))
                meta = (int(event_id), orig_run, orig_event)
        return meta, rows

    def load_pad_rows(
        self,
        event_id: int,
        *,
        trace_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        _, rows = self.read_event(int(event_id))
        row_array = np.asarray(rows, dtype=np.float32)
        if trace_ids is None:
            return row_array

        indices = np.asarray(trace_ids, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"trace_ids must be 1D, got shape {indices.shape}")
        if indices.size == 0:
            return row_array[indices]
        if int(indices.min()) < 0 or int(indices.max()) >= int(row_array.shape[0]):
            raise LookupError(f"trace event {event_id} is not available")

        order = np.argsort(indices, kind="stable")
        sorted_indices = indices[order]
        unique_indices, inverse = np.unique(sorted_indices, return_inverse=True)
        sorted_rows = np.asarray(row_array[unique_indices], dtype=np.float32)[inverse]
        selected = np.empty_like(sorted_rows)
        selected[order] = sorted_rows
        return selected

    def close(self) -> None:
        self._reader.close()
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None


class CompatPointcloudReader:
    def __init__(self, *, workspace: str, run: int, path: str) -> None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(workspace) / resolved
        self._handle = h5py.File(resolved, "r")
        if "cloud" not in self._handle:
            raise KeyError("pointcloud file is missing cloud group")
        self._group = self._handle["cloud"]

    def get_range(self) -> tuple[int, int]:
        return int(self._group.attrs["min_event"]), int(self._group.attrs["max_event"])

    def read_event(self, event_id: int) -> tuple[tuple[int, int, int], np.ndarray]:
        dataset_name = f"cloud_{int(event_id)}"
        if dataset_name not in self._group:
            raise LookupError(f"pointcloud event not found: {event_id}")
        dataset = self._group[dataset_name]
        orig_run = int(dataset.attrs.get("orig_run", int(event_id)))
        orig_event = int(dataset.attrs.get("orig_event", int(event_id)))
        return (int(event_id), orig_run, orig_event), np.asarray(dataset[:])

    def read_processing_attrs(self) -> dict[str, object]:
        return {
            key: self._group.attrs[key]
            for key in PROCESSING_ATTR_KEYS
            if key in self._group.attrs
        }

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None


class CompatPointcloudWriter:
    def __init__(self, *, workspace: str, run: int, path: str) -> None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(workspace) / resolved
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._handle = h5py.File(resolved, "w")
        self._group = self._handle.create_group("cloud")
        self._min_event: int | None = None
        self._max_event: int | None = None

    def write_processing_attrs(self, attrs: dict[str, object]) -> None:
        for key, value in attrs.items():
            self._group.attrs[key] = value

    def write(self, meta: tuple[int, int, int], event: np.ndarray) -> None:
        event_id, orig_run, orig_event = meta
        dataset = self._group.create_dataset(f"cloud_{int(event_id)}", data=event)
        dataset.attrs["orig_run"] = int(orig_run)
        dataset.attrs["orig_event"] = int(orig_event)
        self._min_event = int(event_id) if self._min_event is None else min(self._min_event, int(event_id))
        self._max_event = int(event_id) if self._max_event is None else max(self._max_event, int(event_id))

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._group.attrs["min_event"] = -1 if self._min_event is None else int(self._min_event)
            self._group.attrs["max_event"] = -1 if self._max_event is None else int(self._max_event)
            self._handle.close()
            self._handle = None
