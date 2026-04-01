from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..models import TraceRecord
from .utils import collect_run_files, load_trace_record


class FilteredTraceSource:
    def __init__(self, trace_path: Path, baseline_window_scale: float = 10.0) -> None:
        self.trace_path = trace_path
        self.baseline_window_scale = baseline_window_scale
        self.run_files = collect_run_files(trace_path)
        self._handles: dict[int, h5py.File] = {}
        self._rows = np.empty((0, 3), dtype=np.int64)
        self._pointer = 0

    def set_filter_rows(self, rows: np.ndarray) -> int:
        rows_array = np.asarray(rows, dtype=np.int64)
        if rows_array.ndim != 2 or rows_array.shape[1] != 3:
            raise ValueError("filter rows must have shape (N, 3) with columns run,event_id,trace_id")
        for run_id in rows_array[:, 0].tolist():
            if int(run_id) not in self.run_files:
                raise ValueError(f"filter row references missing run {int(run_id)}")
        self._rows = rows_array
        self._pointer = 0
        return len(self._rows)

    def next_trace(self) -> TraceRecord:
        if len(self._rows) == 0:
            raise LookupError("no traces match the selected filter")
        if self._pointer >= len(self._rows):
            row = self._rows[-1]
        else:
            row = self._rows[self._pointer]
            self._pointer += 1
        return self._load_record(run=int(row[0]), event_id=int(row[1]), trace_id=int(row[2]))

    def previous_trace(self) -> TraceRecord:
        if len(self._rows) == 0:
            raise LookupError("no traces match the selected filter")
        if self._pointer <= 1:
            self._pointer = min(1, len(self._rows))
            row = self._rows[0]
        else:
            self._pointer -= 1
            row = self._rows[self._pointer - 1]
        return self._load_record(run=int(row[0]), event_id=int(row[1]), trace_id=int(row[2]))

    def get_review_progress(self) -> dict[str, int] | None:
        if len(self._rows) == 0 or self._pointer == 0:
            return None
        return {
            "current": min(self._pointer, len(self._rows)),
            "total": len(self._rows),
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _load_record(self, run: int, event_id: int, trace_id: int) -> TraceRecord:
        return load_trace_record(
            self._get_handle(run),
            run=run,
            event_id=event_id,
            trace_id=trace_id,
            baseline_window_scale=self.baseline_window_scale,
        )

    def _get_handle(self, run: int) -> h5py.File:
        handle = self._handles.get(run)
        if handle is None:
            handle = h5py.File(self.run_files[run], "r")
            self._handles[run] = handle
        return handle
