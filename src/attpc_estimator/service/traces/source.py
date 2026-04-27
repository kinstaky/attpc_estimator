from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import time

import numpy as np

from ...model.label import StoredLabel
from ...model.trace import TraceRecord, TraceRef
from ...storage.run_paths import collect_run_files, extract_run_id
from .loader import TraceLoader
from .navigation import Navigator
from .prefetch import TracePrefetcher
from .selection import (
    FilterRowsSelector,
    LabeledReviewSelector,
    RandomUnlabeledSelector,
    trace_refs_from_filter_rows,
)

CACHE_RADIUS = 5
logger = logging.getLogger("attpc_estimator.trace_source")


def _serialize_ref(ref: TraceRef) -> dict[str, int]:
    return {
        "run": int(ref.run),
        "eventId": int(ref.event_id),
        "traceId": int(ref.trace_id),
    }


def _deserialize_ref(payload: object) -> TraceRef | None:
    if not isinstance(payload, dict):
        return None
    run = payload.get("run")
    event_id = payload.get("eventId")
    trace_id = payload.get("traceId")
    if not isinstance(run, int) or not isinstance(event_id, int) or not isinstance(trace_id, int):
        return None
    return TraceRef(run=run, event_id=event_id, trace_id=trace_id)


class TraceSource:
    def __init__(
        self,
        *,
        run_files: Mapping[int, Path],
        selector: RandomUnlabeledSelector | LabeledReviewSelector | FilterRowsSelector,
        labels: Mapping[TraceRef, StoredLabel] | None = None,
        baseline_window_scale: float = 10.0,
        prefetch_radius: int = CACHE_RADIUS,
        verbose: bool = False,
    ) -> None:
        self.run_files = {int(run): path.resolve() for run, path in run_files.items()}
        self.selector = selector
        self.baseline_window_scale = baseline_window_scale
        self.prefetch_radius = prefetch_radius
        self.verbose = verbose
        self._labels = dict(labels or {})
        self._navigator = Navigator(
            review_mode=isinstance(selector, (LabeledReviewSelector, FilterRowsSelector))
        )
        self._navigator.replace_stack(selector.initial_refs(self._labels))
        self._loader = TraceLoader(
            run_files=self.run_files,
            labels=self._labels,
            baseline_window_scale=baseline_window_scale,
        )
        self._prefetcher = TracePrefetcher(self._loader)

    @classmethod
    def for_label_mode(
        cls,
        trace_file: Path,
        *,
        labels: Mapping[TraceRef, StoredLabel] | None = None,
        baseline_window_scale: float = 10.0,
        prefetch_radius: int = CACHE_RADIUS,
        verbose: bool = False,
    ) -> TraceSource:
        trace_file_resolved = trace_file.resolve()
        run = extract_run_id(trace_file_resolved)
        return cls(
            run_files={run: trace_file_resolved},
            selector=RandomUnlabeledSelector(trace_file_resolved, verbose=verbose),
            labels=labels,
            baseline_window_scale=baseline_window_scale,
            prefetch_radius=prefetch_radius,
            verbose=verbose,
        )

    @classmethod
    def for_review_mode(
        cls,
        trace_file: Path,
        *,
        family: str,
        label: str | None,
        labels: Mapping[TraceRef, StoredLabel] | None = None,
        baseline_window_scale: float = 10.0,
        prefetch_radius: int = CACHE_RADIUS,
        verbose: bool = False,
    ) -> TraceSource:
        trace_file_resolved = trace_file.resolve()
        run = extract_run_id(trace_file_resolved)
        return cls(
            run_files={run: trace_file_resolved},
            selector=LabeledReviewSelector(run=run, family=family, label=label),
            labels=labels,
            baseline_window_scale=baseline_window_scale,
            prefetch_radius=prefetch_radius,
            verbose=verbose,
        )

    @classmethod
    def for_filter_rows(
        cls,
        trace_path: Path | Mapping[int, Path],
        rows: np.ndarray,
        *,
        labels: Mapping[TraceRef, StoredLabel] | None = None,
        baseline_window_scale: float = 10.0,
        prefetch_radius: int = CACHE_RADIUS,
        verbose: bool = False,
    ) -> TraceSource:
        run_files = (
            {int(run): path.resolve() for run, path in trace_path.items()}
            if isinstance(trace_path, Mapping)
            else collect_run_files(trace_path)
        )
        for ref in trace_refs_from_filter_rows(rows):
            if ref.run not in run_files:
                raise ValueError(f"filter row references missing run {ref.run}")
        return cls(
            run_files=run_files,
            selector=FilterRowsSelector(rows),
            labels=labels,
            baseline_window_scale=baseline_window_scale,
            prefetch_radius=prefetch_radius,
            verbose=verbose,
        )

    @property
    def trace_cache(self) -> dict[TraceRef, TraceRecord]:
        return self._prefetcher.cache_snapshot()

    def current_trace(self) -> TraceRecord | None:
        current_ref = self._navigator.current_ref()
        if current_ref is None:
            return None
        return self.get_trace(current_ref)

    def current_trace_or_raise(self) -> TraceRecord:
        record = self.current_trace()
        if record is None:
            raise LookupError("no trace is selected")
        return record

    def trace_count(self) -> int:
        return len(self._navigator.stack)

    def next_trace(self) -> TraceRecord:
        started = time.perf_counter()
        self._ensure_forward_capacity(self._navigator.index + self.prefetch_radius + 2)
        try:
            ref = self._navigator.next_ref(clamp_at_end=self.selector.clamp_at_end)
        except LookupError as exc:
            raise LookupError(self.selector.empty_message) from exc
        record = self._require_trace(ref)
        self._schedule_prefetch()
        if self.verbose:
            logger.debug(
                "trace_source next_trace run=%s event=%s trace=%s took=%.3fs",
                record.run,
                record.event_id,
                record.trace_id,
                time.perf_counter() - started,
            )
        return record

    def previous_trace(self) -> TraceRecord:
        try:
            ref = self._navigator.previous_ref()
        except LookupError as exc:
            raise LookupError("no trace history is available") from exc
        self._ensure_forward_capacity(self._navigator.index + self.prefetch_radius + 2)
        record = self._require_trace(ref)
        self._schedule_prefetch()
        return record

    def get_trace(self, ref: TraceRef) -> TraceRecord | None:
        cached = self._prefetcher.get_cached(ref)
        if cached is not None:
            return cached
        record = self._loader.try_load(ref)
        if record is None:
            return None
        self._prefetcher.store_current(ref, record)
        return record

    def get_progress(self) -> dict[str, int] | None:
        return self._navigator.progress()

    def replace_labels(self, labels: Mapping[TraceRef, StoredLabel]) -> None:
        self._labels = dict(labels)
        self._loader.replace_labels(self._labels)
        self._prefetcher.replace_labels(self._labels)

    def apply_label(self, ref: TraceRef, family: str, label: str) -> None:
        self._labels[ref] = StoredLabel(family=family, label=label)
        self._loader.update_label(ref, family, label)
        self._prefetcher.update_cached_label(ref, family, label)

        rebuilt = self.selector.on_label_updated(
            ref,
            family,
            label,
            list(self._navigator.stack),
            self._navigator.index,
            self._labels,
        )
        if rebuilt is not None:
            current_ref = self._navigator.current_ref()
            self._navigator.replace_stack(rebuilt, keep_current_ref=current_ref)
        self._schedule_prefetch()

    def close(self) -> None:
        self._prefetcher.close()
        self._loader.close()
        self.selector.close()

    def snapshot_state(self) -> dict[str, object]:
        return {
            "stack": [_serialize_ref(ref) for ref in self._navigator.stack],
            "index": int(self._navigator.index),
        }

    def restore_state(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        stack_payload = payload.get("stack")
        index = payload.get("index")
        if not isinstance(stack_payload, list) or not isinstance(index, int):
            return
        refs = [ref for item in stack_payload if (ref := _deserialize_ref(item)) is not None]
        self._navigator.replace_stack(refs)
        if not refs:
            self._navigator.index = -1
            return
        self._navigator.index = min(max(index, -1), len(refs) - 1)
        if self._navigator.index >= 0:
            self._schedule_prefetch()

    def _wait_for_prefetch(self, timeout: float = 1.0) -> bool:
        return self._prefetcher.wait(timeout=timeout)

    def _require_trace(self, ref: TraceRef) -> TraceRecord:
        record = self.get_trace(ref)
        if record is None:
            raise LookupError(
                f"trace {ref.run}/{ref.event_id}/{ref.trace_id} is not available"
            )
        return record

    def _ensure_forward_capacity(self, target_size: int) -> None:
        additions = self.selector.ensure_forward_size(
            list(self._navigator.stack),
            self._navigator.index,
            max(0, target_size),
            self._labels,
        )
        self._navigator.extend_stack(additions)

    def _schedule_prefetch(self) -> None:
        self._ensure_forward_capacity(self._navigator.index + self.prefetch_radius + 2)
        self._prefetcher.schedule(self._navigator.window(self.prefetch_radius))
