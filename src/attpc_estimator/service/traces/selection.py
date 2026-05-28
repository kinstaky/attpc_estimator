from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import random
import time

import numpy as np

from ...model.label import StoredLabel
from ...model.trace import TraceRef
from ...storage.run_paths import extract_run_id
from ...utils.trace_data import (
    collect_reader_event_counts,
    open_storage_trace_reader,
    reader_event_trace_count,
)

MAX_RANDOM_TRACE_ATTEMPTS = 1024
FULL_SCAN_EVENT_THRESHOLD = 200
SEQUENTIAL_EVENT_FALLBACK_LIMIT = 32

logger = logging.getLogger("attpc_estimator.selection")


def match_review_label(
    family: str,
    stored_label: str,
    requested_label: str | None,
) -> bool:
    if requested_label is None:
        return True
    if family == "normal" and requested_label == "4+":
        return stored_label in {"4", "5", "6", "7", "8", "9"}
    return stored_label == requested_label


def sort_trace_refs(refs: list[TraceRef]) -> list[TraceRef]:
    return sorted(refs)


def trace_refs_from_filter_rows(rows: np.ndarray) -> list[TraceRef]:
    rows_array = np.asarray(rows, dtype=np.int64)
    if rows_array.ndim != 2 or rows_array.shape[1] != 3:
        raise ValueError(
            "filter rows must have shape (N, 3) with columns run,event_id,trace_id"
        )
    return [
        TraceRef(run=int(row[0]), event_id=int(row[1]), trace_id=int(row[2]))
        for row in rows_array.tolist()
    ]


def rebuild_review_refs(
    *,
    run: int,
    labels: Mapping[TraceRef, StoredLabel],
    family: str,
    label: str | None,
) -> list[TraceRef]:
    return sort_trace_refs(
        [
            ref
            for ref, stored_label in labels.items()
            if ref.run == run
            and stored_label.family == family
            and match_review_label(family, stored_label.label, label)
        ]
    )


def random_unlabeled_ref(
    *,
    run: int,
    event_counts: list[tuple[int, int]],
    labeled_refs: set[TraceRef],
    excluded_refs: set[TraceRef],
    max_attempts: int = MAX_RANDOM_TRACE_ATTEMPTS,
) -> TraceRef:
    if not event_counts:
        raise LookupError("no unlabeled trace was found after repeated random sampling")

    for _ in range(max_attempts):
        event_id, trace_count = random.choice(event_counts)
        trace_id = random.randrange(trace_count)
        ref = TraceRef(run=run, event_id=event_id, trace_id=trace_id)
        if ref in labeled_refs or ref in excluded_refs:
            continue
        return ref
    raise LookupError("no unlabeled trace was found after repeated random sampling")


class RandomUnlabeledSelector:
    clamp_at_end = False
    empty_message = "no unlabeled trace was found after repeated random sampling"

    def __init__(self, trace_file: Path, *, verbose: bool = False) -> None:
        self.trace_file = trace_file
        self.run = extract_run_id(trace_file)
        self.verbose = verbose
        self._event_counts: list[tuple[int, int]] = []
        self._event_count_cache: dict[int, int] = {}
        self._reader = None

        started = time.perf_counter()
        reader = open_storage_trace_reader(run=self.run, path=str(trace_file))
        self._layout = reader.layout
        self._min_event = int(reader.min_event)
        self._max_event = int(reader.max_event)
        self._bad_events = set(getattr(reader, "bad_events", set()))
        self._strategy = (
            "full_scan"
            if (self._max_event - self._min_event + 1 - len(self._bad_events))
            <= FULL_SCAN_EVENT_THRESHOLD
            else "sparse_random"
        )
        if self._strategy == "full_scan":
            self._event_counts = collect_reader_event_counts(reader)
            reader.close()
        else:
            self._reader = reader
        self._debug(
            "selector init run=%s layout=%s strategy=%s events=%s-%s bad=%s took=%.3fs",
            self.run,
            self._layout.value,
            self._strategy,
            self._min_event,
            self._max_event,
            len(self._bad_events),
            time.perf_counter() - started,
        )

    def initial_refs(self, labels: Mapping[TraceRef, StoredLabel]) -> list[TraceRef]:
        return []

    def ensure_forward_size(
        self,
        existing: list[TraceRef],
        pointer: int,
        target_size: int,
        labels: Mapping[TraceRef, StoredLabel],
    ) -> list[TraceRef]:
        additions: list[TraceRef] = []
        excluded_refs = set(existing)
        labeled_refs = {ref for ref in labels if ref.run == self.run}
        started = time.perf_counter()
        total_event_probes = 0
        total_trace_probes = 0
        while len(existing) + len(additions) < target_size:
            try:
                if self._strategy == "full_scan":
                    ref = random_unlabeled_ref(
                        run=self.run,
                        event_counts=self._event_counts,
                        labeled_refs=labeled_refs,
                        excluded_refs=excluded_refs,
                    )
                    event_probes = 1
                    trace_probes = 1
                else:
                    ref, event_probes, trace_probes = self._random_sparse_ref(
                        labeled_refs=labeled_refs,
                        excluded_refs=excluded_refs,
                    )
            except LookupError:
                break
            additions.append(ref)
            excluded_refs.add(ref)
            total_event_probes += event_probes
            total_trace_probes += trace_probes
        if additions:
            self._debug(
                "selector filled %s refs run=%s strategy=%s event_probes=%s trace_probes=%s took=%.3fs",
                len(additions),
                self.run,
                self._strategy,
                total_event_probes,
                total_trace_probes,
                time.perf_counter() - started,
            )
        return additions

    def on_label_updated(
        self,
        ref: TraceRef,
        family: str,
        label: str,
        existing: list[TraceRef],
        pointer: int,
        labels: Mapping[TraceRef, StoredLabel],
    ) -> list[TraceRef] | None:
        # Keep the already-generated label-mode history stable when a user rewinds and
        # relabels an earlier trace. Newly appended refs are still filtered in
        # ensure_forward_size() against the current labels map.
        return None

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _random_sparse_ref(
        self,
        *,
        labeled_refs: set[TraceRef],
        excluded_refs: set[TraceRef],
    ) -> tuple[TraceRef, int, int]:
        if self._min_event > self._max_event:
            raise LookupError("no unlabeled trace was found after repeated random sampling")

        total_event_probes = 0
        total_trace_probes = 0
        event_span = self._max_event - self._min_event + 1
        fallback_limit = min(event_span, SEQUENTIAL_EVENT_FALLBACK_LIMIT)

        for _ in range(MAX_RANDOM_TRACE_ATTEMPTS):
            seed_event = random.randint(self._min_event, self._max_event)
            for event_offset in range(fallback_limit):
                event_id = self._wrap_event(seed_event + event_offset)
                total_event_probes += 1
                if event_id in self._bad_events:
                    continue
                trace_count = self._get_event_trace_count(event_id)
                if trace_count <= 0:
                    continue
                start_trace = random.randrange(trace_count) if event_offset == 0 else 0
                for trace_offset in range(trace_count):
                    trace_id = (start_trace + trace_offset) % trace_count
                    total_trace_probes += 1
                    ref = TraceRef(run=self.run, event_id=event_id, trace_id=trace_id)
                    if ref in labeled_refs or ref in excluded_refs:
                        continue
                    return ref, total_event_probes, total_trace_probes

        raise LookupError("no unlabeled trace was found after repeated random sampling")

    def _get_event_trace_count(self, event_id: int) -> int:
        cached = self._event_count_cache.get(event_id)
        if cached is not None:
            return cached
        if self._reader is None:
            return 0
        count = reader_event_trace_count(self._reader, event_id)
        self._event_count_cache[event_id] = count
        return count

    def _wrap_event(self, event_id: int) -> int:
        event_span = self._max_event - self._min_event + 1
        if event_span <= 0:
            return self._min_event
        return self._min_event + ((event_id - self._min_event) % event_span)

    def _debug(self, message: str, *args: object) -> None:
        if self.verbose:
            logger.debug(message, *args)


class LabeledReviewSelector:
    clamp_at_end = True
    empty_message = "no traces match the selected review filter"

    def __init__(self, *, run: int, family: str, label: str | None) -> None:
        self.run = run
        self.family = family
        self.label = label

    def initial_refs(self, labels: Mapping[TraceRef, StoredLabel]) -> list[TraceRef]:
        return rebuild_review_refs(
            run=self.run,
            labels=labels,
            family=self.family,
            label=self.label,
        )

    def ensure_forward_size(
        self,
        existing: list[TraceRef],
        pointer: int,
        target_size: int,
        labels: Mapping[TraceRef, StoredLabel],
    ) -> list[TraceRef]:
        return []

    def on_label_updated(
        self,
        ref: TraceRef,
        family: str,
        label: str,
        existing: list[TraceRef],
        pointer: int,
        labels: Mapping[TraceRef, StoredLabel],
    ) -> list[TraceRef] | None:
        return rebuild_review_refs(
            run=self.run,
            labels=labels,
            family=self.family,
            label=self.label,
        )

    def close(self) -> None:
        return None


class FilterRowsSelector:
    clamp_at_end = True
    empty_message = "no traces match the selected filter"

    def __init__(self, rows: np.ndarray) -> None:
        self._rows = np.asarray(rows, dtype=np.int64)

    def initial_refs(self, labels: Mapping[TraceRef, StoredLabel]) -> list[TraceRef]:
        return trace_refs_from_filter_rows(self._rows)

    def ensure_forward_size(
        self,
        existing: list[TraceRef],
        pointer: int,
        target_size: int,
        labels: Mapping[TraceRef, StoredLabel],
    ) -> list[TraceRef]:
        return []

    def on_label_updated(
        self,
        ref: TraceRef,
        family: str,
        label: str,
        existing: list[TraceRef],
        pointer: int,
        labels: Mapping[TraceRef, StoredLabel],
    ) -> list[TraceRef] | None:
        return None

    def close(self) -> None:
        return None
