from __future__ import annotations

from pathlib import Path
import random
from typing import Literal

import h5py

from .models import TraceRecord

TraceKey = tuple[int, int]


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
            return len(self.label_event_stack)
        self.review_filter = {"family": family, "label": label or ""}
        self.review_event_stack = self._build_review_stack(family=family, label=label)
        self.review_stack_pointer = 0
        return len(self.review_event_stack)

    def get_trace(self, event_id: int, trace_id: int) -> TraceRecord | None:
        if (
            event_id < self.min_event
            or event_id > self.max_event
            or event_id in self.bad_events
        ):
            return None
        pads = self.file["events"][f"event_{event_id}"]["get"]["pads"]
        if trace_id >= pads.shape[0]:
            return None
        if (event_id, trace_id) in self.labeled_traces:
            family, label = self.labeled_traces[(event_id, trace_id)]
        else:
            family = None
            label = None
        hardware, trace = self._extract_trace_parts(event_id, trace_id)
        return TraceRecord(
            run=self.run,
            event_id=event_id,
            trace_id=trace_id,
            detector="pad",
            hardware_id=hardware,
            trace=trace,
            family=family,
            label=label,
        )

    def _extract_trace_parts(self, event_id: int, trace_id: int) -> tuple:
        pads = self.file["events"][f"event_{event_id}"]["get"]["pads"]
        row = pads[trace_id]
        hardware = row[:5].copy()
        trace = row[5:].copy()
        if len(trace) > 1:
            trace[0] = trace[1]
            trace[-1] = trace[-2]
        return hardware, trace

    def _unlabeled_trace_keys(self) -> list[TraceKey]:
        trace_keys: list[TraceKey] = []
        for event_id in range(int(self.min_event), int(self.max_event) + 1):
            if event_id in self.bad_events:
                continue
            pads = self.file["events"][f"event_{event_id}"]["get"]["pads"]
            for trace_id in range(int(pads.shape[0])):
                key = (event_id, trace_id)
                if key not in self.labeled_traces:
                    trace_keys.append(key)
        return trace_keys

    def _random_trace(self) -> TraceKey:
        candidates = self._unlabeled_trace_keys()
        if not candidates:
            raise LookupError("all traces have been labeled")
        return random.choice(candidates)

    def _build_review_stack(self, family: str, label: str | None) -> list[TraceKey]:
        return sorted(
            key
            for key, stored_label in self.labeled_traces.items()
            if stored_label[0] == family and (label is None or stored_label[1] == label)
        )

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
            return self._next_review_trace()
        return self._next_label_trace()

    def previous_trace(self) -> TraceRecord:
        if self.trace_mode == "review":
            return self._previous_review_trace()
        return self._previous_label_trace()

    def label_trace(self, event_id: int, trace_id: int, family: str, label: str) -> None:
        self.labeled_traces[(event_id, trace_id)] = (family, label)
        if self.trace_mode == "review":
            self._refresh_review_stack_after_relabel((event_id, trace_id))

    def set_labeled(self, labeled: dict[tuple[int, int], tuple[str, str]]) -> None:
        self.labeled_traces = labeled.copy()
