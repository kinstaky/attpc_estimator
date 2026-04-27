from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class PointcloudEventRef:
    run: int
    event_id: int


def _serialize_ref(ref: PointcloudEventRef) -> dict[str, int]:
    return {
        "run": int(ref.run),
        "eventId": int(ref.event_id),
    }


def _deserialize_ref(payload: object) -> PointcloudEventRef | None:
    if not isinstance(payload, dict):
        return None
    run = payload.get("run")
    event_id = payload.get("eventId")
    if not isinstance(run, int) or not isinstance(event_id, int):
        return None
    return PointcloudEventRef(run=run, event_id=event_id)


class PointcloudLabelSource:
    def __init__(
        self,
        *,
        event_ranges: dict[int, tuple[int, int]],
        run: int,
        labeled_event_ids: set[int],
    ) -> None:
        self.event_ranges = event_ranges
        self.run = int(run)
        self._labeled_event_ids = set(int(value) for value in labeled_event_ids)
        self.stack: list[PointcloudEventRef] = []
        self.index = -1

    def current_ref(self) -> PointcloudEventRef | None:
        if self.index < 0 or self.index >= len(self.stack):
            return None
        return self.stack[self.index]

    def current_ref_or_raise(self) -> PointcloudEventRef:
        current = self.current_ref()
        if current is None:
            raise LookupError("no pointcloud event is selected")
        return current

    def next_ref(self) -> PointcloudEventRef:
        if self.index + 1 < len(self.stack):
            self.index += 1
            return self.stack[self.index]

        ref = self._random_unlabeled_ref(set(self.stack))
        self.stack.append(ref)
        self.index = len(self.stack) - 1
        return ref

    def previous_ref(self) -> PointcloudEventRef:
        if not self.stack or self.index < 0:
            raise LookupError("no pointcloud event history is available")
        if self.index > 0:
            self.index -= 1
        return self.stack[self.index]

    def update_labeled_event_ids(self, labeled_event_ids: set[int]) -> None:
        self._labeled_event_ids = set(int(value) for value in labeled_event_ids)

    def snapshot_state(self) -> dict[str, object]:
        return {
            "stack": [_serialize_ref(ref) for ref in self.stack],
            "index": int(self.index),
        }

    def restore_state(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        stack_payload = payload.get("stack")
        index = payload.get("index")
        if not isinstance(stack_payload, list) or not isinstance(index, int):
            return
        refs = [ref for item in stack_payload if (ref := _deserialize_ref(item)) is not None]
        self.stack = [ref for ref in refs if ref.run == self.run]
        if not self.stack:
            self.index = -1
            return
        self.index = min(max(index, -1), len(self.stack) - 1)

    def _random_unlabeled_ref(self, excluded_refs: set[PointcloudEventRef]) -> PointcloudEventRef:
        event_range = self.event_ranges.get(self.run)
        if event_range is None:
            raise LookupError(f"pointcloud run not found: {self.run}")
        candidates = [
            PointcloudEventRef(run=self.run, event_id=event_id)
            for event_id in range(int(event_range[0]), int(event_range[1]) + 1)
            if event_id not in self._labeled_event_ids
            and PointcloudEventRef(run=self.run, event_id=event_id) not in excluded_refs
        ]
        if not candidates:
            raise LookupError("no unlabeled pointcloud event is available")
        return random.choice(candidates)
