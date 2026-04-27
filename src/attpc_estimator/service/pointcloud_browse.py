from __future__ import annotations

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


class PointcloudBrowseSource:
    def __init__(
        self,
        *,
        event_ranges: dict[int, tuple[int, int]],
        run: int,
        source: str,
        labeled_event_ids: list[int] | None = None,
    ) -> None:
        self.event_ranges = event_ranges
        self.run = int(run)
        self.source = str(source)
        self._labeled_event_ids = sorted({int(value) for value in labeled_event_ids or []})
        self._current_event_id: int | None = None

    def current_ref(self) -> PointcloudEventRef | None:
        if self._current_event_id is None:
            return None
        return PointcloudEventRef(run=self.run, event_id=self._current_event_id)

    def current_ref_or_raise(self) -> PointcloudEventRef:
        current = self.current_ref()
        if current is None:
            raise LookupError("no pointcloud event is selected")
        return current

    def set_current(self, event_id: int) -> PointcloudEventRef:
        resolved_event_id = self._normalize_event_id(int(event_id))
        self._current_event_id = resolved_event_id
        return self.current_ref_or_raise()

    def next_ref(self) -> PointcloudEventRef:
        if self.source == "label_set":
            return self._step_labeled(+1)
        return self._step_direct(+1)

    def previous_ref(self) -> PointcloudEventRef:
        if self.source == "label_set":
            return self._step_labeled(-1)
        return self._step_direct(-1)

    def update_labeled_event_ids(self, labeled_event_ids: list[int]) -> None:
        self._labeled_event_ids = sorted({int(value) for value in labeled_event_ids})
        if self.source != "label_set":
            return
        if not self._labeled_event_ids:
            self._current_event_id = None
            return
        if self._current_event_id is None:
            self._current_event_id = self._labeled_event_ids[0]
            return
        if self._current_event_id not in self._labeled_event_ids:
            self._current_event_id = self._closest_labeled_event_id(self._current_event_id)

    def snapshot_state(self) -> dict[str, object]:
        return {
            "current": _serialize_ref(self.current_ref_or_raise()) if self.current_ref() else None,
        }

    def restore_state(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        current_payload = payload.get("current")
        ref = _deserialize_ref(current_payload)
        if ref is None or ref.run != self.run:
            return
        try:
            self.set_current(ref.event_id)
        except LookupError:
            return

    def _normalize_event_id(self, event_id: int) -> int:
        if self.source == "label_set":
            if not self._labeled_event_ids:
                raise LookupError("no labeled pointcloud events match the selected filter")
            if event_id in self._labeled_event_ids:
                return int(event_id)
            return self._closest_labeled_event_id(event_id)

        event_range = self.event_ranges.get(self.run)
        if event_range is None:
            raise LookupError(f"pointcloud run not found: {self.run}")
        return min(max(int(event_id), int(event_range[0])), int(event_range[1]))

    def _closest_labeled_event_id(self, event_id: int) -> int:
        assert self._labeled_event_ids
        for labeled_event_id in self._labeled_event_ids:
            if labeled_event_id >= int(event_id):
                return labeled_event_id
        return self._labeled_event_ids[-1]

    def _step_direct(self, delta: int) -> PointcloudEventRef:
        event_range = self.event_ranges.get(self.run)
        if event_range is None:
            raise LookupError(f"pointcloud run not found: {self.run}")
        if self._current_event_id is None:
            self._current_event_id = int(event_range[0])
            return self.current_ref_or_raise()
        candidate = self._current_event_id + int(delta)
        self._current_event_id = min(max(candidate, int(event_range[0])), int(event_range[1]))
        return self.current_ref_or_raise()

    def _step_labeled(self, delta: int) -> PointcloudEventRef:
        if not self._labeled_event_ids:
            raise LookupError("no labeled pointcloud events match the selected filter")
        if self._current_event_id is None:
            self._current_event_id = self._labeled_event_ids[0]
            return self.current_ref_or_raise()
        try:
            index = self._labeled_event_ids.index(self._current_event_id)
        except ValueError:
            self._current_event_id = self._closest_labeled_event_id(self._current_event_id)
            return self.current_ref_or_raise()
        next_index = min(max(index + int(delta), 0), len(self._labeled_event_ids) - 1)
        self._current_event_id = self._labeled_event_ids[next_index]
        return self.current_ref_or_raise()
