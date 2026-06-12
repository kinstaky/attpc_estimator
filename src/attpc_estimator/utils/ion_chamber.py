from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import h5py
import numpy as np
from attpc_storage import describe_trace_events
from attpc_storage.trace import TraceEventMetadata


IC_COLUMN = 0
FRIB_TRACE_LENGTHS = frozenset((128, 256))


class TriggerType(IntEnum):
    EMPTY_TRIGGER = 0
    MESH_TRIGGER = 1
    IC_DOWNSCALE_TRIGGER = 2
    AND_TRIGGER = 3


@dataclass(frozen=True, slots=True)
class IonChamberEvent:
    event_id: int
    orig_run: int
    orig_event: int
    trigger_type: int
    raw_trace: np.ndarray


def describe_ion_chamber_events(file_handle: h5py.File) -> TraceEventMetadata:
    return describe_trace_events(file_handle)


def event_has_ion_chamber(file_handle: h5py.File, event_id: int) -> bool:
    try:
        _ic_objects(file_handle, event_id)
    except LookupError:
        return False
    return True


def ion_chamber_event_count(file_handle: h5py.File, event_id: int) -> int:
    return 1 if event_has_ion_chamber(file_handle, event_id) else 0


def collect_ion_chamber_event_counts(file_handle: h5py.File) -> list[tuple[int, int]]:
    metadata = describe_ion_chamber_events(file_handle)
    event_counts: list[tuple[int, int]] = []
    for event_id in range(metadata.min_event, metadata.max_event + 1):
        if event_id in metadata.bad_events:
            continue
        if event_has_ion_chamber(file_handle, event_id):
            event_counts.append((event_id, 1))
    return event_counts


def load_ion_chamber_event(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
) -> IonChamberEvent:
    metadata = describe_ion_chamber_events(file_handle)
    if (
        event_id < metadata.min_event
        or event_id > metadata.max_event
        or event_id in metadata.bad_events
    ):
        raise LookupError(f"IC event {run}/{event_id} is not available")

    event_group, trace_dataset, trigger_dataset = _ic_objects(file_handle, event_id)
    trace_matrix = np.asarray(trace_dataset)
    if trace_matrix.ndim != 2:
        raise LookupError(f"IC event {run}/{event_id} has invalid 1903 shape {trace_matrix.shape}")
    if IC_COLUMN >= trace_matrix.shape[1]:
        raise LookupError(f"IC event {run}/{event_id} does not contain IC column {IC_COLUMN}")

    trigger_array = np.asarray(trigger_dataset)
    if trigger_array.size < 1:
        raise LookupError(f"IC event {run}/{event_id} has empty 977 trigger data")

    orig_run = int(event_group.attrs.get("orig_run", run))
    orig_event = int(event_group.attrs.get("orig_event", event_id))
    raw_trace = np.asarray(trace_matrix[:, IC_COLUMN], dtype=np.float32)
    if int(raw_trace.shape[0]) not in FRIB_TRACE_LENGTHS:
        raise LookupError(
            f"IC event {run}/{event_id} trace has length {raw_trace.shape[0]}, "
            f"expected one of {sorted(FRIB_TRACE_LENGTHS)}"
        )

    return IonChamberEvent(
        event_id=int(event_id),
        orig_run=orig_run,
        orig_event=orig_event,
        trigger_type=int(trigger_array.reshape(-1)[0]),
        raw_trace=raw_trace,
    )


def load_ion_chamber_trace(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    trace_ids: np.ndarray | None = None,
) -> np.ndarray:
    if trace_ids is not None:
        indices = np.asarray(trace_ids, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"trace_ids must be 1D, got shape {indices.shape}")
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) > 0):
            raise LookupError(f"IC trace {run}/{event_id} is not available")
        if indices.size == 0:
            return np.empty((0, 0), dtype=np.float32)
    event = load_ion_chamber_event(file_handle, run=run, event_id=event_id)
    return event.raw_trace[np.newaxis, :]


def _ic_objects(
    file_handle: h5py.File,
    event_id: int,
) -> tuple[h5py.Group, h5py.Dataset, h5py.Dataset]:
    if "events" in file_handle:
        events = file_handle["events"]
        event_name = f"event_{event_id}"
        if event_name not in events:
            raise LookupError(f"event {event_id} is not available")
        event_group = events[event_name]
        if "frib_physics" not in event_group:
            raise LookupError(f"event {event_id} has no frib_physics group")
        frib_group = event_group["frib_physics"]
        if "1903" not in frib_group or "977" not in frib_group:
            raise LookupError(f"event {event_id} is missing 1903 or 977 data")
        return event_group, frib_group["1903"], frib_group["977"]

    if "frib" in file_handle and "evt" in file_handle["frib"]:
        frib_group = file_handle["frib"]["evt"]
        trace_name = f"evt{event_id}_1903"
        trigger_name = f"evt{event_id}_977"
        if trace_name not in frib_group or trigger_name not in frib_group:
            raise LookupError(f"event {event_id} is missing 1903 or 977 data")
        return file_handle, frib_group[trace_name], frib_group[trigger_name]

    raise LookupError("unsupported IC trace layout")
