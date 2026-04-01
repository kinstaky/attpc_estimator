from __future__ import annotations

from dataclasses import dataclass
import numpy as np

NORMAL_BUCKETS = tuple(range(10))

@dataclass(slots=True)
class TraceRecord:
    run: str
    event_id: int
    trace_id: int
    detector: str
    hardware_id: np.ndarray
    raw: np.ndarray
    trace: np.ndarray
    transformed: np.ndarray
    family: str | None
    label: str | None

@dataclass(slots=True)
class StoredLabel:
    family: str
    label: str