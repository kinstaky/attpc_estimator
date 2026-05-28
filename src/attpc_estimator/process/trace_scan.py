from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

from .progress import ProgressReporter, emit_progress
from ..utils.trace_data import (
    DETECTOR_ATTPC,
    DETECTOR_IC,
    describe_trace_events,
    load_detector_traces,
    load_pad_traces,
    normalize_detector,
    preprocess_traces,
)


def scan_cleaned_trace_batches(
    trace_file_path: Path,
    *,
    baseline_window_scale: float,
    handler: Callable[[int, np.ndarray], bool | None],
    progress: ProgressReporter | None = None,
    detector: str = DETECTOR_ATTPC,
) -> None:
    resolved_detector = normalize_detector(detector)
    with h5py.File(trace_file_path, "r") as handle:
        metadata = describe_trace_events(handle)
        processed_events = 0
        emit_progress(
            progress,
            current=0,
            total=metadata.valid_event_span,
            unit="event",
        )
        for event_id in range(metadata.min_event, metadata.max_event + 1):
            if event_id in metadata.bad_events:
                continue
            should_continue = True
            try:
                traces = (
                    load_pad_traces(handle, run=0, event_id=event_id)
                    if resolved_detector == DETECTOR_ATTPC
                    else load_detector_traces(
                        handle,
                        run=0,
                        event_id=event_id,
                        detector=DETECTOR_IC,
                    )
                )
            except LookupError:
                processed_events += 1
                emit_progress(
                    progress,
                    current=processed_events,
                    total=metadata.valid_event_span,
                    unit="event",
                    message=f"event={event_id}",
                )
                continue
            cleaned = preprocess_traces(
                traces, baseline_window_scale=baseline_window_scale
            )
            should_continue = handler(event_id, cleaned) is not False
            processed_events += 1
            emit_progress(
                progress,
                current=processed_events,
                total=metadata.valid_event_span,
                unit="event",
                message=f"event={event_id}",
            )
            if not should_continue:
                break
