from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np

from .progress import ProgressReporter, emit_progress
from .filter_core import AmplitudeFilterCore, FilterCore
from ..storage.run_paths import resolve_run_file
from .trace_scan import scan_cleaned_trace_batches

DEFAULT_TRACE_LIMIT = 1000
UNLIMITED_TRACE_LIMIT = 0


def build_filter_rows(
    trace_path: Path,
    run: int,
    filter_cores: Sequence[FilterCore],
    baseline_window_scale: float = 10.0,
    limit: int = DEFAULT_TRACE_LIMIT,
    progress: ProgressReporter | None = None,
) -> np.ndarray:
    active_cores = list(filter_cores)
    if not active_cores:
        raise ValueError("at least one filter core is required")
    if limit < UNLIMITED_TRACE_LIMIT:
        raise ValueError("limit must be non-negative")

    run_file = resolve_run_file(trace_path, run)

    selected_rows: list[tuple[int, int, int]] = []
    unlimited = limit == UNLIMITED_TRACE_LIMIT
    if not unlimited:
        emit_progress(
            progress,
            current=0,
            total=limit,
            unit="trace",
        )

    def handle_batch(event_id: int, cleaned: np.ndarray) -> bool | None:
        if not unlimited and len(selected_rows) >= limit:
            return False
        prepared_batches = [core.prepare_batch(cleaned) for core in active_cores]

        for trace_id, row in enumerate(cleaned):
            if not all(
                core.matches(
                    trace_id=trace_id,
                    row=row,
                    prepared=prepared,
                )
                for core, prepared in zip(active_cores, prepared_batches, strict=True)
            ):
                continue
            selected_rows.append((run, event_id, trace_id))
            if not unlimited:
                emit_progress(
                    progress,
                    current=len(selected_rows),
                    total=limit,
                    unit="trace",
                    message=f"event={event_id},trace={trace_id}",
                )
            if not unlimited and len(selected_rows) >= limit:
                return False
        return None

    scan_cleaned_trace_batches(
        run_file,
        baseline_window_scale=baseline_window_scale,
        handler=handle_batch,
        progress=progress if unlimited else None,
    )

    if not selected_rows:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(selected_rows, dtype=np.int64)


def build_amplitude_filter_rows(
    trace_path: Path,
    run: int,
    min_amplitude: float,
    max_amplitude: float,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
    limit: int = DEFAULT_TRACE_LIMIT,
    progress: ProgressReporter | None = None,
) -> np.ndarray:
    return build_filter_rows(
        trace_path=trace_path,
        run=run,
        filter_cores=[
            AmplitudeFilterCore(
                min_amplitude=min_amplitude,
                max_amplitude=max_amplitude,
                peak_separation=peak_separation,
                peak_prominence=peak_prominence,
                peak_width=peak_width,
            )
        ],
        baseline_window_scale=baseline_window_scale,
        limit=limit,
        progress=progress,
    )


def normalize_amplitude_range(
    amplitude: list[float] | tuple[float, float] | None,
) -> tuple[float, float] | None:
    if amplitude is None:
        return None
    if len(amplitude) != 2:
        raise ValueError("amplitude must contain two values: minimum and maximum")
    minimum = float(amplitude[0])
    maximum = float(amplitude[1])
    if minimum > maximum:
        raise ValueError("amplitude minimum must be less than or equal to the maximum")
    return minimum, maximum


def default_output_name(
    run_token: str,
    filter_cores: Sequence[FilterCore],
) -> str:
    parts = [f"filter_run_{run_token}"]
    for core in filter_cores:
        token = core.output_token()
        if token:
            parts.append(token)
    return "_".join(parts) + ".npy"
