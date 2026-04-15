from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from .progress import ProgressReporter
from .trace_metrics import compute_peak_amplitudes
from ..process.labeled import (
    NORMAL_LABEL_GROUPS,
    load_grouped_labeled_run,
    scan_grouped_labeled_trace_batches,
)
from ..process.trace_scan import scan_cleaned_trace_batches

AMPLITUDE_BIN_COUNT = 8192

__all__ = [
    "AMPLITUDE_BIN_COUNT",
    "NORMAL_LABEL_GROUPS",
    "build_amplitude_histogram",
    "build_labeled_amplitude_histograms",
    "max_peak_amplitude",
    "_accumulate_peak_histogram",
]


def build_amplitude_histogram(
    trace_file_path: Path,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
    progress: ProgressReporter | None = None,
) -> np.ndarray:
    histogram = np.zeros(AMPLITUDE_BIN_COUNT, dtype=np.int64)

    def handle_batch(_event_id: int, cleaned: np.ndarray) -> None:
        for row in cleaned:
            _accumulate_peak_histogram(
                row=row,
                histogram=histogram,
                peak_separation=peak_separation,
                peak_prominence=peak_prominence,
                peak_width=peak_width,
            )

    scan_cleaned_trace_batches(
        trace_file_path,
        baseline_window_scale=baseline_window_scale,
        handler=handle_batch,
        progress=progress,
    )
    return histogram


def build_labeled_amplitude_histograms(
    trace_path: Path,
    workspace: Path,
    run: int,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
    progress: ProgressReporter | None = None,
) -> dict[str, np.ndarray | np.int64]:
    grouped_run = load_grouped_labeled_run(
        trace_path=trace_path, workspace=workspace, run=run
    )
    histograms = np.zeros(
        (len(grouped_run.label_titles), AMPLITUDE_BIN_COUNT),
        dtype=np.int64,
    )

    def handle_batch(
        _event_id: int, cleaned: np.ndarray, label_indices: np.ndarray
    ) -> None:
        for row, label_index in zip(cleaned, label_indices, strict=True):
            _accumulate_peak_histogram(
                row=row,
                histogram=histograms[int(label_index)],
                peak_separation=peak_separation,
                peak_prominence=peak_prominence,
                peak_width=peak_width,
            )

    scan_grouped_labeled_trace_batches(
        grouped_run,
        baseline_window_scale=baseline_window_scale,
        handler=handle_batch,
        progress=progress,
    )

    return {
        "run_id": np.int64(run),
        "label_keys": np.asarray(grouped_run.label_keys, dtype=np.str_),
        "label_titles": np.asarray(grouped_run.label_titles, dtype=np.str_),
        "trace_counts": grouped_run.trace_counts,
        "histograms": histograms,
    }


def max_peak_amplitude(
    row: np.ndarray,
    peak_separation: float,
    peak_prominence: float,
    peak_width: float,
) -> float:
    amplitudes = compute_peak_amplitudes(
        np.asarray([row]),
        peak_separation=peak_separation,
        peak_prominence=peak_prominence,
        peak_width=peak_width,
    )
    return float(amplitudes[0])


def _accumulate_peak_histogram(
    row: np.ndarray,
    histogram: np.ndarray,
    peak_separation: float,
    peak_prominence: float,
    peak_width: float,
) -> None:
    peaks, _ = signal.find_peaks(
        row,
        distance=peak_separation,
        prominence=peak_prominence,
        width=(1.0, peak_width),
        rel_height=0.95,
    )
    for peak in peaks:
        amplitude = int(np.clip(row[peak], 0, histogram.shape[0] - 1))
        histogram[amplitude] += 1
