from __future__ import annotations

from pathlib import Path

import numpy as np

from .labeled import (
    NORMAL_LABEL_GROUPS,
    load_grouped_labeled_run,
    scan_grouped_labeled_trace_batches,
)
from .progress import ProgressReporter
from .trace_scan import scan_cleaned_trace_batches

BASELINE_ABS_MAX = 4096
BASELINE_BIN_CENTERS = np.arange(-BASELINE_ABS_MAX, BASELINE_ABS_MAX + 1, dtype=np.int64)
BASELINE_BIN_COUNT = int(BASELINE_BIN_CENTERS.shape[0])
BASELINE_BIN_LABEL = "Baseline value"
BASELINE_COUNT_LABEL = "Sample count"

__all__ = [
    "BASELINE_BIN_CENTERS",
    "BASELINE_BIN_COUNT",
    "BASELINE_BIN_LABEL",
    "BASELINE_COUNT_LABEL",
    "NORMAL_LABEL_GROUPS",
    "build_baseline_histogram",
    "build_labeled_baseline_histograms",
    "accumulate_baseline_histogram",
    "accumulate_grouped_baseline_histograms",
]


def build_baseline_histogram(
    trace_file_path: Path,
    *,
    baseline_window_scale: float = 10.0,
    progress: ProgressReporter | None = None,
) -> dict[str, np.ndarray | np.int64]:
    histogram = np.zeros(BASELINE_BIN_COUNT, dtype=np.int64)
    trace_count = np.int64(0)

    def handle_batch(_event_id: int, cleaned: np.ndarray) -> None:
        nonlocal trace_count
        accumulate_baseline_histogram(cleaned, histogram=histogram)
        trace_count += np.int64(cleaned.shape[0])

    scan_cleaned_trace_batches(
        trace_file_path,
        baseline_window_scale=baseline_window_scale,
        handler=handle_batch,
        progress=progress,
    )
    return {
        "trace_count": trace_count,
        "histogram": histogram,
        "bin_centers": BASELINE_BIN_CENTERS.copy(),
    }


def build_labeled_baseline_histograms(
    trace_path: Path,
    workspace: Path,
    run: int,
    *,
    baseline_window_scale: float = 10.0,
    progress: ProgressReporter | None = None,
) -> dict[str, np.ndarray | np.int64]:
    grouped_run = load_grouped_labeled_run(
        trace_path=trace_path,
        workspace=workspace,
        run=run,
    )
    label_count = len(grouped_run.label_titles)
    histograms = np.zeros((label_count, BASELINE_BIN_COUNT), dtype=np.int64)

    def handle_batch(
        _event_id: int, cleaned: np.ndarray, label_indices: np.ndarray
    ) -> None:
        accumulate_grouped_baseline_histograms(
            cleaned,
            label_indices=label_indices,
            histograms=histograms,
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
        "bin_centers": BASELINE_BIN_CENTERS.copy(),
    }


def accumulate_baseline_histogram(
    cleaned: np.ndarray,
    *,
    histogram: np.ndarray,
) -> None:
    _accumulate_baseline_values(histogram, np.asarray(cleaned, dtype=np.float32).reshape(-1))


def accumulate_grouped_baseline_histograms(
    cleaned: np.ndarray,
    *,
    label_indices: np.ndarray,
    histograms: np.ndarray,
) -> None:
    for trace_id, label_index in enumerate(label_indices):
        _accumulate_baseline_values(
            histograms[int(label_index)],
            np.asarray(cleaned[trace_id], dtype=np.float32),
        )


def _accumulate_baseline_values(histogram: np.ndarray, values: np.ndarray) -> None:
    if values.size == 0:
        return
    rounded = np.rint(np.asarray(values, dtype=np.float32)).astype(np.int64, copy=False)
    indices = np.clip(rounded + BASELINE_ABS_MAX, 0, BASELINE_BIN_COUNT - 1)
    np.add.at(histogram, indices, 1)
