from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from .progress import ProgressReporter
from ..process.labeled import (
    NORMAL_LABEL_GROUPS,
    load_grouped_labeled_run,
    scan_grouped_labeled_trace_batches,
)
from ..process.trace_scan import scan_cleaned_trace_batches
from ..utils.trace_data import (
    CDF_THRESHOLDS,
    CDF_VALUE_BINS,
    compute_frequency_distribution,
    sample_cdf_points,
)

__all__ = [
    "CDF_THRESHOLDS",
    "CDF_VALUE_BINS",
    "NORMAL_LABEL_GROUPS",
    "build_trace_cdf_histogram",
    "build_labeled_cdf_histograms",
    "_accumulate_cdf_histogram_numba",
    "_accumulate_grouped_histograms_numba",
]


def build_trace_cdf_histogram(
    trace_file_path: Path,
    baseline_window_scale: float = 20.0,
    thresholds: np.ndarray = CDF_THRESHOLDS,
    progress: ProgressReporter | None = None,
) -> np.ndarray:
    histogram = np.zeros((len(thresholds), CDF_VALUE_BINS), dtype=np.int64)

    def handle_batch(_event_id: int, cleaned: np.ndarray) -> None:
        spectrum = compute_frequency_distribution(cleaned)
        samples = sample_cdf_points(spectrum, thresholds=thresholds)
        _accumulate_cdf_histogram_numba(samples, histogram)

    scan_cleaned_trace_batches(
        trace_file_path,
        baseline_window_scale=baseline_window_scale,
        handler=handle_batch,
        progress=progress,
    )
    return histogram


def build_labeled_cdf_histograms(
    trace_path: Path,
    workspace: Path,
    run: int,
    baseline_window_scale: float = 20.0,
    progress: ProgressReporter | None = None,
) -> dict[str, np.ndarray | np.int64]:
    grouped_run = load_grouped_labeled_run(
        trace_path=trace_path, workspace=workspace, run=run
    )
    histograms = np.zeros(
        (len(grouped_run.label_titles), len(CDF_THRESHOLDS), CDF_VALUE_BINS),
        dtype=np.int64,
    )

    def handle_batch(
        _event_id: int, cleaned: np.ndarray, label_indices: np.ndarray
    ) -> None:
        spectrum = compute_frequency_distribution(cleaned)
        samples = sample_cdf_points(spectrum, thresholds=CDF_THRESHOLDS)
        _accumulate_grouped_histograms_numba(samples, label_indices, histograms)

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
        "histograms": histograms,
        "trace_counts": grouped_run.trace_counts,
    }


@njit(cache=False)
def _accumulate_cdf_histogram_numba(samples: np.ndarray, histogram: np.ndarray) -> None:
    row_count, column_count = samples.shape
    value_bin_count = histogram.shape[1]

    for row_index in range(row_count):
        for column_index in range(column_count):
            value = float(samples[row_index, column_index])
            if value <= 0.0:
                value_bin_index = 0
            elif value >= 1.0:
                value_bin_index = value_bin_count - 1
            else:
                value_bin_index = int(value * value_bin_count)
            histogram[column_index, value_bin_index] += 1


@njit(cache=False)
def _accumulate_grouped_histograms_numba(
    samples: np.ndarray,
    label_indices: np.ndarray,
    histograms: np.ndarray,
) -> None:
    row_count, column_count = samples.shape
    value_bin_count = histograms.shape[2]

    for row_index in range(row_count):
        label_index = int(label_indices[row_index])
        for column_index in range(column_count):
            value = float(samples[row_index, column_index])
            if value <= 0.0:
                value_bin_index = 0
            elif value >= 1.0:
                value_bin_index = value_bin_count - 1
            else:
                value_bin_index = int(value * value_bin_count)
            histograms[label_index, column_index, value_bin_index] += 1
