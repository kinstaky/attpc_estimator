from __future__ import annotations

from pathlib import Path

import numpy as np

from .filter_core import analyze_saturation_batch
from .labeled import (
    NORMAL_LABEL_GROUPS,
    load_grouped_labeled_run,
    scan_grouped_labeled_trace_batches,
)
from .progress import ProgressReporter
from .trace_scan import scan_cleaned_trace_batches

SATURATION_VARIANTS = ("drop", "length")
SATURATION_DROP_BIN_COUNT = 500
SATURATION_LENGTH_BIN_COUNT = 256
SATURATION_BIN_COUNTS = {
    "drop": SATURATION_DROP_BIN_COUNT,
    "length": SATURATION_LENGTH_BIN_COUNT,
}
SATURATION_BIN_LABELS = {
    "drop": "Drop from local maximum",
    "length": "Plateau length",
}

__all__ = [
    "SATURATION_VARIANTS",
    "SATURATION_BIN_COUNTS",
    "SATURATION_BIN_LABELS",
    "NORMAL_LABEL_GROUPS",
    "build_saturation_histograms",
    "build_labeled_saturation_histograms",
    "accumulate_saturation_histograms",
    "accumulate_grouped_saturation_histograms",
]


def build_saturation_histograms(
    trace_file_path: Path,
    *,
    baseline_window_scale: float = 10.0,
    threshold: float = 2000.0,
    drop_threshold: float,
    window_radius: int = 16,
    progress: ProgressReporter | None = None,
) -> dict[str, np.ndarray | np.int64]:
    drop_histogram = np.zeros(SATURATION_DROP_BIN_COUNT, dtype=np.int64)
    length_histogram = np.zeros(SATURATION_LENGTH_BIN_COUNT, dtype=np.int64)
    trace_count = np.int64(0)

    def handle_batch(_event_id: int, cleaned: np.ndarray) -> None:
        nonlocal trace_count
        accumulate_saturation_histograms(
            cleaned,
            drop_histogram=drop_histogram,
            length_histogram=length_histogram,
            threshold=threshold,
            drop_threshold=drop_threshold,
            window_radius=window_radius,
        )
        trace_count += np.int64(cleaned.shape[0])

    scan_cleaned_trace_batches(
        trace_file_path,
        baseline_window_scale=baseline_window_scale,
        handler=handle_batch,
        progress=progress,
    )
    return {
        "trace_count": trace_count,
        "drop_histogram": drop_histogram,
        "length_histogram": length_histogram,
    }


def build_labeled_saturation_histograms(
    trace_path: Path,
    workspace: Path,
    run: int,
    *,
    baseline_window_scale: float = 10.0,
    threshold: float = 2000.0,
    drop_threshold: float,
    window_radius: int = 16,
    progress: ProgressReporter | None = None,
) -> dict[str, np.ndarray | np.int64]:
    grouped_run = load_grouped_labeled_run(
        trace_path=trace_path,
        workspace=workspace,
        run=run,
    )
    label_count = len(grouped_run.label_titles)
    drop_histograms = np.zeros((label_count, SATURATION_DROP_BIN_COUNT), dtype=np.int64)
    length_histograms = np.zeros((label_count, SATURATION_LENGTH_BIN_COUNT), dtype=np.int64)

    def handle_batch(
        _event_id: int, cleaned: np.ndarray, label_indices: np.ndarray
    ) -> None:
        accumulate_grouped_saturation_histograms(
            cleaned,
            label_indices=label_indices,
            drop_histograms=drop_histograms,
            length_histograms=length_histograms,
            threshold=threshold,
            drop_threshold=drop_threshold,
            window_radius=window_radius,
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
        "drop_histograms": drop_histograms,
        "length_histograms": length_histograms,
    }


def accumulate_saturation_histograms(
    cleaned: np.ndarray,
    *,
    drop_histogram: np.ndarray,
    length_histogram: np.ndarray,
    threshold: float,
    drop_threshold: float,
    window_radius: int,
) -> None:
    batch = analyze_saturation_batch(
        cleaned,
        threshold=threshold,
        drop_threshold=drop_threshold,
        window_radius=window_radius,
    )
    for drop_values in batch.drop_values_by_trace:
        _accumulate_values(drop_histogram, drop_values)
    _accumulate_values(
        length_histogram,
        batch.plateau_lengths[batch.plateau_lengths >= 2].astype(np.float32, copy=False),
    )


def accumulate_grouped_saturation_histograms(
    cleaned: np.ndarray,
    *,
    label_indices: np.ndarray,
    drop_histograms: np.ndarray,
    length_histograms: np.ndarray,
    threshold: float,
    drop_threshold: float,
    window_radius: int,
) -> None:
    batch = analyze_saturation_batch(
        cleaned,
        threshold=threshold,
        drop_threshold=drop_threshold,
        window_radius=window_radius,
    )
    for trace_id, label_index in enumerate(label_indices):
        resolved_label_index = int(label_index)
        _accumulate_values(
            drop_histograms[resolved_label_index],
            batch.drop_values_by_trace[trace_id],
        )
        _accumulate_values(
            length_histograms[resolved_label_index],
            np.asarray([batch.plateau_lengths[trace_id]], dtype=np.float32)
            if int(batch.plateau_lengths[trace_id]) >= 2
            else np.empty(0, dtype=np.float32),
        )


def _accumulate_values(histogram: np.ndarray, values: np.ndarray) -> None:
    for value in np.asarray(values, dtype=np.float32):
        if value < 0 or value >= histogram.shape[0]:
            continue
        bin_index = int(value)
        histogram[bin_index] += 1
