from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from .baseline import (
    BASELINE_ABS_MAX,
    BASELINE_BIN_COUNT,
    BASELINE_COUNT_LABEL,
)
from .labeled import (
    NORMAL_LABEL_GROUPS,
    load_grouped_labeled_run,
    scan_grouped_labeled_trace_batches,
)
from .progress import ProgressReporter
from .trace_metrics import compute_second_derivative, compute_second_derivative_batch
from .trace_scan import scan_cleaned_trace_batches

BITFLIP_BASELINE_DEFAULT = 10.0
BITFLIP_FILTER_MIN_COUNT_DEFAULT = 1
BITFLIP_ALLOWED_ABS_VALUES = (
    61.0,
    121.0,
    450.0,
    512.0,
    574.0,
    902.0,
    1024.0,
    3584.0,
    3646.0,
    4096.0,
)
BITFLIP_ALLOWED_ABS_TOLERANCE = 15.0
BITFLIP_ALLOWED_ABS_TARGETS = np.asarray(BITFLIP_ALLOWED_ABS_VALUES, dtype=np.float32)
BITFLIP_VARIANTS = ("baseline", "value", "length", "count")
BITFLIP_BASELINE_BIN_COUNT = BASELINE_BIN_COUNT
BITFLIP_VALUE_BIN_COUNT = 8192
BITFLIP_LENGTH_BIN_COUNT = 256
BITFLIP_COUNT_BIN_COUNT = 256
BITFLIP_BIN_COUNTS = {
    "baseline": BITFLIP_BASELINE_BIN_COUNT,
    "value": BITFLIP_VALUE_BIN_COUNT,
    "length": BITFLIP_LENGTH_BIN_COUNT,
    "count": BITFLIP_COUNT_BIN_COUNT,
}
BITFLIP_BIN_LABELS = {
    "baseline": "Second-derivative value",
    "value": "Absolute second-derivative value",
    "length": "Qualified alternating-run length",
    "count": "Found bitflip structures",
}
BITFLIP_COUNT_LABELS = {
    "baseline": BASELINE_COUNT_LABEL,
    "value": "Count",
    "length": "Trace count",
    "count": "Trace count",
}

__all__ = [
    "BITFLIP_BASELINE_DEFAULT",
    "BITFLIP_FILTER_MIN_COUNT_DEFAULT",
    "BITFLIP_VARIANTS",
    "BITFLIP_BIN_COUNTS",
    "BITFLIP_BIN_LABELS",
    "NORMAL_LABEL_GROUPS",
    "BitFlipTraceAnalysis",
    "build_bitflip_histograms",
    "build_labeled_bitflip_histograms",
    "accumulate_bitflip_histograms",
    "accumulate_grouped_bitflip_histograms",
    "analyze_bitflip_trace",
    "count_qualified_bitflip_segments_batch",
]


@dataclass(frozen=True, slots=True)
class BitFlipTraceAnalysis:
    second_derivative: np.ndarray
    segment_value_sets: tuple[np.ndarray, ...]
    qualified_segment_lengths: np.ndarray
    structures: tuple["BitFlipStructure", ...]


@dataclass(frozen=True, slots=True)
class BitFlipStructure:
    start_baseline_index: int
    end_baseline_index: int


def build_bitflip_histograms(
    trace_file_path: Path,
    *,
    baseline_window_scale: float = 10.0,
    baseline_threshold: float = BITFLIP_BASELINE_DEFAULT,
    progress: ProgressReporter | None = None,
) -> dict[str, np.ndarray | np.int64]:
    baseline_histogram = np.zeros(BITFLIP_BASELINE_BIN_COUNT, dtype=np.int64)
    value_histogram = np.zeros(BITFLIP_VALUE_BIN_COUNT, dtype=np.int64)
    length_histogram = np.zeros(BITFLIP_LENGTH_BIN_COUNT, dtype=np.int64)
    count_histogram = np.zeros(BITFLIP_COUNT_BIN_COUNT, dtype=np.int64)
    trace_count = np.int64(0)

    def handle_batch(_event_id: int, cleaned: np.ndarray) -> None:
        nonlocal trace_count
        accumulate_bitflip_histograms(
            cleaned,
            baseline_histogram=baseline_histogram,
            value_histogram=value_histogram,
            length_histogram=length_histogram,
            count_histogram=count_histogram,
            baseline_threshold=baseline_threshold,
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
        "baseline_histogram": baseline_histogram,
        "value_histogram": value_histogram,
        "length_histogram": length_histogram,
        "count_histogram": count_histogram,
    }


def build_labeled_bitflip_histograms(
    trace_path: Path,
    workspace: Path,
    run: int,
    *,
    baseline_window_scale: float = 10.0,
    baseline_threshold: float = BITFLIP_BASELINE_DEFAULT,
    progress: ProgressReporter | None = None,
) -> dict[str, np.ndarray | np.int64]:
    grouped_run = load_grouped_labeled_run(
        trace_path=trace_path,
        workspace=workspace,
        run=run,
    )
    label_count = len(grouped_run.label_titles)
    baseline_histograms = np.zeros((label_count, BITFLIP_BASELINE_BIN_COUNT), dtype=np.int64)
    value_histograms = np.zeros((label_count, BITFLIP_VALUE_BIN_COUNT), dtype=np.int64)
    length_histograms = np.zeros((label_count, BITFLIP_LENGTH_BIN_COUNT), dtype=np.int64)
    count_histograms = np.zeros((label_count, BITFLIP_COUNT_BIN_COUNT), dtype=np.int64)

    def handle_batch(
        _event_id: int, cleaned: np.ndarray, label_indices: np.ndarray
    ) -> None:
        accumulate_grouped_bitflip_histograms(
            cleaned,
            label_indices=label_indices,
            baseline_histograms=baseline_histograms,
            value_histograms=value_histograms,
            length_histograms=length_histograms,
            count_histograms=count_histograms,
            baseline_threshold=baseline_threshold,
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
        "baseline_histograms": baseline_histograms,
        "value_histograms": value_histograms,
        "length_histograms": length_histograms,
        "count_histograms": count_histograms,
    }


def accumulate_bitflip_histograms(
    cleaned: np.ndarray,
    *,
    baseline_histogram: np.ndarray,
    value_histogram: np.ndarray,
    length_histogram: np.ndarray,
    count_histogram: np.ndarray,
    baseline_threshold: float,
) -> None:
    second_derivatives = compute_second_derivative_batch(cleaned)
    _accumulate_bitflip_baseline_values(
        baseline_histogram,
        second_derivatives.reshape(-1),
    )
    for second_derivative in second_derivatives:
        _accumulate_bitflip_row(
            second_derivative,
            value_histogram=value_histogram,
            length_histogram=length_histogram,
            count_histogram=count_histogram,
            baseline_threshold=baseline_threshold,
        )


def accumulate_grouped_bitflip_histograms(
    cleaned: np.ndarray,
    *,
    label_indices: np.ndarray,
    baseline_histograms: np.ndarray,
    value_histograms: np.ndarray,
    length_histograms: np.ndarray,
    count_histograms: np.ndarray,
    baseline_threshold: float,
) -> None:
    second_derivatives = compute_second_derivative_batch(cleaned)
    for trace_id, label_index in enumerate(label_indices):
        resolved_label_index = int(label_index)
        _accumulate_bitflip_baseline_values(
            baseline_histograms[resolved_label_index],
            second_derivatives[trace_id],
        )
        _accumulate_bitflip_row(
            second_derivatives[trace_id],
            value_histogram=value_histograms[resolved_label_index],
            length_histogram=length_histograms[resolved_label_index],
            count_histogram=count_histograms[resolved_label_index],
            baseline_threshold=baseline_threshold,
        )


def count_qualified_bitflip_segments_batch(
    cleaned: np.ndarray,
    *,
    baseline_threshold: float,
) -> np.ndarray:
    second_derivatives = compute_second_derivative_batch(cleaned)
    counts = np.zeros(second_derivatives.shape[0], dtype=np.int64)
    for trace_id, second_derivative in enumerate(second_derivatives):
        counts[trace_id] = _count_qualified_bitflip_segments(
            second_derivative,
            baseline_threshold=baseline_threshold,
        )
    return counts


def analyze_bitflip_trace(
    row: np.ndarray,
    *,
    baseline_threshold: float,
) -> BitFlipTraceAnalysis:
    second_derivative = compute_second_derivative(row)
    if second_derivative.size == 0:
        return BitFlipTraceAnalysis(
            second_derivative=second_derivative,
            segment_value_sets=(),
            qualified_segment_lengths=np.empty(0, dtype=np.float32),
            structures=(),
        )

    segment_value_sets: list[np.ndarray] = []
    qualified_segment_lengths: list[float] = []
    structures: list[BitFlipStructure] = []
    for run_start, run_stop in _iter_valid_bitflip_segments(
        second_derivative,
        baseline_threshold=baseline_threshold,
    ):
        run_values = second_derivative[run_start:run_stop]
        segment_value_sets.append(np.abs(run_values).astype(np.float32, copy=False))
        if _segment_matches_allowed_abs_values(run_values):
            qualified_segment_lengths.append(float(run_values.size))
            structures.append(
                BitFlipStructure(
                    start_baseline_index=int(run_start - 1),
                    end_baseline_index=int(run_stop),
                )
            )

    return BitFlipTraceAnalysis(
        second_derivative=second_derivative,
        segment_value_sets=tuple(segment_value_sets),
        qualified_segment_lengths=np.asarray(qualified_segment_lengths, dtype=np.float32),
        structures=tuple(structures),
    )


def _accumulate_bitflip_row(
    second_derivative: np.ndarray,
    *,
    value_histogram: np.ndarray,
    length_histogram: np.ndarray,
    count_histogram: np.ndarray,
    baseline_threshold: float,
) -> None:
    values = np.asarray(second_derivative, dtype=np.float32)
    if values.size == 0:
        return
    _accumulate_bitflip_row_numba(
        values,
        value_histogram,
        length_histogram,
        count_histogram,
        float(baseline_threshold),
        BITFLIP_ALLOWED_ABS_TARGETS,
        float(BITFLIP_ALLOWED_ABS_TOLERANCE),
    )


def _count_qualified_bitflip_segments(
    second_derivative: np.ndarray,
    *,
    baseline_threshold: float,
) -> int:
    values = np.asarray(second_derivative, dtype=np.float32)
    if values.size == 0:
        return 0
    return int(
        _count_qualified_bitflip_segments_numba(
            values,
            float(baseline_threshold),
            BITFLIP_ALLOWED_ABS_TARGETS,
            float(BITFLIP_ALLOWED_ABS_TOLERANCE),
        )
    )


def _iter_valid_bitflip_segments(
    second_derivative: np.ndarray,
    *,
    baseline_threshold: float,
):
    row = _prepare_bitflip_row(second_derivative, baseline_threshold=baseline_threshold)
    if row is None:
        return

    _, positive, nonbaseline, _ = row
    yield from _iter_valid_bitflip_segments_from_masks(
        nonbaseline=nonbaseline,
        positive=positive,
    )


def _prepare_bitflip_row(
    second_derivative: np.ndarray,
    *,
    baseline_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    values = np.asarray(second_derivative, dtype=np.float32)
    if values.size == 0:
        return None
    abs_values = np.abs(values).astype(np.float32, copy=False)
    nonbaseline = abs_values > float(baseline_threshold)
    if not bool(np.any(nonbaseline)):
        return None
    positive = values > 0.0
    allowed_mask = _compute_allowed_abs_mask(abs_values)
    return abs_values, positive, nonbaseline, allowed_mask


def _iter_valid_bitflip_segments_from_masks(
    *,
    nonbaseline: np.ndarray,
    positive: np.ndarray,
):
    if nonbaseline.size == 0 or not bool(np.any(nonbaseline)):
        return

    transitions = np.diff(
        np.pad(nonbaseline.astype(np.int8, copy=False), (1, 1), constant_values=0)
    )
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    sample_count = int(nonbaseline.shape[0])

    for start, stop in zip(starts, stops):
        run_start = int(start)
        run_stop = int(stop)
        if run_start == 0 or run_stop == sample_count:
            continue
        if (run_stop - run_start) < 2:
            continue
        run_signs = positive[run_start:run_stop]
        if bool(np.any(run_signs[1:] == run_signs[:-1])):
            continue
        yield run_start, run_stop


def _segment_matches_allowed_abs_values(run_values: np.ndarray) -> bool:
    if run_values.size == 0:
        return False
    return bool(np.all(_compute_allowed_abs_mask(np.abs(np.asarray(run_values, dtype=np.float32)))))


def _accumulate_bitflip_baseline_values(histogram: np.ndarray, values: np.ndarray) -> None:
    if values.size == 0:
        return
    rounded = np.rint(np.asarray(values, dtype=np.float32)).astype(np.int64, copy=False)
    indices = np.clip(rounded + BASELINE_ABS_MAX, 0, BITFLIP_BASELINE_BIN_COUNT - 1)
    np.add.at(histogram, indices, 1)


def _compute_allowed_abs_mask(magnitudes: np.ndarray) -> np.ndarray:
    values = np.asarray(magnitudes, dtype=np.float32)
    if values.size == 0:
        return np.zeros(0, dtype=bool)

    insertion_points = np.searchsorted(BITFLIP_ALLOWED_ABS_TARGETS, values, side="left")
    left_indices = np.clip(insertion_points - 1, 0, BITFLIP_ALLOWED_ABS_TARGETS.shape[0] - 1)
    right_indices = np.clip(insertion_points, 0, BITFLIP_ALLOWED_ABS_TARGETS.shape[0] - 1)
    left_distances = np.abs(values - BITFLIP_ALLOWED_ABS_TARGETS[left_indices])
    right_distances = np.abs(values - BITFLIP_ALLOWED_ABS_TARGETS[right_indices])
    return np.minimum(left_distances, right_distances) <= BITFLIP_ALLOWED_ABS_TOLERANCE


@njit(cache=False)
def _matches_allowed_abs_value_numba(
    magnitude: float,
    allowed_targets: np.ndarray,
    tolerance: float,
) -> bool:
    left = 0
    right = allowed_targets.shape[0]
    while left < right:
        mid = (left + right) // 2
        if magnitude > float(allowed_targets[mid]):
            left = mid + 1
        else:
            right = mid

    best_distance = 1.0e30
    if left < allowed_targets.shape[0]:
        distance = abs(magnitude - float(allowed_targets[left]))
        if distance < best_distance:
            best_distance = distance
    if left > 0:
        distance = abs(magnitude - float(allowed_targets[left - 1]))
        if distance < best_distance:
            best_distance = distance
    return best_distance <= tolerance


@njit(cache=False)
def _accumulate_bitflip_row_numba(
    second_derivative: np.ndarray,
    value_histogram: np.ndarray,
    length_histogram: np.ndarray,
    count_histogram: np.ndarray,
    baseline_threshold: float,
    allowed_targets: np.ndarray,
    tolerance: float,
) -> None:
    size = second_derivative.shape[0]
    if size == 0:
        return

    value_bins_limit = value_histogram.shape[0] - 1
    length_bins_limit = length_histogram.shape[0] - 1
    count_bins_limit = count_histogram.shape[0] - 1
    index = 0
    qualified_count = 0

    while index < size:
        magnitude = abs(float(second_derivative[index]))
        if magnitude <= baseline_threshold:
            index += 1
            continue

        run_start = index
        previous_positive = float(second_derivative[index]) > 0.0
        alternating = True
        qualified = _matches_allowed_abs_value_numba(
            magnitude,
            allowed_targets,
            tolerance,
        )
        index += 1

        while index < size:
            magnitude = abs(float(second_derivative[index]))
            if magnitude <= baseline_threshold:
                break
            current_positive = float(second_derivative[index]) > 0.0
            if current_positive == previous_positive:
                alternating = False
            previous_positive = current_positive
            if qualified and not _matches_allowed_abs_value_numba(
                magnitude,
                allowed_targets,
                tolerance,
            ):
                qualified = False
            index += 1

        run_stop = index
        left_has_baseline = run_start > 0
        right_has_baseline = run_stop < size
        run_length = run_stop - run_start
        if left_has_baseline and right_has_baseline and alternating and run_length >= 2:
            for point_index in range(run_start, run_stop):
                value = abs(float(second_derivative[point_index]))
                bin_index = int(value)
                if bin_index < 0:
                    bin_index = 0
                elif bin_index > value_bins_limit:
                    bin_index = value_bins_limit
                value_histogram[bin_index] += 1
            if qualified:
                length_index = run_length
                if length_index < 0:
                    length_index = 0
                elif length_index > length_bins_limit:
                    length_index = length_bins_limit
                length_histogram[length_index] += 1
                qualified_count += 1

    if qualified_count > 0:
        count_index = qualified_count
        if count_index > count_bins_limit:
            count_index = count_bins_limit
        count_histogram[count_index] += 1


@njit(cache=False)
def _count_qualified_bitflip_segments_numba(
    second_derivative: np.ndarray,
    baseline_threshold: float,
    allowed_targets: np.ndarray,
    tolerance: float,
) -> int:
    size = second_derivative.shape[0]
    if size == 0:
        return 0

    index = 0
    count = 0
    while index < size:
        magnitude = abs(float(second_derivative[index]))
        if magnitude <= baseline_threshold:
            index += 1
            continue

        run_start = index
        previous_positive = float(second_derivative[index]) > 0.0
        alternating = True
        qualified = _matches_allowed_abs_value_numba(
            magnitude,
            allowed_targets,
            tolerance,
        )
        index += 1

        while index < size:
            magnitude = abs(float(second_derivative[index]))
            if magnitude <= baseline_threshold:
                break
            current_positive = float(second_derivative[index]) > 0.0
            if current_positive == previous_positive:
                alternating = False
            previous_positive = current_positive
            if qualified and not _matches_allowed_abs_value_numba(
                magnitude,
                allowed_targets,
                tolerance,
            ):
                qualified = False
            index += 1

        run_stop = index
        left_has_baseline = run_start > 0
        right_has_baseline = run_stop < size
        if left_has_baseline and right_has_baseline and alternating and (run_stop - run_start) >= 2:
            if qualified:
                count += 1
    return count
