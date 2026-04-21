from __future__ import annotations

from functools import lru_cache

import h5py
import numpy as np
from numba import njit
from attpc_storage.hdf5 import (
    TraceEventMetadata,
    TraceLayout,
    collect_event_counts,
    describe_trace_events,
    detect_trace_layout,
    load_pad_rows as _load_storage_pad_rows,
    load_pad_traces as _load_storage_pad_traces,
)

from ..model.trace import TraceRecord

PAD_TRACE_OFFSET = 5
CDF_THRESHOLDS = np.arange(1, 151, dtype=np.int64)
CDF_VALUE_BINS = 100


def load_trace_record(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    trace_id: int,
    baseline_window_scale: float,
) -> TraceRecord:
    rows = load_pad_rows(
        file_handle,
        run=run,
        event_id=event_id,
        trace_ids=np.asarray([trace_id], dtype=np.int64),
    )
    return trace_record_from_pad_row(
        run=run,
        event_id=event_id,
        trace_id=trace_id,
        row=rows[0],
        baseline_window_scale=baseline_window_scale,
    )


def trace_record_from_pad_row(
    *,
    run: int,
    event_id: int,
    trace_id: int,
    row: np.ndarray,
    baseline_window_scale: float,
) -> TraceRecord:
    hardware = np.asarray(row[:PAD_TRACE_OFFSET], dtype=np.float32)
    raw = np.asarray(row[PAD_TRACE_OFFSET:], dtype=np.float32)
    trace = preprocess_traces(
        raw[np.newaxis, :],
        baseline_window_scale=baseline_window_scale,
    )[0]
    transformed = compute_frequency_distribution(trace[np.newaxis, :])[0]
    return TraceRecord(
        run=run,
        event_id=event_id,
        trace_id=trace_id,
        detector="pad",
        hardware_id=hardware,
        raw=raw,
        trace=trace,
        transformed=transformed,
        family=None,
        label=None,
    )


def load_pad_rows(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    trace_ids: np.ndarray | None = None,
) -> np.ndarray:
    try:
        return _load_storage_pad_rows(
            file_handle,
            event_id=event_id,
            trace_ids=trace_ids,
        )
    except LookupError as exc:
        raise LookupError(f"trace {run}/{event_id} is not available") from exc


def load_pad_traces(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    trace_ids: np.ndarray | None = None,
) -> np.ndarray:
    try:
        return _load_storage_pad_traces(
            file_handle,
            event_id=event_id,
            trace_ids=trace_ids,
        )
    except LookupError as exc:
        raise LookupError(f"trace {run}/{event_id} is not available") from exc


def event_trace_count(file_handle: h5py.File, event_id: int) -> int:
    metadata = describe_trace_events(file_handle)
    if event_id < metadata.min_event or event_id > metadata.max_event or event_id in metadata.bad_events:
        return 0
    try:
        return int(_load_storage_pad_rows(file_handle, event_id=event_id).shape[0])
    except LookupError:
        return 0


def _replace_baseline_peaks(trace_matrix: np.ndarray) -> np.ndarray:
    bases = np.array(trace_matrix, copy=True)
    means = np.mean(bases, axis=1, keepdims=True, dtype=np.float32)
    sigmas = np.std(bases, axis=1, keepdims=True, dtype=np.float32)
    cutoff = sigmas * np.float32(1.5)
    valid_mask = np.abs(bases - means) <= cutoff

    valid_sums = np.sum(
        np.where(valid_mask, bases, np.float32(0.0)),
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    valid_counts = np.sum(valid_mask, axis=1, keepdims=True, dtype=np.int32)
    replacements = np.divide(
        valid_sums,
        valid_counts,
        out=means.copy(),
        where=valid_counts > 0,
    ).astype(np.float32, copy=False)

    return np.where(valid_mask, bases, replacements).astype(np.float32, copy=False)


@lru_cache(maxsize=None)
def _get_baseline_filter(sample_count: int, baseline_window_scale: float) -> np.ndarray:
    window = np.arange(sample_count, dtype=np.float32) - (sample_count // 2)
    full_filter = np.fft.ifftshift(np.sinc(window / baseline_window_scale)).astype(
        np.float32, copy=False
    )
    return np.ascontiguousarray(full_filter[: sample_count // 2 + 1])


def preprocess_traces(traces: np.ndarray, baseline_window_scale: float) -> np.ndarray:
    traces_array = np.asarray(traces, dtype=np.float32)
    if traces_array.ndim != 2:
        raise ValueError(f"expected a 2D trace matrix, got shape {traces_array.shape}")

    trace_matrix = np.array(traces_array, copy=True)
    sample_count = trace_matrix.shape[1]

    if sample_count < 2:
        return trace_matrix

    trace_matrix[:, 0] = trace_matrix[:, 1]
    trace_matrix[:, -1] = trace_matrix[:, -2]

    bases = _replace_baseline_peaks(trace_matrix)
    baseline_filter = _get_baseline_filter(
        sample_count=sample_count, baseline_window_scale=baseline_window_scale
    )
    transformed = np.fft.rfft(bases, axis=1)
    filtered = np.fft.irfft(
        transformed * baseline_filter[np.newaxis, :],
        n=sample_count,
        axis=1,
    ).astype(np.float32, copy=False)
    return trace_matrix - filtered


def compute_frequency_distribution(traces: np.ndarray) -> np.ndarray:
    trace_matrix = np.asarray(traces, dtype=np.float32)
    if trace_matrix.ndim != 2:
        raise ValueError(f"expected a 2D trace matrix, got shape {trace_matrix.shape}")
    return np.abs(np.fft.rfft(trace_matrix, axis=1)).astype(np.float32, copy=False)


@njit(cache=False)
def _sample_cdf_points_numba(
    spectrum: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    row_count, bin_count = spectrum.shape
    threshold_count = thresholds.shape[0]
    samples = np.zeros((row_count, threshold_count), dtype=np.float32)

    for row_index in range(row_count):
        total = 0.0
        for bin_index in range(bin_count):
            total += float(spectrum[row_index, bin_index])
        if total <= 0.0:
            continue

        cumulative = np.empty(bin_count, dtype=np.float32)
        running = 0.0
        for bin_index in range(bin_count):
            running += float(spectrum[row_index, bin_index]) / total
            cumulative[bin_index] = running

        for threshold_index in range(threshold_count):
            threshold = thresholds[threshold_index]
            if threshold <= 0:
                samples[row_index, threshold_index] = 0.0
            elif threshold >= bin_count:
                samples[row_index, threshold_index] = 1.0
            else:
                samples[row_index, threshold_index] = cumulative[threshold - 1]

    return samples


def sample_cdf_points(
    spectrum: np.ndarray, thresholds: np.ndarray = CDF_THRESHOLDS
) -> np.ndarray:
    spectrum_array = np.asarray(spectrum, dtype=np.float32)
    if spectrum_array.ndim != 2:
        raise ValueError(
            f"expected a 2D spectrum matrix, got shape {spectrum_array.shape}"
        )
    thresholds_array = np.asarray(thresholds, dtype=np.int64)
    return _sample_cdf_points_numba(spectrum_array, thresholds_array)
