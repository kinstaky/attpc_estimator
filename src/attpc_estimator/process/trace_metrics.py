from __future__ import annotations

import numpy as np
from scipy import signal

from ..utils.trace_data import compute_frequency_distribution, sample_cdf_points


def compute_peak_amplitudes(
    cleaned: np.ndarray,
    *,
    peak_separation: float,
    peak_prominence: float,
    peak_width: float,
) -> np.ndarray:
    amplitudes = np.zeros(cleaned.shape[0], dtype=np.float32)
    for trace_id, row in enumerate(cleaned):
        peaks, _ = signal.find_peaks(
            row,
            distance=peak_separation,
            prominence=peak_prominence,
            width=(1.0, peak_width),
            rel_height=0.95,
        )
        if peaks.size:
            amplitudes[trace_id] = float(np.max(row[peaks]))
    return amplitudes


def compute_cdf_threshold_values(
    cleaned: np.ndarray,
    *,
    cdf_bin: int,
) -> np.ndarray:
    spectrum = compute_frequency_distribution(cleaned)
    return sample_cdf_points(
        spectrum,
        thresholds=np.asarray([cdf_bin], dtype=np.int64),
    )[:, 0]


def compute_second_derivative(row: np.ndarray) -> np.ndarray:
    trace = np.asarray(row, dtype=np.float32)
    if trace.size < 3:
        return np.empty(0, dtype=np.float32)
    return (
        trace[:-2] + trace[2:] - (2.0 * trace[1:-1])
    ).astype(np.float32, copy=False)


def compute_first_derivative(row: np.ndarray) -> np.ndarray:
    trace = np.asarray(row, dtype=np.float32)
    if trace.size < 2:
        return np.empty(0, dtype=np.float32)
    return np.diff(trace).astype(np.float32, copy=False)


def compute_second_derivative_batch(cleaned: np.ndarray) -> np.ndarray:
    trace_matrix = np.asarray(cleaned, dtype=np.float32)
    if trace_matrix.ndim != 2:
        raise ValueError(f"expected a 2D trace matrix, got shape {trace_matrix.shape}")
    if trace_matrix.shape[1] < 3:
        return np.empty((trace_matrix.shape[0], 0), dtype=np.float32)
    return (
        trace_matrix[:, :-2] + trace_matrix[:, 2:] - (2.0 * trace_matrix[:, 1:-1])
    ).astype(np.float32, copy=False)


def pad_second_derivative(second_diff: np.ndarray, trace_length: int) -> np.ndarray:
    if trace_length <= 0:
        return np.empty(0, dtype=np.float32)
    if second_diff.size == 0:
        return np.zeros(trace_length, dtype=np.float32)
    return np.pad(
        np.asarray(second_diff, dtype=np.float32),
        (1, 1),
        mode="constant",
        constant_values=0.0,
    )


def pad_first_derivative(first_diff: np.ndarray, trace_length: int) -> np.ndarray:
    if trace_length <= 0:
        return np.empty(0, dtype=np.float32)
    if first_diff.size == 0:
        return np.zeros(trace_length, dtype=np.float32)
    return np.pad(
        np.asarray(first_diff, dtype=np.float32),
        (1, 0),
        mode="constant",
        constant_values=0.0,
    )
