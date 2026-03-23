from functools import lru_cache
import numpy as np
from numba import njit

PAD_TRACE_OFFSET = 5
CDF_THRESHOLDS = np.arange(1, 151, dtype=np.int64)
CDF_VALUE_BINS = 100

@njit(cache=True)
def _replace_baseline_peaks(trace_matrix: np.ndarray) -> np.ndarray:
    bases = trace_matrix.copy()
    row_count, sample_count = bases.shape

    for row_index in range(row_count):
        row = bases[row_index]

        mean = 0.0
        for sample_index in range(sample_count):
            mean += float(row[sample_index])
        mean /= sample_count

        variance = 0.0
        for sample_index in range(sample_count):
            diff = float(row[sample_index]) - mean
            variance += diff * diff
        sigma = np.sqrt(variance / sample_count)
        cutoff = sigma * 1.5

        valid_sum = 0.0
        valid_count = 0
        for sample_index in range(sample_count):
            if abs(float(row[sample_index]) - mean) <= cutoff:
                valid_sum += float(row[sample_index])
                valid_count += 1

        replacement = mean if valid_count == 0 else valid_sum / valid_count
        for sample_index in range(sample_count):
            if abs(float(row[sample_index]) - mean) > cutoff:
                row[sample_index] = replacement

    return bases

@lru_cache(maxsize=None)
def _get_baseline_filter(sample_count: int, baseline_window_scale: float) -> np.ndarray:
    window = np.arange(sample_count, dtype=np.float32) - (sample_count // 2)
    full_filter = np.fft.ifftshift(
        np.sinc(window / baseline_window_scale)).astype(np.float32, copy=False
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
        sample_count=sample_count,
        baseline_window_scale=baseline_window_scale
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

@njit(cache=True)
def _sample_cdf_points_numba(spectrum: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
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


def sample_cdf_points(spectrum: np.ndarray, thresholds: np.ndarray = CDF_THRESHOLDS) -> np.ndarray:
    spectrum_array = np.asarray(spectrum, dtype=np.float32)
    if spectrum_array.ndim != 2:
        raise ValueError(f"expected a 2D spectrum matrix, got shape {spectrum_array.shape}")
    thresholds_array = np.asarray(thresholds, dtype=np.int64)
    return _sample_cdf_points_numba(spectrum_array, thresholds_array)
