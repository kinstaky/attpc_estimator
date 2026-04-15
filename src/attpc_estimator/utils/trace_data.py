from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import h5py
import numpy as np
from numba import njit

from ..model.trace import TraceRecord

PAD_TRACE_OFFSET = 5
CDF_THRESHOLDS = np.arange(1, 151, dtype=np.int64)
CDF_VALUE_BINS = 100


class TraceLayout(Enum):
    LEGACY_MERGER = "legacy_merger"
    MERGER_V2 = "merger_v2"


@dataclass(frozen=True, slots=True)
class TraceEventMetadata:
    layout: TraceLayout
    min_event: int
    max_event: int
    bad_events: frozenset[int]

    @property
    def event_span(self) -> int:
        return max(0, self.max_event - self.min_event + 1)

    @property
    def valid_event_span(self) -> int:
        return max(0, self.event_span - len(self.bad_events))


def detect_trace_layout(file_handle: h5py.File) -> TraceLayout:
    if "meta" in file_handle and "get" in file_handle:
        return TraceLayout.LEGACY_MERGER
    if "events" in file_handle:
        version_token = _decode_attr_value(file_handle["events"].attrs.get("version"))
        if version_token is not None and version_token != "libattpc_merger:2.0":
            raise ValueError(
                f"unsupported events trace version {version_token!r}; expected "
                "'libattpc_merger:2.0' or a compatible events layout"
            )
        return TraceLayout.MERGER_V2

    root_keys = sorted(str(key) for key in file_handle.keys())
    raise ValueError(
        "unsupported trace file layout; expected top-level 'events' or both "
        f"'meta' and 'get', found keys {root_keys}"
    )


def describe_trace_events(file_handle: h5py.File) -> TraceEventMetadata:
    layout = detect_trace_layout(file_handle)
    min_event, max_event, bad_events = _event_bounds(file_handle)
    return TraceEventMetadata(
        layout=layout,
        min_event=min_event,
        max_event=max_event,
        bad_events=frozenset(bad_events),
    )


def collect_event_counts(file_handle: h5py.File) -> list[tuple[int, int]]:
    min_event, max_event, bad_events = _event_bounds(file_handle)
    event_counts: list[tuple[int, int]] = []
    for event_id in range(min_event, max_event + 1):
        if event_id in bad_events:
            continue
        pads = _event_pad_dataset(file_handle, event_id)
        if pads is None:
            continue
        trace_count = int(pads.shape[0])
        if trace_count > 0:
            event_counts.append((event_id, trace_count))
    return event_counts


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
    min_event, max_event, bad_events = _event_bounds(file_handle)
    if event_id < min_event or event_id > max_event or event_id in bad_events:
        raise LookupError(f"trace {run}/{event_id} is not available")

    pads = _event_pad_dataset(file_handle, event_id)
    if pads is None:
        raise LookupError(f"trace {run}/{event_id} is not available")

    if trace_ids is None:
        rows = pads[:]
    else:
        indices = np.asarray(trace_ids, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"trace_ids must be 1D, got shape {indices.shape}")
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= int(pads.shape[0])
        ):
            raise LookupError(f"trace {run}/{event_id} is not available")
        rows = pads[indices]
    return np.asarray(rows, dtype=np.float32)


def load_pad_traces(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    trace_ids: np.ndarray | None = None,
) -> np.ndarray:
    rows = load_pad_rows(
        file_handle,
        run=run,
        event_id=event_id,
        trace_ids=trace_ids,
    )
    return np.asarray(rows[:, PAD_TRACE_OFFSET:], dtype=np.float32)


def event_trace_count(file_handle: h5py.File, event_id: int) -> int:
    min_event, max_event, bad_events = _event_bounds(file_handle)
    if event_id < min_event or event_id > max_event or event_id in bad_events:
        return 0
    pads = _event_pad_dataset(file_handle, event_id)
    if pads is None:
        return 0
    return int(pads.shape[0])


def _event_bounds(file_handle: h5py.File) -> tuple[int, int, set[int]]:
    layout = detect_trace_layout(file_handle)
    if layout is TraceLayout.LEGACY_MERGER:
        meta_group = file_handle["meta"]
        if "meta" not in meta_group:
            raise ValueError("legacy trace file is missing meta/meta")
        meta_data = np.asarray(meta_group["meta"])
        if meta_data.size < 3:
            raise ValueError(
                f"legacy trace metadata is incomplete: expected at least 3 values, got {meta_data.size}"
            )
        return int(meta_data[0]), int(meta_data[2]), set()

    events = file_handle["events"]
    min_event = int(events.attrs["min_event"])
    max_event = int(events.attrs["max_event"])
    bad_events = {
        int(value) for value in np.asarray(events.attrs.get("bad_events", ()))
    }
    return min_event, max_event, bad_events


def _event_pad_dataset(file_handle: h5py.File, event_id: int) -> h5py.Dataset | None:
    layout = detect_trace_layout(file_handle)
    if layout is TraceLayout.LEGACY_MERGER:
        get_group = file_handle["get"]
        dataset_name = f"evt{event_id}_data"
        if dataset_name not in get_group:
            return None
        return get_group[dataset_name]

    events = file_handle["events"]
    event_name = f"event_{event_id}"
    if event_name not in events:
        return None
    event_group = events[event_name]
    if "get" not in event_group:
        return None
    get_group = event_group["get"]
    if "pads" not in get_group:
        return None
    return get_group["pads"]


def _decode_attr_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


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


@njit(cache=True)
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
