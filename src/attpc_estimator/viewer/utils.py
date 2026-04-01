from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from numba import njit

from ..db import TraceLabelRepository
from ..models import TraceRecord

PAD_TRACE_OFFSET = 5
CDF_THRESHOLDS = np.arange(1, 151, dtype=np.int64)
CDF_VALUE_BINS = 100
DEFAULT_DB_FILENAME = "trace_label.db"


@dataclass(frozen=True, slots=True)
class LabeledTraceRow:
    run: int
    event_id: int
    trace_id: int
    label_key: str


def canonical_label_key(family: str, label: str) -> str:
    return f"{family}:{label}"


def sanitize_label_key(label_key: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in label_key)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def label_title_from_key(label_key: str) -> str:
    family, _, label = label_key.partition(":")
    if family != "normal":
        return label
    if label == "0":
        return "0 peak"
    if label == "4+":
        return "4+ peaks"
    if label == "9":
        return "9+ peaks"
    if label == "1":
        return "1 peak"
    return f"{label} peaks"


def trace_file(trace_path: Path, run: int | str) -> Path:
    if trace_path.is_file():
        return trace_path.resolve()
    run_files = collect_run_files(trace_path)
    normalized_run = int(run)
    resolved = run_files.get(normalized_run)
    if resolved is not None:
        return resolved
    return trace_path / f"run_{run}.h5"


def collect_run_files(trace_path: Path) -> dict[int, Path]:
    if trace_path.is_file():
        resolved = trace_path.resolve()
        return {extract_run_id(resolved): resolved}
    run_files: dict[int, Path] = {}
    for candidate in sorted(trace_path.glob("run_*.h5")):
        run_id = extract_run_id(candidate)
        if run_id in run_files:
            raise ValueError(
                f"multiple workspace files resolve to run {run_id}: {run_files[run_id]} and {candidate}"
            )
        run_files[run_id] = candidate.resolve()
    return run_files


def extract_run_id(trace_file_path: Path) -> int:
    stem = trace_file_path.stem
    prefix, _, run_token = stem.partition("_")
    if prefix != "run" or not run_token.isdigit():
        raise ValueError(
            f"expected input filename like run_<run>.h5, got {trace_file_path.name}"
        )
    return int(run_token)


def db_file(workspace: Path) -> Path:
    preferred = workspace / DEFAULT_DB_FILENAME
    return preferred


def _read_labeled_trace_rows(
    trace_path: Path,
    workspace_path: Path,
    run: int | None,
) -> tuple[np.ndarray, list[LabeledTraceRow]]:
    trace_root = Path(trace_path).expanduser().resolve()
    workspace_root = Path(workspace_path).expanduser().resolve()
    run_files = collect_run_files(trace_root)
    if run is not None and run not in run_files:
        raise ValueError(
            f"trace file not found for run {run}: {trace_root / f'run_{run}.h5'}"
        )

    repository = TraceLabelRepository(db_file(workspace_root))
    repository.initialize()
    try:
        labeled_rows = repository.list_labeled_traces(run=run)
    finally:
        repository.connection.close()

    rows_by_run: dict[int, list[LabeledTraceRow]] = {}
    row_count = 0
    for run_id, event_id, trace_id, family, label in labeled_rows:
        if run_id not in run_files:
            continue
        rows_by_run.setdefault(run_id, []).append(
            LabeledTraceRow(
                run=run_id,
                event_id=event_id,
                trace_id=trace_id,
                label_key=canonical_label_key(family, label),
            )
        )
        row_count += 1

    trace_matrix: np.ndarray | None = None
    ordered_rows: list[LabeledTraceRow] = []
    row_index = 0

    for run_id in sorted(rows_by_run):
        run_rows = rows_by_run[run_id]
        with h5py.File(run_files[run_id], "r") as handle:
            events = handle["events"]
            for row in run_rows:
                pads = events[f"event_{row.event_id}"]["get"]["pads"]
                raw_trace = np.asarray(
                    pads[row.trace_id, PAD_TRACE_OFFSET:], dtype=np.float32
                )
                if trace_matrix is None:
                    trace_matrix = np.empty(
                        (row_count, raw_trace.shape[0]), dtype=np.float32
                    )
                trace_matrix[row_index] = raw_trace
                ordered_rows.append(row)
                row_index += 1

    if trace_matrix is None:
        return np.empty((0, 0), dtype=np.float32), []
    return trace_matrix[:row_index], ordered_rows


def read_labeled_trace(
    trace_path: Path,
    workspace_path: Path,
    run: int | None,
) -> tuple[np.ndarray, list[str]]:
    traces, labeled_rows = _read_labeled_trace_rows(
        trace_path=trace_path, workspace_path=workspace_path, run=run
    )
    return traces, [row.label_key for row in labeled_rows]


def collect_event_counts(
    events: h5py.Group,
    min_event: int,
    max_event: int,
    bad_events: set[int],
) -> list[tuple[int, int]]:
    event_counts: list[tuple[int, int]] = []
    for event_id in range(min_event, max_event + 1):
        if event_id in bad_events:
            continue
        pads = events[f"event_{event_id}"]["get"]["pads"]
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
    events = file_handle["events"]
    min_event = int(events.attrs["min_event"])
    max_event = int(events.attrs["max_event"])
    bad_events = {int(value) for value in events.attrs["bad_events"]}
    if event_id < min_event or event_id > max_event or event_id in bad_events:
        raise LookupError(f"trace {run}/{event_id}/{trace_id} is not available")

    pads = events[f"event_{event_id}"]["get"]["pads"]
    if trace_id < 0 or trace_id >= int(pads.shape[0]):
        raise LookupError(f"trace {run}/{event_id}/{trace_id} is not available")

    row = pads[trace_id]
    hardware = np.asarray(row[:PAD_TRACE_OFFSET], dtype=np.float32)
    raw = np.asarray(row[PAD_TRACE_OFFSET:], dtype=np.float32)
    trace = preprocess_traces(
        raw[np.newaxis, :],
        baseline_window_scale=baseline_window_scale,
    )[0]
    transformed = compute_frequency_distribution(trace[np.newaxis, :])[0]
    return TraceRecord(
        run=str(run),
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
