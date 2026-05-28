from __future__ import annotations

from functools import lru_cache

import h5py
import numpy as np
from numba import njit

from ..model.trace import TraceRecord
from attpc_storage import describe_trace_events, open_trace_reader
from .ion_chamber import (
    ion_chamber_event_count,
    load_ion_chamber_event,
    load_ion_chamber_trace as _load_ic_trace,
)

PAD_TRACE_OFFSET = 5
CDF_THRESHOLDS = np.arange(1, 151, dtype=np.int64)
CDF_VALUE_BINS = 100
DETECTOR_ATTPC = "ATTPC"
DETECTOR_IC = "IC"
DETECTOR_SI = "SI"
DETECTOR_GAGG = "GAGG"

SI_SIDE_ORDER = (
    "upstream_front",
    "upstream_back",
    "downstream_front",
    "downstream_back",
)
SI_SIDE_TO_CODE = {
    "upstream_front": 0,
    "upstream_back": 1,
    "downstream_front": 2,
    "downstream_back": 3,
}
SI_CODE_TO_SIDE = {value: key for key, value in SI_SIDE_TO_CODE.items()}
GAGG_LAYER_1_COUNT = 25
GAGG_LAYER_2_COUNT = 16
GAGG_TRACE_COUNT = GAGG_LAYER_1_COUNT + GAGG_LAYER_2_COUNT


def load_trace_record(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    trace_id: int,
    baseline_window_scale: float,
    detector: str = DETECTOR_ATTPC,
) -> TraceRecord:
    if normalize_detector(detector) == DETECTOR_IC:
        return load_ion_chamber_trace_record(
            file_handle,
            run=run,
            event_id=event_id,
            trace_id=trace_id,
            baseline_window_scale=baseline_window_scale,
        )
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


def load_ion_chamber_trace_record(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    trace_id: int,
    baseline_window_scale: float,
) -> TraceRecord:
    if int(trace_id) != 0:
        raise LookupError(f"IC trace {run}/{event_id}/{trace_id} is not available")
    event = load_ion_chamber_event(file_handle, run=run, event_id=event_id)
    return trace_record_from_ion_chamber_trace(
        run=run,
        event_id=event_id,
        trace_id=0,
        raw=event.raw_trace,
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
        detector=DETECTOR_ATTPC,
        hardware_id=hardware,
        raw=raw,
        trace=trace,
        transformed=transformed,
        family=None,
        label=None,
    )


def trace_record_from_ion_chamber_trace(
    *,
    run: int,
    event_id: int,
    trace_id: int,
    raw: np.ndarray,
    baseline_window_scale: float,
) -> TraceRecord:
    raw_array = np.asarray(raw, dtype=np.float32)
    trace = preprocess_traces(
        raw_array[np.newaxis, :],
        baseline_window_scale=baseline_window_scale,
    )[0]
    transformed = compute_frequency_distribution(trace[np.newaxis, :])[0]
    return TraceRecord(
        run=run,
        event_id=event_id,
        trace_id=trace_id,
        detector=DETECTOR_IC,
        hardware_id=np.asarray([1903, 0, 0, 0, 0], dtype=np.float32),
        raw=raw_array,
        trace=trace,
        transformed=transformed,
        family=None,
        label=None,
    )


def trace_record_from_silicon_row(
    *,
    run: int,
    event_id: int,
    trace_id: int,
    row: np.ndarray,
    baseline_window_scale: float,
) -> TraceRecord:
    row_array = np.asarray(row, dtype=np.float32)
    if row_array.ndim != 1 or int(row_array.shape[0]) != 514:
        raise LookupError(
            f"SI trace {run}/{event_id}/{trace_id} has invalid shape {row_array.shape}"
        )
    side_code = int(row_array[0])
    if side_code not in SI_CODE_TO_SIDE:
        raise LookupError(
            f"SI trace {run}/{event_id}/{trace_id} has invalid side code {side_code}"
        )
    strip_id = int(row_array[1])
    raw = row_array[2:]
    trace = preprocess_traces(
        raw[np.newaxis, :],
        baseline_window_scale=baseline_window_scale,
    )[0]
    transformed = compute_frequency_distribution(trace[np.newaxis, :])[0]
    return TraceRecord(
        run=run,
        event_id=event_id,
        trace_id=trace_id,
        detector=DETECTOR_SI,
        hardware_id=np.asarray([side_code, strip_id, trace_id, 0, 0], dtype=np.float32),
        raw=raw,
        trace=trace,
        transformed=transformed,
        family=None,
        label=None,
    )


def trace_record_from_gagg_row(
    *,
    run: int,
    event_id: int,
    trace_id: int,
    row: np.ndarray,
    baseline_window_scale: float,
) -> TraceRecord:
    row_array = np.asarray(row, dtype=np.float32)
    if row_array.ndim != 1 or int(row_array.shape[0]) != 256:
        raise LookupError(
            f"GAGG trace {run}/{event_id}/{trace_id} has invalid shape {row_array.shape}"
        )
    layer, index = gagg_trace_selector(trace_id)
    trace = preprocess_traces(
        row_array[np.newaxis, :],
        baseline_window_scale=baseline_window_scale,
    )[0]
    transformed = compute_frequency_distribution(trace[np.newaxis, :])[0]
    return TraceRecord(
        run=run,
        event_id=event_id,
        trace_id=trace_id,
        detector=DETECTOR_GAGG,
        hardware_id=np.asarray([layer, index, trace_id, 0, 0], dtype=np.float32),
        raw=row_array,
        trace=trace,
        transformed=transformed,
        family=None,
        label=None,
    )


def normalize_detector(detector: str | None) -> str:
    token = (detector or DETECTOR_ATTPC).strip().upper()
    if token in {"ATTPC", "PAD", "PADS"}:
        return DETECTOR_ATTPC
    if token == DETECTOR_IC:
        return DETECTOR_IC
    if token in {"SI", "SILICON"}:
        return DETECTOR_SI
    if token == DETECTOR_GAGG:
        return DETECTOR_GAGG
    raise ValueError("detector must be 'ATTPC', 'IC', 'SI', or 'GAGG'")


def open_storage_trace_reader(
    *,
    run: int,
    path: str,
    detector: str = DETECTOR_ATTPC,
):
    resolved_detector = normalize_detector(detector)
    return open_trace_reader(
        workspace="",
        run=run,
        path=path,
        read_pad=resolved_detector == DETECTOR_ATTPC,
        read_si=resolved_detector == DETECTOR_SI,
        read_gagg=resolved_detector == DETECTOR_GAGG,
        read_ic=resolved_detector == DETECTOR_IC,
    )


def load_reader_trace_record(
    reader,
    *,
    run: int,
    event_id: int,
    trace_id: int,
    baseline_window_scale: float,
    detector: str = DETECTOR_ATTPC,
) -> TraceRecord:
    resolved_detector = normalize_detector(detector)
    try:
        event = reader.read_event(int(event_id))
    except (LookupError, ValueError, KeyError) as exc:
        trace_label = "IC trace" if resolved_detector == DETECTOR_IC else "trace"
        raise LookupError(f"{trace_label} {run}/{event_id} is not available") from exc
    payload = _reader_event_payload(event, detector=resolved_detector)
    if resolved_detector == DETECTOR_IC:
        if int(trace_id) != 0:
            raise LookupError(f"IC trace {run}/{event_id}/{trace_id} is not available")
        return trace_record_from_ion_chamber_trace(
            run=run,
            event_id=event_id,
            trace_id=0,
            raw=payload,
            baseline_window_scale=baseline_window_scale,
        )
    rows = np.asarray(payload, dtype=np.float32)
    normalized_trace_id = int(trace_id)
    if normalized_trace_id < 0 or normalized_trace_id >= int(rows.shape[0]):
        raise LookupError(f"trace {run}/{event_id}/{normalized_trace_id} is not available")
    if resolved_detector == DETECTOR_SI:
        row = rows[normalized_trace_id]
        return trace_record_from_silicon_row(
            run=run,
            event_id=event_id,
            trace_id=normalized_trace_id,
            row=row,
            baseline_window_scale=baseline_window_scale,
        )
    if resolved_detector == DETECTOR_GAGG:
        if rows.ndim != 2 or int(rows.shape[0]) != GAGG_TRACE_COUNT or int(rows.shape[1]) != 256:
            raise LookupError(
                f"GAGG event {run}/{event_id} has invalid matrix shape {rows.shape}"
            )
        return trace_record_from_gagg_row(
            run=run,
            event_id=event_id,
            trace_id=normalized_trace_id,
            row=rows[normalized_trace_id],
            baseline_window_scale=baseline_window_scale,
        )
    return trace_record_from_pad_row(
        run=run,
        event_id=event_id,
        trace_id=normalized_trace_id,
        row=rows[normalized_trace_id],
        baseline_window_scale=baseline_window_scale,
    )


def collect_reader_event_counts(reader) -> list[tuple[int, int]]:
    event_counts: list[tuple[int, int]] = []
    for event_id in range(int(reader.min_event), int(reader.max_event) + 1):
        if event_id in getattr(reader, "bad_events", set()):
            continue
        trace_count = reader_event_trace_count(reader, event_id)
        if trace_count > 0:
            event_counts.append((event_id, trace_count))
    return event_counts


def reader_event_trace_count(
    reader,
    event_id: int,
    *,
    detector: str = DETECTOR_ATTPC,
) -> int:
    resolved_detector = normalize_detector(detector)
    if (
        int(event_id) < int(reader.min_event)
        or int(event_id) > int(reader.max_event)
        or int(event_id) in getattr(reader, "bad_events", set())
    ):
        return 0
    try:
        event = reader.read_event(int(event_id))
    except (LookupError, ValueError, KeyError):
        return 0
    payload = _reader_event_payload(event, detector=resolved_detector, allow_missing=True)
    if payload is None:
        return 0
    if resolved_detector == DETECTOR_IC:
        return 1
    rows = np.asarray(payload, dtype=np.float32)
    if rows.ndim != 2:
        return 0
    if resolved_detector == DETECTOR_GAGG and (
        int(rows.shape[0]) != GAGG_TRACE_COUNT or int(rows.shape[1]) != 256
    ):
        return 0
    return int(rows.shape[0])


def reader_pad_rows(reader, event_id: int) -> np.ndarray:
    try:
        event = reader.read_event(int(event_id))
    except (LookupError, ValueError, KeyError) as exc:
        raise LookupError(f"trace event {event_id} is not available") from exc
    payload = _reader_event_payload(event, detector=DETECTOR_ATTPC)
    return np.asarray(payload, dtype=np.float32)


def reader_event_rows(
    reader,
    event_id: int,
    *,
    detector: str,
) -> np.ndarray:
    resolved_detector = normalize_detector(detector)
    if resolved_detector == DETECTOR_IC:
        raise LookupError("IC does not expose stacked event rows")
    try:
        event = reader.read_event(int(event_id))
    except (LookupError, ValueError, KeyError) as exc:
        raise LookupError(f"trace event {event_id} is not available") from exc
    payload = _reader_event_payload(event, detector=resolved_detector)
    return np.asarray(payload, dtype=np.float32)


def _reader_event_payload(
    event: object,
    *,
    detector: str,
    allow_missing: bool = False,
):
    if not isinstance(event, dict):
        raise LookupError("reader returned invalid event payload")
    key = {
        DETECTOR_ATTPC: "pads",
        DETECTOR_IC: "ic",
        DETECTOR_SI: "si",
        DETECTOR_GAGG: "gagg",
    }[detector]
    value = event.get(key)
    if not isinstance(value, tuple) or len(value) != 2:
        if allow_missing:
            return None
        raise LookupError(f"reader event payload is missing {key}")
    return value[1]


def si_local_index(rows: np.ndarray, trace_id: int) -> int:
    rows_array = np.asarray(rows, dtype=np.float32)
    normalized_trace_id = int(trace_id)
    if rows_array.ndim != 2 or rows_array.shape[1] != 514:
        raise LookupError(f"SI event has invalid matrix shape {rows_array.shape}")
    if normalized_trace_id < 0 or normalized_trace_id >= int(rows_array.shape[0]):
        raise LookupError(f"SI trace {normalized_trace_id} is not available")
    side_code = int(rows_array[normalized_trace_id, 0])
    return int(np.count_nonzero(rows_array[:normalized_trace_id, 0] == side_code))


def si_trace_selector(rows: np.ndarray, trace_id: int) -> dict[str, object]:
    rows_array = np.asarray(rows, dtype=np.float32)
    normalized_trace_id = int(trace_id)
    if rows_array.ndim != 2 or rows_array.shape[1] != 514:
        raise LookupError(f"SI event has invalid matrix shape {rows_array.shape}")
    if normalized_trace_id < 0 or normalized_trace_id >= int(rows_array.shape[0]):
        raise LookupError(f"SI trace {normalized_trace_id} is not available")
    side_code = int(rows_array[normalized_trace_id, 0])
    if side_code not in SI_CODE_TO_SIDE:
        raise LookupError(f"SI trace {normalized_trace_id} has invalid side code {side_code}")
    return {
        "kind": "si",
        "side": SI_CODE_TO_SIDE[side_code],
        "index": si_local_index(rows_array, normalized_trace_id),
    }


def si_side_counts(rows: np.ndarray) -> dict[str, int]:
    rows_array = np.asarray(rows, dtype=np.float32)
    if rows_array.ndim != 2 or rows_array.shape[1] != 514:
        raise LookupError(f"SI event has invalid matrix shape {rows_array.shape}")
    return {
        side: int(np.count_nonzero(rows_array[:, 0] == code))
        for side, code in SI_SIDE_TO_CODE.items()
    }


def si_trace_id_from_selector(rows: np.ndarray, *, side: str, index: int) -> int:
    rows_array = np.asarray(rows, dtype=np.float32)
    if rows_array.ndim != 2 or rows_array.shape[1] != 514:
        raise LookupError(f"SI event has invalid matrix shape {rows_array.shape}")
    normalized_side = str(side).strip().lower()
    if normalized_side not in SI_SIDE_TO_CODE:
        raise LookupError(f"SI side {side!r} is not available")
    normalized_index = int(index)
    if normalized_index < 0:
        raise LookupError(f"SI index {normalized_index} is not available")
    side_code = SI_SIDE_TO_CODE[normalized_side]
    side_positions = np.flatnonzero(rows_array[:, 0] == side_code)
    if normalized_index >= int(side_positions.shape[0]):
        raise LookupError(f"SI index {normalized_index} is not available")
    return int(side_positions[normalized_index])


def gagg_trace_selector(trace_id: int) -> tuple[int, int]:
    normalized_trace_id = int(trace_id)
    if normalized_trace_id < 0 or normalized_trace_id >= GAGG_TRACE_COUNT:
        raise LookupError(f"GAGG trace {normalized_trace_id} is not available")
    if normalized_trace_id < GAGG_LAYER_1_COUNT:
        return 1, normalized_trace_id
    return 2, normalized_trace_id - GAGG_LAYER_1_COUNT


def gagg_event_selector(rows: np.ndarray, trace_id: int) -> dict[str, object]:
    rows_array = np.asarray(rows, dtype=np.float32)
    if rows_array.ndim != 2 or rows_array.shape != (GAGG_TRACE_COUNT, 256):
        raise LookupError(f"GAGG event has invalid matrix shape {rows_array.shape}")
    layer, index = gagg_trace_selector(trace_id)
    return {"kind": "gagg", "layer": layer, "index": index}


def gagg_layer_counts(rows: np.ndarray) -> dict[str, int]:
    rows_array = np.asarray(rows, dtype=np.float32)
    if rows_array.ndim != 2 or rows_array.shape != (GAGG_TRACE_COUNT, 256):
        raise LookupError(f"GAGG event has invalid matrix shape {rows_array.shape}")
    return {"layer1": GAGG_LAYER_1_COUNT, "layer2": GAGG_LAYER_2_COUNT}


def gagg_trace_id_from_selector(*, layer: int, index: int) -> int:
    normalized_layer = int(layer)
    normalized_index = int(index)
    if normalized_layer == 1:
        if normalized_index < 0 or normalized_index >= GAGG_LAYER_1_COUNT:
            raise LookupError(f"GAGG layer 1 index {normalized_index} is not available")
        return normalized_index
    if normalized_layer == 2:
        if normalized_index < 0 or normalized_index >= GAGG_LAYER_2_COUNT:
            raise LookupError(f"GAGG layer 2 index {normalized_index} is not available")
        return GAGG_LAYER_1_COUNT + normalized_index
    raise LookupError(f"GAGG layer {normalized_layer} is not available")


def _event_pad_dataset(file_handle: h5py.File, event_id: int) -> h5py.Dataset | None:
    if "events" in file_handle:
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

    if "meta" in file_handle and "get" in file_handle:
        get_group = file_handle["get"]
        dataset_name = f"evt{event_id}_data"
        if dataset_name not in get_group:
            return None
        return get_group[dataset_name]

    raise LookupError("unsupported ATTPC trace layout")


def collect_event_counts(file_handle: h5py.File) -> list[tuple[int, int]]:
    metadata = describe_trace_events(file_handle)
    event_counts: list[tuple[int, int]] = []
    for event_id in range(metadata.min_event, metadata.max_event + 1):
        if event_id in metadata.bad_events:
            continue
        pads = _event_pad_dataset(file_handle, event_id)
        if pads is None:
            continue
        trace_count = int(pads.shape[0])
        if trace_count > 0:
            event_counts.append((event_id, trace_count))
    return event_counts


def load_pad_rows(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    trace_ids: np.ndarray | None = None,
) -> np.ndarray:
    metadata = describe_trace_events(file_handle)
    if (
        event_id < metadata.min_event
        or event_id > metadata.max_event
        or event_id in metadata.bad_events
    ):
        raise LookupError(f"trace {run}/{event_id} is not available")

    try:
        pads = _event_pad_dataset(file_handle, event_id)
    except LookupError as exc:
        raise LookupError(f"trace {run}/{event_id} is not available") from exc
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
        if indices.size == 0:
            rows = pads[indices]
        else:
            order = np.argsort(indices, kind="stable")
            sorted_indices = indices[order]
            unique_indices, inverse = np.unique(sorted_indices, return_inverse=True)
            sorted_rows = np.asarray(pads[unique_indices], dtype=np.float32)[inverse]
            rows = np.empty_like(sorted_rows)
            rows[order] = sorted_rows
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


def load_detector_traces(
    file_handle: h5py.File,
    *,
    run: int,
    event_id: int,
    detector: str = DETECTOR_ATTPC,
    trace_ids: np.ndarray | None = None,
) -> np.ndarray:
    if normalize_detector(detector) == DETECTOR_IC:
        try:
            return _load_ic_trace(
                file_handle,
                run=run,
                event_id=event_id,
                trace_ids=trace_ids,
            )
        except LookupError as exc:
            raise LookupError(f"IC trace {run}/{event_id} is not available") from exc
    return load_pad_traces(file_handle, run=run, event_id=event_id, trace_ids=trace_ids)


def event_trace_count(file_handle: h5py.File, event_id: int, *, detector: str = DETECTOR_ATTPC) -> int:
    if normalize_detector(detector) == DETECTOR_IC:
        return ion_chamber_event_count(file_handle, event_id)
    metadata = describe_trace_events(file_handle)
    if (
        event_id < metadata.min_event
        or event_id > metadata.max_event
        or event_id in metadata.bad_events
    ):
        return 0
    try:
        pads = _event_pad_dataset(file_handle, event_id)
    except LookupError:
        return 0
    if pads is None:
        return 0
    return int(pads.shape[0])


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
