from __future__ import annotations

from pathlib import Path

import numpy as np

from ..storage.labeled_traces import iter_labeled_trace_batches
from .bitflip import (
    BITFLIP_BASELINE_DEFAULT,
    BITFLIP_FILTER_MIN_COUNT_DEFAULT,
    count_qualified_bitflip_segments_batch,
)
from .filter_core import (
    SATURATION_THRESHOLD_DEFAULT,
    SATURATION_WINDOW_RADIUS_DEFAULT,
    analyze_saturation_batch,
)
from .progress import ProgressReporter, emit_progress
from .trace_metrics import compute_peak_amplitudes
from ..utils.trace_data import preprocess_traces

ZERO_PEAK_AMPLITUDE_CUTOFF = 40.0
SATURATION_DROP_THRESHOLD_DEFAULT = 10.0
SATURATION_RELABEL_MIN_PLATEAU_LENGTH_DEFAULT = 2
RELABEL_LABEL_CHOICES = ("noise", "oscillation", "saturation")
NOISE_RELABEL = "noise"
OSCILLATION_RELABEL = "oscillation"
SATURATION_RELABEL = "saturation"
OSCILLATION_LABEL_KEY = "strange:oscillation"
SATURATION_LABEL_KEY = "strange:saturation"
ZERO_PEAK_LABEL_KEY = "normal:0"
ONE_PEAK_LABEL_KEY = "normal:1"


def build_relabel_rows(
    trace_path: Path,
    workspace: Path,
    label: str,
    run: int | None = None,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
    bitflip_baseline_threshold: float = BITFLIP_BASELINE_DEFAULT,
    bitflip_min_count: int = BITFLIP_FILTER_MIN_COUNT_DEFAULT,
    saturation_threshold: float = SATURATION_THRESHOLD_DEFAULT,
    saturation_drop_threshold: float = SATURATION_DROP_THRESHOLD_DEFAULT,
    saturation_window_radius: int = SATURATION_WINDOW_RADIUS_DEFAULT,
    saturation_min_plateau_length: int = SATURATION_RELABEL_MIN_PLATEAU_LENGTH_DEFAULT,
    progress: ProgressReporter | None = None,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    validate_relabel_label(label)
    batches, total_traces = iter_labeled_trace_batches(
        trace_path=trace_path,
        workspace_path=workspace,
        run=run,
    )
    if total_traces == 0:
        emit_progress(
            progress,
            current=0,
            total=0,
            unit="trace",
        )
        return _build_structured_rows([], []), _build_ratio_metrics(label, [], [])

    processed_traces = 0
    ordered_rows = []
    new_labels: list[str] = []
    old_labels: list[str] = []
    emit_progress(
        progress,
        current=0,
        total=total_traces,
        unit="trace",
    )

    for event_rows, traces in batches:
        cleaned = preprocess_traces(
            traces,
            baseline_window_scale=baseline_window_scale,
        )
        batch_old_labels = [row.label_key for row in event_rows]
        if label == NOISE_RELABEL:
            amplitudes = compute_peak_amplitudes(
                cleaned,
                peak_separation=peak_separation,
                peak_prominence=peak_prominence,
                peak_width=peak_width,
            )
            batch_new_labels = [
                _relabel_noise(old_label=old_label, amplitude=float(amplitude))
                for old_label, amplitude in zip(
                    batch_old_labels, amplitudes, strict=True
                )
            ]
        elif label == OSCILLATION_RELABEL:
            segment_counts = count_qualified_bitflip_segments_batch(
                cleaned,
                baseline_threshold=bitflip_baseline_threshold,
            )
            batch_new_labels = [
                _relabel_oscillation(
                    old_label=old_label,
                    segment_count=int(segment_count),
                    min_count=bitflip_min_count,
                )
                for old_label, segment_count in zip(
                    batch_old_labels, segment_counts, strict=True
                )
            ]
        elif label == SATURATION_RELABEL:
            saturation_batch = analyze_saturation_batch(
                cleaned,
                threshold=saturation_threshold,
                drop_threshold=saturation_drop_threshold,
                window_radius=saturation_window_radius,
            )
            batch_new_labels = [
                _relabel_saturation(
                    old_label=old_label,
                    plateau_length=int(plateau_length),
                    min_plateau_length=saturation_min_plateau_length,
                )
                for old_label, plateau_length in zip(
                    batch_old_labels,
                    saturation_batch.plateau_lengths,
                    strict=True,
                )
            ]
        else:
            raise ValueError(f"unsupported relabel label: {label}")

        ordered_rows.extend(event_rows)
        old_labels.extend(batch_old_labels)
        new_labels.extend(batch_new_labels)
        processed_traces += len(event_rows)
        emit_progress(
            progress,
            current=processed_traces,
            total=total_traces,
            unit="trace",
            message=(
                f"run={event_rows[0].run},event={event_rows[0].event_id}"
                if event_rows
                else ""
            ),
        )

    return _build_structured_rows(ordered_rows, new_labels), _build_ratio_metrics(
        label,
        old_labels,
        new_labels,
    )


def validate_relabel_label(label: str) -> None:
    if label not in RELABEL_LABEL_CHOICES:
        raise ValueError(f"unsupported relabel label: {label}")


def ratio_items_for_label(
    label: str,
    metrics: dict[str, tuple[int, int]],
) -> list[tuple[str, tuple[int, int]]]:
    if label == NOISE_RELABEL:
        return [
            ("old normal:0 -> new normal:0", metrics["normal0_to_normal0"]),
            ("old normal:1 -> new normal:0", metrics["normal1_to_normal0"]),
        ]
    if label == OSCILLATION_RELABEL:
        return [
            (
                "old strange:oscillation -> new strange:oscillation",
                metrics["oscillation_to_oscillation"],
            ),
            (
                "old normal:1 -> new strange:oscillation",
                metrics["normal1_to_oscillation"],
            ),
        ]
    if label == SATURATION_RELABEL:
        return [
            (
                "old strange:saturation -> new strange:saturation",
                metrics["saturation_to_saturation"],
            ),
            (
                "old normal:1 -> new strange:saturation",
                metrics["normal1_to_saturation"],
            ),
        ]
    raise ValueError(f"unsupported relabel label: {label}")


def print_ratio(name: str, ratio: tuple[int, int]) -> None:
    numerator, denominator = ratio
    if denominator == 0:
        print(f"{name}: 0/0 = nan")
        return
    print(f"{name}: {numerator}/{denominator} = {numerator / denominator:.6f}")


def _relabel_noise(old_label: str, amplitude: float) -> str:
    if amplitude < ZERO_PEAK_AMPLITUDE_CUTOFF:
        return ZERO_PEAK_LABEL_KEY
    return old_label


def _relabel_oscillation(old_label: str, segment_count: int, min_count: int) -> str:
    if segment_count >= min_count:
        return OSCILLATION_LABEL_KEY
    return old_label


def _relabel_saturation(
    old_label: str,
    plateau_length: int,
    min_plateau_length: int,
) -> str:
    if plateau_length >= min_plateau_length:
        return SATURATION_LABEL_KEY
    return old_label


def _build_structured_rows(
    labeled_rows: list,
    new_labels: list[str],
) -> np.ndarray:
    max_label_length = max(
        [len(row.label_key) for row in labeled_rows]
        + [len(label) for label in new_labels]
        + [1]
    )
    dtype = np.dtype(
        [
            ("run", np.int64),
            ("event_id", np.int64),
            ("trace_id", np.int64),
            ("old_label", f"U{max_label_length}"),
            ("new_label", f"U{max_label_length}"),
        ]
    )
    rows = np.empty(len(labeled_rows), dtype=dtype)
    for index, (row, new_label) in enumerate(
        zip(labeled_rows, new_labels, strict=True)
    ):
        rows[index] = (row.run, row.event_id, row.trace_id, row.label_key, new_label)
    return rows


def _build_ratio_metrics(
    label: str,
    old_labels: list[str],
    new_labels: list[str],
) -> dict[str, tuple[int, int]]:
    metrics: dict[str, tuple[int, int]] = {}
    if label == NOISE_RELABEL:
        metrics["normal0_to_normal0"] = _label_transition_ratio(
            old_labels, new_labels, ZERO_PEAK_LABEL_KEY, ZERO_PEAK_LABEL_KEY
        )
        metrics["normal1_to_normal0"] = _label_transition_ratio(
            old_labels, new_labels, ONE_PEAK_LABEL_KEY, ZERO_PEAK_LABEL_KEY
        )
        return metrics

    if label == SATURATION_RELABEL:
        metrics["saturation_to_saturation"] = _label_transition_ratio(
            old_labels,
            new_labels,
            SATURATION_LABEL_KEY,
            SATURATION_LABEL_KEY,
        )
        metrics["normal1_to_saturation"] = _label_transition_ratio(
            old_labels,
            new_labels,
            ONE_PEAK_LABEL_KEY,
            SATURATION_LABEL_KEY,
        )
        return metrics

    metrics["oscillation_to_oscillation"] = _label_transition_ratio(
        old_labels,
        new_labels,
        OSCILLATION_LABEL_KEY,
        OSCILLATION_LABEL_KEY,
    )
    metrics["normal1_to_oscillation"] = _label_transition_ratio(
        old_labels,
        new_labels,
        ONE_PEAK_LABEL_KEY,
        OSCILLATION_LABEL_KEY,
    )
    return metrics


def _label_transition_ratio(
    old_labels: list[str],
    new_labels: list[str],
    source_label: str,
    target_label: str,
) -> tuple[int, int]:
    numerator = 0
    denominator = 0
    for old_label, new_label in zip(old_labels, new_labels, strict=True):
        if old_label != source_label:
            continue
        denominator += 1
        if new_label == target_label:
            numerator += 1
    return numerator, denominator
