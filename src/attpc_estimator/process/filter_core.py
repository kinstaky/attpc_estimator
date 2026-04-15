from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .bitflip import (
    BITFLIP_BASELINE_DEFAULT,
    BITFLIP_FILTER_MIN_COUNT_DEFAULT,
    count_qualified_bitflip_segments_batch,
)
from .trace_metrics import compute_cdf_threshold_values, compute_peak_amplitudes

OSCILLATION_CDF_BIN = 60
OSCILLATION_CUTOFF = 0.6
SATURATION_THRESHOLD_DEFAULT = 2000.0
SATURATION_WINDOW_RADIUS_DEFAULT = 16


class FilterCore(Protocol):
    def prepare_batch(self, cleaned: np.ndarray) -> object | None:
        ...

    def matches(
        self,
        *,
        trace_id: int,
        row: np.ndarray,
        prepared: object | None,
    ) -> bool:
        ...

    def output_token(self) -> str | None:
        ...


@dataclass(frozen=True, slots=True)
class SaturationBatchStats:
    drop_values_by_trace: tuple[np.ndarray, ...]
    plateau_lengths: np.ndarray


@dataclass(frozen=True, slots=True)
class BitFlipBatchStats:
    qualified_segment_counts: np.ndarray


@dataclass(frozen=True, slots=True)
class AmplitudeFilterCore:
    min_amplitude: float
    max_amplitude: float
    peak_separation: float = 50.0
    peak_prominence: float = 20.0
    peak_width: float = 50.0

    def __post_init__(self) -> None:
        if self.min_amplitude > self.max_amplitude:
            raise ValueError("amplitude minimum must be less than or equal to the maximum")

    def prepare_batch(self, cleaned: np.ndarray) -> np.ndarray:
        return compute_peak_amplitudes(
            cleaned,
            peak_separation=self.peak_separation,
            peak_prominence=self.peak_prominence,
            peak_width=self.peak_width,
        )

    def matches(
        self,
        *,
        trace_id: int,
        row: np.ndarray,
        prepared: object | None,
    ) -> bool:
        _ = row
        if prepared is None:
            return False
        amplitudes = np.asarray(prepared)
        amplitude = float(amplitudes[trace_id])
        return self.min_amplitude <= amplitude <= self.max_amplitude

    def output_token(self) -> str:
        return f"amp_{_format_bound(self.min_amplitude)}_{_format_bound(self.max_amplitude)}"


@dataclass(frozen=True, slots=True)
class CdfFilterCore:
    cdf_bin: int = OSCILLATION_CDF_BIN
    cutoff: float = OSCILLATION_CUTOFF

    def prepare_batch(self, cleaned: np.ndarray) -> np.ndarray:
        return compute_cdf_threshold_values(cleaned, cdf_bin=self.cdf_bin)

    def matches(
        self,
        *,
        trace_id: int,
        row: np.ndarray,
        prepared: object | None,
    ) -> bool:
        _ = row
        if prepared is None:
            return False
        cdf_values = np.asarray(prepared)
        return float(cdf_values[trace_id]) < self.cutoff

    def output_token(self) -> str:
        return "cdf"


OscillationFilterCore = CdfFilterCore


@dataclass(frozen=True, slots=True)
class BitFlipFilterCore:
    baseline_threshold: float = BITFLIP_BASELINE_DEFAULT
    min_segment_count: int = BITFLIP_FILTER_MIN_COUNT_DEFAULT

    def __post_init__(self) -> None:
        if self.baseline_threshold < 0:
            raise ValueError("bitflip baseline threshold must be non-negative")
        if self.min_segment_count <= 0:
            raise ValueError("bitflip minimum segment count must be positive")

    def prepare_batch(self, cleaned: np.ndarray) -> BitFlipBatchStats:
        return BitFlipBatchStats(
            qualified_segment_counts=count_qualified_bitflip_segments_batch(
                cleaned,
                baseline_threshold=self.baseline_threshold,
            )
        )

    def matches(
        self,
        *,
        trace_id: int,
        row: np.ndarray,
        prepared: object | None,
    ) -> bool:
        _ = row
        if not isinstance(prepared, BitFlipBatchStats):
            return False
        return int(prepared.qualified_segment_counts[trace_id]) >= self.min_segment_count

    def output_token(self) -> str:
        return (
            f"bitflip_base_{_format_bound(self.baseline_threshold)}"
            f"_count_{self.min_segment_count}"
        )


@dataclass(frozen=True, slots=True)
class SaturationFilterCore:
    drop_threshold: float
    min_plateau_length: int
    threshold: float = SATURATION_THRESHOLD_DEFAULT

    def __post_init__(self) -> None:
        if self.drop_threshold < 0:
            raise ValueError("saturation drop threshold must be non-negative")
        if self.min_plateau_length < 2:
            raise ValueError("saturation plateau length must be at least 2")
        if self.threshold < 0:
            raise ValueError("saturation threshold must be non-negative")

    def prepare_batch(self, cleaned: np.ndarray) -> SaturationBatchStats:
        return analyze_saturation_batch(
            cleaned,
            threshold=self.threshold,
            drop_threshold=self.drop_threshold,
            window_radius=SATURATION_WINDOW_RADIUS_DEFAULT,
        )

    def matches(
        self,
        *,
        trace_id: int,
        row: np.ndarray,
        prepared: object | None,
    ) -> bool:
        _ = row
        if not isinstance(prepared, SaturationBatchStats):
            return False
        return int(prepared.plateau_lengths[trace_id]) >= self.min_plateau_length

    def output_token(self) -> str:
        return (
            f"saturation_thr_{_format_bound(self.threshold)}"
            f"_drop_{_format_bound(self.drop_threshold)}"
            f"_len_{self.min_plateau_length}"
        )


def analyze_saturation_batch(
    cleaned: np.ndarray,
    *,
    threshold: float,
    drop_threshold: float,
    window_radius: int,
) -> SaturationBatchStats:
    drop_values_by_trace: list[np.ndarray] = []
    plateau_lengths = np.zeros(cleaned.shape[0], dtype=np.int64)

    for trace_id, row in enumerate(cleaned):
        drop_values, plateau_length = analyze_saturation_trace(
            row,
            threshold=threshold,
            drop_threshold=drop_threshold,
            window_radius=window_radius,
        )
        drop_values_by_trace.append(drop_values)
        plateau_lengths[trace_id] = plateau_length

    return SaturationBatchStats(
        drop_values_by_trace=tuple(drop_values_by_trace),
        plateau_lengths=plateau_lengths,
    )


def analyze_saturation_trace(
    row: np.ndarray,
    *,
    threshold: float,
    drop_threshold: float,
    window_radius: int,
) -> tuple[np.ndarray, int]:
    trace = np.asarray(row, dtype=np.float32)
    if trace.size == 0:
        return np.empty(0, dtype=np.float32), 0

    max_index = int(np.argmax(trace))
    max_value = float(trace[max_index])
    if max_value < threshold:
        return np.empty(0, dtype=np.float32), 0

    start = max(0, max_index - int(window_radius))
    stop = min(trace.shape[0], max_index + int(window_radius) + 1)
    drops = (max_value - trace[start:stop]).astype(np.float32, copy=False)

    left = max_index
    while left > 0 and (max_value - float(trace[left - 1])) <= drop_threshold:
        left -= 1
    right = max_index
    while right + 1 < trace.shape[0] and (max_value - float(trace[right + 1])) <= drop_threshold:
        right += 1
    return drops, (right - left + 1)


def _format_bound(value: float) -> str:
    token = f"{value:g}"
    return token.replace("-", "neg").replace(".", "p")
