from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .trace_data import GAGG_TRACE_COUNT


@dataclass(frozen=True, slots=True)
class GaggTraceProcessingRule:
    baseline_window_scale: float
    peak_config: GaggPeakFindingConfig | None = None


@dataclass(frozen=True, slots=True)
class GaggPeakFindingConfig:
    peak_separation: float
    peak_prominence: float
    peak_width: float
    peak_threshold: float
    rel_height: float

    def as_dict(self) -> dict[str, float]:
        return {
            "peak_separation": self.peak_separation,
            "peak_prominence": self.peak_prominence,
            "peak_width": self.peak_width,
            "peak_threshold": self.peak_threshold,
            "peak_rel_height": self.rel_height,
        }


def build_gagg_trace_rules(
    *,
    baseline_entries: list[dict[str, Any]],
    amplitude_entries: list[dict[str, Any]] | None = None,
    time_entries: list[dict[str, Any]] | None = None,
    baseline_override: float | None = None,
    peak_overrides: dict[str, float] | None = None,
) -> dict[int, GaggTraceProcessingRule]:
    baseline_by_index = _indexed_values(
        entries=baseline_entries,
        value_keys=("fft_window_scale",),
        section="gagg.baseline",
    )
    amplitude_by_index = _indexed_values(
        entries=amplitude_entries or [],
        value_keys=("peak_separation", "peak_prominence", "peak_width", "peak_threshold", "rel_height"),
        section="gagg.amplitude",
    )

    rules: dict[int, GaggTraceProcessingRule] = {}
    for index in sorted(baseline_by_index):
        scale = float(baseline_by_index[index]["fft_window_scale"])
        if baseline_override is not None:
            scale = float(baseline_override)
        if scale <= 0:
            raise ValueError(f"gagg.baseline index {index} must have a positive fft_window_scale")
        peak_values = amplitude_by_index.get(index)
        peak_config = (
            _peak_config(peak_values, section=f"gagg.amplitude index {index}")
            if peak_values is not None
            else None
        )
        if peak_config is not None and peak_overrides:
            peak_config = replace(peak_config, **peak_overrides)
            _validate_peak_config(peak_config, section=f"gagg.amplitude index {index}")
        rules[index] = GaggTraceProcessingRule(scale, peak_config)
    return rules


def default_gagg_trace_rules(*, baseline_window_scale: float) -> dict[int, GaggTraceProcessingRule]:
    peak_config = GaggPeakFindingConfig(50.0, 20.0, 50.0, 0.0, 0.95)
    return {
        index: GaggTraceProcessingRule(float(baseline_window_scale), peak_config)
        for index in range(GAGG_TRACE_COUNT)
    }


def require_gagg_peak_config(rule: GaggTraceProcessingRule, *, index: int) -> GaggPeakFindingConfig:
    if rule.peak_config is None:
        raise ValueError(f"gagg.amplitude has no settings for baseline-configured index {index}")
    return rule.peak_config


def _peak_config(values: dict[str, Any], *, section: str) -> GaggPeakFindingConfig:
    config = GaggPeakFindingConfig(
        peak_separation=float(values["peak_separation"]),
        peak_prominence=float(values["peak_prominence"]),
        peak_width=float(values["peak_width"]),
        peak_threshold=float(values["peak_threshold"]),
        rel_height=float(values["rel_height"]),
    )
    _validate_peak_config(config, section=section)
    return config


def _validate_peak_config(config: GaggPeakFindingConfig, *, section: str) -> None:
    if config.peak_separation <= 0 or config.peak_width <= 0:
        raise ValueError(f"{section} peak_separation and peak_width must be positive")


def _indexed_values(
    *, entries: list[dict[str, Any]], value_keys: tuple[str, ...], section: str
) -> dict[int, dict[str, Any]]:
    values: dict[int, dict[str, Any]] = {}
    for entry in entries:
        indexes = entry.get("index")
        if indexes is None:
            indexes = list(range(GAGG_TRACE_COUNT))
        if not isinstance(indexes, list) or not indexes:
            raise ValueError(f"{section}.index must be a non-empty list")
        if any(key not in entry for key in value_keys):
            raise ValueError(f"{section} entries must include {', '.join(value_keys)}")
        for raw_index in indexes:
            if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                raise ValueError(f"{section}.index entries must be integers")
            if raw_index < 0 or raw_index >= GAGG_TRACE_COUNT:
                raise ValueError(f"{section}.index {raw_index} is outside 0..{GAGG_TRACE_COUNT - 1}")
            if raw_index in values:
                raise ValueError(f"{section}.index {raw_index} is configured more than once")
            values[raw_index] = {key: entry[key] for key in value_keys}
    return values
