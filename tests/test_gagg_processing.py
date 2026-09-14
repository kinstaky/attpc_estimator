from __future__ import annotations

import pytest

from attpc_estimator.utils.gagg_processing import (
    GaggPeakFindingConfig,
    GaggTraceProcessingRule,
    build_gagg_trace_rules,
    require_gagg_peak_config,
)


def test_gagg_rules_allow_amplitude_only_indexes_and_ignore_time_entries() -> None:
    rules = build_gagg_trace_rules(
        baseline_entries=[{"index": [0], "fft_window_scale": 8.0}],
        amplitude_entries=[{
            "index": [0, 1],
            "peak_separation": 10.0,
            "peak_prominence": 5.0,
            "peak_width": 20.0,
            "peak_threshold": 30.0,
            "rel_height": 0.9,
        }],
        time_entries=[{"index": [1], "min": 10, "max": 20}],
    )

    assert rules == {
        0: GaggTraceProcessingRule(
            8.0, GaggPeakFindingConfig(10.0, 5.0, 20.0, 30.0, 0.9)
        )
    }


def test_gagg_rules_reject_duplicate_indexes() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_gagg_trace_rules(
            baseline_entries=[
                {"index": [0], "fft_window_scale": 8.0},
                {"index": [0], "fft_window_scale": 16.0},
            ],
            amplitude_entries=[],
        )


def test_gagg_rules_apply_peak_overrides_without_eagerly_requiring_every_index() -> None:
    rules = build_gagg_trace_rules(
        baseline_entries=[{"index": [0, 1], "fft_window_scale": 8.0}],
        amplitude_entries=[{
            "index": [0],
            "peak_separation": 10.0,
            "peak_prominence": 5.0,
            "peak_width": 20.0,
            "peak_threshold": 30.0,
            "rel_height": 0.9,
        }],
        peak_overrides={"peak_threshold": 40.0},
    )

    assert require_gagg_peak_config(rules[0], index=0).peak_threshold == 40.0
    with pytest.raises(ValueError, match="index 1"):
        require_gagg_peak_config(rules[1], index=1)
