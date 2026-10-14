from __future__ import annotations


import numpy as np

from attpc_estimator.process.bitflip import (
    analyze_bitflip_trace,
    accumulate_bitflip_histograms,
)
from attpc_estimator.process.saturation import accumulate_saturation_histograms
from attpc_estimator.process.filter_core import BitFlipFilterCore, SaturationFilterCore


def _trace_from_second_derivative(second_diff: list[float]) -> np.ndarray:
    values = np.asarray(second_diff, dtype=np.float32)
    trace = np.zeros(values.size + 2, dtype=np.float32)
    for index, value in enumerate(values):
        trace[index + 2] = value + (2.0 * trace[index + 1]) - trace[index]
    return trace


def _qualified_bitflip_trace() -> np.ndarray:
    return _trace_from_second_derivative([0.0, 61.0, -121.0, 450.0, -512.0, 0.0, 0.0, 0.0])


def _multi_segment_qualified_bitflip_trace() -> np.ndarray:
    return _trace_from_second_derivative(
        [0.0, 61.0, -121.0, 0.0, 0.0, -512.0, 574.0, 0.0]
    )


def _off_band_bitflip_trace() -> np.ndarray:
    return _trace_from_second_derivative([0.0, 200.0, -205.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _invalid_bitflip_trace() -> np.ndarray:
    return _trace_from_second_derivative([0.0, 61.0, 121.0, -450.0, 0.0])


def _flat_trace(sample_count: int = 8) -> np.ndarray:
    return np.zeros(sample_count, dtype=np.float32)


def _saturation_trace() -> np.ndarray:
    row = np.zeros(64, dtype=np.float32)
    row[24:28] = 2105.0
    row[28] = 2098.0
    return row


def _single_peak_trace() -> np.ndarray:
    row = np.zeros(64, dtype=np.float32)
    row[24] = 2105.0
    row[25] = 2090.0
    return row


def _high_drop_trace() -> np.ndarray:
    row = np.zeros(64, dtype=np.float32)
    row[24:28] = 2105.0
    return row


def test_analyze_bitflip_trace_collects_all_values_and_segment_length() -> None:
    analysis = analyze_bitflip_trace(_qualified_bitflip_trace(), baseline_threshold=1.0)

    assert analysis.qualified_segment_lengths.tolist() == [4.0]
    assert [(item.start_baseline_index, item.end_baseline_index) for item in analysis.structures] == [
        (0, 5)
    ]
    assert len(analysis.segment_value_sets) == 1
    np.testing.assert_allclose(
        analysis.segment_value_sets[0],
        np.asarray([61.0, 121.0, 450.0, 512.0], dtype=np.float32),
    )


def test_analyze_bitflip_trace_collects_multiple_non_overlapping_segments() -> None:
    analysis = analyze_bitflip_trace(
        _multi_segment_qualified_bitflip_trace(),
        baseline_threshold=1.0,
    )

    assert analysis.qualified_segment_lengths.tolist() == [2.0, 2.0]
    assert [
        (item.start_baseline_index, item.end_baseline_index)
        for item in analysis.structures
    ] == [(0, 3), (4, 7)]
    assert len(analysis.segment_value_sets) == 2
    np.testing.assert_allclose(
        analysis.segment_value_sets[0],
        np.asarray([61.0, 121.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        analysis.segment_value_sets[1],
        np.asarray([512.0, 574.0], dtype=np.float32),
    )


def test_analyze_bitflip_trace_rejects_non_alternating_runs() -> None:
    analysis = analyze_bitflip_trace(_invalid_bitflip_trace(), baseline_threshold=1.0)

    assert analysis.qualified_segment_lengths.size == 0
    assert analysis.structures == ()
    assert analysis.segment_value_sets == ()


def test_analyze_bitflip_trace_keeps_off_band_values_but_not_qualified_lengths() -> None:
    analysis = analyze_bitflip_trace(_off_band_bitflip_trace(), baseline_threshold=1.0)

    assert len(analysis.segment_value_sets) == 1
    np.testing.assert_allclose(
        analysis.segment_value_sets[0],
        np.asarray([200.0, 205.0], dtype=np.float32),
    )
    assert analysis.qualified_segment_lengths.size == 0
    assert analysis.structures == ()


def test_accumulate_bitflip_histograms_only_counts_qualified_lengths() -> None:
    cleaned = np.asarray(
        [_qualified_bitflip_trace(), _off_band_bitflip_trace()],
        dtype=np.float32,
    )
    baseline_histogram = np.zeros(8193, dtype=np.int64)
    value_histogram = np.zeros(8192, dtype=np.int64)
    length_histogram = np.zeros(256, dtype=np.int64)
    count_histogram = np.zeros(256, dtype=np.int64)

    accumulate_bitflip_histograms(
        cleaned,
        baseline_histogram=baseline_histogram,
        value_histogram=value_histogram,
        length_histogram=length_histogram,
        count_histogram=count_histogram,
        baseline_threshold=1.0,
    )

    assert baseline_histogram[4096] >= 1
    assert value_histogram[61] == 1
    assert value_histogram[121] == 1
    assert value_histogram[200] == 1
    assert value_histogram[205] == 1
    assert length_histogram[4] == 1
    assert length_histogram[2] == 0
    assert count_histogram[1] == 1
    assert count_histogram[0] == 0


def test_bitflip_filter_core_requires_qualified_segments() -> None:
    core = BitFlipFilterCore(baseline_threshold=1.0, min_segment_count=1)
    cleaned = np.asarray(
        [_qualified_bitflip_trace(), _off_band_bitflip_trace()],
        dtype=np.float32,
    )

    prepared = core.prepare_batch(cleaned)

    assert prepared.qualified_segment_counts.tolist() == [1, 0]
    assert core.matches(trace_id=0, row=cleaned[0], prepared=prepared) is True
    assert core.matches(trace_id=1, row=cleaned[1], prepared=prepared) is False


def test_saturation_filter_core_matches_when_plateau_length_is_met() -> None:
    core = SaturationFilterCore(
        drop_threshold=10.0,
        min_plateau_length=4,
        threshold=2000.0,
    )
    cleaned = np.asarray([_saturation_trace(), _flat_trace(64)], dtype=np.float32)

    prepared = core.prepare_batch(cleaned)

    assert prepared.plateau_lengths.tolist() == [5, 0]
    assert core.matches(trace_id=0, row=cleaned[0], prepared=prepared) is True
    assert core.matches(trace_id=1, row=cleaned[1], prepared=prepared) is False


def test_saturation_filter_core_requires_plateau_length_at_least_two() -> None:
    try:
        SaturationFilterCore(
            drop_threshold=10.0,
            min_plateau_length=1,
            threshold=2000.0,
        )
    except ValueError as exc:
        assert str(exc) == "saturation plateau length must be at least 2"
    else:
        raise AssertionError("expected ValueError for min_plateau_length < 2")


def test_accumulate_saturation_histograms_ignores_single_point_peaks_and_large_drops() -> None:
    cleaned = np.asarray([_single_peak_trace(), _high_drop_trace()], dtype=np.float32)
    drop_histogram = np.zeros(500, dtype=np.int64)
    length_histogram = np.zeros(256, dtype=np.int64)

    accumulate_saturation_histograms(
        cleaned,
        drop_histogram=drop_histogram,
        length_histogram=length_histogram,
        threshold=2000.0,
        drop_threshold=10.0,
        window_radius=16,
    )

    assert length_histogram[1] == 0
    assert length_histogram[2:].sum() == 1
    assert drop_histogram[499] == 0
