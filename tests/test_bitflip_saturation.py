from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

import attpc_estimator.cli.bitflip as bitflip_cli
from attpc_estimator.cli.baseline import main as baseline_main
from attpc_estimator.cli.saturation import main as saturation_main
from attpc_estimator.process.bitflip import (
    analyze_bitflip_trace,
    accumulate_bitflip_histograms,
)
from attpc_estimator.process.saturation import accumulate_saturation_histograms
from attpc_estimator.process.filter_core import BitFlipFilterCore, SaturationFilterCore
from attpc_estimator.service.estimator import EstimatorService


def _pad_rows(traces: list[np.ndarray]) -> np.ndarray:
    rows = []
    for trace_id, trace in enumerate(traces):
        hardware = np.asarray(
            [10 + trace_id, 20 + trace_id, 30 + trace_id, 40 + trace_id, 50 + trace_id],
            dtype=np.float32,
        )
        rows.append(np.concatenate([hardware, trace]).astype(np.float32))
    return np.asarray(rows, dtype=np.float32)


def write_run_file(path: Path, events: dict[int, list[np.ndarray]]) -> None:
    with h5py.File(path, "w") as handle:
        event_ids = sorted(events)
        group = handle.create_group("events")
        group.attrs["min_event"] = min(event_ids)
        group.attrs["max_event"] = max(event_ids)
        group.attrs["bad_events"] = np.asarray([], dtype=np.int64)
        for event_id in event_ids:
            event_group = group.create_group(f"event_{event_id}")
            get_group = event_group.create_group("get")
            get_group.create_dataset("pads", data=_pad_rows(events[event_id]))


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


def test_baseline_main_writes_expected_artifact(tmp_path, monkeypatch) -> None:
    trace_root = tmp_path / "traces"
    workspace = tmp_path / "workspace"
    trace_root.mkdir()
    workspace.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {1: [_qualified_bitflip_trace(), _flat_trace(_qualified_bitflip_trace().shape[0])]},
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline",
            "-t",
            str(trace_root),
            "-w",
            str(workspace),
            "-r",
            "8",
        ],
    )
    baseline_main()

    payload = np.load(workspace / "run_0008_baseline.npz")
    assert int(payload["trace_count"]) == 2
    assert payload["histogram"].sum() == 20
    assert payload["bin_centers"][0] < 0
    assert payload["bin_centers"][-1] > 0


def test_bitflip_main_writes_expected_artifact(tmp_path, monkeypatch) -> None:
    trace_root = tmp_path / "traces"
    workspace = tmp_path / "workspace"
    trace_root.mkdir()
    workspace.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {1: [_qualified_bitflip_trace()]},
    )
    monkeypatch.setattr(
        bitflip_cli,
        "build_bitflip_histograms",
        lambda **_: {
            "trace_count": np.int64(3),
            "baseline_histogram": np.arange(8, dtype=np.int64) + 30,
            "value_histogram": np.arange(8, dtype=np.int64),
            "length_histogram": np.arange(8, dtype=np.int64) + 10,
            "count_histogram": np.arange(8, dtype=np.int64) + 20,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bitflip",
            "-t",
            str(trace_root),
            "-w",
            str(workspace),
            "-r",
            "8",
                "--baseline",
                "1",
            ],
        )
    bitflip_cli.main()

    payload = np.load(workspace / "run_0008_bitflip.npz")
    assert int(payload["trace_count"]) == 3
    assert payload["baseline_histogram"].sum() == 268
    assert payload["value_histogram"].sum() == 28
    assert payload["length_histogram"].sum() == 108
    assert payload["count_histogram"].sum() == 188


def test_saturation_main_writes_expected_artifact(tmp_path, monkeypatch) -> None:
    trace_root = tmp_path / "traces"
    workspace = tmp_path / "workspace"
    trace_root.mkdir()
    workspace.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {1: [_saturation_trace(), _flat_trace(64)]},
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saturation",
            "-t",
            str(trace_root),
            "-w",
            str(workspace),
            "-r",
            "8",
            "--threshold",
            "2000",
            "--drop-threshold",
            "10",
            "--window-radius",
            "4",
        ],
    )
    saturation_main()

    payload = np.load(workspace / "run_0008_saturation.npz")
    assert int(payload["trace_count"]) == 2
    assert payload["length_histogram"].sum() == 1
    assert payload["drop_histogram"].sum() > 0
    assert payload["drop_histogram"].shape == (500,)


def test_histogram_service_loads_baseline_bitflip_and_saturation_variants(tmp_path) -> None:
    trace_root = tmp_path / "traces"
    workspace = tmp_path / "workspace"
    trace_root.mkdir()
    workspace.mkdir()
    (trace_root / "run_0008.h5").touch()
    np.save(workspace / "filter_placeholder.npy", np.asarray([[8, 1, 0]], dtype=np.int64))
    np.savez(
        workspace / "run_0008_baseline.npz",
        trace_count=np.int64(4),
        histogram=np.arange(8, dtype=np.int64),
        bin_centers=np.arange(-4, 4, dtype=np.int64),
    )
    np.savez(
        workspace / "run_0008_labeled_baseline.npz",
        run_id=np.int64(8),
        label_keys=np.asarray(["normal:0", "strange:Noise"], dtype=np.str_),
        label_titles=np.asarray(["0 peak", "Noise"], dtype=np.str_),
        trace_counts=np.asarray([1, 2], dtype=np.int64),
        histograms=np.asarray(
            [np.arange(8, dtype=np.int64), np.arange(8, dtype=np.int64) + 10]
        ),
        bin_centers=np.arange(-4, 4, dtype=np.int64),
    )
    np.savez(
        workspace / "run_0008_bitflip.npz",
        trace_count=np.int64(4),
        baseline_histogram=np.arange(8, dtype=np.int64) + 60,
        value_histogram=np.arange(8, dtype=np.int64),
        length_histogram=np.arange(8, dtype=np.int64) + 10,
        count_histogram=np.arange(8, dtype=np.int64) + 20,
    )
    np.savez(
        workspace / "run_0008_labeled_bitflip.npz",
        run_id=np.int64(8),
        label_keys=np.asarray(["normal:0", "strange:Noise"], dtype=np.str_),
        label_titles=np.asarray(["0 peak", "Noise"], dtype=np.str_),
        trace_counts=np.asarray([1, 2], dtype=np.int64),
        baseline_histograms=np.asarray(
            [np.arange(8, dtype=np.int64) + 80, np.arange(8, dtype=np.int64) + 90]
        ),
        value_histograms=np.asarray(
            [np.arange(8, dtype=np.int64), np.arange(8, dtype=np.int64) + 10]
        ),
        length_histograms=np.asarray(
            [np.arange(8, dtype=np.int64) + 20, np.arange(8, dtype=np.int64) + 30]
        ),
        count_histograms=np.asarray(
            [np.arange(8, dtype=np.int64) + 40, np.arange(8, dtype=np.int64) + 50]
        ),
    )
    np.savez(
        workspace / "run_0008_saturation.npz",
        trace_count=np.int64(5),
        drop_histogram=np.arange(500, dtype=np.int64),
        length_histogram=np.arange(6, dtype=np.int64) + 5,
    )
    np.savez(
        workspace / "run_0008_labeled_saturation.npz",
        run_id=np.int64(8),
        label_keys=np.asarray(["normal:0", "strange:Noise"], dtype=np.str_),
        label_titles=np.asarray(["0 peak", "Noise"], dtype=np.str_),
        trace_counts=np.asarray([2, 3], dtype=np.int64),
        drop_histograms=np.asarray(
            [np.arange(500, dtype=np.int64), np.arange(500, dtype=np.int64) + 10]
        ),
        length_histograms=np.asarray(
            [np.arange(6, dtype=np.int64) + 20, np.arange(6, dtype=np.int64) + 30]
        ),
    )

    service = EstimatorService(trace_path=trace_root, workspace=workspace)
    try:
        bootstrap = service.bootstrap_state()
        assert bootstrap["histogramAvailability"]["8"]["baseline"]["all"] is True
        assert bootstrap["histogramAvailability"]["8"]["baseline"]["filtered"] is True
        assert bootstrap["histogramAvailability"]["8"]["bitflip"]["all"] is True
        assert bootstrap["histogramAvailability"]["8"]["bitflip"]["labeled"] is True
        assert bootstrap["histogramAvailability"]["8"]["saturation"]["all"] is True
        assert bootstrap["histogramAvailability"]["8"]["saturation"]["filtered"] is True

        baseline_payload = service.get_histogram(
            metric="baseline",
            mode="all",
            run=8,
        )
        assert baseline_payload["binLabel"] == "Baseline value"
        assert baseline_payload["binCenters"][0] == -4096
        assert baseline_payload["series"][0]["traceCount"] == 4

        bitflip_payload = service.get_histogram(
            metric="bitflip",
            variant="baseline",
            mode="all",
            run=8,
        )
        assert bitflip_payload["variant"] == "baseline"
        assert bitflip_payload["binLabel"] == "Second-derivative value"
        assert bitflip_payload["binCenters"][0] == -4096
        assert bitflip_payload["series"][0]["histogram"][0] == 60

        bitflip_value_payload = service.get_histogram(
            metric="bitflip",
            variant="value",
            mode="all",
            run=8,
        )
        assert bitflip_value_payload["variant"] == "value"
        assert bitflip_value_payload["binLabel"] == "Absolute second-derivative value"
        assert bitflip_value_payload["series"][0]["traceCount"] == 4
        assert bitflip_value_payload["series"][0]["histogram"][0] == 0

        bitflip_count_payload = service.get_histogram(
            metric="bitflip",
            variant="count",
            mode="all",
            run=8,
        )
        assert bitflip_count_payload["variant"] == "count"
        assert bitflip_count_payload["binLabel"] == "Found bitflip structures"
        assert bitflip_count_payload["series"][0]["histogram"][0] == 20
    finally:
        service.close()
