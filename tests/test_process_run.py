from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import attpc_estimator.utils.trace_data as trace_data
from attpc_estimator.cli.cdf import main
from attpc_estimator.process.cdf import (
    CDF_THRESHOLDS,
    CDF_VALUE_BINS,
    build_trace_cdf_histogram,
)
from attpc_estimator.utils.trace_data import (
    TraceLayout,
    collect_event_counts,
    detect_trace_layout,
    load_trace_record,
    preprocess_traces,
    sample_cdf_points,
)
from tests.hdf5_fixtures import write_events_hdf5, write_legacy_hdf5


def write_hdf5_input(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["min_event"] = 1
        events.attrs["max_event"] = 2
        events.attrs["bad_events"] = np.array([], dtype=np.int64)

        event_1 = events.create_group("event_1")
        get_1 = event_1.create_group("get")
        get_1.create_dataset(
            "pads",
            data=np.array(
                [
                    [10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8],
                    [20, 21, 22, 23, 24, 8, 7, 6, 5, 4, 3, 2, 1],
                ],
                dtype=np.float32,
            ),
        )

        event_2 = events.create_group("event_2")
        get_2 = event_2.create_group("get")
        get_2.create_dataset(
            "pads",
            data=np.array(
                [
                    [30, 31, 32, 33, 34, 0, 1, 0, 1, 0, 1, 0, 1],
                ],
                dtype=np.float32,
            ),
        )


def preprocess_traces_reference(traces: np.ndarray, baseline_window_scale: float) -> np.ndarray:
    traces_array = np.asarray(traces, dtype=np.float32)
    if traces_array.ndim != 2:
        raise ValueError(f"expected a 2D trace matrix, got shape {traces_array.shape}")

    trace_matrix = np.array(traces_array, copy=True)
    sample_count = trace_matrix.shape[1]

    if sample_count < 2:
        return trace_matrix

    trace_matrix[:, 0] = trace_matrix[:, 1]
    trace_matrix[:, -1] = trace_matrix[:, -2]

    bases = trace_matrix.copy()
    for row in bases:
        mean = np.mean(row)
        sigma = np.std(row)
        mask = np.abs(row - mean) > sigma * 1.5
        if np.any(~mask):
            row[mask] = np.mean(row[~mask])
        else:
            row.fill(mean)

    window = np.arange(sample_count, dtype=np.float32) - (sample_count // 2)
    baseline_filter = np.fft.ifftshift(np.sinc(window / baseline_window_scale)).astype(np.float32, copy=False)
    transformed = np.fft.fft(bases, axis=1)
    filtered = np.real(np.fft.ifft(transformed * baseline_filter[np.newaxis, :], axis=1)).astype(np.float32, copy=False)
    return trace_matrix - filtered


def test_sample_cdf_points_uses_under_frequency_convention() -> None:
    spectrum = np.array([[0.0, 1.0, 3.0, 6.0]], dtype=np.float32)
    thresholds = np.array([0, 1, 2, 3, 4, 10], dtype=np.int64)

    samples = sample_cdf_points(spectrum, thresholds=thresholds)

    np.testing.assert_allclose(
        samples[0],
        np.array([0.0, 0.0, 0.1, 0.4, 1.0, 1.0], dtype=np.float32),
    )


def test_preprocess_traces_matches_existing_reader_implementation(tmp_path) -> None:
    trace_path = tmp_path / "run_0007.h5"
    write_hdf5_input(trace_path)
    _ = trace_path

    traces = np.array(
        [
            [1, 2, 3, 9, 3, 2, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.float32,
    )

    expected = preprocess_traces_reference(traces, baseline_window_scale=20.0)
    actual = preprocess_traces(traces, baseline_window_scale=20.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_preprocess_traces_matches_reference_fft_implementation() -> None:
    rng = np.random.default_rng(0)
    traces = rng.normal(size=(32, 128)).astype(np.float32)

    expected = preprocess_traces_reference(traces, baseline_window_scale=20.0)
    actual = preprocess_traces(traces, baseline_window_scale=20.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_build_trace_cdf_histogram_returns_expected_shape_and_count(tmp_path) -> None:
    trace_path = tmp_path / "run_0005.h5"
    write_hdf5_input(trace_path)

    histogram = build_trace_cdf_histogram(trace_file_path=trace_path)

    assert histogram.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)
    assert np.all(histogram >= 0)
    assert int(histogram.sum()) == 3 * len(CDF_THRESHOLDS)


def test_build_trace_cdf_histogram_supports_legacy_hdf5_layout(tmp_path) -> None:
    trace_path = tmp_path / "run_0005.h5"
    write_legacy_hdf5(
        trace_path,
        {
            1: np.asarray(
                [
                    [10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8],
                    [20, 21, 22, 23, 24, 8, 7, 6, 5, 4, 3, 2, 1],
                ],
                dtype=np.float32,
            ),
            2: np.asarray(
                [
                    [30, 31, 32, 33, 34, 0, 1, 0, 1, 0, 1, 0, 1],
                ],
                dtype=np.float32,
            ),
        },
    )

    histogram = build_trace_cdf_histogram(trace_file_path=trace_path)

    assert histogram.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)
    assert np.all(histogram >= 0)
    assert int(histogram.sum()) == 3 * len(CDF_THRESHOLDS)


def test_trace_data_honors_bad_events_in_v2_layout(tmp_path) -> None:
    trace_path = tmp_path / "run_0008.h5"
    write_events_hdf5(
        trace_path,
        {
            1: np.asarray(
                [
                    [10, 11, 12, 13, 14, 1, 2, 3],
                    [20, 21, 22, 23, 24, 4, 5, 6],
                ],
                dtype=np.float32,
            ),
            2: np.asarray(
                [
                    [30, 31, 32, 33, 34, 7, 8, 9],
                ],
                dtype=np.float32,
            ),
        },
        bad_events=np.asarray([2], dtype=np.int64),
    )

    with h5py.File(trace_path, "r") as handle:
        assert detect_trace_layout(handle) is TraceLayout.MERGER_V2
        assert collect_event_counts(handle) == [(1, 2)]
        with pytest.raises(LookupError, match="trace 8/2"):
            load_trace_record(
                handle,
                run=8,
                event_id=2,
                trace_id=0,
                baseline_window_scale=20.0,
            )


def test_trace_data_reports_unsupported_layout(tmp_path) -> None:
    trace_path = tmp_path / "run_0009.h5"
    with h5py.File(trace_path, "w") as handle:
        handle.create_group("unexpected")

    with h5py.File(trace_path, "r") as handle:
        with pytest.raises(ValueError, match="unsupported trace file layout"):
            collect_event_counts(handle)


def test_build_trace_cdf_histogram_does_not_require_collect_event_counts(
    tmp_path, monkeypatch
) -> None:
    trace_path = tmp_path / "run_0005.h5"
    write_hdf5_input(trace_path)

    def fail_collect_event_counts(_handle: h5py.File) -> list[tuple[int, int]]:
        raise AssertionError("collect_event_counts should not be used")

    monkeypatch.setattr(
        trace_data,
        "collect_event_counts",
        fail_collect_event_counts,
    )

    histogram = build_trace_cdf_histogram(trace_file_path=trace_path)

    assert histogram.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)
    assert int(histogram.sum()) == 3 * len(CDF_THRESHOLDS)


def test_cdf_main_writes_default_output_file(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "run_0006.h5"
    write_hdf5_input(trace_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(sys, "argv", ["cdf", "-t", str(trace_path), "-w", str(workspace), "-r", "0006"])
    main()

    output_path = workspace / "histograms" / "run_0006_cdf.npy"
    saved = np.load(output_path)

    assert output_path.is_file()
    assert saved.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)
    assert int(saved.sum()) == 3 * len(CDF_THRESHOLDS)


def test_cdf_main_reads_options_from_config_file(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "run_0006.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)
    config_path = tmp_path / "batch.toml"
    config_path.write_text(
        "\n".join(
            [
                f'trace_path = "{trace_path}"',
                f'workspace = "{workspace}"',
                'run = "0006"',
                "",
                "[cdf]",
                "baseline_window_scale = 12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["cdf", "-c", str(config_path)])
    main()

    output_path = workspace / "histograms" / "run_0006_cdf.npy"
    saved = np.load(output_path)
    assert output_path.is_file()
    assert saved.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)


def test_cdf_main_zero_pads_integer_run_from_config_file(tmp_path, monkeypatch) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_hdf5_input(trace_root / "run_0106.h5")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "batch.toml"
    config_path.write_text(
        "\n".join(
            [
                f'trace_path = "{trace_root}"',
                f'workspace = "{workspace}"',
                "run = 106",
                "",
                "[cdf]",
                "baseline_window_scale = 12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["cdf", "-c", str(config_path)])
    main()

    output_path = workspace / "histograms" / "run_0106_cdf.npy"
    saved = np.load(output_path)
    assert output_path.is_file()
    assert saved.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)


def test_cdf_main_cli_arguments_override_config_file(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "run_0006.h5"
    write_hdf5_input(trace_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "batch.toml"
    config_path.write_text(
        "\n".join(
            [
                f'trace_path = "{trace_path}"',
                f'workspace = "{workspace}"',
                'run = "9999"',
                "",
                "[cdf]",
                "baseline_window_scale = 12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["cdf", "-c", str(config_path), "-r", "0006", "--baseline-window-scale", "20.0"],
    )
    main()

    output_path = workspace / "histograms" / "run_0006_cdf.npy"
    assert output_path.is_file()
