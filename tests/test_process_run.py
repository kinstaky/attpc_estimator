from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from trace_label.batch.process_run import CDF_THRESHOLDS, CDF_VALUE_BINS, build_trace_cdf_histogram, main, preprocess_traces, sample_cdf_points
from trace_label.input_reader import TraceSource


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
    input_path = tmp_path / "run_0007.h5"
    write_hdf5_input(input_path)

    traces = np.array(
        [
            [1, 2, 3, 9, 3, 2, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.float32,
    )

    source = TraceSource(input_path)
    try:
        expected = source.preprocess_traces(traces, baseline_window_scale=20.0)
    finally:
        source.file.close()

    actual = preprocess_traces(traces, baseline_window_scale=20.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_preprocess_traces_matches_reference_fft_implementation() -> None:
    rng = np.random.default_rng(0)
    traces = rng.normal(size=(32, 128)).astype(np.float32)

    expected = preprocess_traces_reference(traces, baseline_window_scale=20.0)
    actual = preprocess_traces(traces, baseline_window_scale=20.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_build_trace_cdf_histogram_returns_expected_shape_and_count(tmp_path) -> None:
    input_path = tmp_path / "run_0005.h5"
    write_hdf5_input(input_path)

    histogram = build_trace_cdf_histogram(input_path=input_path)

    assert histogram.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)
    assert np.all(histogram >= 0)
    assert int(histogram.sum()) == 3 * len(CDF_THRESHOLDS)


def test_process_run_main_writes_default_output_file(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "run_0006.h5"
    write_hdf5_input(input_path)

    monkeypatch.setattr(sys, "argv", ["process-run", "-i", str(input_path)])
    main()

    output_path = tmp_path / "run_0006_cdf_hist2d.npy"
    saved = np.load(output_path)

    assert output_path.is_file()
    assert saved.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)
    assert int(saved.sum()) == 3 * len(CDF_THRESHOLDS)


def test_process_run_main_reads_options_from_config_file(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "run_0006.h5"
    output_path = tmp_path / "from_config.npy"
    write_hdf5_input(input_path)
    config_path = tmp_path / "batch.toml"
    config_path.write_text(
        "\n".join(
            [
                "[batch]",
                f'input_file = "{input_path}"',
                f'output_file = "{output_path}"',
                "baseline_window_scale = 12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["batch", "-c", str(config_path)])
    main()

    saved = np.load(output_path)
    assert output_path.is_file()
    assert saved.shape == (len(CDF_THRESHOLDS), CDF_VALUE_BINS)


def test_process_run_main_cli_arguments_override_config_file(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "run_0006.h5"
    write_hdf5_input(input_path)
    config_output = tmp_path / "from_config.npy"
    cli_output = tmp_path / "from_cli.npy"
    config_path = tmp_path / "batch.toml"
    config_path.write_text(
        "\n".join(
            [
                "[batch]",
                f'input_file = "{input_path}"',
                f'output_file = "{config_output}"',
                "baseline_window_scale = 12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["batch", "-c", str(config_path), "-o", str(cli_output), "--baseline-window-scale", "20.0"],
    )
    main()

    assert not config_output.exists()
    assert cli_output.is_file()
