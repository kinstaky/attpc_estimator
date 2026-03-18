from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from trace_label.batch import build_trace_cdf_samples, main, preprocess_traces, sample_cdf_points
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


def test_build_trace_cdf_samples_returns_one_row_per_trace(tmp_path) -> None:
    input_path = tmp_path / "run_0005.h5"
    write_hdf5_input(input_path)

    samples = build_trace_cdf_samples(input_path=input_path)

    assert samples.shape == (3, 10)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    assert np.all(np.diff(samples, axis=1) >= -1e-6)


def test_batch_main_writes_default_output_file(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "run_0006.h5"
    write_hdf5_input(input_path)

    monkeypatch.setattr(sys, "argv", ["batch", "-i", str(input_path)])
    main()

    output_path = tmp_path / "run_0006_cdf.npy"
    saved = np.load(output_path)

    assert output_path.is_file()
    assert saved.shape == (3, 10)
