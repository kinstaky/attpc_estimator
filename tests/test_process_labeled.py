from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from attpc_estimator.cli.cdf import main
from attpc_estimator.storage.labels_db import LabelRepository
from attpc_estimator.process.cdf import build_labeled_cdf_histograms
from tests.hdf5_fixtures import write_legacy_hdf5


def write_hdf5_input(path: Path, traces: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["min_event"] = 1
        events.attrs["max_event"] = 2
        events.attrs["bad_events"] = np.array([], dtype=np.int64)

        event_1 = events.create_group("event_1")
        get_1 = event_1.create_group("get")
        get_1.create_dataset("pads", data=traces[:2])

        event_2 = events.create_group("event_2")
        get_2 = event_2.create_group("get")
        get_2.create_dataset("pads", data=traces[2:])


def seed_labels(workspace: Path) -> None:
    repository = LabelRepository(workspace / "labels.db")
    repository.initialize()
    repository.create_strange_label("Noise", "n")
    repository.create_strange_label("Burst", "b")
    repository.save_label(8, 1, 0, "pad", 10, 11, 12, 13, 14, "normal", "0")
    repository.save_label(8, 1, 1, "pad", 20, 21, 22, 23, 24, "strange", "Noise")
    repository.save_label(8, 2, 0, "pad", 30, 31, 32, 33, 34, "normal", "4")
    repository.save_label(9, 1, 0, "pad", 40, 41, 42, 43, 44, "normal", "2")
    repository.connection.close()


def make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    traces_run8 = np.array(
        [
            [10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8],
            [20, 21, 22, 23, 24, 8, 7, 6, 5, 4, 3, 2, 1],
            [30, 31, 32, 33, 34, 0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    traces_run9 = np.array(
        [
            [40, 41, 42, 43, 44, 1, 1, 2, 2, 3, 3, 4, 4],
            [50, 51, 52, 53, 54, 2, 3, 4, 5, 6, 7, 8, 9],
            [60, 61, 62, 63, 64, 9, 8, 7, 6, 5, 4, 3, 2],
        ],
        dtype=np.float32,
    )
    write_hdf5_input(trace_root / "run_0008.h5", traces_run8)
    write_hdf5_input(trace_root / "run_0009.h5", traces_run9)
    return workspace, trace_root


def test_build_labeled_cdf_histograms_filters_selected_run(tmp_path) -> None:
    workspace, trace_root = make_workspace(tmp_path)
    seed_labels(workspace)

    payload = build_labeled_cdf_histograms(
        trace_path=trace_root, workspace=workspace, run=8
    )

    assert payload["run_id"] == 8
    assert payload["label_titles"].tolist() == [
        "0 peak",
        "1 peak",
        "2 peaks",
        "3 peaks",
        "4+ peaks",
        "Burst",
        "Noise",
    ]
    assert payload["trace_counts"].tolist() == [1, 0, 0, 0, 1, 0, 1]


def test_build_labeled_cdf_histograms_supports_legacy_trace_layout(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_legacy_hdf5(
        trace_root / "run_0008.h5",
        {
            1: np.array(
                [
                    [10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8],
                    [20, 21, 22, 23, 24, 8, 7, 6, 5, 4, 3, 2, 1],
                ],
                dtype=np.float32,
            ),
            2: np.array(
                [
                    [30, 31, 32, 33, 34, 0, 1, 0, 1, 0, 1, 0, 1],
                ],
                dtype=np.float32,
            ),
        },
    )
    seed_labels(workspace)

    payload = build_labeled_cdf_histograms(
        trace_path=trace_root, workspace=workspace, run=8
    )

    assert payload["run_id"] == 8
    assert payload["trace_counts"].tolist() == [1, 0, 0, 0, 1, 0, 1]


def test_cdf_main_writes_labeled_output_for_selected_run(tmp_path, monkeypatch) -> None:
    workspace, trace_root = make_workspace(tmp_path)
    seed_labels(workspace)

    monkeypatch.setattr(
        sys,
        "argv",
        ["cdf", "-t", str(trace_root), "-w", str(workspace), "-r", "0008", "--labeled"],
    )
    main()

    output_path = workspace / "histograms" / "run_0008_labeled_cdf.npz"
    payload = np.load(output_path)

    assert output_path.is_file()
    assert payload["run_id"] == 8
    assert payload["trace_counts"].tolist() == [1, 0, 0, 0, 1, 0, 1]


def test_cdf_labeled_main_reads_options_from_config_file(tmp_path, monkeypatch) -> None:
    workspace, trace_root = make_workspace(tmp_path)
    seed_labels(workspace)
    config_path = tmp_path / "cdf.toml"
    config_path.write_text(
        "\n".join(
            [
                f'trace_path = "{trace_root}"',
                f'workspace = "{workspace}"',
                'run = "0008"',
                "",
                "[cdf]",
                "labeled = true",
                "baseline_window_scale = 12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["cdf", "-c", str(config_path)])
    main()

    output_path = workspace / "histograms" / "run_0008_labeled_cdf.npz"
    payload = np.load(output_path)
    assert output_path.is_file()
    assert payload["run_id"] == 8
