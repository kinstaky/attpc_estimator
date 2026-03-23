from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from trace_label.batch.process_labeled import build_labeled_cdf_histograms, main
from trace_label.db import TraceLabelRepository


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


def seed_labels(db_dir: Path) -> None:
    repository = TraceLabelRepository(db_dir / "trace_label.sqlite3")
    repository.initialize()
    repository.create_strange_label("Noise", "n")
    repository.create_strange_label("Burst", "b")
    repository.save_label(8, 1, 0, "pad", 10, 11, 12, 13, 14, "normal", "0")
    repository.save_label(8, 1, 1, "pad", 20, 21, 22, 23, 24, "strange", "Noise")
    repository.save_label(8, 2, 0, "pad", 30, 31, 32, 33, 34, "normal", "4")
    repository.save_label(9, 1, 0, "pad", 40, 41, 42, 43, 44, "normal", "2")
    repository.connection.close()


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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
    write_hdf5_input(workspace / "run_0008.h5", traces_run8)
    write_hdf5_input(workspace / "run_0009.h5", traces_run9)
    return workspace


def test_build_labeled_cdf_histograms_aggregates_all_workspace_runs(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    db_dir = tmp_path / "db"
    seed_labels(db_dir)

    payload = build_labeled_cdf_histograms(workspace=workspace, db_dir=db_dir)

    assert payload["run_ids"].tolist() == [8, 9]
    assert payload["label_titles"].tolist() == [
        "0 peak",
        "1 peak",
        "2 peak",
        "3 peak",
        "4+ peaks",
        "Burst",
        "Noise",
    ]
    assert payload["histograms"].shape == (7, 150, 100)
    assert payload["trace_counts"].tolist() == [1, 0, 1, 0, 1, 0, 1]
    assert payload["histograms"].sum(axis=(1, 2)).tolist() == [150, 0, 150, 0, 150, 0, 150]


def test_build_labeled_cdf_histograms_filters_selected_run(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    db_dir = tmp_path / "db"
    seed_labels(db_dir)

    payload = build_labeled_cdf_histograms(workspace=workspace, db_dir=db_dir, run=8)

    assert payload["run_ids"].tolist() == [8]
    assert payload["trace_counts"].tolist() == [1, 0, 0, 0, 1, 0, 1]


def test_process_labeled_main_writes_default_output_file_for_selected_run(tmp_path, monkeypatch) -> None:
    workspace = make_workspace(tmp_path)
    db_dir = tmp_path / "db"
    seed_labels(db_dir)

    monkeypatch.setattr(sys, "argv", ["process-labeled", "-w", str(workspace), "-r", "0008", "-d", str(db_dir)])
    main()

    output_path = workspace / "run_0008_labeled_hist2d.npy"
    payload = np.load(output_path, allow_pickle=True).item()

    assert output_path.is_file()
    assert payload["run_ids"].tolist() == [8]
    assert payload["trace_counts"].tolist() == [1, 0, 0, 0, 1, 0, 1]


def test_process_labeled_main_writes_default_output_file_for_all_runs(tmp_path, monkeypatch) -> None:
    workspace = make_workspace(tmp_path)
    db_dir = tmp_path / "db"
    seed_labels(db_dir)

    monkeypatch.setattr(sys, "argv", ["process-labeled", "-w", str(workspace), "-d", str(db_dir)])
    main()

    output_path = workspace / "all_runs_labeled_hist2d.npy"
    payload = np.load(output_path, allow_pickle=True).item()

    assert output_path.is_file()
    assert payload["run_ids"].tolist() == [8, 9]
    assert payload["trace_counts"].tolist() == [1, 0, 1, 0, 1, 0, 1]


def test_process_labeled_main_reads_options_from_config_file(tmp_path, monkeypatch) -> None:
    workspace = make_workspace(tmp_path)
    db_dir = tmp_path / "db"
    seed_labels(db_dir)
    output_path = workspace / "from_config.npy"
    config_path = tmp_path / "process_labeled.toml"
    config_path.write_text(
        "\n".join(
            [
                "[process_labeled]",
                f'workspace = "{workspace}"',
                'run = "0008"',
                f'database_dir = "{db_dir}"',
                f'output_file = "{output_path}"',
                "baseline_window_scale = 12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["process_labeled", "-c", str(config_path)])
    main()

    payload = np.load(output_path, allow_pickle=True).item()
    assert output_path.is_file()
    assert payload["run_ids"].tolist() == [8]


def test_process_labeled_main_cli_arguments_override_config_file(tmp_path, monkeypatch) -> None:
    workspace = make_workspace(tmp_path)
    db_dir = tmp_path / "db"
    seed_labels(db_dir)
    config_output = workspace / "from_config.npy"
    cli_output = workspace / "from_cli.npy"
    config_path = tmp_path / "process_labeled.toml"
    config_path.write_text(
        "\n".join(
            [
                "[process_labeled]",
                f'workspace = "{workspace}"',
                'run = "0008"',
                f'database_dir = "{db_dir}"',
                f'output_file = "{config_output}"',
                "baseline_window_scale = 12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["process-labeled", "-c", str(config_path), "-o", str(cli_output), "-d", str(db_dir)],
    )
    main()

    assert not config_output.exists()
    assert cli_output.is_file()
