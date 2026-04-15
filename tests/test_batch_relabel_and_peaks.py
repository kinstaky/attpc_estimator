from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from attpc_estimator.cli.amplitude import main as find_peaks_main
from attpc_estimator.storage.labels_db import LabelRepository
from attpc_estimator.process.amplitude import build_labeled_amplitude_histograms
from attpc_estimator.process.relabel import build_relabel_rows
from attpc_estimator.cli.relabel import main as relabel_main
from attpc_estimator.storage.labeled_traces import read_labeled_trace
from tests.hdf5_fixtures import write_legacy_hdf5


def _trace_from_second_derivative(
    second_diff: list[float],
    sample_count: int = 256,
) -> np.ndarray:
    values = np.asarray(second_diff, dtype=np.float32)
    trace = np.zeros(sample_count, dtype=np.float32)
    for index, value in enumerate(values):
        trace[index + 2] = value + (2.0 * trace[index + 1]) - trace[index]
    return trace


def _low_amplitude_trace() -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return (20.0 * np.exp(-0.5 * ((x - 120.0) / 8.0) ** 2)).astype(np.float32)


def _high_amplitude_trace() -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return (80.0 * np.exp(-0.5 * ((x - 120.0) / 8.0) ** 2)).astype(np.float32)


def _qualified_bitflip_trace() -> np.ndarray:
    return _trace_from_second_derivative(
        [0.0, 61.0, -121.0, 450.0, -512.0, 0.0, 0.0, 0.0]
    )


def _saturation_trace() -> np.ndarray:
    row = np.zeros(256, dtype=np.float32)
    row[120:124] = 2105.0
    row[124] = 2098.0
    return row


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


def seed_workspace_db(workspace: Path) -> None:
    repository = LabelRepository(workspace / "labels.db")
    repository.initialize()
    repository.create_strange_label("oscillation", "o")
    repository.save_label(8, 1, 0, "pad", 1, 1, 1, 1, 1, "strange", "oscillation")
    repository.save_label(8, 1, 1, "pad", 2, 2, 2, 2, 2, "normal", "1")
    repository.save_label(8, 1, 2, "pad", 3, 3, 3, 3, 3, "normal", "1")
    repository.save_label(8, 2, 0, "pad", 4, 4, 4, 4, 4, "normal", "1")
    repository.save_label(8, 2, 1, "pad", 5, 5, 5, 5, 5, "normal", "0")
    repository.save_label(9, 1, 0, "pad", 6, 6, 6, 6, 6, "normal", "1")
    repository.connection.close()


def seed_saturation_workspace_db(workspace: Path) -> None:
    repository = LabelRepository(workspace / "labels.db")
    repository.initialize()
    repository.create_strange_label("oscillation", "o")
    repository.create_strange_label("saturation", "s")
    repository.save_label(8, 1, 0, "pad", 1, 1, 1, 1, 1, "strange", "saturation")
    repository.save_label(8, 1, 1, "pad", 2, 2, 2, 2, 2, "normal", "1")
    repository.save_label(8, 1, 2, "pad", 3, 3, 3, 3, 3, "normal", "1")
    repository.save_label(8, 1, 3, "pad", 4, 4, 4, 4, 4, "normal", "0")
    repository.connection.close()


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_run_file(
        workspace / "run_0008.h5",
        {
            1: [_qualified_bitflip_trace(), _low_amplitude_trace(), _high_amplitude_trace()],
            2: [_qualified_bitflip_trace(), _low_amplitude_trace()],
        },
    )
    write_run_file(
        workspace / "run_0009.h5",
        {
            1: [_high_amplitude_trace()],
        },
    )
    seed_workspace_db(workspace)
    return workspace


def make_saturation_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_run_file(
        workspace / "run_0008.h5",
        {
            1: [
                _saturation_trace(),
                _saturation_trace(),
                _high_amplitude_trace(),
                _low_amplitude_trace(),
            ],
        },
    )
    seed_saturation_workspace_db(workspace)
    return workspace


def test_read_labeled_trace_returns_storage_keys_for_selected_run_and_all_runs(
    tmp_path,
) -> None:
    workspace = make_workspace(tmp_path)

    traces_run8, labels_run8 = read_labeled_trace(
        trace_path=workspace, workspace_path=workspace, run=8
    )
    all_traces, all_labels = read_labeled_trace(
        trace_path=workspace, workspace_path=workspace, run=None
    )

    assert traces_run8.shape == (5, 256)
    assert labels_run8 == [
        "strange:oscillation",
        "normal:1",
        "normal:1",
        "normal:1",
        "normal:0",
    ]
    assert all_traces.shape == (6, 256)
    assert all_labels == labels_run8 + ["normal:1"]


def test_read_labeled_trace_supports_legacy_trace_layout(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_legacy_hdf5(
        workspace / "run_0008.h5",
        {
            1: _pad_rows([_qualified_bitflip_trace(), _low_amplitude_trace(), _high_amplitude_trace()]),
            2: _pad_rows([_qualified_bitflip_trace(), _low_amplitude_trace()]),
        },
    )
    seed_workspace_db(workspace)

    traces, labels = read_labeled_trace(
        trace_path=workspace, workspace_path=workspace, run=8
    )

    assert traces.shape == (5, 256)
    assert labels == [
        "strange:oscillation",
        "normal:1",
        "normal:1",
        "normal:1",
        "normal:0",
    ]


def test_read_labeled_trace_uses_labels_db_filename(tmp_path) -> None:
    workspace = make_workspace(tmp_path)

    traces, labels = read_labeled_trace(
        trace_path=workspace, workspace_path=workspace, run=8
    )

    assert traces.shape == (5, 256)
    assert labels[0] == "strange:oscillation"


def test_build_relabel_rows_noise_applies_zero_peak_rule_and_reports_noise_ratios(
    tmp_path,
) -> None:
    workspace = make_workspace(tmp_path)

    rows, metrics = build_relabel_rows(
        trace_path=workspace,
        workspace=workspace,
        run=8,
        label="noise",
    )

    assert rows.dtype.names == ("run", "event_id", "trace_id", "old_label", "new_label")
    assert rows["old_label"].tolist() == [
        "strange:oscillation",
        "normal:1",
        "normal:1",
        "normal:1",
        "normal:0",
    ]
    assert rows["new_label"].tolist() == [
        "strange:oscillation",
        "normal:0",
        "normal:1",
        "normal:1",
        "normal:0",
    ]
    assert set(metrics) == {"normal0_to_normal0", "normal1_to_normal0"}
    assert metrics["normal0_to_normal0"] == (1, 1)
    assert metrics["normal1_to_normal0"] == (1, 3)


def test_build_relabel_rows_oscillation_applies_bitflip_rule_and_reports_oscillation_ratios(
    tmp_path,
) -> None:
    workspace = make_workspace(tmp_path)

    rows, metrics = build_relabel_rows(
        trace_path=workspace,
        workspace=workspace,
        run=8,
        label="oscillation",
    )

    assert rows["new_label"].tolist() == [
        "strange:oscillation",
        "normal:1",
        "normal:1",
        "strange:oscillation",
        "normal:0",
    ]
    assert set(metrics) == {
        "oscillation_to_oscillation",
        "normal1_to_oscillation",
    }
    assert metrics["oscillation_to_oscillation"] == (1, 1)
    assert metrics["normal1_to_oscillation"] == (1, 3)


def test_build_relabel_rows_reports_trace_progress(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    progress_updates = []

    rows, _ = build_relabel_rows(
        trace_path=workspace,
        workspace=workspace,
        run=8,
        label="noise",
        progress=progress_updates.append,
    )

    assert len(rows) == 5
    assert [update.current for update in progress_updates] == [0, 3, 5]
    assert all(update.total == 5 for update in progress_updates)
    assert all(update.unit == "trace" for update in progress_updates)


def test_relabel_main_noise_writes_rows_and_prints_noise_ratios(
    tmp_path, monkeypatch, capsys
) -> None:
    workspace = make_workspace(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relabel",
            "-t",
            str(workspace),
            "-w",
            str(workspace),
            "-r",
            "0008",
            "--label",
            "noise",
        ],
    )

    relabel_main()

    output_path = workspace / "run_0008_labeled_relabel.npy"
    saved = np.load(output_path)
    stdout = capsys.readouterr().out

    assert output_path.is_file()
    assert saved["new_label"].tolist() == [
        "strange:oscillation",
        "normal:0",
        "normal:1",
        "normal:1",
        "normal:0",
    ]
    assert "old normal:0 -> new normal:0: 1/1 = 1.000000" in stdout
    assert "old normal:1 -> new normal:0: 1/3 = 0.333333" in stdout
    assert "old strange:oscillation -> new strange:oscillation" not in stdout
    assert "old normal:1 -> new strange:oscillation" not in stdout


def test_relabel_main_oscillation_writes_rows_and_prints_oscillation_ratios(
    tmp_path, monkeypatch, capsys
) -> None:
    workspace = make_workspace(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relabel",
            "-t",
            str(workspace),
            "-w",
            str(workspace),
            "-r",
            "0008",
            "--label",
            "oscillation",
        ],
    )

    relabel_main()

    output_path = workspace / "run_0008_labeled_relabel.npy"
    saved = np.load(output_path)
    stdout = capsys.readouterr().out

    assert output_path.is_file()
    assert saved["new_label"].tolist() == [
        "strange:oscillation",
        "normal:1",
        "normal:1",
        "strange:oscillation",
        "normal:0",
    ]
    assert "old normal:0 -> new normal:0" not in stdout
    assert "old normal:1 -> new normal:0" not in stdout
    assert (
        "old strange:oscillation -> new strange:oscillation: 1/1 = 1.000000" in stdout
    )
    assert "old normal:1 -> new strange:oscillation: 1/3 = 0.333333" in stdout


def test_relabel_main_zero_pads_integer_run_from_config_file(
    tmp_path, monkeypatch
) -> None:
    workspace = make_workspace(tmp_path)
    config_path = tmp_path / "relabel.toml"
    config_path.write_text(
        "\n".join(
            [
                f'trace_path = "{workspace}"',
                f'workspace = "{workspace}"',
                "run = 8",
                "",
                "[relabel]",
                'label = "noise"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["relabel", "-c", str(config_path)])
    relabel_main()

    output_path = workspace / "run_0008_labeled_relabel.npy"
    saved = np.load(output_path)

    assert output_path.is_file()
    assert saved["new_label"].tolist() == [
        "strange:oscillation",
        "normal:0",
        "normal:1",
        "normal:1",
        "normal:0",
    ]


def test_relabel_main_requires_label_option(tmp_path, monkeypatch, capsys) -> None:
    workspace = make_workspace(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relabel",
            "-t",
            str(workspace),
            "-w",
            str(workspace),
            "-r",
            "0008",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        relabel_main()

    assert exc.value.code == 2
    assert "the following arguments are required: --label" in capsys.readouterr().err


def test_build_relabel_rows_saturation_applies_plateau_rule_and_reports_saturation_ratios(
    tmp_path,
) -> None:
    workspace = make_saturation_workspace(tmp_path)

    rows, metrics = build_relabel_rows(
        trace_path=workspace,
        workspace=workspace,
        run=8,
        label="saturation",
    )

    assert rows["new_label"].tolist() == [
        "strange:saturation",
        "strange:saturation",
        "normal:1",
        "normal:0",
    ]
    assert set(metrics) == {
        "saturation_to_saturation",
        "normal1_to_saturation",
    }
    assert metrics["saturation_to_saturation"] == (1, 1)
    assert metrics["normal1_to_saturation"] == (1, 2)


def test_relabel_main_saturation_writes_rows_and_prints_saturation_ratios(
    tmp_path, monkeypatch, capsys
) -> None:
    workspace = make_saturation_workspace(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relabel",
            "-t",
            str(workspace),
            "-w",
            str(workspace),
            "-r",
            "0008",
            "--label",
            "saturation",
        ],
    )

    relabel_main()

    output_path = workspace / "run_0008_labeled_relabel.npy"
    saved = np.load(output_path)
    stdout = capsys.readouterr().out

    assert output_path.is_file()
    assert saved["new_label"].tolist() == [
        "strange:saturation",
        "strange:saturation",
        "normal:1",
        "normal:0",
    ]
    assert "old strange:saturation -> new strange:saturation: 1/1 = 1.000000" in stdout
    assert "old normal:1 -> new strange:saturation: 1/2 = 0.500000" in stdout


def test_build_labeled_amplitude_histograms_groups_histograms_by_label(
    tmp_path,
) -> None:
    workspace = make_workspace(tmp_path)

    payload = build_labeled_amplitude_histograms(
        trace_path=workspace, workspace=workspace, run=8
    )

    assert payload["run_id"] == 8
    assert payload["label_keys"].tolist() == [
        "normal:0",
        "normal:1",
        "normal:2",
        "normal:3",
        "normal:4+",
        "strange:oscillation",
    ]
    assert payload["label_titles"].tolist() == [
        "0 peak",
        "1 peak",
        "2 peaks",
        "3 peaks",
        "4+ peaks",
        "oscillation",
    ]
    assert payload["trace_counts"].tolist() == [1, 3, 0, 0, 0, 1]
    assert int(payload["histograms"][0].sum()) == 0
    assert int(payload["histograms"][1].sum()) == 2
    assert int(payload["histograms"][5].sum()) == 1


def test_find_peaks_main_labeled_writes_payload_npz(tmp_path, monkeypatch) -> None:
    workspace = make_workspace(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "find-peak",
            "-t",
            str(workspace),
            "-w",
            str(workspace),
            "-r",
            "0008",
            "--labeled",
        ],
    )

    find_peaks_main()

    output_path = workspace / "run_0008_labeled_amp.npz"
    payload = np.load(output_path)

    assert output_path.is_file()
    assert payload["label_keys"].tolist() == [
        "normal:0",
        "normal:1",
        "normal:2",
        "normal:3",
        "normal:4+",
        "strange:oscillation",
    ]
    assert payload["trace_counts"].tolist() == [1, 3, 0, 0, 0, 1]
    assert int(payload["histograms"][1].sum()) == 2


def test_find_peaks_main_zero_pads_integer_run_from_config_file(
    tmp_path, monkeypatch
) -> None:
    workspace = make_workspace(tmp_path)
    config_path = tmp_path / "amplitude.toml"
    config_path.write_text(
        "\n".join(
            [
                f'trace_path = "{workspace}"',
                f'workspace = "{workspace}"',
                "run = 8",
                "",
                "[amplitude]",
                "labeled = true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["find-peak", "-c", str(config_path)])
    find_peaks_main()

    output_path = workspace / "run_0008_labeled_amp.npz"
    payload = np.load(output_path)

    assert output_path.is_file()
    assert payload["trace_counts"].tolist() == [1, 3, 0, 0, 0, 1]
