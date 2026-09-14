from __future__ import annotations

import sys
from pathlib import Path

import pytest

from attpc_estimator.cli import gagg
from attpc_estimator.utils.gagg_processing import (
    GaggPeakFindingConfig,
    GaggTraceProcessingRule,
)


def test_parse_args_reads_gagg_config_and_command_line_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
trace_path = "traces"
workspace = "workspace"
run = 42

[gagg.baseline]
fft_window_scale = 12.0

[gagg.amplitude]
peak_separation = 11.0
peak_prominence = 12.0
peak_width = 13.0
peak_threshold = 14.0
rel_height = 0.7

[gagg.time]
min = 15
max = 16
""".strip(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gagg",
            "--config",
            str(config_path),
            "--peak-threshold",
            "25.0",
            "--max-time",
            "26",
        ],
    )

    args = gagg._parse_args()

    assert args.trace_path == "traces"
    assert args.workspace == "workspace"
    assert args.run == 42
    assert args.baseline_window_scale == 12.0
    assert args.peak_separation == 11.0
    assert args.peak_prominence == 12.0
    assert args.peak_width == 13.0
    assert args.peak_threshold == 25.0
    assert args.peak_rel_height == 0.7
    assert args.min_time == 0
    assert args.max_time == 26


def test_main_forwards_gagg_arguments_to_process_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    captured: dict[str, object] = {}

    def process_run(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(gagg, "process_run", process_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gagg", "--trace-path", "traces", "--workspace", str(workspace), "--run", "42"],
    )

    gagg.main()

    assert captured == {
        "workspace": str(workspace.resolve()),
        "run": 42,
        "input_path": "hdf5/run_<run>.h5",
        "output_path": "ingot/gagg_<run>.root",
        "fft_window_scale": 20.0,
        "peak_separation": 50.0,
        "peak_prominence": 20.0,
        "peak_max_width": 50.0,
        "peak_threshold": 300.0,
        "rel_height": 0.95,
        "min_time": 0,
        "max_time": 255,
        "gagg_trace_rules": {
            index: GaggTraceProcessingRule(
                20.0, GaggPeakFindingConfig(50.0, 20.0, 50.0, 300.0, 0.95)
            )
            for index in range(41)
        },
        "reporter": captured["reporter"],
    }
    assert captured["reporter"].description == "Processing GaGG"
    output = capsys.readouterr().out
    assert "Starting GAGG run 42." in output
    assert "Finished GAGG run 42: success." in output


def test_main_builds_per_index_rules_and_applies_scalar_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
trace_path = "traces"
workspace = "workspace"
run = 42

[[gagg.baseline]]
index = [0, 25]
fft_window_scale = 12.0

[[gagg.time]]
index = [0, 25]
min = 15
max = 16
""".strip(),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(gagg, "process_run", lambda **kwargs: captured.update(kwargs) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gagg", "--config", str(config_path), "--baseline-window-scale", "8",
            "--min-time", "20", "--max-time", "30",
        ],
    )

    gagg.main()

    assert captured["gagg_trace_rules"] == {
        0: GaggTraceProcessingRule(
            8.0, GaggPeakFindingConfig(50.0, 20.0, 50.0, 300.0, 0.95)
        ),
        25: GaggTraceProcessingRule(
            8.0, GaggPeakFindingConfig(50.0, 20.0, 50.0, 300.0, 0.95)
        ),
    }


def test_main_requires_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gagg", "--trace-path", "traces", "--workspace", "workspace"])

    with pytest.raises(SystemExit) as error:
        gagg.main()

    assert error.value.code == 2
