from __future__ import annotations

import sys
from pathlib import Path

import pytest

from attpc_estimator.cli import ic, silicon
from attpc_estimator.cli.config import table_config_values


def test_table_config_values_reads_nested_dotted_tables() -> None:
    payload = {
        "findline": {
            "ransac": {
                "residual_threshold": 20.0,
                "max_trials": 50,
                "ignored": 123,
            },
            "mergeline": {
                "distance_threshold": 15,
                "angle_threshold": 3,
            },
        }
    }

    ransac = table_config_values(
        payload,
        table="findline.ransac",
        allowed_keys={"residual_threshold", "max_trials"},
    )
    mergeline = table_config_values(
        payload,
        table="findline.mergeline",
        allowed_keys={"distance_threshold", "angle_threshold"},
    )

    assert ransac == {
        "residual_threshold": 20.0,
        "max_trials": 50,
    }
    assert mergeline == {
        "distance_threshold": 15,
        "angle_threshold": 3,
    }


def test_table_config_values_returns_empty_dict_for_missing_nested_table() -> None:
    payload = {"findline": {}}

    assert table_config_values(
        payload,
        table="findline.ransac",
        allowed_keys={"residual_threshold"},
    ) == {}


def test_detector_clis_share_parser_and_keep_detector_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("""
trace_path = "traces"
workspace = "workspace"
run = 42

[run_tags]
path = "runs.xlsx"

[ic.baseline]
fft_window_scale = 12.0

[si.time]
min = 90
max = 98
""")
    monkeypatch.setattr(sys, "argv", ["detector", "-c", str(config_path)])

    ic_args = ic._parse_args()
    silicon_args = silicon._parse_args()

    assert ic_args.baseline_window_scale == 12.0
    assert ic_args.min_time == 62.0
    assert ic_args.max_time == 68
    assert silicon_args.baseline_window_scale == 20.0
    assert silicon_args.min_time == 90
    assert silicon_args.max_time == 98
    assert ic_args.run_tags_path == silicon_args.run_tags_path == tmp_path / "runs.xlsx"
