from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import Workbook

from attpc_estimator.pipeline import gagg, ic, silicon
from attpc_runs import RunDatabase


def _write_workbook(workspace: Path, rows: list[tuple[object, ...]], *, silicon_header: str = "slilcon") -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(("run", "experiment", silicon_header, "gagg"))
    for row in rows:
        worksheet.append(row)
    workbook.save(workspace / "runs.xlsx")
    workbook.close()


@pytest.mark.parametrize(
    ("detector", "run", "valid"),
    [
        ("silicon", 1, True),
        ("gagg", 1, True),
        ("ic", 1, True),
        ("silicon", 2, False),
        ("gagg", 2, True),
        ("ic", 2, True),
        ("silicon", 3, True),
        ("gagg", 3, False),
        ("ic", 3, True),
        ("silicon", 4, False),
        ("gagg", 4, False),
        ("ic", 4, False),
        ("silicon", 5, False),
    ],
)
def test_run_tags_enforce_experiment_and_detector_tags(
    tmp_path: Path,
    detector: str,
    run: int,
    valid: bool,
) -> None:
    _write_workbook(
        tmp_path,
        [
            (1, "Normal", "Normal", "Normal"),
            (2, "Beam Stop", "Invalid", "Normal"),
            (3, "Normal", "Normal", "Invalid"),
            (4, "Invalid", "Normal", "Normal"),
        ],
    )

    eligibility = _run_database(tmp_path).check(run=run, detector=detector)

    assert eligibility.valid is valid


def test_run_tags_accepts_correctly_spelled_silicon_header(tmp_path: Path) -> None:
    _write_workbook(tmp_path, [(1, "Normal", "Normal", "Normal")], silicon_header="silicon")

    assert _run_database(tmp_path).check(run=1, detector="silicon").valid


def test_run_tags_rejects_missing_workbook_and_malformed_rows(tmp_path: Path) -> None:
    assert not _run_database(tmp_path).check(run=1, detector="ic").valid

    _write_workbook(tmp_path, [("not-a-run", "Normal", "Normal", "Normal")])

    eligibility = _run_database(tmp_path).check(run=1, detector="ic")
    assert not eligibility.valid
    assert "invalid run value" in eligibility.reason


def _run_database(workspace: Path) -> RunDatabase:
    return RunDatabase(
        workbook_path=workspace / "runs.xlsx",
        database_path=workspace / "runs.sqlite",
    )


@pytest.mark.parametrize(
    ("pipeline", "detector", "output_path"),
    [
        (silicon, "silicon", ["ingot/t0d1_<run>.root", "ingot/t0d2_<run>.root"]),
        (gagg, "gagg", "ingot/gagg_<run>.root"),
        (ic, "ic", "ingot/ic_<run>.root"),
    ],
)
def test_pipelines_skip_invalid_runs_before_opening_trace_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pipeline: object,
    detector: str,
    output_path: object,
) -> None:
    _write_workbook(tmp_path, [(1, "Invalid", "Normal", "Normal")])

    def fail_if_opened(**_: object) -> None:
        raise AssertionError("invalid runs must not open trace data")

    monkeypatch.setattr(pipeline, "open_trace_reader", fail_if_opened)

    result = pipeline.process_run(
        workspace=str(tmp_path),
        run=1,
        input_path="hdf5/run_<run>.h5",
        output_path=output_path,
        fft_window_scale=20.0,
        peak_separation=50.0,
        peak_prominence=20.0,
        peak_max_width=50.0,
        peak_threshold=20.0,
        rel_height=0.95,
        min_time=0,
        max_time=255,
        reporter=object(),
    )

    assert result == 1
