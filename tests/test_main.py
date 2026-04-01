from __future__ import annotations

import sys

import pytest

from attpc_estimator.label_trace import __main__


def test_main_reconstructs_trace_path_from_workspace_and_run(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "run_0042.h5"
    trace_path.touch()
    db_dir = tmp_path / "db"
    captured: dict[str, object] = {}

    class DummyService:
        def __init__(self, trace_path, db_dir) -> None:
            captured["trace_path"] = trace_path
            captured["db_dir"] = db_dir

    monkeypatch.setattr(
        sys,
        "argv",
        ["label", "-w", str(tmp_path), "-r", "0042", "-d", str(db_dir)],
    )
    monkeypatch.setattr(__main__, "TraceLabelService", DummyService)
    monkeypatch.setattr(__main__, "create_app", lambda service, frontend_dist: object())
    monkeypatch.setattr(__main__, "_pick_port", lambda preferred_port: 8765)
    monkeypatch.setattr(__main__.uvicorn, "run", lambda *args, **kwargs: None)

    __main__.main()

    assert captured["trace_path"] == trace_path.resolve()
    assert captured["db_dir"] == db_dir.resolve()


def test_parse_args_rejects_non_numeric_run(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["label", "-w", "/tmp/workspace", "-r", "run42", "-d", "/tmp/db"],
    )

    with pytest.raises(SystemExit):
        __main__._parse_args()
