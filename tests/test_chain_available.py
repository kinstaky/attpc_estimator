from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import pytest
import uproot

import attpc_estimator.service.chain as chain_module
from attpc_estimator.service.chain import ChainService


def _write_tree(path: Path, **branches: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as root_file:
        root_file["tree"] = {
            name: np.asarray(values, dtype=np.int32)
            for name, values in branches.items()
        }


def _service(workspace: Path) -> ChainService:
    service = object.__new__(ChainService)
    service.workspace = workspace
    service._availability_indexes = {}
    service._availability_lock = threading.Lock()
    service._current = (42, 3)
    return service


def _write_availability_inputs(workspace: Path) -> None:
    _write_tree(
        workspace / "ingot" / "t0d1_42.root",
        entry=[1, 2, 5, 8], nfront=[0, 1, 0, 0], nback=[0, 0, 1, 0],
    )
    _write_tree(
        workspace / "ingot" / "t0d2_42.root",
        entry=[1, 4, 5, 7], nfront=[0, 1, 0, 0], nback=[1, 0, 0, 0],
    )
    _write_tree(
        workspace / "match" / "t0d1_42.root",
        entry=[1, 2, 4, 5, 7], num=[1, 1, 0, 1, 1],
    )
    _write_tree(
        workspace / "match" / "t0d2_42.root",
        entry=[1, 2, 4, 5, 8], num=[0, 1, 1, 1, 1],
    )


def test_chain_available_entries_follow_stage_rules(tmp_path: Path) -> None:
    _write_availability_inputs(tmp_path)
    service = _service(tmp_path)

    assert service._available_entries(42, "raw").tolist() == [1, 2, 4, 5]
    assert service._available_entries(42, "ingot").tolist() == [1, 2, 4, 5]
    assert service._available_entries(42, "match").tolist() == [2, 5]


def test_chain_available_navigation_and_unavailable_window(tmp_path: Path) -> None:
    _write_availability_inputs(tmp_path)
    service = _service(tmp_path)
    loaded: list[int] = []
    warmed: list[tuple[int, str]] = []
    service.load = lambda *, run, event_id: loaded.append(event_id) or {"run": run, "eventId": event_id}
    service._schedule_prefetch = lambda run, event_id, stage: warmed.append((event_id, stage))

    assert service._available_window(42, 3, "raw") == [2, 1, 4, 5]
    assert service.previous_available("raw")["eventId"] == 2
    assert service.next_available("raw")["eventId"] == 4
    assert loaded == [2, 4]
    assert warmed == [(2, "raw"), (4, "raw")]


def test_chain_available_prefetch_merges_windows_without_duplicates(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._availability_indexes[(42, "raw")] = np.arange(10, 101, 10)

    class Prefetcher:
        scheduled: list[tuple[int, int]] = []

        def schedule(self, keys: list[tuple[int, int]]) -> None:
            self.scheduled = keys

    prefetcher = Prefetcher()
    service._event_prefetcher = prefetcher
    service._schedule_prefetch(42, 50, "raw")

    assert len(prefetcher.scheduled) == len(set(prefetcher.scheduled))
    assert (42, 50) in prefetcher.scheduled
    assert {(42, 45), (42, 55), (42, 10), (42, 100)} <= set(prefetcher.scheduled)


def test_chain_event_window_skips_negative_ids_and_previous_stops_at_zero(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._current = (42, 0)

    assert all(event_id >= 0 for _, event_id in service._event_window(42, 0))
    with pytest.raises(LookupError, match="before event 0"):
        service.previous()
    with pytest.raises(ValueError, match="nonnegative"):
        service.load(run=42, event_id=-1)


def test_chain_raw_traces_open_only_selected_detectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    service.trace_path = tmp_path
    service.silicon_baseline_window_scale = 10.0
    opened: list[str] = []

    class Reader:
        def __init__(self, detector: str) -> None:
            self.detector = detector

        def close(self) -> None:
            opened.append(f"closed:{self.detector}")

    def open_reader(*, run: int, path: str, detector: str) -> Reader:
        opened.append(detector)
        return Reader(detector)

    monkeypatch.setattr(chain_module, "resolve_run_file", lambda _path, _run: tmp_path / "run_42.h5")
    monkeypatch.setattr(chain_module, "open_storage_trace_reader", open_reader)
    monkeypatch.setattr(chain_module, "reader_event_rows", lambda *_args, **_kwargs: [])

    assert service._raw_traces(42, 0, {"d1": None, "d2": None, "gagg": None}) == []
    assert opened == []

    assert service._raw_traces(42, 0, {
        "d1": {"front_strip": [3]}, "d2": None, "gagg": None,
    }) == []
    assert opened == ["SI", "closed:SI"]

    def fail_gagg(*, run: int, path: str, detector: str) -> Reader:
        if detector == "GAGG":
            raise OSError("reader unavailable")
        return open_reader(run=run, path=path, detector=detector)

    monkeypatch.setattr(chain_module, "open_storage_trace_reader", fail_gagg)
    with pytest.raises(OSError, match="reader unavailable"):
        service._raw_traces(42, 0, {
            "d1": {"front_strip": [3]}, "d2": None, "gagg": {"index": [0]},
        })
    assert opened[-2:] == ["SI", "closed:SI"]
