from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import uproot

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
