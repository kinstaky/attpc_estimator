from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

from attpc_estimator.service.chain import ChainService
from attpc_estimator.service.estimator import EstimatorService


class _Guess:
    def __init__(self, value: float, weight: float, time: float, guess_type: int, flag: int) -> None:
        self.value = value
        self.weight = weight
        self.time = time
        self.type = guess_type
        self.flag = flag


class _Viewer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def strip_combination_guesses(self, entry: int, detector: int) -> tuple[list[_Guess], list[_Guess]]:
        self.calls.append((entry, detector))
        if detector == 0:
            return (
                [_Guess(100.0, 1.0, 11.0, 1, 0b11)],
                [_Guess(103.0, 2.0, 12.0, 4, 0b11 << 8)],
            )
        return (
            [_Guess(200.0, 3.0, 21.0, 0, 0b1)],
            [_Guess(199.0, 4.0, 22.0, 8, 0b10 << 8)],
        )

    @staticmethod
    def distances(front: list[_Guess], back: list[_Guess]) -> list[float]:
        return [abs(first.value - second.value) for first in front for second in back]

    @staticmethod
    def energies(front: list[_Guess], back: list[_Guess]) -> list[float]:
        return [
            (first.value * second.weight + second.value * first.weight)
            / (first.weight + second.weight)
            for first in front
            for second in back
        ]


def _service(tmp_path: Path) -> ChainService:
    service = object.__new__(ChainService)
    service.workspace = tmp_path
    service.brill_config_path = Path("brill.toml")
    service._ingot_viewers = {}
    service._ingot_viewer_lock = threading.RLock()
    return service


def test_chain_ingot_stage_serializes_guesses_and_matrices(tmp_path: Path) -> None:
    service = _service(tmp_path)
    viewer = _Viewer()
    service._ingot_viewer = lambda _run: viewer
    service._root_event_with_position = lambda _path, _event: ({}, 7)
    ingot = {
        "d1": {"front_strip": [10, 11], "back_strip": [20, 21]},
        "d2": {"front_strip": [30, 31], "back_strip": [40, 41]},
        "gagg": None,
    }

    stage = service._ingot_stage(42, 99, ingot)

    assert viewer.calls == [(7, 0), (7, 1)]
    assert stage["guesses"]["d1"]["front"] == [{
        "strips": [10, 11], "type": "charge-shared", "time": 11.0,
        "energy": 100.0, "weight": 1.0,
    }]
    assert stage["guesses"]["d1"]["back"][0]["strips"] == [20, 21]
    assert stage["guesses"]["d1"]["back"][0]["type"] == "short"
    assert stage["guesses"]["d2"]["back"][0]["type"] == "piecewise"
    assert stage["matrices"]["d1"] == {
        "distances": [[3.0]], "energies": [[101.0]],
    }
    assert stage["matrices"]["d2"] == {
        "distances": [[1.0]], "energies": [[199.57142857142858]],
    }


def test_chain_ingot_guess_type_decodes_bitmask() -> None:
    assert ChainService._guess_type_name(0) == "normal"
    assert ChainService._guess_type_name(5) == "charge-shared + short"
    assert ChainService._guess_type_name(18) == "broken-adjacent + unknown (0x10)"


def test_chain_ingot_stage_keeps_empty_review_without_brill_config(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.brill_config_path = None
    service._ingot_table = lambda _ingot: [{"detector": "d1"}]

    stage = service._ingot_stage(42, 99, {"d1": None, "d2": None, "gagg": None})

    assert stage["rows"] == [{"detector": "d1"}]
    assert stage["guesses"] == {
        "d1": {"front": [], "back": []},
        "d2": {"front": [], "back": []},
    }
    assert stage["matrices"] == {
        "d1": {"distances": [], "energies": []},
        "d2": {"distances": [], "energies": []},
    }


def test_chain_ingot_ui_state_persists_valid_threshold_and_selection(tmp_path: Path) -> None:
    service = object.__new__(EstimatorService)
    service.workspace = tmp_path
    service.run_files = {42: tmp_path / "run_42.h5"}
    service.pointcloud = SimpleNamespace(pointcloud_files={})

    state = service._normalize_ui_state({
        "chain": {
            "selectedRun": 42,
            "ingotDistanceThreshold": 750.5,
            "selectedD1FrontGuessIndex": 1,
            "selectedD1BackGuessIndex": 2,
            "selectedD2FrontGuessIndex": 3,
            "selectedD2BackGuessIndex": 4,
        },
    })

    assert state["chain"]["ingotDistanceThreshold"] == 750.5
    assert [
        state["chain"][key]
        for key in (
            "selectedD1FrontGuessIndex", "selectedD1BackGuessIndex",
            "selectedD2FrontGuessIndex", "selectedD2BackGuessIndex",
        )
    ] == [1, 2, 3, 4]
