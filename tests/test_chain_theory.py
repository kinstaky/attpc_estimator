from __future__ import annotations

from pathlib import Path

import pytest

from attpc_estimator.cli.webui import _parse_chain_theory_lines
from attpc_estimator.service.chain import ChainService


class _FakeResult:
    def __init__(self, loss: float, residual_per_u: float) -> None:
        self.Eloss = loss
        self.Eout = residual_per_u


class _FakeProjectile:
    def __init__(self, mass: float, _charge: int, _state: int, energy_per_u: float) -> None:
        self.mass = mass
        self.energy_per_u = energy_per_u


class _FakeMaterial:
    def __init__(self, _elements, _density: float) -> None:
        self.thickness_cm_value = 0.0

    def thickness_cm(self, value: float) -> None:
        self.thickness_cm_value = value


class _FakeCatima:
    Material = _FakeMaterial
    Projectile = _FakeProjectile

    @staticmethod
    def calculate(projectile: _FakeProjectile, material: _FakeMaterial) -> _FakeResult:
        incident = projectile.energy_per_u * projectile.mass
        capacity = material.thickness_cm_value * 40.0
        loss = min(incident, capacity)
        return _FakeResult(loss, (incident - loss) / projectile.mass)


def _service(tmp_path: Path, thicknesses: tuple[float, float] = (500.0, 1000.0)) -> ChainService:
    service = object.__new__(ChainService)
    service.workspace = tmp_path
    service.brill_config_path = Path("brill.toml")
    service.theory_lines = ({"particle": "4He", "max": 8.0, "step": 1.0},)
    service._brill_api = lambda: (None, None, None, lambda _path: thicknesses)
    service._catima_api = lambda: _FakeCatima
    return service


def test_chain_theory_cache_preserves_stopping_layers_and_reuses_cache(tmp_path: Path) -> None:
    first = _service(tmp_path)
    payload = first._theory_payload()

    assert payload[0]["particle"] == "4He"
    assert payload[0]["d1d2"]["x"] == [1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    assert payload[0]["d1d2"]["y"] == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    assert payload[0]["d2gagg"] == {"x": [1.0, 2.0], "y": [4.0, 4.0]}

    cache_files = list((tmp_path / "estimate").glob("chain_pid_theory_*.json"))
    assert len(cache_files) == 1

    second = _service(tmp_path)
    second._catima_api = lambda: (_ for _ in ()).throw(AssertionError("cache was not reused"))
    assert second._theory_payload() == payload


def test_chain_theory_cache_key_includes_silicon_thickness(tmp_path: Path) -> None:
    _service(tmp_path)._theory_payload()
    changed = _service(tmp_path, (600.0, 1000.0))
    changed._theory_payload()

    assert len(list((tmp_path / "estimate").glob("chain_pid_theory_*.json"))) == 2


def test_chain_theory_config_defaults_and_validates_step() -> None:
    assert _parse_chain_theory_lines([{"particle": "4He", "max": 20}]) == (
        {"particle": "4He", "max": 20.0, "step": 0.1},
    )
    with pytest.raises(ValueError, match="step"):
        _parse_chain_theory_lines([{"particle": "4He", "max": 20, "step": 0}])
