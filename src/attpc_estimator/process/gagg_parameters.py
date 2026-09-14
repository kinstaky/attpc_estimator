from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


DEFAULT_GAGG_PARAMETERS_PATH = Path("config/gagg_fit_parameters.toml")


@dataclass(frozen=True, slots=True)
class GaggFitParameters:
    p0: tuple[float, ...]
    p1: tuple[float, ...]
    linear_window: float
    linear_clipped: float
    tau: tuple[float, ...]

    def tau_for(self, index: int) -> float:
        try:
            value = self.tau[index]
        except IndexError as exc:
            raise ValueError(f"GAGG tau is missing for index {index}") from exc
        if value <= 0.0:
            raise ValueError(f"GAGG tau must be positive for index {index}")
        return value


def load_gagg_fit_parameters(path: str | Path = DEFAULT_GAGG_PARAMETERS_PATH) -> GaggFitParameters:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ValueError(f"GAGG calibration parameters file not found: {config_path}")
    try:
        with config_path.open("rb") as handle:
            values = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid GAGG calibration parameters TOML: {config_path}") from exc

    def numbers(key: str, minimum: int) -> tuple[float, ...]:
        raw = values.get(key)
        if not isinstance(raw, list) or len(raw) < minimum:
            raise ValueError(f"GAGG calibration parameter {key!r} must contain at least {minimum} values")
        try:
            result = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GAGG calibration parameter {key!r} must contain numbers") from exc
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"GAGG calibration parameter {key!r} must contain finite numbers")
        return result

    p0 = numbers("p0", 16)
    p1 = numbers("p1", 16)
    tau = numbers("tau", 41)
    try:
        linear_window = float(values["linear_window"])
        linear_clipped = float(values["linear_clipped"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("GAGG calibration parameters require numeric linear_window and linear_clipped") from exc
    if linear_window < 0.0 or linear_clipped <= 0.0:
        raise ValueError("GAGG linear_window must be non-negative and linear_clipped must be positive")
    return GaggFitParameters(p0, p1, linear_window, linear_clipped, tau)
