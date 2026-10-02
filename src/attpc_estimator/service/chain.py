from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any, Callable

import awkward as ak
import numpy as np
import uproot

from ..process.gagg_fit import FIXED_GAMMA_N
from ..storage.run_paths import resolve_run_file
from ..utils.trace_data import (
    DETECTOR_GAGG,
    DETECTOR_SI,
    SI_CODE_TO_SIDE,
    open_storage_trace_reader,
    reader_event_rows,
    trace_record_from_gagg_row,
    trace_record_from_silicon_row,
)
from .gagg_fit_cache import GaggFitCache

CHAIN_EVENT_PREFETCH_RADIUS = 5
THEORY_CACHE_VERSION = 1
THEORY_DEFAULT_STEP_MEV = 0.1
SILICON_DENSITY_G_CM3 = 2.329
THEORY_STOP_EPSILON_MEV = 1e-8
ChainEventKey = tuple[int, int]

# CATIMA only needs Z and A. Keep this local rather than adding a periodic-table
# dependency solely to turn configured isotope labels into projectiles.
ELEMENT_CHARGES = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
    "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
}


class _ChainEventPrefetcher:
    """Keep a small, complete Chain-event window warm without blocking navigation."""

    def __init__(self, loader: Callable[[ChainEventKey], dict[str, Any]]) -> None:
        self._loader = loader
        self._cache: dict[ChainEventKey, dict[str, Any]] = {}
        self._desired_keys: tuple[ChainEventKey, ...] = ()
        self._scheduled_keys: tuple[ChainEventKey, ...] = ()
        self._generation = 0
        self._completed_generation = 0
        self._active_generation: int | None = None
        self._cache_lock = threading.Lock()
        self._condition = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="chain-event-prefetch",
            daemon=True,
        )
        self._thread.start()

    def get_cached(self, key: ChainEventKey) -> dict[str, Any] | None:
        with self._cache_lock:
            return self._cache.get(key)

    def store_current(self, key: ChainEventKey, payload: dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = payload

    def schedule(self, keys: list[ChainEventKey]) -> None:
        scheduled = tuple(dict.fromkeys(keys))
        desired = set(scheduled)
        with self._cache_lock:
            self._desired_keys = scheduled
            self._cache = {
                key: payload for key, payload in self._cache.items() if key in desired
            }
        with self._condition:
            self._generation += 1
            self._scheduled_keys = scheduled
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        with self._cache_lock:
            self._cache.clear()
            self._desired_keys = ()

    def _is_stale(self, generation: int) -> bool:
        with self._condition:
            return self._closed or generation != self._generation

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._closed
                    and self._generation == self._completed_generation
                    and self._active_generation is None
                ):
                    self._condition.wait()
                if self._closed:
                    return
                generation = self._generation
                keys = self._scheduled_keys
                self._active_generation = generation

            self._prefetch_window(generation, keys)

            with self._condition:
                if self._active_generation == generation:
                    self._active_generation = None
                if generation == self._generation:
                    self._completed_generation = generation
                self._condition.notify_all()

    def _prefetch_window(
        self, generation: int, keys: tuple[ChainEventKey, ...]
    ) -> None:
        for key in keys:
            if self._is_stale(generation):
                return
            if self.get_cached(key) is not None:
                continue
            try:
                payload = self._loader(key)
            except Exception:
                # A missing neighbor must not make the foreground Chain review fail.
                continue
            if self._is_stale(generation):
                return
            with self._cache_lock:
                if key in self._desired_keys:
                    self._cache[key] = payload
                    desired = set(self._desired_keys)
                    self._cache = {
                        cached_key: cached_payload
                        for cached_key, cached_payload in self._cache.items()
                        if cached_key in desired
                    }


class ChainService:
    """Read one event across the raw, ingot, and silicon-match stages."""

    def __init__(
        self,
        *,
        trace_path: Path,
        workspace: Path,
        silicon_baseline_window_scale: float,
        gagg_baseline_window_scale: float,
        gagg_fit_cache: GaggFitCache,
        brill_config_path: Path | None = None,
        pid_run: int | None = None,
        pid_end_run: int | None = None,
        theory_lines: tuple[dict[str, float | str], ...] = (),
    ) -> None:
        self.trace_path = trace_path
        self.workspace = workspace
        self.silicon_baseline_window_scale = silicon_baseline_window_scale
        self.gagg_baseline_window_scale = gagg_baseline_window_scale
        self.gagg_fit_cache = gagg_fit_cache
        self.brill_config_path = brill_config_path
        self.pid_run = pid_run
        self.pid_end_run = pid_end_run
        self.theory_lines = tuple(dict(line) for line in theory_lines)
        self._current: tuple[int, int] | None = None
        self._match_viewers: dict[int, Any] = {}
        self._pid_background: dict[str, Any] | None = None
        self._pid_background_lock = threading.Lock()
        # T0MatchViewer owns native ROOT state, so only its calibration calls
        # are serialized. Full event construction remains concurrent with the
        # background prefetch worker.
        self._match_viewer_lock = threading.RLock()
        self._event_prefetcher = _ChainEventPrefetcher(self._load_prefetched_event)

    def load(self, *, run: int, event_id: int) -> dict[str, Any]:
        run, event_id = int(run), int(event_id)
        key = (run, event_id)
        payload = self._event_prefetcher.get_cached(key)
        if payload is None:
            # Never wait for a neighbor that is being prefetched. The worker
            # may duplicate this work in a narrow race, but foreground event
            # navigation always wins over cache efficiency.
            payload = self._load_event_payload(run, event_id)
            self._event_prefetcher.store_current(key, payload)
        self._current = key
        self._event_prefetcher.schedule(self._event_window(run, event_id))
        return payload

    def close(self) -> None:
        self._event_prefetcher.close()
        self._match_viewers.clear()

    def _event_window(self, run: int, event_id: int) -> list[ChainEventKey]:
        keys = [(run, event_id)]
        for distance in range(1, CHAIN_EVENT_PREFETCH_RADIUS + 1):
            keys.extend(((run, event_id + distance), (run, event_id - distance)))
        return keys

    def _load_prefetched_event(self, key: ChainEventKey) -> dict[str, Any]:
        cached = self._event_prefetcher.get_cached(key)
        if cached is not None:
            return cached
        return self._load_event_payload(*key)

    def _load_event_payload(self, run: int, event_id: int) -> dict[str, Any]:
        """Build every Chain stage once so stage changes are cache-only reads."""
        ingot = self._ingot_rows(run, event_id)
        match = self._match_stage(run, event_id, ingot)
        return {
            "run": run,
            "eventId": event_id,
            "stages": {
                "raw": {"traces": self._raw_traces(run, event_id, ingot)},
                "ingot": {
                    "columns": ["detector", "side", "index", "energy", "time"],
                    "rows": self._ingot_table(ingot),
                },
                "match": match,
            },
        }

    def current(self) -> dict[str, Any]:
        if self._current is None:
            raise LookupError("no chain event is selected")
        return self.load(run=self._current[0], event_id=self._current[1])

    def next(self) -> dict[str, Any]:
        if self._current is None:
            raise LookupError("no chain event is selected")
        return self.load(run=self._current[0], event_id=self._current[1] + 1)

    def previous(self) -> dict[str, Any]:
        if self._current is None:
            raise LookupError("no chain event is selected")
        return self.load(run=self._current[0], event_id=self._current[1] - 1)

    def _root_event(self, path: Path, event_id: int) -> dict[str, Any] | None:
        event, _ = self._root_event_with_position(path, event_id)
        return event

    def _root_event_with_position(
        self, path: Path, event_id: int
    ) -> tuple[dict[str, Any] | None, int | None]:
        if not path.is_file():
            return None, None
        with uproot.open(path) as root_file:
            tree = root_file["tree"]
            entries = tree["entry"].array(library="np")
            matches = np.flatnonzero(entries == int(event_id))
            if not len(matches):
                return None, None
            position = int(matches[0])
            arrays = tree.arrays(entry_start=position, entry_stop=position + 1, library="ak")
            return {name: ak.to_list(arrays[name][0]) for name in arrays.fields}, position

    def _ingot_rows(self, run: int, event_id: int) -> dict[str, dict[str, Any] | None]:
        return {
            "d1": self._root_event(self.workspace / "ingot" / f"t0d1_{run}.root", event_id),
            "d2": self._root_event(self.workspace / "ingot" / f"t0d2_{run}.root", event_id),
            "gagg": self._root_event(self.workspace / "ingot" / f"gagg_{run}.root", event_id),
        }

    @staticmethod
    def _values(event: dict[str, Any] | None, key: str) -> list[Any]:
        value = (event or {}).get(key, [])
        return value if isinstance(value, list) else []

    def _ingot_table(self, ingot: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for detector in ("d1", "d2"):
            event = ingot[detector]
            for side in ("front", "back"):
                strips = self._values(event, f"{side}_strip")
                energy = self._values(event, f"{side}_integral")
                time = self._values(event, f"{side}_time")
                rows.extend({"detector": detector, "side": side, "index": strip, "energy": energy[i] if i < len(energy) else None, "time": time[i] if i < len(time) else None} for i, strip in enumerate(strips))
        event = ingot["gagg"]
        for i, index in enumerate(self._values(event, "index")):
            tau, t0 = self._values(event, "tau"), self._values(event, "t0")
            rows.append({"detector": "gagg", "side": "", "index": index, "energy": self._values(event, "amp")[i] if i < len(self._values(event, "amp")) else None, "time": (t0[i] + FIXED_GAMMA_N * tau[i]) if i < len(t0) and i < len(tau) else None})
        return rows

    def _match_stage(
        self,
        run: int,
        event_id: int,
        ingot: dict[str, dict[str, Any] | None],
    ) -> dict[str, Any]:
        d1_event, entry_position = self._root_event_with_position(
            self.workspace / "match" / f"t0d1_{run}.root", event_id
        )
        d2_event = self._root_event(
            self.workspace / "match" / f"t0d2_{run}.root", event_id
        )
        candidate_rows = {
            "d1": self._silicon_match_rows(d1_event),
            "d2": self._silicon_match_rows(d2_event),
            "gagg": self._gagg_match_rows(ingot["gagg"]),
        }
        return {
            "tables": {
                detector: {"rows": rows}
                for detector, rows in candidate_rows.items()
            },
            "calibrated": self._calibrated_energies(
                run=run,
                entry_position=entry_position,
                tables=candidate_rows,
            ),
        }

    def _silicon_match_rows(self, event: dict[str, Any] | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        front, back = self._values(event, "front_strip"), self._values(event, "back_strip")
        energy, time = self._values(event, "energy"), self._values(event, "time")
        for index, value in enumerate(front):
            rows.append(
                {
                    "frontStrip": value,
                    "backStrip": back[index] if index < len(back) else None,
                    "energy": energy[index] if index < len(energy) else None,
                    "time": time[index] if index < len(time) else None,
                }
            )
        return rows

    def _gagg_match_rows(self, event: dict[str, Any] | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        indexes = self._values(event, "index")
        amplitude = self._values(event, "amp")
        tau, t0 = self._values(event, "tau"), self._values(event, "t0")
        for index, value in enumerate(indexes):
            rows.append(
                {
                    "index": value,
                    "energy": amplitude[index] if index < len(amplitude) else None,
                    "time": (
                        t0[index] + FIXED_GAMMA_N * tau[index]
                        if index < len(t0) and index < len(tau)
                        else None
                    ),
                }
            )
        return rows

    def _calibrated_energies(
        self,
        *,
        run: int,
        entry_position: int | None,
        tables: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[float | None]]:
        empty = {detector: [None] * len(rows) for detector, rows in tables.items()}
        if entry_position is None or self.brill_config_path is None:
            return empty
        try:
            with self._match_viewer_lock:
                viewer = self._match_viewer(run)
                counts = viewer.meta(entry_position)
                values: dict[str, list[float | None]] = {}
                for detector, count, component in (
                    ("d1", int(counts[0]), 0),
                    ("d2", int(counts[1]), 1),
                    ("gagg", int(counts[2]), 2),
                ):
                    calibrated: list[float | None] = []
                    for index in range(len(tables[detector])):
                        if index >= count:
                            calibrated.append(None)
                            continue
                        selected = [-1, -1, -1]
                        selected[component] = index
                        result = viewer.calibrated_energy(entry_position, *selected)
                        value = result[component]
                        calibrated.append(float(value) if value is not None else None)
                    values[detector] = calibrated
                return values
        except Exception as exc:
            raise ValueError(f"Unable to load calibrated Chain energies for run {run}: {exc}") from exc

    def _match_viewer(self, run: int) -> Any:
        viewer = self._match_viewers.get(run)
        if viewer is not None:
            return viewer
        _, _, viewer_type, _ = self._brill_api()
        viewer = viewer_type(str(self.brill_config_path), int(run))
        self._match_viewers[run] = viewer
        return viewer

    def _brill_api(self) -> tuple[Any, Any, Any, Any]:
        try:
            from py_brill_rcnp2025 import (  # type: ignore[import-not-found]
                T0MatchViewer,
                generate_t0_pid,
                get_t0_pid_path,
                get_t0_thickness,
            )
        except ImportError as exc:
            raise ValueError(
                "py_brill_rcnp2025 is required for Chain Match PID review; run uv sync"
            ) from exc
        return get_t0_pid_path, generate_t0_pid, T0MatchViewer, get_t0_thickness

    def match_pid_background(self) -> dict[str, Any]:
        if self._pid_background is not None:
            return self._pid_background
        if self.brill_config_path is None or self.pid_run is None or self.pid_end_run is None:
            raise ValueError("Chain Match PID configuration is incomplete")
        with self._pid_background_lock:
            if self._pid_background is not None:
                return self._pid_background
            get_path, _, _, _ = self._brill_api()
            path = Path(get_path(str(self.brill_config_path), self.pid_run, self.pid_end_run))
            if not path.is_file():
                self._generate_pid_background()
            if not path.is_file():
                raise ValueError(f"Generated Chain PID background was not found: {path}")
            try:
                with uproot.open(path) as root_file:
                    self._pid_background = {
                        "d1d2": self._pid_histogram(root_file, "d1d2"),
                        "d2gagg": self._pid_histogram(root_file, "d2gagg"),
                        "theoryLines": self._theory_payload(),
                    }
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "Chain PID background is incomplete or invalid at "
                    f"{path}. Regenerate it after fixing the PID binding."
                ) from exc
            return self._pid_background

    def _theory_payload(self) -> list[dict[str, Any]]:
        """Load or derive cached CATIMA curves and project them onto both PID plots."""
        if not self.theory_lines:
            return []
        thicknesses = self._t0_thicknesses()
        descriptor = self._theory_cache_descriptor(thicknesses)
        cache_path = self._theory_cache_path(descriptor)
        raw_lines = self._read_theory_cache(cache_path, descriptor)
        if raw_lines is None:
            raw_lines = self._generate_and_cache_theory(
                cache_path, descriptor, thicknesses
            )

        payload: list[dict[str, Any]] = []
        for line in raw_lines:
            d1d2_x: list[float] = []
            d1d2_y: list[float] = []
            d2gagg_x: list[float] = []
            d2gagg_y: list[float] = []
            for d1, d2, gagg in line["samples"]:
                if d1 is not None and d2 is not None:
                    d1d2_x.append(float(d2))
                    d1d2_y.append(float(d1))
                if d2 is not None and gagg is not None:
                    d2gagg_x.append(float(gagg))
                    d2gagg_y.append(float(d2))
            payload.append({
                "particle": line["particle"],
                "d1d2": {"x": d1d2_x, "y": d1d2_y},
                "d2gagg": {"x": d2gagg_x, "y": d2gagg_y},
            })
        return payload

    def _t0_thicknesses(self) -> tuple[float, float]:
        if self.brill_config_path is None:
            raise ValueError("Chain Match theory configuration is incomplete")
        _, _, _, get_t0_thickness = self._brill_api()
        try:
            values = tuple(
                float(value)
                for value in get_t0_thickness(str(self.brill_config_path))
            )
        except Exception as exc:
            raise ValueError(f"Unable to read T0 silicon thicknesses: {exc}") from exc
        if len(values) < 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in values[:2]
        ):
            raise ValueError(
                "Brill T0 configuration must define positive D1 and D2 silicon thicknesses"
            )
        return values[0], values[1]

    def _theory_cache_descriptor(
        self, thicknesses: tuple[float, float]
    ) -> dict[str, Any]:
        return {
            "version": THEORY_CACHE_VERSION,
            "siliconThicknessUm": list(thicknesses),
            "lines": [
                {
                    "particle": str(line["particle"]),
                    "max": float(line["max"]),
                    "step": float(line.get("step", THEORY_DEFAULT_STEP_MEV)),
                }
                for line in self.theory_lines
            ],
        }

    def _theory_cache_path(self, descriptor: dict[str, Any]) -> Path:
        encoded = json.dumps(
            descriptor, sort_keys=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return self.workspace / "estimate" / f"chain_pid_theory_{digest}.json"

    @staticmethod
    def _read_theory_cache(
        path: Path, descriptor: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("descriptor") != descriptor or not isinstance(
                payload.get("lines"), list
            ):
                return None
            lines = payload["lines"]
            for line in lines:
                if not isinstance(line, dict) or not isinstance(
                    line.get("particle"), str
                ):
                    return None
                samples = line.get("samples")
                if not isinstance(samples, list) or any(
                    not isinstance(sample, list)
                    or len(sample) != 3
                    or any(
                        value is not None and not isinstance(value, (int, float))
                        for value in sample
                    )
                    for sample in samples
                ):
                    return None
            return lines
        except (OSError, ValueError, TypeError):
            return None

    def _generate_and_cache_theory(
        self,
        path: Path,
        descriptor: dict[str, Any],
        thicknesses: tuple[float, float],
    ) -> list[dict[str, Any]]:
        lines = self._generate_theory_lines(thicknesses)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.json")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"descriptor": descriptor, "lines": lines},
                    handle,
                    separators=(",", ":"),
                )
            os.replace(temporary, path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ValueError(
                f"Unable to store Chain PID theory cache at {path}: {exc}"
            ) from exc
        return lines

    def _generate_theory_lines(
        self, thicknesses: tuple[float, float]
    ) -> list[dict[str, Any]]:
        catima = self._catima_api()
        materials = [
            self._silicon_material(catima, thickness) for thickness in thicknesses
        ]
        lines: list[dict[str, Any]] = []
        for config in self.theory_lines:
            particle = str(config["particle"])
            mass, charge = self._parse_particle(particle)
            maximum = float(config["max"])
            step = float(config.get("step", THEORY_DEFAULT_STEP_MEV))
            samples: list[list[float | None]] = []
            count = int(math.floor(maximum / step))
            energies = [index * step for index in range(count + 1)]
            if not math.isclose(
                energies[-1], maximum, rel_tol=0.0, abs_tol=1e-10
            ):
                energies.append(maximum)
            for incident in energies:
                d1 = self._catima_layer(
                    catima, mass, charge, incident, materials[0]
                )
                if d1[1] is None:
                    samples.append([d1[0], None, None])
                    continue
                d2 = self._catima_layer(
                    catima, mass, charge, d1[1], materials[1]
                )
                samples.append([d1[0], d2[0], d2[1]])
            lines.append({"particle": particle, "samples": samples})
        return lines

    @staticmethod
    def _silicon_material(catima: Any, thickness_um: float) -> Any:
        material = catima.Material([[28.0, 14, 1.0]], SILICON_DENSITY_G_CM3)
        material.thickness_cm(thickness_um * 1e-4)
        return material

    @staticmethod
    def _catima_layer(
        catima: Any,
        mass: int,
        charge: int,
        incident_mev: float,
        material: Any,
    ) -> tuple[float, float | None]:
        if incident_mev <= THEORY_STOP_EPSILON_MEV:
            return 0.0, None
        projectile = catima.Projectile(float(mass), charge, 0, incident_mev / mass)
        result = catima.calculate(projectile, material)
        deposited = max(0.0, float(result.Eloss))
        residual = max(0.0, float(result.Eout) * mass)
        return deposited, residual if residual > THEORY_STOP_EPSILON_MEV else None

    @staticmethod
    def _parse_particle(particle: str) -> tuple[int, int]:
        matched = re.fullmatch(r"([1-9][0-9]*)([A-Z][a-z]?)", particle)
        if matched is None or matched.group(2) not in ELEMENT_CHARGES:
            raise ValueError(f"Unsupported Chain PID theory isotope: {particle!r}")
        return int(matched.group(1)), ELEMENT_CHARGES[matched.group(2)]

    @staticmethod
    def _catima_api() -> Any:
        try:
            import pycatima  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(
                "pycatima is required to generate Chain PID theory curves; run uv sync"
            ) from exc
        return pycatima

    def _generate_pid_background(self) -> None:
        """Run the native PID generator out of process.

        ROOT errors inside the binding must not terminate the WebUI process.
        The output path is deterministic, so the following existence check is
        still the source of truth for a successful generation.
        """
        command = (
            "from py_brill_rcnp2025 import generate_t0_pid; "
            "import sys; "
            "raise SystemExit(generate_t0_pid(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))"
        )
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    command,
                    str(self.brill_config_path),
                    str(self.pid_run),
                    str(self.pid_end_run),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Generating Chain PID background timed out after 15 minutes") from exc
        if result.returncode != 0:
            raise ValueError(
                "Generating Chain PID background failed "
                f"(native process exit status {result.returncode})."
            )

    @staticmethod
    def _pid_histogram(root_file: Any, name: str) -> dict[str, Any]:
        if name not in root_file:
            raise ValueError(f"PID ROOT file is missing histogram {name!r}")
        values, x_edges, y_edges = root_file[name].to_numpy(flow=False)
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or len(x_edges) != values.shape[0] + 1 or len(y_edges) != values.shape[1] + 1:
            raise ValueError(f"PID histogram {name!r} is not a valid TH2")
        return {
            "x": [float((left + right) / 2) for left, right in zip(x_edges[:-1], x_edges[1:], strict=True)],
            "y": [float((left + right) / 2) for left, right in zip(y_edges[:-1], y_edges[1:], strict=True)],
            # uproot returns TH2 values in x/y order; Plotly heatmaps expect y/x.
            "z": values.T.tolist(),
        }

    def _raw_traces(self, run: int, event_id: int, ingot: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
        selected_si = {(0 if detector == "d1" else 2) + (0 if side == "front" else 1): set(self._values(ingot[detector], f"{side}_strip")) for detector in ("d1", "d2") for side in ("front", "back")}
        output: list[dict[str, Any]] = []
        si_reader = open_storage_trace_reader(run=run, path=str(resolve_run_file(self.trace_path, run)), detector=DETECTOR_SI)
        gagg_reader = open_storage_trace_reader(run=run, path=str(resolve_run_file(self.trace_path, run)), detector=DETECTOR_GAGG)
        try:
            try:
                si_rows = reader_event_rows(si_reader, event_id, detector=DETECTOR_SI)
                for side_code in range(4):
                    selected_rows = sorted(
                        ((trace_id, raw_row) for trace_id, raw_row in enumerate(si_rows) if int(raw_row[0]) == side_code and int(raw_row[1]) in selected_si[side_code]),
                        key=lambda item: int(item[1][1]),
                    )
                    for trace_id, raw_row in selected_rows:
                        record = trace_record_from_silicon_row(run=run, event_id=event_id, trace_id=trace_id, row=raw_row, baseline_window_scale=self.silicon_baseline_window_scale)
                        output.append({"detector": "silicon", "side": SI_CODE_TO_SIDE[side_code], "index": int(raw_row[1]), "raw": record.raw.tolist(), "fit": None})
            except LookupError:
                pass
            try:
                gagg_rows = reader_event_rows(gagg_reader, event_id, detector=DETECTOR_GAGG)
                selected_gagg = set(self._values(ingot["gagg"], "index"))
                for index in sorted(int(value) for value in selected_gagg):
                    if index < 0 or index >= len(gagg_rows):
                        continue
                    record = trace_record_from_gagg_row(run=run, event_id=event_id, trace_id=index, row=gagg_rows[index], baseline_window_scale=self.gagg_baseline_window_scale)
                    fit = self.gagg_fit_cache.get_or_compute(record).recovery
                    output.append({"detector": "gagg", "side": "", "index": index, "raw": record.raw.tolist(), "fit": {"status": fit.status, "fitted": fit.fitted.tolist() if fit.fitted is not None else None, "reason": fit.reason}})
            except LookupError:
                pass
        finally:
            si_reader.close()
            gagg_reader.close()
        return output
