from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import h5py
import numpy as np
import polars as pl
from tqdm import tqdm

from pointcloud import TraceLength, fft_filter_traces, find_trace_peaks

from ..cli.config import parse_run, parse_toml_config, root_config_values, table_config_values
from ..storage.run_paths import ion_chamber_run_path, resolve_run_file
from ..utils.ion_chamber import describe_ion_chamber_events, load_ion_chamber_event


@dataclass(frozen=True, slots=True)
class ProgressState:
    total: int
    unit: str
    description: str


class ProgressReporter:
    def report_start(self, *, total: int, unit: str, description: str) -> None:
        raise NotImplementedError

    def report_progress(self, current: int, *, message: str = "") -> None:
        raise NotImplementedError

    def report_finish(self) -> None:
        raise NotImplementedError


class TqdmProgressReporter(ProgressReporter):
    def __init__(self) -> None:
        self._bar: tqdm | None = None
        self._current = 0

    def report_start(self, *, total: int, unit: str, description: str) -> None:
        self.report_finish()
        self._current = 0
        self._bar = tqdm(total=max(int(total), 0), desc=description, unit=unit)

    def report_progress(self, current: int, *, message: str = "") -> None:
        if self._bar is None:
            return
        bounded = max(int(current), 0)
        delta = max(0, bounded - self._current)
        if delta:
            self._bar.update(delta)
        self._current = max(self._current, bounded)
        if message:
            self._bar.set_postfix_str(message)

    def report_finish(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        self._current = 0


@dataclass(frozen=True, slots=True)
class IcBaselineConfig:
    fft_window_scale: float


@dataclass(frozen=True, slots=True)
class IcAmplitudeConfig:
    peak_separation: float
    peak_prominence: float
    peak_width: float
    peak_threshold: float
    rel_height: float
    ic_delay_time_bucket: float
    ic_multiplicity: int


def process_run(
    *,
    trace_path: Path,
    workspace: Path,
    run: int,
    output_path: Path,
    baseline_config: IcBaselineConfig,
    amplitude_config: IcAmplitudeConfig,
    progress: ProgressReporter,
) -> int:
    run_file = resolve_run_file(trace_path, run)
    rows: list[dict[str, int | float]] = []
    with h5py.File(run_file, "r") as handle:
        metadata = describe_ion_chamber_events(handle)
        progress.report_start(
            total=metadata.valid_event_span,
            unit="event",
            description=f"Ion chamber run {run:04d}",
        )
        processed_events = 0
        for event_id in range(metadata.min_event, metadata.max_event + 1):
            if event_id in metadata.bad_events:
                continue
            row = _sentinel_row(run=run, event_id=event_id)
            try:
                event = load_ion_chamber_event(handle, run=run, event_id=event_id)
            except LookupError:
                rows.append(row)
                processed_events += 1
                progress.report_progress(processed_events, message=f"event={event_id},missing=IC")
                continue

            row["orig_run"] = int(event.orig_run)
            row["orig_event"] = int(event.orig_event)
            row["trigger_type"] = int(event.trigger_type)
            _fill_selected_peak(
                row,
                raw_trace=event.raw_trace,
                baseline_config=baseline_config,
                amplitude_config=amplitude_config,
            )
            rows.append(row)
            processed_events += 1
            progress.report_progress(
                processed_events,
                message=f"event={event_id},trigger={event.trigger_type}",
            )
    progress.report_finish()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _dataframe_from_rows(rows).write_parquet(output_path)
    return len(rows)


def _fill_selected_peak(
    row: dict[str, int | float],
    *,
    raw_trace: np.ndarray,
    baseline_config: IcBaselineConfig,
    amplitude_config: IcAmplitudeConfig,
) -> None:
    filtered = fft_filter_traces(
        raw_trace[np.newaxis, :].astype(np.float64, copy=False),
        trace_length=TraceLength.TB256,
        baseline_window_scale=baseline_config.fft_window_scale,
    )
    peaks = find_trace_peaks(
        filtered,
        peak_separation=amplitude_config.peak_separation,
        peak_prominence=amplitude_config.peak_prominence,
        peak_max_width=amplitude_config.peak_width,
        peak_threshold=amplitude_config.peak_threshold,
        rel_height=amplitude_config.rel_height,
    )
    if peaks.shape[0] == 0:
        row["ic_multiplicity"] = 0.0
        return

    order = np.argsort(peaks[:, 3], kind="stable")
    sorted_peaks = peaks[order]
    trigger_candidates = sorted_peaks[
        sorted_peaks[:, 3] > float(amplitude_config.ic_delay_time_bucket)
    ]
    multiplicity = int(trigger_candidates.shape[0])
    row["ic_multiplicity"] = float(multiplicity)
    if not (0 < multiplicity <= int(amplitude_config.ic_multiplicity)):
        return

    selected = trigger_candidates[0]
    row["ic_amplitude"] = float(selected[1])
    row["ic_integral"] = float(selected[2])
    row["ic_centroid"] = float(selected[3])


def _sentinel_row(*, run: int, event_id: int) -> dict[str, int | float]:
    return {
        "run": int(run),
        "event_id": int(event_id),
        "orig_run": int(run),
        "orig_event": int(event_id),
        "trigger_type": -1,
        "ic_amplitude": -1.0,
        "ic_integral": -1.0,
        "ic_centroid": -1.0,
        "ic_multiplicity": -1.0,
    }


def _dataframe_from_rows(rows: list[dict[str, int | float]]) -> pl.DataFrame:
    schema = {
        "run": pl.Int64,
        "event_id": pl.Int64,
        "orig_run": pl.Int64,
        "orig_event": pl.Int64,
        "trigger_type": pl.Int64,
        "ic_amplitude": pl.Float64,
        "ic_integral": pl.Float64,
        "ic_centroid": pl.Float64,
        "ic_multiplicity": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(payload, allowed_keys={"trace_path", "workspace", "run"})
    baseline_config = table_config_values(
        payload,
        table="ic.baseline",
        allowed_keys={"fft_window_scale"},
    )
    amplitude_config = table_config_values(
        payload,
        table="ic.amplitude",
        allowed_keys={
            "peak_separation",
            "peak_prominence",
            "peak_width",
            "peak_threshold",
            "rel_height",
            "ic_delay_time_bucket",
            "ic_multiplicity",
        },
    )
    configured_runs = _config_runs(config.get("run"))

    parser = argparse.ArgumentParser(description="Build phase-1 ion-chamber parquet for raw runs")
    parser.add_argument("-c", "--config", dest="config_file", default=str(config_path))
    parser.add_argument(
        "-t",
        "--trace-path",
        required="trace_path" not in config,
        default=config.get("trace_path"),
        help="Path to a trace file or a directory containing run_<run>.h5 files",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        required="workspace" not in config,
        default=config.get("workspace"),
        help="Workspace directory used for ion-chamber parquet output",
    )
    parser.add_argument(
        "-r",
        "--run",
        action="append",
        type=parse_run,
        default=configured_runs,
        help="Run identifier to process. May be repeated.",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=baseline_config.get("fft_window_scale", 20.0),
    )
    parser.add_argument(
        "--peak-separation",
        type=float,
        default=amplitude_config.get("peak_separation", 50.0),
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=amplitude_config.get("peak_prominence", 20.0),
    )
    parser.add_argument(
        "--peak-width",
        type=float,
        default=amplitude_config.get("peak_width", 500.0),
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=amplitude_config.get("peak_threshold", 100.0),
    )
    parser.add_argument("--peak-rel-height", type=float, default=amplitude_config.get("rel_height", 0.85))
    parser.add_argument(
        "--ic-delay-time-bucket",
        type=float,
        default=amplitude_config.get("ic_delay_time_bucket", 1100.0),
    )
    parser.add_argument(
        "--ic-multiplicity",
        type=int,
        default=amplitude_config.get("ic_multiplicity", 1),
    )
    return parser.parse_args()


def _config_runs(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [parse_run(str(item)) for item in value]
    return [parse_run(str(value))]


def main() -> None:
    args = _parse_args()
    if not args.run:
        raise SystemExit("no runs provided; pass --run for each run to process")

    trace_path = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    reporter = TqdmProgressReporter()
    baseline_config = IcBaselineConfig(fft_window_scale=float(args.baseline_window_scale))
    amplitude_config = IcAmplitudeConfig(
        peak_separation=float(args.peak_separation),
        peak_prominence=float(args.peak_prominence),
        peak_width=float(args.peak_width),
        peak_threshold=float(args.peak_threshold),
        rel_height=float(args.peak_rel_height),
        ic_delay_time_bucket=float(args.ic_delay_time_bucket),
        ic_multiplicity=int(args.ic_multiplicity),
    )
    for run_token in args.run:
        run = int(run_token)
        output_path = ion_chamber_run_path(workspace, run).resolve()
        written = process_run(
            trace_path=trace_path,
            workspace=workspace,
            run=run,
            output_path=output_path,
            baseline_config=baseline_config,
            amplitude_config=amplitude_config,
            progress=reporter,
        )
        print(f"wrote {written} ion-chamber events to {output_path}")


if __name__ == "__main__":
    main()
