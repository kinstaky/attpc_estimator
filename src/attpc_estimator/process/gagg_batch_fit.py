from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import uproot
from pointcloud import find_trace_peaks

from ..process.gagg_fit import fit_gamma_pulse, is_gagg_fit_eligible
from ..process.gagg_parameters import GaggFitParameters
from ..pipeline.progress_reporter import ProgressReporter
from ..storage.run_paths import collect_run_files, format_run_id
from ..utils.gagg_processing import GaggTraceProcessingRule, require_gagg_peak_config
from ..utils.trace_data import (
    DETECTOR_GAGG,
    GAGG_TRACE_COUNT,
    open_storage_trace_reader,
    trace_record_from_gagg_row,
)

GAGG_FIT_DIRNAME = "trace"
GAGG_FIT_COLUMNS = {
    "index": pl.Int16,
    "amplitude": pl.Float64,
    "integral": pl.Float64,
    "time": pl.Int16,
    "A": pl.Float64,
    "tau": pl.Float64,
    "n": pl.Float64,
    "t0": pl.Float64,
    "run": pl.Int32,
    "event": pl.Int32,
}


def gagg_fit_output_dir(workspace: Path) -> Path:
    return workspace / GAGG_FIT_DIRNAME


def gagg_fit_parquet_path(workspace: Path, run: int) -> Path:
    return gagg_fit_output_dir(workspace) / f"gagg_fit_{format_run_id(run)}.parquet"


def gagg_fit_root_path(workspace: Path, run: int) -> Path:
    return gagg_fit_output_dir(workspace) / f"gagg_fit_{format_run_id(run)}.root"


def gagg_fit_parquet_paths(workspace: Path) -> list[Path]:
    return sorted(gagg_fit_output_dir(workspace).glob("gagg_fit_*.parquet"))


def fit_gagg_run(
    *,
    trace_path: Path,
    workspace: Path,
    run: int,
    rules: dict[int, GaggTraceProcessingRule],
    parameters: GaggFitParameters,
    reporter: ProgressReporter | None = None,
    output_root: bool = False,
) -> tuple[Path | None, int]:
    run_files = collect_run_files(trace_path)
    path = run_files.get(run)
    if path is None:
        return None, 0
    if reporter is not None:
        reporter.report_start(message="Preparing GAGG fits")
    rows: list[dict[str, float | int]] = []
    reader = open_storage_trace_reader(run=run, path=str(path), detector=DETECTOR_GAGG)
    try:
            min_event = int(reader.min_event)
            max_event = int(reader.max_event)
            event_count = max(1, max_event - min_event + 1)
            for event_id in range(min_event, max_event + 1):
                if reporter is not None:
                    reporter.report_progress(
                        int((event_id - min_event + 1) * 100 / event_count),
                        message=f"run {run} · event {event_id} · {len(rows)} fits",
                    )
                if event_id in getattr(reader, "bad_events", set()):
                    continue
                try:
                    event = reader.read_event(event_id)
                    _, gagg_rows = event["gagg"]
                except (KeyError, LookupError, ValueError):
                    continue
                matrix = np.asarray(gagg_rows, dtype=np.float32)
                if matrix.ndim != 2 or matrix.shape[0] != GAGG_TRACE_COUNT:
                    continue
                for index, rule in rules.items():
                    if index >= matrix.shape[0] or float(matrix[index, 0]) != 0.0:
                        continue
                    try:
                        peak_config = require_gagg_peak_config(rule, index=index)
                        record = trace_record_from_gagg_row(
                            run=run, event_id=event_id, trace_id=index,
                            row=matrix[index], baseline_window_scale=rule.baseline_window_scale,
                        )
                    except (LookupError, ValueError):
                        continue
                    peaks = find_trace_peaks(record.trace[np.newaxis, :],
                        peak_separation=peak_config.peak_separation, peak_prominence=peak_config.peak_prominence,
                        peak_max_width=peak_config.peak_width, peak_threshold=peak_config.peak_threshold,
                        rel_height=peak_config.rel_height)
                    if not len(peaks):
                        continue
                    peak = max(peaks, key=lambda item: float(item[1]))
                    peak_amplitude, peak_integral, peak_time = float(peak[1]), float(peak[2]), int(peak[3])
                    if not is_gagg_fit_eligible(
                        index, record.trace, high_amplitude_threshold=parameters.linear_clipped
                    ):
                        continue
                    fit = fit_gamma_pulse(record.trace)
                    if fit.status != "ok":
                        continue
                    rows.append({
                        "index": index,
                        "amplitude": peak_amplitude,
                        "integral": peak_integral,
                        "time": peak_time,
                        "A": float(fit.amplitude),
                        "tau": float(fit.tau),
                        "n": float(fit.n),
                        "t0": float(fit.t0),
                        "run": run,
                        "event": event_id,
                    })
    finally:
        reader.close()
    output = gagg_fit_parquet_path(workspace, run)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows, schema=GAGG_FIT_COLUMNS)
    temporary = output.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary)
    temporary.replace(output)
    if output_root:
        _write_root(gagg_fit_root_path(workspace, run), frame)
    if reporter is not None:
        reporter.report_finish(message=f"Wrote {frame.height} fits")
    return output, frame.height


def _write_root(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.root")
    branch_types = {
        "index": "int16",
        "amplitude": "float64",
        "integral": "float64",
        "time": "int16",
        "A": "float64",
        "tau": "float64",
        "n": "float64",
        "t0": "float64",
        "run": "int32",
        "event": "int32",
    }
    with uproot.recreate(temporary) as root_file:
        root_file.mktree("tree", branch_types)
        if frame.height:
            root_file["tree"].extend({
                name: frame.get_column(name).to_numpy() for name in GAGG_FIT_COLUMNS
            })
    temporary.replace(path)
