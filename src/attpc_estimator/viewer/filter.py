from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from tqdm import tqdm

from .amplitude import max_peak_amplitude
from .cli_config import parse_toml_config
from .utils import (
    PAD_TRACE_OFFSET,
    collect_run_files,
    compute_frequency_distribution,
    preprocess_traces,
    sample_cdf_points,
)

DEFAULT_TRACE_LIMIT = 1000
OSCILLATION_CDF_BIN = 60
OSCILLATION_CUTOFF = 0.6
UNLIMITED_TRACE_LIMIT = -1


def main() -> None:
    args = _parse_args()
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    run_token = args.run
    run_id = int(run_token)
    amplitude_range = _normalize_amplitude_range(args.amplitude)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else workspace / _default_output_name(run_token, amplitude_range, args.oscillation)
    )

    rows = build_filter_rows(
        trace_path=trace_root,
        run=run_id,
        amplitude_range=amplitude_range,
        oscillation=args.oscillation,
        baseline_window_scale=args.baseline_window_scale,
        peak_separation=args.peak_separation,
        peak_prominence=args.peak_prominence,
        peak_width=args.peak_width,
        limit=args.limit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, rows)

    print(f"saved {len(rows)} filter rows to {output_path}")


def _parse_args() -> argparse.Namespace:
    config_path, config = parse_toml_config(
        sys.argv[1:],
        allowed_keys={
            "trace_path",
            "workspace",
            "run",
            "amplitude",
            "oscillation",
            "unlimit",
            "baseline_window_scale",
            "peak_separation",
            "peak_prominence",
            "peak_width",
            "limit",
            "output",
        },
    )
    parser = argparse.ArgumentParser(
        description="Generate a filter file containing the first matching traces in one run",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=str(config_path),
        help="Path to a TOML config file. Defaults to config.toml.",
    )
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
        help="Workspace directory where the filter file will be written",
    )
    parser.add_argument(
        "-r",
        "--run",
        required="run" not in config,
        type=_parse_run,
        default=config.get("run"),
        help="Run identifier to filter",
    )
    parser.add_argument(
        "--amplitude",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=config.get("amplitude"),
        help="Inclusive lower and upper bounds for the highest detected peak amplitude",
    )
    parser.add_argument(
        "--oscillation",
        action="store_true",
        default=bool(config.get("oscillation", False)),
        help=f"Keep traces whose CDF F({OSCILLATION_CDF_BIN}) is below {OSCILLATION_CUTOFF}",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=config.get("baseline_window_scale", 10.0),
        help="Baseline-removal filter scale used before peak detection and FFT",
    )
    parser.add_argument(
        "--peak-separation",
        type=float,
        default=config.get("peak_separation", 50.0),
        help="Minimum separation between peaks",
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=config.get("peak_prominence", 20.0),
        help="Prominence of peaks",
    )
    parser.add_argument(
        "--peak-width",
        type=float,
        default=config.get("peak_width", 50.0),
        help="Maximum width of peaks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.get("limit", DEFAULT_TRACE_LIMIT),
        help=f"Maximum number of matching traces to keep, default {DEFAULT_TRACE_LIMIT}",
    )
    parser.add_argument(
        "--unlimit",
        action="store_true",
        default=bool(config.get("unlimit", False)),
        help="Disable the row limit and keep every matching trace in the selected run",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=config.get("output"),
        help="Optional explicit output .npy path",
    )
    args = parser.parse_args()
    if args.amplitude is None and not args.oscillation:
        parser.error("at least one filter criterion is required: --amplitude MIN MAX and/or --oscillation")
    if args.unlimit:
        args.limit = UNLIMITED_TRACE_LIMIT
    return args


def build_filter_rows(
    trace_path: Path,
    run: int,
    amplitude_range: tuple[float, float] | None = None,
    oscillation: bool = False,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
    limit: int = DEFAULT_TRACE_LIMIT,
) -> np.ndarray:
    if amplitude_range is None and not oscillation:
        raise ValueError("at least one filter criterion is required")
    if amplitude_range is not None and amplitude_range[0] > amplitude_range[1]:
        raise ValueError("amplitude minimum must be less than or equal to the maximum")
    if limit == 0 or limit < UNLIMITED_TRACE_LIMIT:
        raise ValueError("limit must be positive or use --unlimit")

    run_files = collect_run_files(trace_path)
    if run not in run_files:
        raise ValueError(f"trace file not found for run {run}: {trace_path / f'run_{run}.h5'}")

    selected_rows: list[tuple[int, int, int]] = []
    unlimited = limit == UNLIMITED_TRACE_LIMIT
    with tqdm(total=1, desc="Scanning run", unit="run") as progress:
        with h5py.File(run_files[run], "r") as handle:
            events = handle["events"]
            min_event = int(events.attrs["min_event"])
            max_event = int(events.attrs["max_event"])
            bad_events = {int(event_id) for event_id in events.attrs["bad_events"]}

            for event_id in range(min_event, max_event + 1):
                if event_id in bad_events or (not unlimited and len(selected_rows) >= limit):
                    continue
                pads = events[f"event_{event_id}"]["get"]["pads"]
                if pads.shape[0] == 0:
                    continue

                traces = np.asarray(pads[:, PAD_TRACE_OFFSET:], dtype=np.float32)
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=baseline_window_scale
                )
                oscillation_values = _compute_oscillation_values(cleaned) if oscillation else None

                for trace_id, row in enumerate(cleaned):
                    if not _matches_filter(
                        row=row,
                        trace_id=trace_id,
                        amplitude_range=amplitude_range,
                        oscillation=oscillation,
                        oscillation_values=oscillation_values,
                        peak_separation=peak_separation,
                        peak_prominence=peak_prominence,
                        peak_width=peak_width,
                    ):
                        continue
                    selected_rows.append((run, event_id, trace_id))
                    if not unlimited and len(selected_rows) >= limit:
                        break
        progress.update(1)
        progress.set_postfix_str(f"run={run},selected={len(selected_rows)}")

    if not selected_rows:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(selected_rows, dtype=np.int64)


def build_amplitude_filter_rows(
    trace_path: Path,
    run: int,
    min_amplitude: float,
    max_amplitude: float,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
    limit: int = DEFAULT_TRACE_LIMIT,
) -> np.ndarray:
    return build_filter_rows(
        trace_path=trace_path,
        run=run,
        amplitude_range=(min_amplitude, max_amplitude),
        oscillation=False,
        baseline_window_scale=baseline_window_scale,
        peak_separation=peak_separation,
        peak_prominence=peak_prominence,
        peak_width=peak_width,
        limit=limit,
    )


def _compute_oscillation_values(cleaned: np.ndarray) -> np.ndarray:
    spectrum = compute_frequency_distribution(cleaned)
    return sample_cdf_points(
        spectrum,
        thresholds=np.asarray([OSCILLATION_CDF_BIN], dtype=np.int64),
    )[:, 0]


def _matches_filter(
    *,
    row: np.ndarray,
    trace_id: int,
    amplitude_range: tuple[float, float] | None,
    oscillation: bool,
    oscillation_values: np.ndarray | None,
    peak_separation: float,
    peak_prominence: float,
    peak_width: float,
) -> bool:
    if oscillation:
        if oscillation_values is None:
            return False
        if float(oscillation_values[trace_id]) >= OSCILLATION_CUTOFF:
            return False

    if amplitude_range is not None:
        amplitude = max_peak_amplitude(
            row=row,
            peak_separation=peak_separation,
            peak_prominence=peak_prominence,
            peak_width=peak_width,
        )
        if not (amplitude_range[0] <= amplitude <= amplitude_range[1]):
            return False

    return True


def _parse_run(value: str) -> str:
    run = value.strip()
    if not run or not run.isdigit():
        raise argparse.ArgumentTypeError("run must contain only digits")
    return run


def _normalize_amplitude_range(
    amplitude: list[float] | tuple[float, float] | None,
) -> tuple[float, float] | None:
    if amplitude is None:
        return None
    if len(amplitude) != 2:
        raise ValueError("amplitude must contain two values: minimum and maximum")
    minimum = float(amplitude[0])
    maximum = float(amplitude[1])
    if minimum > maximum:
        raise ValueError("amplitude minimum must be less than or equal to the maximum")
    return minimum, maximum


def _default_output_name(
    run_token: str,
    amplitude_range: tuple[float, float] | None,
    oscillation: bool,
) -> str:
    parts = [f"filter_run_{run_token}"]
    if oscillation:
        parts.append("oscillation")
    if amplitude_range is not None:
        parts.append(
            f"amp_{_format_bound(amplitude_range[0])}_{_format_bound(amplitude_range[1])}"
        )
    return "_".join(parts) + ".npy"


def _format_bound(value: float) -> str:
    token = f"{value:g}"
    return token.replace("-", "neg").replace(".", "p")


if __name__ == "__main__":
    main()
