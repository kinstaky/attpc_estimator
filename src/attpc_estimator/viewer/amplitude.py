from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy import signal
from tqdm import tqdm

from .cli_config import parse_toml_config
from .utils import (
    PAD_TRACE_OFFSET,
    collect_event_counts,
    preprocess_traces,
    read_labeled_trace,
    trace_file,
)

AMPLITUDE_BIN_COUNT = 8192


def main() -> None:
    args = _parse_args()
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    run_token = args.run
    run_id = int(run_token)

    if args.labeled:
        payload = build_labeled_amplitude_histograms(
            trace_path=trace_root,
            workspace=workspace,
            run=run_id,
            baseline_window_scale=args.baseline_window_scale,
            peak_separation=args.peak_separation,
            peak_prominence=args.peak_prominence,
            peak_width=args.peak_width,
        )
        output_path = workspace / f"run_{run_token}_labeled_amp.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, payload)
        print(f"saved labeled amplitude histograms to {output_path}")
        print(f"labels: {payload['label_keys'].tolist()}")
        print(f"trace counts: {payload['trace_counts'].tolist()}")
        return

    trace_file_path = trace_file(trace_root, run_token)
    if not trace_file_path.is_file():
        raise SystemExit(f"trace file not found: {trace_file_path}")

    histogram = build_amplitude_histogram(
        trace_file_path=trace_file_path,
        baseline_window_scale=args.baseline_window_scale,
        peak_separation=args.peak_separation,
        peak_prominence=args.peak_prominence,
        peak_width=args.peak_width,
    )
    output_path = workspace / f"run_{run_token}_amp.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, histogram)

    print(f"saved amplitude histogram with shape {histogram.shape} to {output_path}")
    print(f"total histogram count: {int(histogram.sum())}")


def _parse_args() -> argparse.Namespace:
    config_path, config = parse_toml_config(
        sys.argv[1:],
        allowed_keys={
            "trace_path",
            "run",
            "workspace",
            "baseline_window_scale",
            "peak_separation",
            "peak_prominence",
            "peak_width",
            "labeled",
        },
    )
    parser = argparse.ArgumentParser(
        description="Compute peak-amplitude histograms for all traces or labeled traces",
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
        "-r",
        "--run",
        required="run" not in config,
        default=config.get("run"),
        help="Run number",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        required="workspace" not in config,
        default=config.get("workspace"),
        help="Path to store result files and locate the labels database",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=config.get("baseline_window_scale", 10.0),
        help="Baseline-removal filter scale used before taking the FFT",
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
        "--labeled",
        action="store_true",
        default=bool(config.get("labeled", False)),
        help="Process only labeled traces for the selected run and save one histogram per label",
    )
    return parser.parse_args()


def build_amplitude_histogram(
    trace_file_path: Path,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
) -> np.ndarray:
    with h5py.File(trace_file_path, "r") as handle:
        events = handle["events"]
        min_event = int(events.attrs["min_event"])
        max_event = int(events.attrs["max_event"])
        bad_events = {int(event_id) for event_id in events.attrs["bad_events"]}
        event_counts = collect_event_counts(events=events, min_event=min_event, max_event=max_event, bad_events=bad_events)
        total_traces = sum(trace_count for _, trace_count in event_counts)
        histogram = np.zeros(AMPLITUDE_BIN_COUNT, dtype=np.int64)

        with tqdm(total=total_traces, desc="Processing pad traces", unit="trace") as progress:
            for event_id, trace_count in event_counts:
                pads = events[f"event_{event_id}"]["get"]["pads"]
                traces = np.asarray(pads[:, PAD_TRACE_OFFSET:], dtype=np.float32)
                cleaned = preprocess_traces(traces, baseline_window_scale=baseline_window_scale)
                for row in cleaned:
                    _accumulate_peak_histogram(
                        row=row,
                        histogram=histogram,
                        peak_separation=peak_separation,
                        peak_prominence=peak_prominence,
                        peak_width=peak_width,
                    )
                progress.update(trace_count)
                progress.set_postfix_str(f"event={event_id}")

    return histogram


def build_labeled_amplitude_histograms(
    trace_path: Path,
    workspace: Path,
    run: int,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
) -> dict[str, np.ndarray]:
    traces, label_keys = read_labeled_trace(trace_path=trace_path, workspace_path=workspace, run=run)
    if len(label_keys) == 0:
        return {
            "run_id": np.int64(run),
            "label_keys": np.asarray([], dtype=object),
            "trace_counts": np.asarray([], dtype=np.int64),
        }

    cleaned = preprocess_traces(traces, baseline_window_scale=baseline_window_scale)
    unique_label_keys = sorted(set(label_keys))
    histograms = {label_key: np.zeros(AMPLITUDE_BIN_COUNT, dtype=np.int64) for label_key in unique_label_keys}
    trace_counts = np.zeros(len(unique_label_keys), dtype=np.int64)
    label_index_map = {label_key: index for index, label_key in enumerate(unique_label_keys)}

    with tqdm(total=len(label_keys), desc="Processing labeled pad traces", unit="trace") as progress:
        for row, label_key in zip(cleaned, label_keys, strict=True):
            _accumulate_peak_histogram(
                row=row,
                histogram=histograms[label_key],
                peak_separation=peak_separation,
                peak_prominence=peak_prominence,
                peak_width=peak_width,
            )
            trace_counts[label_index_map[label_key]] += 1
            progress.update(1)

    payload: dict[str, np.ndarray] = {
        "run_id": np.int64(run),
        "label_keys": np.asarray(unique_label_keys, dtype=object),
        "trace_counts": trace_counts,
    }
    payload.update(histograms)
    return payload


def max_peak_amplitude(
    row: np.ndarray,
    peak_separation: float,
    peak_prominence: float,
    peak_width: float,
) -> float:
    peaks, _ = signal.find_peaks(
        row,
        distance=peak_separation,
        prominence=peak_prominence,
        width=(1.0, peak_width),
        rel_height=0.95,
    )
    if peaks.size == 0:
        return 0.0
    return float(np.max(row[peaks]))


def _accumulate_peak_histogram(
    row: np.ndarray,
    histogram: np.ndarray,
    peak_separation: float,
    peak_prominence: float,
    peak_width: float,
) -> None:
    peaks, _ = signal.find_peaks(
        row,
        distance=peak_separation,
        prominence=peak_prominence,
        width=(1.0, peak_width),
        rel_height=0.95,
    )
    for peak in peaks:
        amplitude = int(np.clip(row[peak], 0, histogram.shape[0] - 1))
        histogram[amplitude] += 1


if __name__ == "__main__":
    main()
