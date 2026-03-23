from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from numba import njit
from tqdm import tqdm

from .cli_config import parse_toml_config
from .utils import (
	CDF_THRESHOLDS,
	CDF_VALUE_BINS,
	PAD_TRACE_OFFSET,
    preprocess_traces,
    sample_cdf_points,
    compute_frequency_distribution,
)

def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    output_path = _resolve_output_path(input_path=input_path, output_file=args.output_file)

    if not input_path.is_file():
        raise SystemExit(f"input file not found: {input_path}")

    histogram = build_trace_cdf_histogram(
        input_path=input_path,
        baseline_window_scale=args.baseline_window_scale,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, histogram)

    print(f"saved CDF histogram with shape {histogram.shape} to {output_path}")
    print(f"total histogram count: {int(histogram.sum())}")
    print(f"thresholds: {CDF_THRESHOLDS.tolist()}")


def _parse_args() -> argparse.Namespace:
    config_path, config = parse_toml_config(
        sys.argv[1:],
        section_names=("batch",),
        allowed_keys={"input_file", "output_file", "baseline_window_scale"},
    )
    parser = argparse.ArgumentParser(
        description="Compute a 2D histogram of transformed CDF values for all pad traces",
    )
    parser.add_argument(
        "-c",
        "--connfig",
        "--config",
        dest="config_file",
        default=str(config_path),
        help="Path to a TOML config file. Defaults to config.toml.",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required="input_file" not in config,
        default=config.get("input_file"),
        help="Path to the trace input file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=config.get("output_file"),
        help="Optional output .npy path. Defaults to <input-stem>_cdf_hist2d.npy next to the input file.",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=config.get("baseline_window_scale", 20.0),
        help="Baseline-removal filter scale used before taking the FFT",
    )
    return parser.parse_args()


def _resolve_output_path(input_path: Path, output_file: str | None) -> Path:
    if output_file:
        return Path(output_file).expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_cdf_hist2d.npy")


def build_trace_cdf_histogram(
    input_path: Path,
    baseline_window_scale: float = 20.0,
    thresholds: np.ndarray = CDF_THRESHOLDS,
) -> np.ndarray:
    with h5py.File(input_path, "r") as handle:
        events = handle["events"]
        min_event = int(events.attrs["min_event"])
        max_event = int(events.attrs["max_event"])
        bad_events = {int(event_id) for event_id in events.attrs["bad_events"]}
        event_counts = _collect_event_counts(events=events, min_event=min_event, max_event=max_event, bad_events=bad_events)
        total_traces = sum(trace_count for _, trace_count in event_counts)
        histogram = np.zeros((len(thresholds), CDF_VALUE_BINS), dtype=np.int64)

        with tqdm(total=total_traces, desc="Processing pad traces", unit="trace") as progress:
            for event_id, trace_count in event_counts:
                pads = events[f"event_{event_id}"]["get"]["pads"]
                traces = np.asarray(pads[:, PAD_TRACE_OFFSET:], dtype=np.float32)
                cleaned = preprocess_traces(traces, baseline_window_scale=baseline_window_scale)
                spectrum = compute_frequency_distribution(cleaned)
                samples = sample_cdf_points(spectrum, thresholds=thresholds)
                _accumulate_cdf_histogram_numba(samples, histogram)
                progress.update(trace_count)
                progress.set_postfix_str(f"event={event_id}")

    return histogram


def _collect_event_counts(
    events: h5py.Group,
    min_event: int,
    max_event: int,
    bad_events: set[int],
) -> list[tuple[int, int]]:
    event_counts: list[tuple[int, int]] = []
    for event_id in range(min_event, max_event + 1):
        if event_id in bad_events:
            continue
        pads = events[f"event_{event_id}"]["get"]["pads"]
        trace_count = int(pads.shape[0])
        if trace_count > 0:
            event_counts.append((event_id, trace_count))
    return event_counts



@njit(cache=True)
def _accumulate_cdf_histogram_numba(samples: np.ndarray, histogram: np.ndarray) -> None:
    row_count, column_count = samples.shape
    value_bin_count = histogram.shape[1]

    for row_index in range(row_count):
        for column_index in range(column_count):
            value = float(samples[row_index, column_index])
            if value <= 0.0:
                value_bin_index = 0
            elif value >= 1.0:
                value_bin_index = value_bin_count - 1
            else:
                value_bin_index = int(value * value_bin_count)

            histogram[column_index, value_bin_index] += 1


if __name__ == "__main__":
    main()
