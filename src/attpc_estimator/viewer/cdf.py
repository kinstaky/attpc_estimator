from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from numba import njit
from tqdm import tqdm

from ..db import TraceLabelRepository
from .cli_config import parse_toml_config
from .utils import (
    CDF_THRESHOLDS,
    CDF_VALUE_BINS,
    PAD_TRACE_OFFSET,
    collect_event_counts,
    collect_run_files,
    compute_frequency_distribution,
    db_file,
    preprocess_traces,
    sample_cdf_points,
    trace_file,
)

NORMAL_LABEL_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("normal:0", "0 peak", ("0",)),
    ("normal:1", "1 peak", ("1",)),
    ("normal:2", "2 peaks", ("2",)),
    ("normal:3", "3 peaks", ("3",)),
    ("normal:4+", "4+ peaks", ("4", "5", "6", "7", "8", "9")),
)


def main() -> None:
    args = _parse_args()
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    run_token = args.run
    run_id = int(run_token)

    if args.labeled:
        payload = build_labeled_cdf_histograms(
            trace_path=trace_root,
            workspace=workspace,
            run=run_id,
            baseline_window_scale=args.baseline_window_scale,
        )
        output_path = workspace / f"run_{run_token}_labeled_cdf.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, payload)
        print(
            f"saved labeled CDF histograms with shape {payload['histograms'].shape} to {output_path}"
        )
        print(f"labels: {payload['label_titles'].tolist()}")
        print(f"trace counts: {payload['trace_counts'].tolist()}")
        return

    trace_file_path = trace_file(trace_root, run_token)
    if not trace_file_path.is_file():
        raise SystemExit(f"trace file not found: {trace_file_path}")

    histogram = build_trace_cdf_histogram(
        trace_file_path=trace_file_path,
        baseline_window_scale=args.baseline_window_scale,
    )
    output_path = workspace / f"run_{run_token}_cdf.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, histogram)

    print(f"saved CDF histogram with shape {histogram.shape} to {output_path}")
    print(f"total histogram count: {int(histogram.sum())}")
    print(f"thresholds: {CDF_THRESHOLDS.tolist()}")


def _parse_args() -> argparse.Namespace:
    config_path, config = parse_toml_config(
        sys.argv[1:],
        allowed_keys={
            "trace_path",
            "run",
            "workspace",
            "baseline_window_scale",
            "labeled",
        },
    )
    parser = argparse.ArgumentParser(
        description="Compute CDF histograms for all traces or labeled traces in a single run",
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
        help="Workspace directory containing the labels database and output files",
    )
    parser.add_argument(
        "-r",
        "--run",
        required="run" not in config,
        type=_parse_run,
        default=config.get("run"),
        help="Run identifier",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=config.get("baseline_window_scale", 20.0),
        help="Baseline-removal filter scale used before taking the FFT",
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        default=bool(config.get("labeled", False)),
        help="Build one CDF histogram per trace label for the selected run",
    )
    return parser.parse_args()


def _parse_run(value: str) -> str:
    run = value.strip()
    if not run or not run.isdigit():
        raise argparse.ArgumentTypeError("run must contain only digits")
    return run


def build_trace_cdf_histogram(
    trace_file_path: Path,
    baseline_window_scale: float = 20.0,
    thresholds: np.ndarray = CDF_THRESHOLDS,
) -> np.ndarray:
    with h5py.File(trace_file_path, "r") as handle:
        events = handle["events"]
        min_event = int(events.attrs["min_event"])
        max_event = int(events.attrs["max_event"])
        bad_events = {int(event_id) for event_id in events.attrs["bad_events"]}
        event_counts = collect_event_counts(
            events=events,
            min_event=min_event,
            max_event=max_event,
            bad_events=bad_events,
        )
        total_traces = sum(trace_count for _, trace_count in event_counts)
        histogram = np.zeros((len(thresholds), CDF_VALUE_BINS), dtype=np.int64)

        with tqdm(
            total=total_traces, desc="Processing pad traces", unit="trace"
        ) as progress:
            for event_id, trace_count in event_counts:
                pads = events[f"event_{event_id}"]["get"]["pads"]
                traces = np.asarray(pads[:, PAD_TRACE_OFFSET:], dtype=np.float32)
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=baseline_window_scale
                )
                spectrum = compute_frequency_distribution(cleaned)
                samples = sample_cdf_points(spectrum, thresholds=thresholds)
                _accumulate_cdf_histogram_numba(samples, histogram)
                progress.update(trace_count)
                progress.set_postfix_str(f"event={event_id}")

    return histogram


def build_labeled_cdf_histograms(
    trace_path: Path,
    workspace: Path,
    run: int,
    baseline_window_scale: float = 20.0,
) -> dict[str, np.ndarray]:
    run_files = collect_run_files(trace_path)
    if run not in run_files:
        raise ValueError(
            f"trace file not found for run {run}: {trace_path / f'run_{run}.h5'}"
        )

    repository = TraceLabelRepository(db_file(workspace))
    repository.initialize()
    try:
        labeled_rows = repository.list_labeled_traces(run=run)
        strange_label_names = [row["name"] for row in repository.list_strange_labels()]
    finally:
        repository.connection.close()

    label_keys, label_titles = _build_label_metadata(strange_label_names)
    histograms = np.zeros(
        (len(label_titles), len(CDF_THRESHOLDS), CDF_VALUE_BINS), dtype=np.int64
    )
    trace_counts = np.zeros(len(label_titles), dtype=np.int64)

    grouped = _group_labeled_traces(
        labeled_rows=labeled_rows,
        strange_label_names=strange_label_names,
        trace_counts=trace_counts,
    )

    total_traces = sum(len(grouped_traces) for grouped_traces in grouped.values())
    with h5py.File(run_files[run], "r") as handle:
        events = handle["events"]
        with tqdm(
            total=total_traces, desc="Processing labeled pad traces", unit="trace"
        ) as progress:
            for event_id in sorted(grouped):
                grouped_traces = sorted(grouped[event_id], key=lambda item: item[0])
                trace_ids = np.array(
                    [trace_id for trace_id, _ in grouped_traces], dtype=np.int64
                )
                label_indices = np.array(
                    [label_index for _, label_index in grouped_traces], dtype=np.int64
                )
                pads = events[f"event_{event_id}"]["get"]["pads"]
                traces = np.asarray(
                    pads[trace_ids, PAD_TRACE_OFFSET:], dtype=np.float32
                )
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=baseline_window_scale
                )
                spectrum = compute_frequency_distribution(cleaned)
                samples = sample_cdf_points(spectrum, thresholds=CDF_THRESHOLDS)
                _accumulate_grouped_histograms_numba(samples, label_indices, histograms)
                progress.update(len(grouped_traces))
                progress.set_postfix_str(f"run={run},event={event_id}")

    return {
        "run_id": np.int64(run),
        "label_keys": np.asarray(label_keys, dtype=object),
        "label_titles": np.asarray(label_titles, dtype=object),
        "histograms": histograms,
        "trace_counts": trace_counts,
    }


def _build_label_metadata(
    strange_label_names: list[str],
) -> tuple[list[str], list[str]]:
    label_keys = [entry[0] for entry in NORMAL_LABEL_GROUPS]
    label_titles = [entry[1] for entry in NORMAL_LABEL_GROUPS]
    label_keys.extend(f"strange:{name}" for name in strange_label_names)
    label_titles.extend(strange_label_names)
    return label_keys, label_titles


def _group_labeled_traces(
    labeled_rows: list[tuple[int, int, int, str, str]],
    strange_label_names: list[str],
    trace_counts: np.ndarray,
) -> dict[int, list[tuple[int, int]]]:
    strange_index_map = {
        name: len(NORMAL_LABEL_GROUPS) + idx
        for idx, name in enumerate(strange_label_names)
    }
    grouped: dict[int, list[tuple[int, int]]] = {}

    for _, event_id, trace_id, family, label in labeled_rows:
        label_index = _resolve_label_index(
            family=family, label=label, strange_index_map=strange_index_map
        )
        if label_index is None:
            continue
        grouped.setdefault(event_id, []).append((trace_id, label_index))
        trace_counts[label_index] += 1

    return grouped


def _resolve_label_index(
    family: str, label: str, strange_index_map: dict[str, int]
) -> int | None:
    if family == "normal":
        if label in {"0", "1", "2", "3"}:
            return int(label)
        if label in {"4", "5", "6", "7", "8", "9"}:
            return 4
        return None
    if family == "strange":
        return strange_index_map.get(label)
    return None


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


@njit(cache=True)
def _accumulate_grouped_histograms_numba(
    samples: np.ndarray,
    label_indices: np.ndarray,
    histograms: np.ndarray,
) -> None:
    row_count, column_count = samples.shape
    value_bin_count = histograms.shape[2]

    for row_index in range(row_count):
        label_index = int(label_indices[row_index])
        for column_index in range(column_count):
            value = float(samples[row_index, column_index])
            if value <= 0.0:
                value_bin_index = 0
            elif value >= 1.0:
                value_bin_index = value_bin_count - 1
            else:
                value_bin_index = int(value * value_bin_count)
            histograms[label_index, column_index, value_bin_index] += 1


if __name__ == "__main__":
    main()
