from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from numba import njit
from tqdm import tqdm

from .cli_config import parse_toml_config
from ..db import TraceLabelRepository
from .utils import (
    compute_frequency_distribution,
    preprocess_traces,
    sample_cdf_points,
)
from .utils import CDF_THRESHOLDS, CDF_VALUE_BINS, PAD_TRACE_OFFSET

NORMAL_LABEL_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("normal:0", "0 peak", ("0",)),
    ("normal:1", "1 peak", ("1",)),
    ("normal:2", "2 peak", ("2",)),
    ("normal:3", "3 peak", ("3",)),
    ("normal:4+", "4+ peaks", ("4", "5", "6", "7", "8", "9")),
)


def main() -> None:
    args = _parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    db_dir = Path(args.database_dir).expanduser().resolve()
    selected_run = int(args.run) if args.run is not None else None

    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")

    payload = build_labeled_cdf_histograms(
        workspace=workspace,
        db_dir=db_dir,
        run=selected_run,
        baseline_window_scale=args.baseline_window_scale,
    )
    output_path = _resolve_output_path(output_file=args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload)

    histograms = payload["histograms"]
    trace_counts = payload["trace_counts"]
    print(f"saved labeled CDF histograms with shape {histograms.shape} to {output_path}")
    print(f"runs: {payload['run_ids'].tolist()}")
    print(f"labels: {payload['label_titles'].tolist()}")
    print(f"trace counts: {trace_counts.tolist()}")


def _parse_args() -> argparse.Namespace:
    config_path, config = parse_toml_config(
        sys.argv[1:],
        section_names=("batch_labeled", "batch-labeled"),
        allowed_keys={"workspace", "run", "database_dir", "output_file", "baseline_window_scale"},
    )
    parser = argparse.ArgumentParser(
        description="Compute one 2D CDF histogram per trace label from workspace run files and SQLite labels",
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
        "-w",
        "--workspace",
        required="workspace" not in config,
        default=config.get("workspace"),
        help="Workspace directory containing run_<run>.h5 files",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=_parse_run,
        default=config.get("run"),
        help="Optional run identifier. When omitted, aggregate labeled traces across all runs present in both workspace and database.",
    )
    parser.add_argument(
        "-d",
        "--database-dir",
        required="database_dir" not in config,
        default=config.get("database_dir"),
        help="Directory containing trace_label.sqlite3",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=config.get("output_file"),
        help="Required output .npy path.",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=config.get("baseline_window_scale", 20.0),
        help="Baseline-removal filter scale used before taking the FFT",
    )
    return parser.parse_args()


def _parse_run(value: str) -> str:
    run = value.strip()
    if not run or not run.isdigit():
        raise argparse.ArgumentTypeError("run must contain only digits")
    return run


def _resolve_output_path(output_file: str | None) -> Path | None:
    if output_file:
        return Path(output_file).expanduser().resolve()
    return None


def build_labeled_cdf_histograms(
    workspace: Path,
    db_dir: Path,
    run: int | None = None,
    baseline_window_scale: float = 20.0,
) -> dict[str, np.ndarray]:
    workspace_run_files = _collect_workspace_run_files(workspace)
    if run is not None and run not in workspace_run_files:
        raise ValueError(f"input file not found for run {run}: {workspace / f'run_{run}.h5'}")

    repository = TraceLabelRepository(db_dir / "trace_label.sqlite3")
    repository.initialize()
    try:
        labeled_rows = repository.list_labeled_traces(run=run)
        strange_label_names = [row["name"] for row in repository.list_strange_labels()]
    finally:
        repository.connection.close()

    label_keys, label_titles = _build_label_metadata(strange_label_names)
    histograms = np.zeros((len(label_titles), len(CDF_THRESHOLDS), CDF_VALUE_BINS), dtype=np.int64)
    trace_counts = np.zeros(len(label_titles), dtype=np.int64)
    traces_by_run = _group_labeled_traces(
        labeled_rows=labeled_rows,
        strange_label_names=strange_label_names,
        trace_counts=trace_counts,
        workspace_run_files=workspace_run_files,
        selected_run=run,
    )
    run_ids = np.asarray(sorted(traces_by_run), dtype=np.int64)

    if run is not None and run_ids.size == 0:
        run_ids = np.asarray([run], dtype=np.int64)

    total_traces = sum(
        len(grouped_traces)
        for event_map in traces_by_run.values()
        for grouped_traces in event_map.values()
    )

    with tqdm(total=total_traces, desc="Processing labeled pad traces", unit="trace") as progress:
        for run_id in sorted(traces_by_run):
            input_path = workspace_run_files[run_id]
            with h5py.File(input_path, "r") as handle:
                events = handle["events"]
                for event_id in sorted(traces_by_run[run_id]):
                    grouped_traces = sorted(traces_by_run[run_id][event_id], key=lambda item: item[0])
                    trace_ids = np.array([trace_id for trace_id, _ in grouped_traces], dtype=np.int64)
                    label_indices = np.array([label_index for _, label_index in grouped_traces], dtype=np.int64)
                    pads = events[f"event_{event_id}"]["get"]["pads"]
                    traces = np.asarray(pads[trace_ids, PAD_TRACE_OFFSET:], dtype=np.float32)
                    cleaned = preprocess_traces(traces, baseline_window_scale=baseline_window_scale)
                    spectrum = compute_frequency_distribution(cleaned)
                    samples = sample_cdf_points(spectrum, thresholds=CDF_THRESHOLDS)
                    _accumulate_grouped_histograms_numba(samples, label_indices, histograms)
                    progress.update(len(grouped_traces))
                    progress.set_postfix_str(f"run={run_id},event={event_id}")

    return {
        "run_ids": run_ids,
        "label_keys": np.asarray(label_keys, dtype=object),
        "label_titles": np.asarray(label_titles, dtype=object),
        "histograms": histograms,
        "trace_counts": trace_counts,
    }


def _collect_workspace_run_files(workspace: Path) -> dict[int, Path]:
    run_files: dict[int, Path] = {}
    for input_path in sorted(workspace.glob("run_*.h5")):
        run_id = _extract_run_id(input_path)
        if run_id in run_files:
            raise ValueError(f"multiple workspace files resolve to run {run_id}: {run_files[run_id]} and {input_path}")
        run_files[run_id] = input_path.resolve()
    return run_files


def _extract_run_id(input_path: Path) -> int:
    stem = input_path.stem
    prefix, _, run_token = stem.partition("_")
    if prefix != "run" or not run_token.isdigit():
        raise ValueError(f"expected input filename like run_<run>.h5, got {input_path.name}")
    return int(run_token)


def _build_label_metadata(strange_label_names: list[str]) -> tuple[list[str], list[str]]:
    label_keys = [entry[0] for entry in NORMAL_LABEL_GROUPS]
    label_titles = [entry[1] for entry in NORMAL_LABEL_GROUPS]
    label_keys.extend(f"strange:{name}" for name in strange_label_names)
    label_titles.extend(strange_label_names)
    return label_keys, label_titles


def _group_labeled_traces(
    labeled_rows: list[tuple[int, int, int, str, str]],
    strange_label_names: list[str],
    trace_counts: np.ndarray,
    workspace_run_files: dict[int, Path],
    selected_run: int | None,
) -> dict[int, dict[int, list[tuple[int, int]]]]:
    strange_index_map = {name: len(NORMAL_LABEL_GROUPS) + idx for idx, name in enumerate(strange_label_names)}
    grouped: dict[int, dict[int, list[tuple[int, int]]]] = {}

    for run_id, event_id, trace_id, family, label in labeled_rows:
        if run_id not in workspace_run_files:
            continue
        if selected_run is not None and run_id != selected_run:
            continue
        label_index = _resolve_label_index(family=family, label=label, strange_index_map=strange_index_map)
        if label_index is None:
            continue
        grouped.setdefault(run_id, {}).setdefault(event_id, []).append((trace_id, label_index))
        trace_counts[label_index] += 1

    return grouped


def _resolve_label_index(family: str, label: str, strange_index_map: dict[str, int]) -> int | None:
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
