from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..process.cdf import (
    CDF_THRESHOLDS,
    build_labeled_cdf_histograms,
    build_trace_cdf_histogram,
)
from ..storage.run_paths import format_run_id, histogram_dir, resolve_run_file
from .config import (
    parse_run,
    parse_toml_config,
    root_config_values,
    table_config_values,
)
from .progress import tqdm_reporter


def main() -> None:
    args = _parse_args()
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_root = histogram_dir(workspace)
    run_token = args.run
    run_id = int(run_token)
    run_name = format_run_id(run_id)

    if args.labeled:
        with tqdm_reporter("Processing labeled pad traces") as progress:
            payload = build_labeled_cdf_histograms(
                trace_path=trace_root,
                workspace=workspace,
                run=run_id,
                baseline_window_scale=args.baseline_window_scale,
                progress=progress,
            )
        output_path = output_root / f"run_{run_name}_labeled_cdf.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            run_id=payload["run_id"],
            label_keys=payload["label_keys"],
            label_titles=payload["label_titles"],
            histograms=payload["histograms"],
            trace_counts=payload["trace_counts"],
        )
        print(
            f"saved labeled CDF histograms with shape {payload['histograms'].shape} to {output_path}"
        )
        print(f"labels: {payload['label_titles'].tolist()}")
        print(f"trace counts: {payload['trace_counts'].tolist()}")
        return

    try:
        trace_file_path = resolve_run_file(trace_root, run_token)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with tqdm_reporter("Processing pad traces") as progress:
        histogram = build_trace_cdf_histogram(
            trace_file_path=trace_file_path,
            baseline_window_scale=args.baseline_window_scale,
            progress=progress,
        )
    output_path = output_root / f"run_{run_name}_cdf.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, histogram)

    print(f"saved CDF histogram with shape {histogram.shape} to {output_path}")
    print(f"total histogram count: {int(histogram.sum())}")
    print(f"thresholds: {CDF_THRESHOLDS.tolist()}")


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(
        payload,
        allowed_keys={"trace_path", "run", "workspace"},
    )
    cdf_config = table_config_values(
        payload,
        table="cdf",
        allowed_keys={"baseline_window_scale", "labeled"},
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
        type=parse_run,
        default=config.get("run"),
        help="Run identifier",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=cdf_config.get("baseline_window_scale", 20.0),
        help="Baseline-removal filter scale used before taking the FFT",
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        default=bool(cdf_config.get("labeled", False)),
        help="Build one CDF histogram per trace label for the selected run",
    )
    return parser.parse_args()
