from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..process.saturation import (
    build_labeled_saturation_histograms,
    build_saturation_histograms,
)
from ..storage.run_paths import format_run_id, resolve_run_file
from .config import parse_run, parse_toml_config, root_config_values, table_config_values
from .progress import tqdm_reporter


def main() -> None:
    args = _parse_args()
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    run_token = args.run
    run_id = int(run_token)
    run_name = format_run_id(run_id)

    if args.labeled:
        with tqdm_reporter("Processing labeled pad traces") as progress:
            payload = build_labeled_saturation_histograms(
                trace_path=trace_root,
                workspace=workspace,
                run=run_id,
                baseline_window_scale=args.baseline_window_scale,
                threshold=args.threshold,
                drop_threshold=args.drop_threshold,
                window_radius=args.window_radius,
                progress=progress,
            )
        output_path = workspace / f"run_{run_name}_labeled_saturation.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            run_id=payload["run_id"],
            label_keys=payload["label_keys"],
            label_titles=payload["label_titles"],
            trace_counts=payload["trace_counts"],
            drop_histograms=payload["drop_histograms"],
            length_histograms=payload["length_histograms"],
        )
        print(f"saved labeled saturation histograms to {output_path}")
        print(f"labels: {payload['label_keys'].tolist()}")
        print(f"trace counts: {payload['trace_counts'].tolist()}")
        return

    try:
        trace_file_path = resolve_run_file(trace_root, run_token)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with tqdm_reporter("Processing pad traces") as progress:
        payload = build_saturation_histograms(
            trace_file_path=trace_file_path,
            baseline_window_scale=args.baseline_window_scale,
            threshold=args.threshold,
            drop_threshold=args.drop_threshold,
            window_radius=args.window_radius,
            progress=progress,
        )
    output_path = workspace / f"run_{run_name}_saturation.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        trace_count=payload["trace_count"],
        drop_histogram=payload["drop_histogram"],
        length_histogram=payload["length_histogram"],
    )
    print(f"saved saturation histograms to {output_path}")
    print(f"trace count: {int(payload['trace_count'])}")


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(
        payload,
        allowed_keys={"trace_path", "run", "workspace"},
    )
    saturation_config = table_config_values(
        payload,
        table="saturation",
        allowed_keys={
            "threshold",
            "drop_threshold",
            "window_radius",
            "min_plateau_length",
        },
    )
    baseline_config = table_config_values(
        payload,
        table="baseline",
        allowed_keys={"fft_window_scale"},
    )
    parser = argparse.ArgumentParser(
        description="Compute saturation plateau histograms for all traces or labeled traces",
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
        type=parse_run,
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
        default=baseline_config.get("fft_window_scale", 10.0),
        help="Baseline-removal filter scale used before saturation analysis",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=saturation_config.get("threshold", 2000.0),
        help="Minimum trace maximum required before evaluating a saturation plateau",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        required="drop_threshold" not in saturation_config,
        default=saturation_config.get("drop_threshold"),
        help="Drop threshold D used when measuring plateau length",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=saturation_config.get("window_radius", 16),
        help="Radius of the local window used when accumulating drop-from-maximum statistics",
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        default=False,
        help="Process only labeled traces for the selected run and save one histogram per label",
    )
    return parser.parse_args()
