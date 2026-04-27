from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..process.relabel import (
    BITFLIP_BASELINE_DEFAULT,
    BITFLIP_FILTER_MIN_COUNT_DEFAULT,
    RELABEL_LABEL_CHOICES,
    SATURATION_DROP_THRESHOLD_DEFAULT,
    SATURATION_RELABEL_MIN_PLATEAU_LENGTH_DEFAULT,
    build_relabel_rows,
    confused_trace_key_sections_for_label,
    print_ratio,
    ratio_items_for_label,
)
from ..process.filter_core import (
    SATURATION_THRESHOLD_DEFAULT,
    SATURATION_WINDOW_RADIUS_DEFAULT,
)
from ..storage.run_paths import format_run_id
from .config import (
    argument_config_kwargs,
    parse_run,
    parse_toml_config,
    root_config_values,
    table_config_values,
)
from .progress import tqdm_reporter


def main() -> None:
    args = _parse_args()
    trace_path = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    run_token = args.run
    selected_run = int(run_token) if run_token is not None else None
    run_name = format_run_id(selected_run) if selected_run is not None else None

    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")

    with tqdm_reporter("Relabeling traces") as progress:
        rows, metrics = build_relabel_rows(
            trace_path=trace_path,
            workspace=workspace,
            run=selected_run,
            label=args.label,
            baseline_window_scale=args.baseline_window_scale,
            peak_separation=args.peak_separation,
            peak_prominence=args.peak_prominence,
            peak_width=args.peak_width,
            bitflip_baseline_threshold=args.bitflip_baseline,
            bitflip_min_count=args.bitflip_min_count,
            saturation_threshold=args.saturation_threshold,
            saturation_drop_threshold=args.saturation_drop_threshold,
            saturation_window_radius=args.saturation_window_radius,
            saturation_min_plateau_length=args.saturation_min_plateau_length,
            progress=progress,
        )

    if run_token is None:
        output_path = workspace / "labeled_relabel.npy"
    else:
        output_path = workspace / f"run_{run_name}_labeled_relabel.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, rows)

    print(f"saved relabel rows with shape {rows.shape} to {output_path}")
    print(f"total traces: {len(rows)}")
    for name, ratio in ratio_items_for_label(args.label, metrics):
        print_ratio(name, ratio)
    for title, trace_keys in confused_trace_key_sections_for_label(args.label, rows):
        print(title)
        if not trace_keys:
            print("none")
            continue
        for run, event_id, trace_id in trace_keys:
            print(f"{run}/{event_id}/{trace_id}")


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(
        payload,
        allowed_keys={"trace_path", "workspace", "run"},
    )
    relabel_config = table_config_values(
        payload,
        table="relabel",
        allowed_keys={"label", "baseline_window_scale"},
    )
    amplitude_config = table_config_values(
        payload,
        table="amplitude",
        allowed_keys={"peak_separation", "peak_prominence", "peak_width"},
    )
    bitflip_config = table_config_values(
        payload,
        table="bitflip",
        allowed_keys={"baseline", "min_count"},
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
    parser = argparse.ArgumentParser(
        description="Relabel labeled traces using amplitude, bitflip, and saturation heuristics",
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
        **argument_config_kwargs(config, "trace_path"),
        help="Path to a trace file or a directory containing run_<run>.h5 files",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        **argument_config_kwargs(config, "workspace"),
        help="Workspace directory containing the SQLite labels database and outputs",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=parse_run,
        default=config.get("run"),
        help="Optional run identifier. When omitted, relabel labeled traces across all runs present in both workspace and database.",
    )
    parser.add_argument(
        "--label",
        choices=RELABEL_LABEL_CHOICES,
        required="label" not in relabel_config,
        default=relabel_config.get("label"),
        help="Relabel mode: noise uses the current zero-peak heuristic, oscillation uses the bitflip rule, saturation uses the saturation plateau rule.",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        **argument_config_kwargs(relabel_config, "baseline_window_scale"),
        help="Baseline-removal filter scale used before taking the FFT",
    )
    parser.add_argument(
        "--peak-separation",
        type=float,
        **argument_config_kwargs(amplitude_config, "peak_separation"),
        help="Minimum separation between peaks",
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        **argument_config_kwargs(amplitude_config, "peak_prominence"),
        help="Prominence of peaks",
    )
    parser.add_argument(
        "--peak-width",
        type=float,
        **argument_config_kwargs(amplitude_config, "peak_width"),
        help="Maximum width of peaks",
    )
    parser.add_argument(
        "--bitflip-baseline",
        type=float,
        **argument_config_kwargs(bitflip_config, "baseline"),
        help="Absolute second-derivative threshold used to classify baseline points for oscillation relabeling",
    )
    parser.add_argument(
        "--bitflip-min-count",
        type=int,
        **argument_config_kwargs(bitflip_config, "min_count"),
        help="Minimum number of qualified bitflip segments required to relabel a trace as oscillation",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        **argument_config_kwargs(saturation_config, "threshold"),
        help="Minimum trace maximum required before evaluating saturation relabeling",
    )
    parser.add_argument(
        "--saturation-drop-threshold",
        type=float,
        **argument_config_kwargs(saturation_config, "drop_threshold"),
        help="Maximum drop from the local maximum when measuring saturation plateau length for relabeling",
    )
    parser.add_argument(
        "--saturation-window-radius",
        type=int,
        **argument_config_kwargs(saturation_config, "window_radius"),
        help="Local window radius used when measuring saturation drops for relabeling",
    )
    parser.add_argument(
        "--saturation-min-plateau-length",
        type=int,
        **argument_config_kwargs(saturation_config, "min_plateau_length"),
        help="Minimum plateau length required to relabel a trace as saturation",
    )
    return parser.parse_args()
