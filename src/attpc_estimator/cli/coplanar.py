from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..process.coplanar import (
    build_coplanar_histogram,
    default_pointcloud_file_path,
)
from ..storage.run_paths import format_run_id, histogram_dir
from .config import parse_run, parse_toml_config, root_config_values
from .progress import tqdm_reporter


def main() -> None:
    args = _parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    run_id = int(args.run)
    run_name = format_run_id(run_id)
    pointcloud_path = default_pointcloud_file_path(workspace, run_id)
    if not pointcloud_path.is_file():
        raise SystemExit(f"pointcloud file not found: {pointcloud_path}")

    with tqdm_reporter("Processing phase-2 pointcloud events") as progress:
        payload = build_coplanar_histogram(
            pointcloud_file_path=pointcloud_path,
            run=run_id,
            progress=progress,
        )

    output_path = histogram_dir(workspace) / f"run_{run_name}_coplanar.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"saved coplanar histogram to {output_path}")
    print(f"processed events: {int(np.asarray(payload['processed_events']).item())}")
    print(f"accepted events: {int(np.asarray(payload['accepted_events']).item())}")
    print(f"skipped events: {int(np.asarray(payload['skipped_events']).item())}")
    valid_events = int(np.asarray(payload["valid_events"]).item())
    thresholds = np.asarray(payload["ratio_thresholds"], dtype=np.float64)
    counts = np.asarray(payload["ratio_threshold_counts"], dtype=np.int64)
    for threshold, count in zip(thresholds.tolist(), counts.tolist(), strict=True):
        ratio = (float(count) / float(valid_events)) if valid_events > 0 else 0.0
        print(f"ratio under {threshold:g}: {count}/{valid_events} ({ratio:.6f})")


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(payload, allowed_keys={"workspace", "run"})
    parser = argparse.ArgumentParser(
        description="Compute phase-2 pointcloud coplanarity histograms for one run",
    )
    parser.add_argument(
        "-c",
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
        help="Workspace directory containing pointcloud data and histogram outputs",
    )
    parser.add_argument(
        "-r",
        "--run",
        required="run" not in config,
        type=parse_run,
        default=config.get("run"),
        help="Run number",
    )
    return parser.parse_args()
