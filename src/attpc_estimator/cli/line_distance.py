from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..process.line_distance import (
    LineDistanceHistogramConfig,
    RansacConfig,
    build_line_distance_histograms,
    default_pointcloud_file_path,
)
from ..storage.run_paths import format_run_id, histogram_dir
from .config import parse_run, parse_toml_config, root_config_values, table_config_values
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
        payload = build_line_distance_histograms(
            pointcloud_file_path=pointcloud_path,
            run=run_id,
            ransac_config=RansacConfig(
                min_samples=int(args.min_samples),
                residual_threshold=float(args.residual_threshold),
                max_trials=int(args.max_trials),
                max_iterations=int(args.max_iterations),
                target_labeled_ratio=float(args.target_labeled_ratio),
                min_inliers=int(args.min_inliers),
                max_start_radius=float(args.max_start_radius),
            ),
            histogram_config=LineDistanceHistogramConfig(),
            progress=progress,
        )

    output_path = histogram_dir(workspace) / f"run_{run_name}_line_distance.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"saved line-distance histograms to {output_path}")
    print(f"processed events: {int(np.asarray(payload['processed_events']).item())}")


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(payload, allowed_keys={"workspace", "run"})
    ransac_config = table_config_values(
        payload,
        table="ransac",
        allowed_keys={
            "min_samples",
            "residual_threshold",
            "max_trials",
            "max_iterations",
            "target_labeled_ratio",
            "min_inliers",
            "max_start_radius",
        },
    )
    parser = argparse.ArgumentParser(
        description="Compute phase-2 pointcloud line-distance histograms for one run",
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
    parser.add_argument(
        "--min-samples",
        type=int,
        default=ransac_config.get("min_samples", 3),
        help="Minimum number of samples required by RANSAC",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=ransac_config.get("residual_threshold", 20.0),
        help="Residual threshold used by RANSAC",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=ransac_config.get("max_trials", 200),
        help="Maximum number of RANSAC trials",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=ransac_config.get("max_iterations", 10),
        help="Maximum number of accepted-cluster search iterations per event",
    )
    parser.add_argument(
        "--target-labeled-ratio",
        type=float,
        default=ransac_config.get("target_labeled_ratio", 0.8),
        help="Stop extracting lines once the labeled point ratio reaches this value",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=ransac_config.get("min_inliers", 20),
        help="Minimum inlier count required to accept a RANSAC cluster",
    )
    parser.add_argument(
        "--max-start-radius",
        type=float,
        default=ransac_config.get("max_start_radius", 40.0),
        help="Reject clusters whose nearest XY radius is not below this value",
    )
    return parser.parse_args()
