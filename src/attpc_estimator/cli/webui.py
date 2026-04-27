from __future__ import annotations

import argparse
import inspect
import logging
import socket
import sys
from pathlib import Path

import uvicorn

from ..server import create_app
from ..process.line_pipeline import MergeConfig, RansacConfig
from ..service.estimator import EstimatorService
from .config import (
    argument_config_kwargs,
    parse_run,
    parse_toml_config,
    root_config_values,
    table_config_values,
)


def main() -> None:
    args = _parse_args()
    _configure_logging(verbose=bool(args.verbose))
    trace_path = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    default_run = int(args.run) if args.run is not None else None

    if not trace_path.exists():
        raise SystemExit(f"trace path not found: {trace_path}")
    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")

    service_kwargs = {
        "trace_path": trace_path,
        "workspace": workspace,
        "baseline_window_scale": args.baseline_window_scale,
        "bitflip_baseline_threshold": args.bitflip_baseline,
        "saturation_threshold": args.saturation_threshold,
        "saturation_drop_threshold": args.saturation_drop_threshold,
        "saturation_window_radius": args.saturation_window_radius,
        "default_run": default_run,
        "pointcloud_micromegas_time_bucket": args.pointcloud_micromegas_time_bucket,
        "pointcloud_window_time_bucket": args.pointcloud_window_time_bucket,
        "pointcloud_detector_length": args.pointcloud_detector_length,
        "ransac_config": RansacConfig(
            residual_threshold=float(args.residual_threshold),
            max_trials=int(args.max_trials),
            max_iterations=int(args.max_iterations),
            target_labeled_ratio=float(args.target_labeled_ratio),
            min_inliers=int(args.min_inliers),
            max_start_radius=float(args.max_start_radius),
        ),
        "merge_config": MergeConfig(
            distance_threshold=float(args.merge_distance_threshold),
            angle_threshold=float(args.merge_angle_threshold),
        ),
        "verbose": bool(args.verbose),
    }
    accepted_parameters = set(inspect.signature(EstimatorService).parameters)
    try:
        service = EstimatorService(
            **{
                key: value
                for key, value in service_kwargs.items()
                if key in accepted_parameters
            }
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    app = create_app(service, frontend_dist)

    port = _pick_port(args.port)
    url = f"http://0.0.0.0:{port}"
    print(f"WebUI running at {url}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="debug" if args.verbose else "info",
    )


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(
        payload,
        allowed_keys={"trace_path", "workspace", "port", "run"},
    )
    baseline_config = table_config_values(
        payload,
        table="baseline",
        allowed_keys={"fft_window_scale"},
    )
    bitflip_config = table_config_values(
        payload,
        table="bitflip",
        allowed_keys={
            "baseline",
        },
    )
    saturation_config = table_config_values(
        payload,
        table="saturation",
        allowed_keys={"threshold", "drop_threshold", "window_radius"},
    )
    pointcloud_config = table_config_values(
        payload,
        table="pointcloud",
        allowed_keys={"micromegas_time_bucket", "window_time_bucket", "detector_length"},
    )
    ransac_config = table_config_values(
        payload,
        table="findline.ransac",
        allowed_keys={
            "residual_threshold",
            "max_trials",
            "max_iterations",
            "target_labeled_ratio",
            "min_inliers",
            "max_start_radius",
        },
    )
    merge_config = table_config_values(
        payload,
        table="findline.mergeline",
        allowed_keys={"distance_threshold", "angle_threshold"},
    )
    parser = argparse.ArgumentParser(description="Launch the merged web UI app")
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
        **argument_config_kwargs(config, "workspace"),
        help="Workspace directory containing histogram artifacts, filters, and the labels database",
    )
    parser.add_argument(
        "-t",
        "--trace-path",
        **argument_config_kwargs(config, "trace_path"),
        help="Path to a trace file or a directory containing run_<run>.h5 files",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=parse_run,
        default=config.get("run"),
        help="Optional default run shown when the WebUI opens",
    )
    parser.add_argument(
        "--port",
        type=int,
        **argument_config_kwargs(config, "port"),
        help="Preferred local HTTP port",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        **argument_config_kwargs(baseline_config, "fft_window_scale"),
        help="Baseline-removal filter scale used for review/label trace preprocessing",
    )
    parser.add_argument(
        "--bitflip-baseline",
        type=float,
        **argument_config_kwargs(bitflip_config, "baseline"),
        help="Absolute second-derivative threshold used when building filtered bitflip histograms",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        **argument_config_kwargs(saturation_config, "threshold"),
        help="Minimum trace maximum required before building filtered saturation histograms",
    )
    parser.add_argument(
        "--saturation-drop-threshold",
        type=float,
        **argument_config_kwargs(saturation_config, "drop_threshold"),
        help="Drop threshold used when measuring filtered saturation plateau lengths",
    )
    parser.add_argument(
        "--saturation-window-radius",
        type=int,
        **argument_config_kwargs(saturation_config, "window_radius"),
        help="Local window radius used when accumulating filtered saturation drop histograms",
    )
    parser.add_argument(
        "--pointcloud-micromegas-time-bucket",
        type=float,
        default=pointcloud_config.get("micromegas_time_bucket"),
        help="Fallback micromegas_time_bucket for pointcloud files that do not store it in HDF5 attrs",
    )
    parser.add_argument(
        "--pointcloud-window-time-bucket",
        type=float,
        default=pointcloud_config.get("window_time_bucket"),
        help="Fallback window_time_bucket for pointcloud files that do not store it in HDF5 attrs",
    )
    parser.add_argument(
        "--pointcloud-detector-length",
        type=float,
        default=pointcloud_config.get("detector_length"),
        help="Fallback detector_length for pointcloud files that do not store it in HDF5 attrs",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=ransac_config.get("residual_threshold", 20.0),
        help="RANSAC residual threshold used for pointcloud label suggestions",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=ransac_config.get("max_trials", 200),
        help="RANSAC max trials used for pointcloud label suggestions",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=ransac_config.get("max_iterations", 10),
        help="RANSAC max iterations used for pointcloud label suggestions",
    )
    parser.add_argument(
        "--target-labeled-ratio",
        type=float,
        default=ransac_config.get("target_labeled_ratio", 0.8),
        help="RANSAC target labeled ratio used for pointcloud label suggestions",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=ransac_config.get("min_inliers", 20),
        help="RANSAC minimum inliers used for pointcloud label suggestions",
    )
    parser.add_argument(
        "--max-start-radius",
        type=float,
        default=ransac_config.get("max_start_radius", 40.0),
        help="RANSAC start radius used for pointcloud label suggestions",
    )
    parser.add_argument(
        "--merge-distance-threshold",
        type=float,
        default=merge_config.get("distance_threshold", 30.0),
        help="Cluster merge distance threshold used for pointcloud label suggestions",
    )
    parser.add_argument(
        "--merge-angle-threshold",
        type=float,
        default=merge_config.get("angle_threshold", 3.0),
        help="Cluster merge angle threshold used for pointcloud label suggestions",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print backend debug messages to the terminal",
    )
    return parser.parse_args()


def _configure_logging(*, verbose: bool) -> None:
    if not verbose:
        return
    logger = logging.getLogger("attpc_estimator")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def _pick_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred_port))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    main()
