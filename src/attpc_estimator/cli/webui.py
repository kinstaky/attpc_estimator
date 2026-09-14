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
from ..process.gagg_parameters import load_gagg_fit_parameters
from ..service.estimator import EstimatorService
from .config import (
    argument_config_kwargs,
    config_relative_path,
    parse_toml_config,
    parse_run,
    root_config_values,
    table_config_entries,
    table_config_values,
)
from ..utils.gagg_processing import build_gagg_trace_rules


def main() -> None:
    args = _parse_args()
    _configure_logging(verbose=bool(args.verbose))
    trace_path = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    default_run = int(args.run) if args.run is not None else None
    try:
        gagg_fit_parameters = (
            load_gagg_fit_parameters(args.gagg_fit_parameters_path)
            if args.gagg_fit_parameters_path is not None
            else None
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    gagg_trace_rules = build_gagg_trace_rules(
        baseline_entries=args.gagg_baseline_entries
        or [{"fft_window_scale": args.gagg_baseline_window_scale}],
        amplitude_entries=args.gagg_amplitude_entries
        or [{
            "peak_separation": args.gagg_peak_separation,
            "peak_prominence": args.gagg_peak_prominence,
            "peak_width": args.gagg_peak_width,
            "peak_threshold": args.gagg_peak_threshold,
            "rel_height": args.gagg_peak_rel_height,
        }],
        baseline_override=(
            args.gagg_baseline_window_scale
            if _option_supplied("--gagg-baseline-window-scale")
            else None
        ),
        peak_overrides={
            field: float(getattr(args, argument))
            for field, argument, option in (
                ("peak_separation", "gagg_peak_separation", "--gagg-peak-separation"),
                ("peak_prominence", "gagg_peak_prominence", "--gagg-peak-prominence"),
                ("peak_width", "gagg_peak_width", "--gagg-peak-width"),
                ("peak_threshold", "gagg_peak_threshold", "--gagg-peak-threshold"),
                ("rel_height", "gagg_peak_rel_height", "--gagg-peak-rel-height"),
            )
            if _option_supplied(option)
        },
    )

    if not trace_path.exists():
        raise SystemExit(f"trace path not found: {trace_path}")
    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")

    service_kwargs = {
        "trace_path": trace_path,
        "workspace": workspace,
        "baseline_window_scale": args.baseline_window_scale,
        "ic_baseline_window_scale": args.ic_baseline_window_scale,
        "detector_baseline_window_scales": {
            "IC": args.ic_baseline_window_scale,
            "SI": args.si_baseline_window_scale,
            "GAGG": args.gagg_baseline_window_scale,
        },
        "peak_separation": args.peak_separation,
        "peak_prominence": args.peak_prominence,
        "peak_width": args.peak_width,
        "peak_threshold": args.peak_threshold,
        "peak_rel_height": args.peak_rel_height,
        "detector_peak_configs": {
            "IC": {
                "peak_separation": args.ic_peak_separation,
                "peak_prominence": args.ic_peak_prominence,
                "peak_width": args.ic_peak_width,
                "peak_threshold": args.ic_peak_threshold,
                "peak_rel_height": args.ic_peak_rel_height,
            },
            "SI": {
                "peak_separation": args.si_peak_separation,
                "peak_prominence": args.si_peak_prominence,
                "peak_width": args.si_peak_width,
                "peak_threshold": args.si_peak_threshold,
                "peak_rel_height": args.si_peak_rel_height,
            },
            "GAGG": {
                "peak_separation": args.gagg_peak_separation,
                "peak_prominence": args.gagg_peak_prominence,
                "peak_width": args.gagg_peak_width,
                "peak_threshold": args.gagg_peak_threshold,
                "peak_rel_height": args.gagg_peak_rel_height,
            },
        },
        "gagg_trace_rules": gagg_trace_rules,
        "gagg_fit_parameters": gagg_fit_parameters,
        "gagg_valid_indexes": args.gagg_valid_indexes,
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
        table="attpc.baseline",
        allowed_keys={"fft_window_scale"},
    )
    ic_baseline_config = table_config_values(
        payload,
        table="ic.baseline",
        allowed_keys={"fft_window_scale"},
    )
    si_baseline_config = table_config_values(
        payload,
        table="si.baseline",
        allowed_keys={"fft_window_scale"},
    )
    gagg_baseline_entries = table_config_entries(
        payload,
        table="gagg.baseline",
        allowed_keys={"index", "fft_window_scale"},
    )
    gagg_time_entries = table_config_entries(
        payload,
        table="gagg.time",
        allowed_keys={"index", "min", "max"},
    )
    gagg_config = table_config_values(
        payload,
        table="gagg",
        allowed_keys={"fit_parameters_path"},
    )
    gagg_indexes_config = table_config_values(
        payload, table="gagg.indexes", allowed_keys={"min", "max", "invalid"},
    )
    amplitude_config = table_config_values(
        payload,
        table="attpc.amplitude",
        allowed_keys={
            "peak_separation",
            "peak_prominence",
            "peak_width",
            "peak_threshold",
            "rel_height",
        },
    )
    ic_amplitude_config = table_config_values(
        payload,
        table="ic.amplitude",
        allowed_keys={
            "peak_separation",
            "peak_prominence",
            "peak_width",
            "peak_threshold",
            "rel_height",
        },
    )
    si_amplitude_config = table_config_values(
        payload,
        table="si.amplitude",
        allowed_keys={
            "peak_separation",
            "peak_prominence",
            "peak_width",
            "peak_threshold",
            "rel_height",
        },
    )
    gagg_amplitude_entries = table_config_entries(
        payload,
        table="gagg.amplitude",
        allowed_keys={
            "index",
            "peak_separation",
            "peak_prominence",
            "peak_width",
            "peak_threshold",
            "rel_height",
        },
    )
    legacy_gagg_amplitude = (
        gagg_amplitude_entries[0] if len(gagg_amplitude_entries) == 1 else {}
    )
    bitflip_config = table_config_values(
        payload,
        table="attpc.bitflip",
        allowed_keys={
            "baseline",
        },
    )
    saturation_config = table_config_values(
        payload,
        table="attpc.saturation",
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
        "--ic-baseline-window-scale",
        type=float,
        default=ic_baseline_config.get("fft_window_scale", baseline_config.get("fft_window_scale")),
        help="Baseline-removal filter scale used for IC trace preprocessing",
    )
    parser.add_argument(
        "--si-baseline-window-scale",
        type=float,
        default=si_baseline_config.get("fft_window_scale", baseline_config.get("fft_window_scale")),
        help="Baseline-removal filter scale used for Si trace preprocessing",
    )
    parser.add_argument(
        "--gagg-baseline-window-scale",
        type=float,
        default=baseline_config.get("fft_window_scale"),
        help="Baseline-removal filter scale used for GAGG trace preprocessing",
    )
    parser.add_argument(
        "--peak-separation",
        type=float,
        default=amplitude_config.get("peak_separation", 50.0),
        help="Minimum separation between peaks for filtered amplitude histograms",
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=amplitude_config.get("peak_prominence", 20.0),
        help="Prominence of peaks for filtered amplitude histograms",
    )
    parser.add_argument(
        "--peak-width",
        type=float,
        default=amplitude_config.get("peak_width", 50.0),
        help="Maximum peak width for filtered amplitude histograms",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=amplitude_config.get("peak_threshold", 0.0),
        help="Minimum peak amplitude for filtered amplitude histograms",
    )
    parser.add_argument(
        "--peak-rel-height",
        type=float,
        default=amplitude_config.get("rel_height", 0.95),
        help="Relative height used when measuring filtered amplitude peak width",
    )
    parser.add_argument(
        "--ic-peak-separation",
        type=float,
        default=ic_amplitude_config.get("peak_separation", amplitude_config.get("peak_separation", 50.0)),
        help="Minimum separation between IC peaks for WebUI review and histograms",
    )
    parser.add_argument(
        "--ic-peak-prominence",
        type=float,
        default=ic_amplitude_config.get("peak_prominence", amplitude_config.get("peak_prominence", 20.0)),
        help="Prominence of IC peaks for WebUI review and histograms",
    )
    parser.add_argument(
        "--ic-peak-width",
        type=float,
        default=ic_amplitude_config.get("peak_width", amplitude_config.get("peak_width", 50.0)),
        help="Maximum IC peak width for WebUI review and histograms",
    )
    parser.add_argument(
        "--ic-peak-threshold",
        type=float,
        default=ic_amplitude_config.get("peak_threshold", amplitude_config.get("peak_threshold", 0.0)),
        help="Minimum IC peak amplitude for WebUI review and histograms",
    )
    parser.add_argument(
        "--ic-peak-rel-height",
        type=float,
        default=ic_amplitude_config.get("rel_height", amplitude_config.get("rel_height", 0.95)),
        help="Relative height used when measuring IC peak width",
    )
    parser.add_argument(
        "--si-peak-separation",
        type=float,
        default=si_amplitude_config.get("peak_separation", amplitude_config.get("peak_separation", 50.0)),
        help="Minimum separation between Si peaks for WebUI review",
    )
    parser.add_argument(
        "--si-peak-prominence",
        type=float,
        default=si_amplitude_config.get("peak_prominence", amplitude_config.get("peak_prominence", 20.0)),
        help="Prominence of Si peaks for WebUI review",
    )
    parser.add_argument(
        "--si-peak-width",
        type=float,
        default=si_amplitude_config.get("peak_width", amplitude_config.get("peak_width", 50.0)),
        help="Maximum Si peak width for WebUI review",
    )
    parser.add_argument(
        "--si-peak-threshold",
        type=float,
        default=si_amplitude_config.get("peak_threshold", amplitude_config.get("peak_threshold", 0.0)),
        help="Minimum Si peak amplitude for WebUI review",
    )
    parser.add_argument(
        "--si-peak-rel-height",
        type=float,
        default=si_amplitude_config.get("rel_height", amplitude_config.get("rel_height", 0.95)),
        help="Relative height used when measuring Si peak width",
    )
    parser.add_argument(
        "--gagg-peak-separation",
        type=float,
        default=legacy_gagg_amplitude.get("peak_separation", amplitude_config.get("peak_separation", 50.0)),
        help="Minimum separation between GAGG peaks for WebUI review",
    )
    parser.add_argument(
        "--gagg-peak-prominence",
        type=float,
        default=legacy_gagg_amplitude.get("peak_prominence", amplitude_config.get("peak_prominence", 20.0)),
        help="Prominence of GAGG peaks for WebUI review",
    )
    parser.add_argument(
        "--gagg-peak-width",
        type=float,
        default=legacy_gagg_amplitude.get("peak_width", amplitude_config.get("peak_width", 50.0)),
        help="Maximum GAGG peak width for WebUI review",
    )
    parser.add_argument(
        "--gagg-peak-threshold",
        type=float,
        default=legacy_gagg_amplitude.get("peak_threshold", amplitude_config.get("peak_threshold", 0.0)),
        help="Minimum GAGG peak amplitude for WebUI review",
    )
    parser.add_argument(
        "--gagg-peak-rel-height",
        type=float,
        default=legacy_gagg_amplitude.get("rel_height", amplitude_config.get("rel_height", 0.95)),
        help="Relative height used when measuring GAGG peak width",
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
    args = parser.parse_args()
    args.gagg_baseline_entries = gagg_baseline_entries
    args.gagg_amplitude_entries = gagg_amplitude_entries
    args.gagg_time_entries = gagg_time_entries
    fit_parameters_path = gagg_config.get("fit_parameters_path")
    if "gagg" in payload and (
        not isinstance(fit_parameters_path, str) or not fit_parameters_path.strip()
    ):
        raise ValueError("gagg.fit_parameters_path must name a GAGG fit-parameters TOML file")
    args.gagg_fit_parameters_path = (
        config_relative_path(config_path, fit_parameters_path)
        if isinstance(fit_parameters_path, str)
        else None
    )
    if gagg_indexes_config:
        try:
            lower, upper = int(gagg_indexes_config["min"]), int(gagg_indexes_config["max"])
            invalid = {int(value) for value in gagg_indexes_config.get("invalid", [])}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("[gagg.indexes] requires min, max, and an optional integer invalid list") from exc
        args.gagg_valid_indexes = tuple(index for index in range(lower, upper + 1) if index not in invalid)
    else:
        args.gagg_valid_indexes = tuple(range(41))
    return args


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


def _option_supplied(option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in sys.argv[1:])


if __name__ == "__main__":
    main()
