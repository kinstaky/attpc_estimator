from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..process.amplitude import build_amplitude_histogram, build_labeled_amplitude_histograms
from ..process.baseline import build_baseline_histogram, build_labeled_baseline_histograms
from ..process.bitflip import build_bitflip_histograms, build_labeled_bitflip_histograms
from ..process.cdf import CDF_THRESHOLDS, build_labeled_cdf_histograms, build_trace_cdf_histogram
from ..process.coplanar import build_coplanar_histogram, default_pointcloud_file_path as default_coplanar_path
from ..process.line_distance import (
    LineDistanceHistogramConfig,
    RansacConfig,
    build_line_distance_histograms,
    default_pointcloud_file_path as default_line_distance_path,
)
from ..process.line_pipeline import MergeConfig
from ..process.line_property import (
    LinePropertyHistogramConfig,
    build_line_property_histograms,
    default_pointcloud_file_path as default_line_property_path,
)
from ..process.saturation import build_labeled_saturation_histograms, build_saturation_histograms
from ..storage.run_paths import format_run_id, histogram_dir, resolve_run_file
from .config import (
    bool_argument_config_kwargs,
    parse_run,
    parse_toml_config,
    root_config_values,
    table_config_values,
)
from .progress import tqdm_reporter


@dataclass(frozen=True, slots=True)
class OptionSpec:
    flags: tuple[str, ...]
    help: str
    config_key: tuple[str, ...]
    config_table: str | None = None
    value_type: Callable[[str], Any] | None = None
    dest: str | None = None
    boolean: bool = False


ROOT_ALLOWED_KEYS = {"trace_path", "workspace", "run"}
ROOT_POINTCLOUD_ALLOWED_KEYS = {"workspace", "run"}

TRACE_PATH_OPTION = OptionSpec(
    flags=("-t", "--trace-path"),
    help="Path to a trace file or a directory containing run_<run>.h5 files",
    config_key=("trace_path",),
)
WORKSPACE_OPTION = OptionSpec(
    flags=("-w", "--workspace"),
    help="Path to store result files and locate the labels database",
    config_key=("workspace",),
)
POINTCLOUD_WORKSPACE_OPTION = OptionSpec(
    flags=("-w", "--workspace"),
    help="Workspace directory containing pointcloud data and histogram outputs",
    config_key=("workspace",),
)
RUN_OPTION = OptionSpec(
    flags=("-r", "--run"),
    help="Run number",
    config_key=("run",),
    value_type=parse_run,
)
BASELINE_WINDOW_SCALE_OPTION = OptionSpec(
    flags=("--baseline-window-scale",),
    help="Baseline-removal filter scale used before processing",
    config_table="baseline",
    config_key=("fft_window_scale",),
    value_type=float,
    dest="baseline_window_scale",
)
AMPLITUDE_SEPARATION_OPTION = OptionSpec(
    flags=("--peak-separation",),
    help="Minimum separation between peaks",
    config_table="amplitude",
    config_key=("separation", "peak_separation"),
    value_type=float,
    dest="peak_separation",
)
AMPLITUDE_PROMINENCE_OPTION = OptionSpec(
    flags=("--peak-prominence",),
    help="Prominence of peaks",
    config_table="amplitude",
    config_key=("prominence", "peak_prominence"),
    value_type=float,
    dest="peak_prominence",
)
AMPLITUDE_WIDTH_OPTION = OptionSpec(
    flags=("--peak-width",),
    help="Maximum width of peaks",
    config_table="amplitude",
    config_key=("max_width", "peak_width"),
    value_type=float,
    dest="peak_width",
)
AMPLITUDE_LABELED_OPTION = OptionSpec(
    flags=("--labeled",),
    help="Process only labeled traces for the selected run and save one histogram per label",
    config_table="amplitude",
    config_key=("labeled",),
    dest="labeled",
    boolean=True,
)
BASELINE_LABELED_OPTION = OptionSpec(
    flags=("--labeled",),
    help="Process only labeled traces for the selected run and save one histogram per label",
    config_table="baseline",
    config_key=("labeled",),
    dest="labeled",
    boolean=True,
)
CDF_BASELINE_WINDOW_SCALE_OPTION = OptionSpec(
    flags=("--baseline-window-scale",),
    help="Baseline-removal filter scale used before taking the FFT",
    config_table="cdf",
    config_key=("baseline_window_scale",),
    value_type=float,
    dest="baseline_window_scale",
)
CDF_LABELED_OPTION = OptionSpec(
    flags=("--labeled",),
    help="Build one CDF histogram per trace label for the selected run",
    config_table="cdf",
    config_key=("labeled",),
    dest="labeled",
    boolean=True,
)
BITFLIP_BASELINE_OPTION = OptionSpec(
    flags=("--baseline",),
    help="Absolute second-derivative threshold used to classify baseline points",
    config_table="bitflip",
    config_key=("baseline",),
    value_type=float,
    dest="baseline",
)
BITFLIP_LABELED_OPTION = OptionSpec(
    flags=("--labeled",),
    help="Process only labeled traces for the selected run and save one histogram per label",
    config_table="bitflip",
    config_key=("labeled",),
    dest="labeled",
    boolean=True,
)
SATURATION_THRESHOLD_OPTION = OptionSpec(
    flags=("--threshold",),
    help="Minimum trace maximum required before evaluating a saturation plateau",
    config_table="saturation",
    config_key=("threshold",),
    value_type=float,
)
SATURATION_DROP_THRESHOLD_OPTION = OptionSpec(
    flags=("--drop-threshold",),
    help="Drop threshold D used when measuring plateau length",
    config_table="saturation",
    config_key=("drop_threshold",),
    value_type=float,
    dest="drop_threshold",
)
SATURATION_WINDOW_RADIUS_OPTION = OptionSpec(
    flags=("--window-radius",),
    help="Radius of the local window used when accumulating drop-from-maximum statistics",
    config_table="saturation",
    config_key=("window_radius",),
    value_type=int,
    dest="window_radius",
)
SATURATION_LABELED_OPTION = OptionSpec(
    flags=("--labeled",),
    help="Process only labeled traces for the selected run and save one histogram per label",
    config_table="saturation",
    config_key=("labeled",),
    dest="labeled",
    boolean=True,
)
RANSAC_RESIDUAL_OPTION = OptionSpec(
    flags=("--residual-threshold",),
    help="Residual threshold used by RANSAC",
    config_table="findline.ransac",
    config_key=("residual_threshold",),
    value_type=float,
    dest="residual_threshold",
)
RANSAC_MAX_TRIALS_OPTION = OptionSpec(
    flags=("--max-trials",),
    help="Maximum number of RANSAC trials",
    config_table="findline.ransac",
    config_key=("max_trials",),
    value_type=int,
    dest="max_trials",
)
RANSAC_MAX_ITERATIONS_OPTION = OptionSpec(
    flags=("--max-iterations",),
    help="Maximum number of accepted-cluster search iterations per event",
    config_table="findline.ransac",
    config_key=("max_iterations",),
    value_type=int,
    dest="max_iterations",
)
RANSAC_TARGET_RATIO_OPTION = OptionSpec(
    flags=("--target-labeled-ratio",),
    help="Stop extracting lines once the labeled point ratio reaches this value",
    config_table="findline.ransac",
    config_key=("target_labeled_ratio",),
    value_type=float,
    dest="target_labeled_ratio",
)
RANSAC_MIN_INLIERS_OPTION = OptionSpec(
    flags=("--min-inliers",),
    help="Minimum inlier count required to accept a RANSAC cluster",
    config_table="findline.ransac",
    config_key=("min_inliers",),
    value_type=int,
    dest="min_inliers",
)
RANSAC_MAX_START_RADIUS_OPTION = OptionSpec(
    flags=("--max-start-radius",),
    help="Reject clusters whose nearest XY radius is not below this value",
    config_table="findline.ransac",
    config_key=("max_start_radius",),
    value_type=float,
    dest="max_start_radius",
)
MERGE_DISTANCE_OPTION = OptionSpec(
    flags=("--merge-distance-threshold",),
    help="Merge lines when centroid-to-line distance is below this threshold",
    config_table="findline.mergeline",
    config_key=("distance_threshold",),
    value_type=float,
    dest="merge_distance_threshold",
)
MERGE_ANGLE_OPTION = OptionSpec(
    flags=("--merge-angle-threshold",),
    help="Merge lines when direction angle is below this threshold in degrees",
    config_table="findline.mergeline",
    config_key=("angle_threshold",),
    value_type=float,
    dest="merge_angle_threshold",
)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    args.handler(args)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    config_path, payload = parse_toml_config(argv)
    parser = argparse.ArgumentParser(description="Compute histogram artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_amplitude_parser(subparsers, config_path, payload)
    _build_baseline_parser(subparsers, config_path, payload)
    _build_cdf_parser(subparsers, config_path, payload)
    _build_bitflip_parser(subparsers, config_path, payload)
    _build_saturation_parser(subparsers, config_path, payload)
    _build_coplanar_parser(subparsers, config_path, payload)
    _build_line_distance_parser(subparsers, config_path, payload)
    _build_line_property_parser(subparsers, config_path, payload)
    return parser.parse_args(argv)


def _add_config_option(
    parser: argparse.ArgumentParser,
    payload: dict[str, Any],
    spec: OptionSpec,
) -> None:
    values = (
        root_config_values(payload, allowed_keys=set(spec.config_key))
        if spec.config_table is None
        else table_config_values(payload, table=spec.config_table, allowed_keys=set(spec.config_key))
    )
    target_key = spec.dest or spec.config_key[0]
    bound_values = _bound_config_values(values, spec.config_key, target_key)
    kwargs = (
        bool_argument_config_kwargs(bound_values, target_key)
        if spec.boolean
        else _argument_kwargs(bound_values, target_key)
    )
    if spec.value_type is not None and not spec.boolean:
        kwargs["type"] = spec.value_type
    if spec.dest is not None:
        kwargs["dest"] = spec.dest
    parser.add_argument(*spec.flags, help=spec.help, **kwargs)


def _bound_config_values(values: dict[str, Any], keys: tuple[str, ...], target_key: str) -> dict[str, Any]:
    if not keys:
        return {}
    for key in keys:
        if key in values:
            return {target_key: values[key]}
    return {}


def _argument_kwargs(values: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "required": key not in values,
        "default": values.get(key),
    }


def _add_config_argument(parser: argparse.ArgumentParser, config_path: Path) -> None:
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=str(config_path),
        help="Path to a TOML config file. Defaults to config.toml.",
    )


def _build_amplitude_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], config_path: Path, payload: dict[str, Any]) -> None:
    parser = subparsers.add_parser("amplitude", description="Compute peak-amplitude histograms for all traces or labeled traces")
    _add_config_argument(parser, config_path)
    for spec in (
        TRACE_PATH_OPTION,
        WORKSPACE_OPTION,
        RUN_OPTION,
        BASELINE_WINDOW_SCALE_OPTION,
        AMPLITUDE_SEPARATION_OPTION,
        AMPLITUDE_PROMINENCE_OPTION,
        AMPLITUDE_WIDTH_OPTION,
        AMPLITUDE_LABELED_OPTION,
    ):
        _add_config_option(parser, payload, spec)
    parser.set_defaults(handler=_run_amplitude)


def _build_baseline_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], config_path: Path, payload: dict[str, Any]) -> None:
    parser = subparsers.add_parser("baseline", description="Compute preprocessed-trace baseline histograms for all traces or labeled traces")
    _add_config_argument(parser, config_path)
    for spec in (
        TRACE_PATH_OPTION,
        WORKSPACE_OPTION,
        RUN_OPTION,
        BASELINE_WINDOW_SCALE_OPTION,
        BASELINE_LABELED_OPTION,
    ):
        _add_config_option(parser, payload, spec)
    parser.set_defaults(handler=_run_baseline)


def _build_cdf_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], config_path: Path, payload: dict[str, Any]) -> None:
    parser = subparsers.add_parser("cdf", description="Compute CDF histograms for all traces or labeled traces in a single run")
    _add_config_argument(parser, config_path)
    for spec in (
        TRACE_PATH_OPTION,
        WORKSPACE_OPTION,
        RUN_OPTION,
        CDF_BASELINE_WINDOW_SCALE_OPTION,
        CDF_LABELED_OPTION,
    ):
        _add_config_option(parser, payload, spec)
    parser.set_defaults(handler=_run_cdf)


def _build_bitflip_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], config_path: Path, payload: dict[str, Any]) -> None:
    parser = subparsers.add_parser("bitflip", description="Compute second-derivative alternating-run bitflip histograms for all traces or labeled traces")
    _add_config_argument(parser, config_path)
    for spec in (
        TRACE_PATH_OPTION,
        WORKSPACE_OPTION,
        RUN_OPTION,
        BASELINE_WINDOW_SCALE_OPTION,
        BITFLIP_BASELINE_OPTION,
        BITFLIP_LABELED_OPTION,
    ):
        _add_config_option(parser, payload, spec)
    parser.set_defaults(handler=_run_bitflip)


def _build_saturation_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], config_path: Path, payload: dict[str, Any]) -> None:
    parser = subparsers.add_parser("saturation", description="Compute saturation plateau histograms for all traces or labeled traces")
    _add_config_argument(parser, config_path)
    for spec in (
        TRACE_PATH_OPTION,
        WORKSPACE_OPTION,
        RUN_OPTION,
        BASELINE_WINDOW_SCALE_OPTION,
        SATURATION_THRESHOLD_OPTION,
        SATURATION_DROP_THRESHOLD_OPTION,
        SATURATION_WINDOW_RADIUS_OPTION,
        SATURATION_LABELED_OPTION,
    ):
        _add_config_option(parser, payload, spec)
    parser.set_defaults(handler=_run_saturation)


def _build_coplanar_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], config_path: Path, payload: dict[str, Any]) -> None:
    parser = subparsers.add_parser("coplanar", description="Compute phase-2 pointcloud coplanarity histograms for one run")
    _add_config_argument(parser, config_path)
    for spec in (POINTCLOUD_WORKSPACE_OPTION, RUN_OPTION):
        _add_config_option(parser, payload, spec)
    parser.set_defaults(handler=_run_coplanar)


def _build_line_distance_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], config_path: Path, payload: dict[str, Any]) -> None:
    parser = subparsers.add_parser("line_distance", description="Compute phase-2 pointcloud line-distance histograms for one run")
    _add_config_argument(parser, config_path)
    for spec in (
        POINTCLOUD_WORKSPACE_OPTION,
        RUN_OPTION,
        RANSAC_RESIDUAL_OPTION,
        RANSAC_MAX_TRIALS_OPTION,
        RANSAC_MAX_ITERATIONS_OPTION,
        RANSAC_TARGET_RATIO_OPTION,
        RANSAC_MIN_INLIERS_OPTION,
        RANSAC_MAX_START_RADIUS_OPTION,
    ):
        _add_config_option(parser, payload, spec)
    parser.set_defaults(handler=_run_line_distance)


def _build_line_property_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], config_path: Path, payload: dict[str, Any]) -> None:
    parser = subparsers.add_parser("line_property", description="Compute phase-2 pointcloud line-property histograms for one run")
    _add_config_argument(parser, config_path)
    for spec in (
        POINTCLOUD_WORKSPACE_OPTION,
        RUN_OPTION,
        RANSAC_RESIDUAL_OPTION,
        RANSAC_MAX_TRIALS_OPTION,
        RANSAC_MAX_ITERATIONS_OPTION,
        RANSAC_TARGET_RATIO_OPTION,
        RANSAC_MIN_INLIERS_OPTION,
        RANSAC_MAX_START_RADIUS_OPTION,
        MERGE_DISTANCE_OPTION,
        MERGE_ANGLE_OPTION,
    ):
        _add_config_option(parser, payload, spec)
    parser.set_defaults(handler=_run_line_property)


def _run_amplitude(args: argparse.Namespace) -> None:
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_root = histogram_dir(workspace)
    run_id = int(args.run)
    run_name = format_run_id(run_id)

    if args.labeled:
        with tqdm_reporter("Processing labeled pad traces") as progress:
            payload = build_labeled_amplitude_histograms(
                trace_path=trace_root,
                workspace=workspace,
                run=run_id,
                baseline_window_scale=args.baseline_window_scale,
                peak_separation=args.peak_separation,
                peak_prominence=args.peak_prominence,
                peak_width=args.peak_width,
                progress=progress,
            )
        output_path = output_root / f"run_{run_name}_labeled_amp.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            run_id=payload["run_id"],
            label_keys=payload["label_keys"],
            label_titles=payload["label_titles"],
            trace_counts=payload["trace_counts"],
            histograms=payload["histograms"],
        )
        print(f"saved labeled amplitude histograms to {output_path}")
        print(f"labels: {payload['label_keys'].tolist()}")
        print(f"trace counts: {payload['trace_counts'].tolist()}")
        return

    try:
        trace_file_path = resolve_run_file(trace_root, args.run)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with tqdm_reporter("Processing pad traces") as progress:
        histogram = build_amplitude_histogram(
            trace_file_path=trace_file_path,
            baseline_window_scale=args.baseline_window_scale,
            peak_separation=args.peak_separation,
            peak_prominence=args.peak_prominence,
            peak_width=args.peak_width,
            progress=progress,
        )
    output_path = output_root / f"run_{run_name}_amp.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, histogram)
    print(f"saved amplitude histogram with shape {histogram.shape} to {output_path}")
    print(f"total histogram count: {int(histogram.sum())}")


def _run_baseline(args: argparse.Namespace) -> None:
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_root = histogram_dir(workspace)
    run_id = int(args.run)
    run_name = format_run_id(run_id)

    if args.labeled:
        with tqdm_reporter("Processing labeled pad traces") as progress:
            payload = build_labeled_baseline_histograms(
                trace_path=trace_root,
                workspace=workspace,
                run=run_id,
                baseline_window_scale=args.baseline_window_scale,
                progress=progress,
            )
        output_path = output_root / f"run_{run_name}_labeled_baseline.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            run_id=payload["run_id"],
            label_keys=payload["label_keys"],
            label_titles=payload["label_titles"],
            trace_counts=payload["trace_counts"],
            histograms=payload["histograms"],
            bin_centers=payload["bin_centers"],
        )
        print(f"saved labeled baseline histograms to {output_path}")
        print(f"labels: {payload['label_keys'].tolist()}")
        print(f"trace counts: {payload['trace_counts'].tolist()}")
        return

    try:
        trace_file_path = resolve_run_file(trace_root, args.run)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with tqdm_reporter("Processing pad traces") as progress:
        payload = build_baseline_histogram(
            trace_file_path=trace_file_path,
            baseline_window_scale=args.baseline_window_scale,
            progress=progress,
        )
    output_path = output_root / f"run_{run_name}_baseline.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        trace_count=payload["trace_count"],
        histogram=payload["histogram"],
        bin_centers=payload["bin_centers"],
    )
    print(f"saved baseline histograms to {output_path}")
    print(f"trace count: {int(payload['trace_count'])}")


def _run_cdf(args: argparse.Namespace) -> None:
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_root = histogram_dir(workspace)
    run_id = int(args.run)
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
        print(f"saved labeled CDF histograms with shape {payload['histograms'].shape} to {output_path}")
        print(f"labels: {payload['label_titles'].tolist()}")
        print(f"trace counts: {payload['trace_counts'].tolist()}")
        return

    try:
        trace_file_path = resolve_run_file(trace_root, args.run)
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


def _run_bitflip(args: argparse.Namespace) -> None:
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_root = histogram_dir(workspace)
    run_id = int(args.run)
    run_name = format_run_id(run_id)

    if args.labeled:
        with tqdm_reporter("Processing labeled pad traces") as progress:
            payload = build_labeled_bitflip_histograms(
                trace_path=trace_root,
                workspace=workspace,
                run=run_id,
                baseline_window_scale=args.baseline_window_scale,
                baseline_threshold=args.baseline,
                progress=progress,
            )
        output_path = output_root / f"run_{run_name}_labeled_bitflip.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            run_id=payload["run_id"],
            label_keys=payload["label_keys"],
            label_titles=payload["label_titles"],
            trace_counts=payload["trace_counts"],
            baseline_histograms=payload["baseline_histograms"],
            value_histograms=payload["value_histograms"],
            length_histograms=payload["length_histograms"],
            count_histograms=payload["count_histograms"],
        )
        print(f"saved labeled bitflip histograms to {output_path}")
        print(f"labels: {payload['label_keys'].tolist()}")
        print(f"trace counts: {payload['trace_counts'].tolist()}")
        return

    try:
        trace_file_path = resolve_run_file(trace_root, args.run)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with tqdm_reporter("Processing pad traces") as progress:
        payload = build_bitflip_histograms(
            trace_file_path=trace_file_path,
            baseline_window_scale=args.baseline_window_scale,
            baseline_threshold=args.baseline,
            progress=progress,
        )
    output_path = output_root / f"run_{run_name}_bitflip.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        trace_count=payload["trace_count"],
        baseline_histogram=payload["baseline_histogram"],
        value_histogram=payload["value_histogram"],
        length_histogram=payload["length_histogram"],
        count_histogram=payload["count_histogram"],
    )
    print(f"saved bitflip histograms to {output_path}")
    print(f"trace count: {int(payload['trace_count'])}")


def _run_saturation(args: argparse.Namespace) -> None:
    trace_root = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_root = histogram_dir(workspace)
    run_id = int(args.run)
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
        output_path = output_root / f"run_{run_name}_labeled_saturation.npz"
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
        trace_file_path = resolve_run_file(trace_root, args.run)
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
    output_path = output_root / f"run_{run_name}_saturation.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        trace_count=payload["trace_count"],
        drop_histogram=payload["drop_histogram"],
        length_histogram=payload["length_histogram"],
    )
    print(f"saved saturation histograms to {output_path}")
    print(f"trace count: {int(payload['trace_count'])}")


def _run_coplanar(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).expanduser().resolve()
    run_id = int(args.run)
    run_name = format_run_id(run_id)
    pointcloud_path = default_coplanar_path(workspace, run_id)
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


def _run_line_distance(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).expanduser().resolve()
    run_id = int(args.run)
    run_name = format_run_id(run_id)
    pointcloud_path = default_line_distance_path(workspace, run_id)
    if not pointcloud_path.is_file():
        raise SystemExit(f"pointcloud file not found: {pointcloud_path}")

    with tqdm_reporter("Processing phase-2 pointcloud events") as progress:
        payload = build_line_distance_histograms(
            pointcloud_file_path=pointcloud_path,
            run=run_id,
            ransac_config=RansacConfig(
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


def _run_line_property(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).expanduser().resolve()
    run_id = int(args.run)
    run_name = format_run_id(run_id)
    pointcloud_path = default_line_property_path(workspace, run_id)
    if not pointcloud_path.is_file():
        raise SystemExit(f"pointcloud file not found: {pointcloud_path}")

    with tqdm_reporter("Processing phase-2 pointcloud events") as progress:
        payload = build_line_property_histograms(
            pointcloud_file_path=pointcloud_path,
            run=run_id,
            ransac_config=RansacConfig(
                residual_threshold=float(args.residual_threshold),
                max_trials=int(args.max_trials),
                max_iterations=int(args.max_iterations),
                target_labeled_ratio=float(args.target_labeled_ratio),
                min_inliers=int(args.min_inliers),
                max_start_radius=float(args.max_start_radius),
            ),
            merge_config=MergeConfig(
                distance_threshold=float(args.merge_distance_threshold),
                angle_threshold=float(args.merge_angle_threshold),
            ),
            histogram_config=LinePropertyHistogramConfig(),
            progress=progress,
        )
    output_path = histogram_dir(workspace) / f"run_{run_name}_line_property.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"saved line-property histograms to {output_path}")
    print(f"processed events: {int(np.asarray(payload['processed_events']).item())}")
    print(f"accepted lines: {int(np.asarray(payload['accepted_line_total']).item())}")
