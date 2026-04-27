from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..process.filter_core import (
    AmplitudeFilterCore,
    BitFlipFilterCore,
    CdfFilterCore,
    SaturationFilterCore,
)
from ..process.filter import (
    DEFAULT_TRACE_LIMIT,
    UNLIMITED_TRACE_LIMIT,
    build_filter_rows,
    default_output_name,
    normalize_amplitude_range,
)
from ..storage.run_paths import filter_dir, format_run_id
from .config import (
    argument_config_kwargs,
    bool_argument_config_kwargs,
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
    output_root = filter_dir(workspace)
    run_token = args.run
    run_id = int(run_token)
    run_name = format_run_id(run_id)
    amplitude_range = normalize_amplitude_range(args.amplitude)
    filter_cores = _build_filter_cores(
        amplitude_range=amplitude_range,
        cdf=args.cdf,
        peak_separation=args.peak_separation,
        peak_prominence=args.peak_prominence,
        peak_width=args.peak_width,
        bitflip=args.bitflip,
        bitflip_baseline=args.bitflip_baseline,
        bitflip_min_count=args.bitflip_min_count,
        saturation=args.saturation,
        saturation_drop_threshold=args.saturation_drop_threshold,
        saturation_min_plateau_length=args.saturation_min_plateau_length,
        saturation_threshold=args.saturation_threshold,
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else output_root / default_output_name(run_name, filter_cores)
    )

    progress_desc = (
        "Scanning run"
        if args.limit == UNLIMITED_TRACE_LIMIT
        else "Collecting filter rows"
    )
    with tqdm_reporter(progress_desc) as progress:
        rows = build_filter_rows(
            trace_path=trace_root,
            run=run_id,
            filter_cores=filter_cores,
            baseline_window_scale=args.baseline_window_scale,
            limit=args.limit,
            progress=progress,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, rows)

    print(f"saved {len(rows)} filter rows to {output_path}")


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(
        payload,
        allowed_keys={"trace_path", "workspace", "run"},
    )
    filter_config = table_config_values(
        payload,
        table="filter",
        allowed_keys={
            "use_amplitude",
            "use_bitflip",
            "use_cdf",
            "use_saturation",
            "min_amplitude",
            "max_amplitude",
            "baseline_window_scale",
            "limit",
            "output",
        },
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
            "drop_threshold",
            "min_plateau_length",
            "threshold",
        },
    )
    amplitude_default = None
    if bool(filter_config.get("use_amplitude", False)):
        min_amplitude = filter_config.get("min_amplitude")
        max_amplitude = filter_config.get("max_amplitude")
        if min_amplitude is not None and max_amplitude is not None:
            amplitude_default = [min_amplitude, max_amplitude]
    parser = argparse.ArgumentParser(
        description="Generate a filter file containing the first matching traces in one run",
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
        help="Workspace directory where the filter file will be written",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=parse_run,
        **argument_config_kwargs(config, "run"),
        help="Run identifier to filter",
    )
    parser.add_argument(
        "--amplitude",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=amplitude_default,
        help="Inclusive lower and upper bounds for the highest detected peak amplitude",
    )
    parser.add_argument(
        "--cdf",
        **bool_argument_config_kwargs(filter_config, "use_cdf"),
        help="Keep traces whose CDF F(60) is below 0.6",
    )
    parser.add_argument(
        "--bitflip",
        **bool_argument_config_kwargs(filter_config, "use_bitflip"),
        help="Keep traces containing at least the requested number of qualified bitflip segments",
    )
    parser.add_argument(
        "--bitflip-baseline",
        type=float,
        **argument_config_kwargs(bitflip_config, "baseline"),
        help="Absolute second-derivative threshold used to classify baseline points for bitflip filtering",
    )
    parser.add_argument(
        "--bitflip-min-count",
        type=int,
        **argument_config_kwargs(bitflip_config, "min_count"),
        help="Minimum number of qualified bitflip segments required to keep a trace",
    )
    parser.add_argument(
        "--saturation",
        **bool_argument_config_kwargs(filter_config, "use_saturation"),
        help="Keep traces with a flat high-amplitude saturation plateau",
    )
    parser.add_argument(
        "--saturation-drop-threshold",
        type=float,
        default=saturation_config.get("drop_threshold"),
        help="Maximum drop from the local maximum when measuring the saturation plateau",
    )
    parser.add_argument(
        "--saturation-min-plateau-length",
        type=int,
        default=saturation_config.get("min_plateau_length"),
        help="Minimum contiguous plateau length required for saturation filtering",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        **argument_config_kwargs(saturation_config, "threshold"),
        help="Minimum trace maximum required before evaluating saturation plateau length",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        **argument_config_kwargs(filter_config, "baseline_window_scale"),
        help="Baseline-removal filter scale used before peak detection and FFT",
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
        "--limit",
        type=int,
        **argument_config_kwargs(filter_config, "limit"),
        help=(
            f"Maximum number of matching traces to keep, default {DEFAULT_TRACE_LIMIT}. "
            "Use 0 to keep every matching trace in the selected run."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=filter_config.get("output"),
        help="Optional explicit output .npy path",
    )
    args = parser.parse_args()
    if (
        args.amplitude is None
        and not args.cdf
        and not args.bitflip
        and not args.saturation
    ):
        parser.error(
            "at least one filter criterion is required: --amplitude MIN MAX, --cdf, --bitflip, and/or --saturation"
        )
    if bool(filter_config.get("use_amplitude", False)) and args.amplitude is None:
        parser.error(
            "[filter] use_amplitude requires min_amplitude and max_amplitude unless --amplitude is provided"
        )
    if args.saturation and args.saturation_drop_threshold is None:
        parser.error("--saturation requires --saturation-drop-threshold")
    if args.saturation and args.saturation_min_plateau_length is None:
        parser.error("--saturation requires --saturation-min-plateau-length")
    if args.limit < UNLIMITED_TRACE_LIMIT:
        parser.error("--limit must be non-negative")
    return args


def _build_filter_cores(
    *,
    amplitude_range: tuple[float, float] | None,
    cdf: bool,
    peak_separation: float,
    peak_prominence: float,
    peak_width: float,
    bitflip: bool,
    bitflip_baseline: float,
    bitflip_min_count: int,
    saturation: bool,
    saturation_drop_threshold: float | None,
    saturation_min_plateau_length: int | None,
    saturation_threshold: float,
) -> list[CdfFilterCore | AmplitudeFilterCore | BitFlipFilterCore | SaturationFilterCore]:
    filter_cores: list[
        CdfFilterCore | AmplitudeFilterCore | BitFlipFilterCore | SaturationFilterCore
    ] = []
    if cdf:
        filter_cores.append(CdfFilterCore())
    if amplitude_range is not None:
        filter_cores.append(
            AmplitudeFilterCore(
                min_amplitude=amplitude_range[0],
                max_amplitude=amplitude_range[1],
                peak_separation=peak_separation,
                peak_prominence=peak_prominence,
                peak_width=peak_width,
            )
        )
    if bitflip:
        filter_cores.append(
            BitFlipFilterCore(
                baseline_threshold=bitflip_baseline,
                min_segment_count=bitflip_min_count,
            )
        )
    if (
        saturation
        and saturation_drop_threshold is not None
        and saturation_min_plateau_length is not None
    ):
        filter_cores.append(
            SaturationFilterCore(
                drop_threshold=saturation_drop_threshold,
                min_plateau_length=saturation_min_plateau_length,
                threshold=saturation_threshold,
            )
        )
    return filter_cores
