import argparse
import sys
from pathlib import Path

from .config import config_relative_path, parse_toml_config, root_config_values, table_config_entries, table_config_values
from ..pipeline.gagg import process_run
from ..pipeline.progress_reporter import TqdmProgressReporter
from ..process.gagg_parameters import load_gagg_fit_parameters
from ..utils.gagg_processing import build_gagg_trace_rules


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(payload, allowed_keys={"trace_path", "workspace", "run"})
    baseline_entries = table_config_entries(
        payload,
        table="gagg.baseline",
        allowed_keys={"index", "fft_window_scale"},
    )
    amplitude_entries = table_config_entries(
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
    time_entries = table_config_entries(
        payload,
        table="gagg.time",
        allowed_keys={"index", "min", "max"},
    )
    index_config = table_config_values(
        payload,
        table="gagg.indexes",
        allowed_keys={"min", "max", "invalid"},
    )
    gagg_config = table_config_values(payload, table="gagg", allowed_keys={"fit_parameters_path"})
    run_tags_config = table_config_values(payload, table="run_tags", allowed_keys={"path"})
    legacy_baseline = baseline_entries[0] if len(baseline_entries) == 1 else {}
    legacy_amplitude = amplitude_entries[0] if len(amplitude_entries) == 1 else {}

    parser = argparse.ArgumentParser(description="Build GaGG ingot root files")
    parser.add_argument("-c", "--config", dest="config_file", default=str(config_path))
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
        help="Workspace directory used for GaGG ROOT output",
    )
    parser.add_argument(
        "-r",
        "--run",
        required="run" not in config,
        default=config.get("run"),
        help="Run identifier to process.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show an interactive progress bar (disabled by default for log-friendly output)",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=legacy_baseline.get("fft_window_scale", 20.0),
    )
    parser.add_argument(
        "--peak-separation",
        type=float,
        default=legacy_amplitude.get("peak_separation", 50.0),
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=legacy_amplitude.get("peak_prominence", 20.0),
    )
    parser.add_argument(
        "--peak-width",
        type=float,
        default=legacy_amplitude.get("peak_width", 50.0),
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=legacy_amplitude.get("peak_threshold", 300.0),
    )
    parser.add_argument(
        "--peak-rel-height",
        type=float,
        default=legacy_amplitude.get("rel_height", 0.95),
    )
    parser.add_argument(
        "--min-time",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=255,
    )
    args = parser.parse_args()
    if "min" not in index_config or "max" not in index_config:
        raise ValueError("gagg.indexes must define min and max")
    fit_parameters_path = gagg_config.get("fit_parameters_path")
    if not isinstance(fit_parameters_path, str) or not fit_parameters_path.strip():
        raise ValueError("gagg.fit_parameters_path must name a GAGG fit-parameters TOML file")
    run_tags_path = run_tags_config.get("path")
    if not isinstance(run_tags_path, str) or not run_tags_path.strip():
        raise ValueError("run_tags.path must name the runs workbook")
    args.gagg_fit_parameters_path = config_relative_path(config_path, fit_parameters_path)
    args.run_tags_path = config_relative_path(config_path, run_tags_path)
    args.gagg_baseline_entries = baseline_entries
    args.gagg_amplitude_entries = amplitude_entries
    args.gagg_time_entries = time_entries
    args.gagg_valid_indexes = tuple(
        index for index in range(int(index_config["min"]), int(index_config["max"]) + 1)
        if index not in set(index_config.get("invalid", []))
    )
    return args


def main() -> None:
    args = _parse_args()
    if not args.run:
        raise SystemExit("no runs provided; pass --run for each run to process")

    workspace = Path(args.workspace).expanduser().resolve()
    try:
        calibration_parameters = load_gagg_fit_parameters(args.gagg_fit_parameters_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rules = build_gagg_trace_rules(
        baseline_entries=args.gagg_baseline_entries
        or [{"fft_window_scale": args.baseline_window_scale}],
        amplitude_entries=args.gagg_amplitude_entries
        or [{
            "peak_separation": args.peak_separation,
            "peak_prominence": args.peak_prominence,
            "peak_width": args.peak_width,
            "peak_threshold": args.peak_threshold,
            "rel_height": args.peak_rel_height,
        }],
        baseline_override=(args.baseline_window_scale if _option_supplied("--baseline-window-scale") else None),
        peak_overrides={
            field: float(getattr(args, argument))
            for field, argument, option in (
                ("peak_separation", "peak_separation", "--peak-separation"),
                ("peak_prominence", "peak_prominence", "--peak-prominence"),
                ("peak_width", "peak_width", "--peak-width"),
                ("peak_threshold", "peak_threshold", "--peak-threshold"),
                ("rel_height", "peak_rel_height", "--peak-rel-height"),
            )
            if _option_supplied(option)
        },
    )

    print(f"Starting GAGG run {args.run}.")
    result = process_run(
        workspace=str(workspace),
        run=int(args.run),
        input_path="hdf5/run_<run>.h5",
        output_path="ingot/gagg_<run>.root",
        fft_window_scale=args.baseline_window_scale,
        peak_separation=args.peak_separation,
        peak_prominence=args.peak_prominence,
        peak_max_width=args.peak_width,
        peak_threshold=args.peak_threshold,
        rel_height=args.peak_rel_height,
        min_time=args.min_time,
        max_time=args.max_time,
        gagg_trace_rules=rules,
        gagg_valid_indexes=args.gagg_valid_indexes,
        gagg_fit_parameters=calibration_parameters,
        run_tags_path=args.run_tags_path,
        reporter=TqdmProgressReporter(description="Processing GaGG") if args.progress else None,
    )
    if result == 0:
        print(f"Finished GAGG run {args.run}: success.")
    else:
        print(f"Finished GAGG run {args.run}: failed.")
        raise SystemExit(1)


def _option_supplied(option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in sys.argv[1:])


if __name__ == "__main__":
    main()
