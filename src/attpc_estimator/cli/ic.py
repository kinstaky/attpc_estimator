import argparse
import sys
from pathlib import Path

from .config import parse_toml_config, root_config_values, table_config_values
from ..pipeline.ic import process_run
from ..pipeline.progress_reporter import TqdmProgressReporter

def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(payload, allowed_keys={"trace_path", "workspace", "run"})
    baseline_config = table_config_values(
        payload,
        table="ic.baseline",
        allowed_keys={"fft_window_scale"},
    )
    amplitude_config = table_config_values(
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
    time_config = table_config_values(
        payload,
        table="ic.time",
        allowed_keys={
            "min",
            "max",
        },
    )

    parser = argparse.ArgumentParser(description="Build phase-1 ion-chamber parquet for raw runs")
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
        help="Workspace directory used for ion-chamber parquet output",
    )
    parser.add_argument(
        "-r",
        "--run",
        required="run" not in config,
        default=config.get("run"),
        help="Run identifier to process. May be repeated.",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=baseline_config.get("fft_window_scale", 20.0),
    )
    parser.add_argument(
        "--peak-separation",
        type=float,
        default=amplitude_config.get("peak_separation", 50.0),
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=amplitude_config.get("peak_prominence", 20.0),
    )
    parser.add_argument(
        "--peak-width",
        type=float,
        default=amplitude_config.get("peak_width", 500.0),
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=amplitude_config.get("peak_threshold", 100.0),
    )
    parser.add_argument(
        "--peak-rel-height",
        type=float,
        default=amplitude_config.get("rel_height", 0.85)
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=time_config.get("min", 62.0),
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=time_config.get("max", 68.0),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.run:
        raise SystemExit("no runs provided; pass --run for each run to process")

    workspace = Path(args.workspace).expanduser().resolve()

    result = process_run(
        workspace=str(workspace),
        run=int(args.run),
        input_path="hdf5/run_<run>.h5",
        output_path="ingot/ic_<run>.root",
        fft_window_scale=args.baseline_window_scale,
        peak_separation=args.peak_separation,
        peak_prominence=args.peak_prominence,
        peak_max_width=args.peak_width,
        peak_threshold=args.peak_threshold,
        rel_height=args.peak_rel_height,
        min_time=args.min_time,
        max_time=args.max_time,
        reporter=TqdmProgressReporter(description="Processing ion chamber"),
    )
    if result == 0:
        print(f"Successfully built run {args.run} ion-chamber root files.")
    else:
        print(f"Failed to build run {args.run} ion-chamber root files.")

if __name__ == "__main__":
    main()