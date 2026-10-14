import argparse
import sys
from pathlib import Path

from .detector_args import parse_detector_pipeline_args
from ..pipeline.silicon import process_run
from ..pipeline.progress_reporter import TqdmProgressReporter


def _parse_args() -> argparse.Namespace:
    return parse_detector_pipeline_args(
        sys.argv[1:],
        detector="si",
        description="Build silicon ingot ROOT files",
        min_time_default=89,
        max_time_default=99,
    )


def main() -> None:
    args = _parse_args()
    if not args.run:
        raise SystemExit("no runs provided; pass --run for each run to process")

    workspace = Path(args.workspace).expanduser().resolve()

    result = process_run(
        workspace=str(workspace),
        run=int(args.run),
        input_path="hdf5/run_<run>.h5",
        output_path=[
            "ingot/t0d1_<run>.root",
            "ingot/t0d2_<run>.root",
        ],
        fft_window_scale=args.baseline_window_scale,
        peak_separation=args.peak_separation,
        peak_prominence=args.peak_prominence,
        peak_max_width=args.peak_width,
        peak_threshold=args.peak_threshold,
        rel_height=args.peak_rel_height,
        min_time=args.min_time,
        max_time=args.max_time,
        reporter=TqdmProgressReporter(description="Processing T0 silicon"),
        run_tags_path=args.run_tags_path,
    )
    if result == 0:
        print(f"Successfully built run {args.run} T0 root files.")
    else:
        print(f"Failed to build run {args.run} T0 root files.")


if __name__ == "__main__":
    main()
