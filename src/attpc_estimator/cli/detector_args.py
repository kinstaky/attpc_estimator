from __future__ import annotations

import argparse

from .config import config_relative_path, parse_toml_config, root_config_values, table_config_values


def parse_detector_pipeline_args(
    argv: list[str],
    *,
    detector: str,
    description: str,
    min_time_default: float | int,
    max_time_default: int,
) -> argparse.Namespace:
    """Parse the shared IC and silicon pipeline options from one detector table."""
    config_path, payload = parse_toml_config(argv)
    config = root_config_values(payload, allowed_keys={"trace_path", "workspace", "run"})
    run_tags = table_config_values(payload, table="run_tags", allowed_keys={"path"})
    baseline = table_config_values(
        payload, table=f"{detector}.baseline", allowed_keys={"fft_window_scale"}
    )
    amplitude = table_config_values(
        payload,
        table=f"{detector}.amplitude",
        allowed_keys={"peak_separation", "peak_prominence", "peak_width", "peak_threshold", "rel_height"},
    )
    time = table_config_values(
        payload, table=f"{detector}.time", allowed_keys={"min", "max"}
    )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config", dest="config_file", default=str(config_path))
    parser.add_argument(
        "-t", "--trace-path", required="trace_path" not in config,
        default=config.get("trace_path"),
        help="Path to a trace file or a directory containing run_<run>.h5 files",
    )
    parser.add_argument(
        "-w", "--workspace", required="workspace" not in config,
        default=config.get("workspace"),
        help="Workspace directory used for detector ROOT output",
    )
    parser.add_argument(
        "-r", "--run", required="run" not in config,
        default=config.get("run"), help="Run identifier to process",
    )
    for option, key, default in (
        ("baseline-window-scale", "fft_window_scale", 20.0),
        ("peak-separation", "peak_separation", 50.0),
        ("peak-prominence", "peak_prominence", 20.0),
        ("peak-width", "peak_width", 500.0),
        ("peak-threshold", "peak_threshold", 100.0),
        ("peak-rel-height", "rel_height", 0.85),
    ):
        values = baseline if key == "fft_window_scale" else amplitude
        parser.add_argument(f"--{option}", type=float, default=values.get(key, default))
    parser.add_argument(
        "--min-time", type=type(min_time_default), default=time.get("min", min_time_default)
    )
    parser.add_argument(
        "--max-time", type=int, default=time.get("max", max_time_default)
    )
    args = parser.parse_args(argv)
    run_tags_path = run_tags.get("path")
    if not isinstance(run_tags_path, str) or not run_tags_path.strip():
        raise ValueError("run_tags.path must name the runs workbook")
    args.run_tags_path = config_relative_path(config_path, run_tags_path)
    return args
