from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    config_relative_path,
    parse_run,
    parse_toml_config,
    root_config_values,
    table_config_entries,
    table_config_values,
)
from ..process.gagg_batch_fit import fit_gagg_run
from ..process.gagg_parameters import load_gagg_fit_parameters
from ..pipeline.progress_reporter import TqdmProgressReporter
from ..utils.gagg_processing import build_gagg_trace_rules


def main() -> None:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(payload, allowed_keys={"trace_path", "workspace", "run"})
    gagg_config = table_config_values(
        payload, table="gagg", allowed_keys={"fit_parameters_path"}
    )
    parser = argparse.ArgumentParser(description="Fit valid GAGG traces and write Parquet results")
    parser.add_argument("-c", "--config", default=str(config_path))
    parser.add_argument("-t", "--trace-path", required="trace_path" not in config, default=config.get("trace_path"))
    parser.add_argument("-w", "--workspace", required="workspace" not in config, default=config.get("workspace"))
    parser.add_argument("-r", "--run", type=parse_run, required="run" not in config, default=config.get("run"), help="First run (inclusive)")
    parser.add_argument("--output-root", action="store_true", help="Also write one CERN ROOT file per run")
    parser.add_argument("--progress", action="store_true", help="Show an interactive progress bar (disabled by default for log-friendly output)")
    args = parser.parse_args()
    fit_parameters_path = gagg_config.get("fit_parameters_path")
    if not isinstance(fit_parameters_path, str) or not fit_parameters_path.strip():
        raise SystemExit("gagg.fit_parameters_path must name a GAGG fit-parameters TOML file")
    try:
        parameters = load_gagg_fit_parameters(
            config_relative_path(config_path, fit_parameters_path)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    baseline_entries = table_config_entries(payload, table="gagg.baseline", allowed_keys={"index", "fft_window_scale"})
    amplitude_entries = table_config_entries(payload, table="gagg.amplitude", allowed_keys={"index", "peak_separation", "peak_prominence", "peak_width", "peak_threshold", "rel_height"})
    rules = build_gagg_trace_rules(baseline_entries=baseline_entries, amplitude_entries=amplitude_entries)
    run = int(args.run)
    output, count = fit_gagg_run(
        trace_path=Path(args.trace_path).expanduser().resolve(),
        workspace=Path(args.workspace).expanduser().resolve(),
        run=run, rules=rules,
        reporter=TqdmProgressReporter(description="Fitting GAGG traces") if args.progress else None,
        output_root=bool(args.output_root),
        parameters=parameters,
    )
    if output is None:
        raise SystemExit(f"trace file for run {run} was not found")
    print(f"Wrote {count} GAGG fit rows to {output}")


if __name__ == "__main__":
    main()
