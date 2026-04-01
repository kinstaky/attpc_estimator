from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import signal

from .cli_config import parse_toml_config
from .utils import (
    _read_labeled_trace_rows,
    compute_frequency_distribution,
    preprocess_traces,
    sample_cdf_points,
)

CDF_BIN = 60
OSCILLATION_CUTOFF = 0.6
ZERO_PEAK_AMPLITUDE_CUTOFF = 40.0
RELABEL_LABEL_CHOICES = ("noise", "oscillation", "saturation")
NOISE_RELABEL = "noise"
OSCILLATION_RELABEL = "oscillation"
SATURATION_RELABEL = "saturation"
OSCILLATION_LABEL_KEY = "strange:oscillation"
ZERO_PEAK_LABEL_KEY = "normal:0"
ONE_PEAK_LABEL_KEY = "normal:1"


def main() -> None:
    args = _parse_args()
    trace_path = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    run_token = args.run
    selected_run = int(run_token) if run_token is not None else None

    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")

    try:
        rows, metrics = build_relabel_rows(
            trace_path=trace_path,
            workspace=workspace,
            run=selected_run,
            label=args.label,
            baseline_window_scale=args.baseline_window_scale,
            peak_separation=args.peak_separation,
            peak_prominence=args.peak_prominence,
            peak_width=args.peak_width,
        )
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc

    if run_token is None:
        output_path = workspace / "labeled_relabel.npy"
    else:
        output_path = workspace / f"run_{run_token}_labeled_relabel.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, rows)

    print(f"saved relabel rows with shape {rows.shape} to {output_path}")
    print(f"total traces: {len(rows)}")
    print(f"relabel changes: {metrics['changed_count']}")
    for name, ratio in _ratio_items_for_label(args.label, metrics):
        _print_ratio(name, ratio)


def _parse_args() -> argparse.Namespace:
    config_path, config = parse_toml_config(
        sys.argv[1:],
        allowed_keys={
            "trace_path",
            "workspace",
            "run",
            "label",
            "baseline_window_scale",
            "peak_separation",
            "peak_prominence",
            "peak_width",
        },
    )
    parser = argparse.ArgumentParser(
        description="Relabel labeled traces using FFT CDF and amplitude heuristics",
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
        required="trace_path" not in config,
        default=config.get("trace_path"),
        help="Path to a trace file or a directory containing run_<run>.h5 files",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        required="workspace" not in config,
        default=config.get("workspace"),
        help="Workspace directory containing the SQLite labels database and outputs",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=_parse_run,
        default=config.get("run"),
        help="Optional run identifier. When omitted, relabel labeled traces across all runs present in both workspace and database.",
    )
    parser.add_argument(
        "--label",
        choices=RELABEL_LABEL_CHOICES,
        required="label" not in config,
        default=config.get("label"),
        help="Relabel mode: noise uses the current zero-peak heuristic, oscillation uses the FFT CDF rule, saturation is reserved for future implementation.",
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=config.get("baseline_window_scale", 10.0),
        help="Baseline-removal filter scale used before taking the FFT",
    )
    parser.add_argument(
        "--peak-separation",
        type=float,
        default=config.get("peak_separation", 50.0),
        help="Minimum separation between peaks",
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=config.get("peak_prominence", 20.0),
        help="Prominence of peaks",
    )
    parser.add_argument(
        "--peak-width",
        type=float,
        default=config.get("peak_width", 50.0),
        help="Maximum width of peaks",
    )
    return parser.parse_args()


def _parse_run(value: str) -> str:
    run = value.strip()
    if not run or not run.isdigit():
        raise argparse.ArgumentTypeError("run must contain only digits")
    return run


def build_relabel_rows(
    trace_path: Path,
    workspace: Path,
    label: str,
    run: int | None = None,
    baseline_window_scale: float = 10.0,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
) -> tuple[np.ndarray, dict[str, tuple[int, int] | int]]:
    _validate_relabel_label(label)
    traces, labeled_rows = _read_labeled_trace_rows(trace_path=trace_path, workspace_path=workspace, run=run)

    if len(labeled_rows) == 0:
        return _build_structured_rows([], []), _build_ratio_metrics(label, [], [])

    cleaned = preprocess_traces(traces, baseline_window_scale=baseline_window_scale)
    old_labels = [row.label_key for row in labeled_rows]
    if label == NOISE_RELABEL:
        amplitudes = np.asarray(
            [
                _max_peak_amplitude(
                    row=trace,
                    peak_separation=peak_separation,
                    peak_prominence=peak_prominence,
                    peak_width=peak_width,
                )
                for trace in cleaned
            ],
            dtype=np.float32,
        )
        new_labels = [
            _relabel_noise(old_label=old_label, amplitude=float(amplitude))
            for old_label, amplitude in zip(old_labels, amplitudes, strict=True)
        ]
    elif label == OSCILLATION_RELABEL:
        spectrum = compute_frequency_distribution(cleaned)
        f60 = sample_cdf_points(
            spectrum,
            thresholds=np.asarray([CDF_BIN], dtype=np.int64),
        )[:, 0]
        new_labels = [
            _relabel_oscillation(old_label=old_label, f60_value=float(f60_value))
            for old_label, f60_value in zip(old_labels, f60, strict=True)
        ]
    else:
        raise ValueError(f"unsupported relabel label: {label}")
    return _build_structured_rows(labeled_rows, new_labels), _build_ratio_metrics(label, old_labels, new_labels)


def _max_peak_amplitude(
    row: np.ndarray,
    peak_separation: float,
    peak_prominence: float,
    peak_width: float,
) -> float:
    peaks, _ = signal.find_peaks(
        row,
        distance=peak_separation,
        prominence=peak_prominence,
        width=(1.0, peak_width),
        rel_height=0.95,
    )
    if peaks.size == 0:
        return 0.0
    return float(np.max(row[peaks]))


def _validate_relabel_label(label: str) -> None:
    if label == SATURATION_RELABEL:
        raise NotImplementedError("relabel label 'saturation' is not implemented yet")
    if label not in RELABEL_LABEL_CHOICES:
        raise ValueError(f"unsupported relabel label: {label}")


def _relabel_noise(old_label: str, amplitude: float) -> str:
    if amplitude < ZERO_PEAK_AMPLITUDE_CUTOFF:
        return ZERO_PEAK_LABEL_KEY
    return old_label


def _relabel_oscillation(old_label: str, f60_value: float) -> str:
    if f60_value < OSCILLATION_CUTOFF:
        return OSCILLATION_LABEL_KEY
    return old_label


def _build_structured_rows(
    labeled_rows: list,
    new_labels: list[str],
) -> np.ndarray:
    max_label_length = max(
        [len(row.label_key) for row in labeled_rows] + [len(label) for label in new_labels] + [1]
    )
    dtype = np.dtype(
        [
            ("run", np.int64),
            ("event_id", np.int64),
            ("trace_id", np.int64),
            ("old_label", f"U{max_label_length}"),
            ("new_label", f"U{max_label_length}"),
        ]
    )
    rows = np.empty(len(labeled_rows), dtype=dtype)
    for index, (row, new_label) in enumerate(zip(labeled_rows, new_labels, strict=True)):
        rows[index] = (row.run, row.event_id, row.trace_id, row.label_key, new_label)
    return rows


def _build_ratio_metrics(
    label: str,
    old_labels: list[str],
    new_labels: list[str],
) -> dict[str, tuple[int, int] | int]:
    changed_count = sum(old_label != new_label for old_label, new_label in zip(old_labels, new_labels, strict=True))
    metrics: dict[str, tuple[int, int] | int] = {"changed_count": changed_count}
    if label == NOISE_RELABEL:
        metrics["normal0_to_normal0"] = _label_transition_ratio(
            old_labels, new_labels, ZERO_PEAK_LABEL_KEY, ZERO_PEAK_LABEL_KEY
        )
        metrics["normal1_to_normal0"] = _label_transition_ratio(
            old_labels, new_labels, ONE_PEAK_LABEL_KEY, ZERO_PEAK_LABEL_KEY
        )
        return metrics

    metrics["oscillation_to_oscillation"] = _label_transition_ratio(
        old_labels,
        new_labels,
        OSCILLATION_LABEL_KEY,
        OSCILLATION_LABEL_KEY,
    )
    metrics["normal1_to_oscillation"] = _label_transition_ratio(
        old_labels,
        new_labels,
        ONE_PEAK_LABEL_KEY,
        OSCILLATION_LABEL_KEY,
    )
    return metrics


def _ratio_items_for_label(
    label: str,
    metrics: dict[str, tuple[int, int] | int],
) -> list[tuple[str, tuple[int, int]]]:
    if label == NOISE_RELABEL:
        return [
            ("old normal:0 -> new normal:0", metrics["normal0_to_normal0"]),
            ("old normal:1 -> new normal:0", metrics["normal1_to_normal0"]),
        ]
    if label == OSCILLATION_RELABEL:
        return [
            ("old strange:oscillation -> new strange:oscillation", metrics["oscillation_to_oscillation"]),
            ("old normal:1 -> new strange:oscillation", metrics["normal1_to_oscillation"]),
        ]
    raise ValueError(f"unsupported relabel label: {label}")


def _label_transition_ratio(
    old_labels: list[str],
    new_labels: list[str],
    source_label: str,
    target_label: str,
) -> tuple[int, int]:
    numerator = 0
    denominator = 0
    for old_label, new_label in zip(old_labels, new_labels, strict=True):
        if old_label != source_label:
            continue
        denominator += 1
        if new_label == target_label:
            numerator += 1
    return numerator, denominator


def _print_ratio(name: str, ratio: tuple[int, int]) -> None:
    numerator, denominator = ratio
    if denominator == 0:
        print(f"{name}: 0/0 = nan")
        return
    print(f"{name}: {numerator}/{denominator} = {numerator / denominator:.6f}")


if __name__ == "__main__":
    main()
