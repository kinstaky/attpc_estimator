from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from attpc_storage.hdf5 import PointcloudWriter, RawTraceReader
from pointcloud import (
    PadMapEntry,
    adapt_attpc,
    build_tpc_hits,
    create_tpc_drift_calibration_config,
    fft_filter_traces,
    find_trace_peaks,
)

from ..cli.config import parse_run, parse_toml_config, root_config_values, table_config_values
from ..detector.pads import PadInfo, PadLookup, load_pad_lookup
from ..process.bitflip import BITFLIP_BASELINE_DEFAULT, count_qualified_bitflip_segments_batch
from ..storage.run_paths import pointcloud_run_path, resolve_run_file


@dataclass(frozen=True, slots=True)
class ProgressState:
    total: int
    unit: str
    description: str


class ProgressReporter:
    def report_start(self, *, total: int, unit: str, description: str) -> None:
        raise NotImplementedError

    def report_progress(self, current: int, *, message: str = "") -> None:
        raise NotImplementedError

    def report_finish(self) -> None:
        raise NotImplementedError


class TqdmProgressReporter(ProgressReporter):
    def __init__(self) -> None:
        self._bar: tqdm | None = None
        self._current = 0

    def report_start(self, *, total: int, unit: str, description: str) -> None:
        self.report_finish()
        self._current = 0
        self._bar = tqdm(total=max(int(total), 0), desc=description, unit=unit)

    def report_progress(self, current: int, *, message: str = "") -> None:
        if self._bar is None:
            return
        bounded = max(int(current), 0)
        delta = max(0, bounded - self._current)
        if delta:
            self._bar.update(delta)
        self._current = max(self._current, bounded)
        if message:
            self._bar.set_postfix_str(message)

    def report_finish(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        self._current = 0


@dataclass(frozen=True, slots=True)
class FftConfig:
    baseline_window_scale: float


@dataclass(frozen=True, slots=True)
class BitflipConfig:
    baseline_threshold: float
    min_count: int


@dataclass(frozen=True, slots=True)
class PeakConfig:
    separation: float
    prominence: float
    max_width: float
    threshold: float
    rel_height: float


@dataclass(frozen=True, slots=True)
class DriftConfig:
    micromegas_time_bucket: float
    window_time_bucket: float
    detector_length: float


def _parse_args() -> argparse.Namespace:
    config_path, payload = parse_toml_config(sys.argv[1:])
    config = root_config_values(payload, allowed_keys={"trace_path", "workspace", "run"})
    pointcloud_config = table_config_values(
        payload,
        table="pointcloud",
        allowed_keys={"output", "micromegas_time_bucket", "window_time_bucket", "detector_length"},
    )
    fft_config = table_config_values(
        payload,
        table="fft",
        allowed_keys={"baseline_window_scale"},
    )
    bitflip_config = table_config_values(
        payload,
        table="bitflip",
        allowed_keys={"baseline", "min_count"},
    )
    peak_config = table_config_values(
        payload,
        table="peak",
        allowed_keys={
            "separation",
            "prominence",
            "max_width",
            "threshold",
            "rel_height",
        },
    )

    parser = argparse.ArgumentParser(description="Build phase-1 pointcloud HDF5 for one ATTPC run")
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
        help="Workspace directory used for default pointcloud output",
    )
    parser.add_argument(
        "-r",
        "--run",
        required="run" not in config,
        type=parse_run,
        default=config.get("run"),
        help="Run identifier to process",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=pointcloud_config.get("output"),
        help="Optional explicit output .h5 path",
    )
    parser.add_argument(
        "--micromegas-time-bucket",
        type=float,
        default=pointcloud_config.get("micromegas_time_bucket", 10.0),
    )
    parser.add_argument(
        "--window-time-bucket",
        type=float,
        default=pointcloud_config.get("window_time_bucket", 560.0),
    )
    parser.add_argument(
        "--detector-length",
        type=float,
        default=pointcloud_config.get("detector_length", 1000.0),
    )
    parser.add_argument(
        "--baseline-window-scale",
        type=float,
        default=fft_config.get("baseline_window_scale", 20.0),
    )
    parser.add_argument(
        "--bitflip-baseline",
        type=float,
        default=bitflip_config.get("baseline", BITFLIP_BASELINE_DEFAULT),
    )
    parser.add_argument(
        "--bitflip-min-count",
        type=int,
        default=bitflip_config.get("min_count", 1),
    )
    parser.add_argument("--peak-separation", type=float, default=peak_config.get("separation", 50.0))
    parser.add_argument("--peak-prominence", type=float, default=peak_config.get("prominence", 20.0))
    parser.add_argument("--peak-max-width", type=float, default=peak_config.get("max_width", 50.0))
    parser.add_argument("--peak-threshold", type=float, default=peak_config.get("threshold", 40.0))
    parser.add_argument("--peak-rel-height", type=float, default=peak_config.get("rel_height", 0.95))
    return parser.parse_args()


def _resolve_output_path(workspace: Path, run: int, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return pointcloud_run_path(workspace, run).resolve()


def _lookup_pad(pad_lookup: PadLookup, hardware_row: np.ndarray) -> PadInfo | None:
    cobo = int(hardware_row[0])
    asad = int(hardware_row[1])
    aget = int(hardware_row[2])
    channel = int(hardware_row[3])
    pad = pad_lookup.get_by_hardware(cobo=cobo, asad=asad, aget=aget, channel=channel)
    if pad is not None:
        return pad
    if hardware_row.shape[0] > 4:
        return pad_lookup.get_by_pad_id(int(hardware_row[4]))
    return None


class _PadLookupAdapter:
    def __init__(self, pad_lookup: PadLookup) -> None:
        self._pad_lookup = pad_lookup

    def get_by_hardware(self, hardware_row: np.ndarray) -> PadMapEntry | None:
        pad = _lookup_pad(self._pad_lookup, hardware_row)
        if pad is None:
            return None
        return PadMapEntry(
            pad_id=int(pad.pad_id),
            x=float(pad.cx),
            y=float(pad.cy),
            scale=float(pad.scale),
            time_offset=0.0,
            is_beam_pad=False,
        )

    def get_by_pad_id(self, pad_id: int) -> PadMapEntry | None:
        pad = self._pad_lookup.get_by_pad_id(int(pad_id))
        if pad is None:
            return None
        return PadMapEntry(
            pad_id=int(pad.pad_id),
            x=float(pad.cx),
            y=float(pad.cy),
            scale=float(pad.scale),
            time_offset=0.0,
            is_beam_pad=False,
        )


def process_run(
    *,
    trace_path: Path,
    workspace: Path,
    run: int,
    output_path: Path,
    fft_config: FftConfig,
    bitflip_config: BitflipConfig,
    peak_config: PeakConfig,
    drift_config: DriftConfig,
    progress: ProgressReporter,
) -> int:
    run_file = resolve_run_file(trace_path, run)
    pad_lookup = load_pad_lookup()
    pad_map = _PadLookupAdapter(pad_lookup)
    drift = create_tpc_drift_calibration_config(
        micromegas_time_bucket=drift_config.micromegas_time_bucket,
        window_time_bucket=drift_config.window_time_bucket,
        detector_length=drift_config.detector_length,
    )
    writer = PointcloudWriter(
        workspace=str(workspace),
        run=run,
        path=str(output_path),
    )
    reader = RawTraceReader(
        workspace=str(workspace),
        run=run,
        path=str(run_file),
    )
    written_events = 0

    try:
        writer.write_processing_attrs(
            {
                "fft_window_scale": float(fft_config.baseline_window_scale),
                "bitflip_baseline": float(bitflip_config.baseline_threshold),
                "bitflip_min_count": int(bitflip_config.min_count),
                "peak_separation": float(peak_config.separation),
                "peak_prominence": float(peak_config.prominence),
                "peak_max_width": float(peak_config.max_width),
                "peak_threshold": float(peak_config.threshold),
                "peak_rel_height": float(peak_config.rel_height),
                "micromegas_time_bucket": float(drift_config.micromegas_time_bucket),
                "window_time_bucket": float(drift_config.window_time_bucket),
                "detector_length": float(drift_config.detector_length),
            }
        )
        metadata = reader.describe_events()
        progress.report_start(
            total=metadata.valid_event_span,
            unit="event",
            description=f"Pointcloud run {run:04d}",
        )
        processed_events = 0
        for event_id in range(metadata.min_event, metadata.max_event + 1):
            if event_id in metadata.bad_events:
                continue
            meta, rows = reader.read_event(event_id)
            hardware, traces = adapt_attpc(rows)
            filtered = fft_filter_traces(
                traces.astype(np.float64, copy=False),
                baseline_window_scale=fft_config.baseline_window_scale,
            )
            source_indices = np.arange(filtered.shape[0], dtype=np.int64)
            if filtered.shape[0] > 0:
                qualified = count_qualified_bitflip_segments_batch(
                    filtered.astype(np.float32, copy=False),
                    baseline_threshold=bitflip_config.baseline_threshold,
                )
                valid_mask = qualified < bitflip_config.min_count
                filtered = filtered[valid_mask]
                source_indices = source_indices[valid_mask]

            if filtered.shape[0] == 0:
                event_hits = np.empty((0, 9), dtype=np.float64)
            else:
                peaks = find_trace_peaks(
                    filtered,
                    peak_separation=peak_config.separation,
                    peak_prominence=peak_config.prominence,
                    peak_max_width=peak_config.max_width,
                    peak_threshold=peak_config.threshold,
                    rel_height=peak_config.rel_height,
                )
                event_hits = build_tpc_hits(
                    hardware[source_indices],
                    peaks,
                    pad_map=pad_map,
                    drift_calibration=drift,
                    trace_ids=source_indices,
                )
            writer.write(meta, event_hits)
            written_events += 1
            processed_events += 1
            progress.report_progress(
                processed_events,
                message=f"event={event_id},hits={event_hits.shape[0]}",
            )
    finally:
        reader.close()
        writer.close()
        progress.report_finish()

    return written_events


def main() -> None:
    args = _parse_args()
    trace_path = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    run = int(args.run)
    output_path = _resolve_output_path(workspace, run, args.output)
    reporter = TqdmProgressReporter()
    written_events = process_run(
        trace_path=trace_path,
        workspace=workspace,
        run=run,
        output_path=output_path,
        fft_config=FftConfig(
            baseline_window_scale=float(args.baseline_window_scale),
        ),
        bitflip_config=BitflipConfig(
            baseline_threshold=float(args.bitflip_baseline),
            min_count=int(args.bitflip_min_count),
        ),
        peak_config=PeakConfig(
            separation=float(args.peak_separation),
            prominence=float(args.peak_prominence),
            max_width=float(args.peak_max_width),
            threshold=float(args.peak_threshold),
            rel_height=float(args.peak_rel_height),
        ),
        drift_config=DriftConfig(
            micromegas_time_bucket=float(args.micromegas_time_bucket),
            window_time_bucket=float(args.window_time_bucket),
            detector_length=float(args.detector_length),
        ),
        progress=reporter,
    )
    print(f"wrote {written_events} pointcloud events to {output_path}")


if __name__ == "__main__":
    main()
