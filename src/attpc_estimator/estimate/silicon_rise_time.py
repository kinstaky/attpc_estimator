import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import uproot

from ..cli.config import (
	parse_toml_config, root_config_values, table_config_values
)
from attpc_storage import (
	MergerLegacyReader, MergerV2Reader, open_trace_reader, resolve_path
)
from ..pipeline.progress_reporter import TqdmProgressReporter
from pointcloud import TraceLength, fft_filter_traces, find_trace_peaks

def _parse_args() -> argparse.Namespace:
	config_path, payload = parse_toml_config(sys.argv[1:])
	config = root_config_values(payload, allowed_keys={"trace_path", "workspace", "run"})
	baseline_config = table_config_values(
		payload,
		table="si.baseline",
		allowed_keys={"fft_window_scale"},
	)
	amplitude_config = table_config_values(
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
	time_config = table_config_values(
		payload,
		table="si.time",
		allowed_keys={
			"min",
			"max",
		},
	)

	parser = argparse.ArgumentParser(description="Build silicon ingot root files")
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
		type=int,
		default=time_config.get("min", 89),
	)
	parser.add_argument(
		"--max-time",
		type=int,
		default=time_config.get("max", 99),
	)
	return parser.parse_args()

@dataclass
class Event:
	side: int = 0,
	strip: int = 0,
	centroid: float = 0.0,
	amplitude: float = 0.0,
	rise_time: float = 0.0,
	fall_time: float = 0.0,
	run: int = 0,
	entry: int = 0,
	trace: int = 0,


def process_run(
	*,
	trace_reader: MergerLegacyReader | MergerV2Reader,
	output_file: uproot.WritableDirectory,
	chunk_size: int = 1000,
	fft_window_scale: float,
	peak_separation: float,
	peak_prominence: float,
	peak_max_width: float,
	peak_threshold: float,
	rel_height: float,
	min_time: int,
	max_time: int,
) -> None:
	min_event, max_event = trace_reader.range()
	# report start
	reporter = TqdmProgressReporter(
		description="Getting silicon rise time"
	)
	reporter.report_start()
	last_percentage: int = 0
	# event list
	events: list[Event] = []
	for event_id in range(min_event, max_event + 1):
		if int(event_id * 100 / max_event) > last_percentage:
			last_percentage = int(event_id * 100 / max_event)
			reporter.report_progress(current=last_percentage)
		try:
			trace = trace_reader.read_event(event_id)
		except LookupError:
			continue
		meta, si_event = trace["si"]
		si_data = si_event[:]
		filtered = fft_filter_traces(
			si_data[:, 2:],
			trace_length=TraceLength.TB512,
			baseline_window_scale=fft_window_scale,
		)
		peaks = find_trace_peaks(
			filtered,
			peak_separation=peak_separation,
			peak_prominence=peak_prominence,
			peak_max_width=peak_max_width,
			peak_threshold=peak_threshold,
			rel_height=rel_height,
		)
		for row in peaks:
			time = row[3]
			if (time < min_time or time > max_time):
				continue
			event = Event()
			trace_id = int(row[0])
			event.side = int(si_data[trace_id, 0])
			event.strip = int(si_data[trace_id, 1])
			peak_idx = int(row[3])
			si_trace = filtered[trace_id]
			peak_pol2_c = si_trace[peak_idx]
			peak_pol2_b = (si_trace[peak_idx+1] - si_trace[peak_idx-1])/2
			peak_pol2_a = (si_trace[peak_idx+1] + si_trace[peak_idx-1])/2 - peak_pol2_c
			relative_centroid = -0.5 * peak_pol2_b / peak_pol2_a
			event.centroid = row[3] + relative_centroid
			event.amplitude = \
				peak_pol2_a * relative_centroid * relative_centroid \
				+ peak_pol2_b * relative_centroid + peak_pol2_c
			event.peak = row[1]
			# print(f"event {event_id}, trace {trace_id}, side {event.side}, strip {event.strip}")
			# print(f"peak index: {peak_idx}, peak valud: {event.peak}")
			# print(f"peak+1: {si_trace[peak_idx+1]}, peak-1: {si_trace[peak_idx-1]}")
			# print(f"a: {peak_pol2_a}, b: {peak_pol2_b}, c: {peak_pol2_c}")
			# print(f"centroid: {event.centroid}, amplitude: {event.amplitude}")
			# calculate rise time
			# get 90% amplitude time in rise
			start = peak_idx
			for idx in range(start-1, 0, -1):
				if (si_trace[idx] < 0.9*event.amplitude):
					event.rise_time = float(idx)
					event.rise_time += \
						(0.9*event.amplitude - si_trace[idx]) \
						/ (si_trace[idx+1] - si_trace[idx])
					start = idx
					break
			# get 10% amplitude time in rise
			for idx in range(start-1, 0, -1):
				# print(f"idx: {idx}, y: {si_trace[idx]}, B: {0.1*event.amplitude}")
				if (si_trace[idx] < 0.1*event.amplitude):
					event.rise_time -= float(idx)
					event.rise_time -= \
						(0.1*event.amplitude - si_trace[idx]) \
						/ (si_trace[idx+1] - si_trace[idx])
					break
			# calculate fall time
			# get 90% amplitude in fall
			start = peak_idx
			for idx in range(start+1, 512):
				if (si_trace[idx] < 0.9*event.amplitude):
					event.fall_time = -float(idx)
					event.fall_time -= \
						(0.9*event.amplitude - si_trace[idx-1]) \
						/ (si_trace[idx] - si_trace[idx-1])
					start = idx
					break
			for idx in range(start+1, 512):
				if (si_trace[idx] < 0.1*event.amplitude):
					event.fall_time += float(idx)
					event.fall_time += \
						(0.1*event.amplitude - si_trace[idx-1]) \
						/ (si_trace[idx] - si_trace[idx-1])
					break
			# print(f"rise time: {event.rise_time}, fall time: {event.fall_time}")
			event.run = meta[1]
			event.entry = meta[2]
			event.trace = trace_id
			events.append(event)

		if (event_id+1) % chunk_size == 0 or event_id == max_event:
			output_file["tree"].extend({
				"side": np.array([event.side for event in events], dtype=np.int32),
				"strip": np.array([event.strip for event in events], dtype=np.int32),
				"centroid": np.array([event.centroid for event in events], dtype=np.float64),
				"amplitude": np.array([event.amplitude for event in events], dtype=np.float64),
				"peak": np.array([event.peak for event in events], dtype=np.float64),
				"rise_time": np.array([event.rise_time for event in events], dtype=np.float64),
				"fall_time": np.array([event.fall_time for event in events], dtype=np.float64),
				"run": np.array([event.run for event in events], dtype=np.int32),
				"entry": np.array([event.entry for event in events], dtype=np.int32),
				"trace": np.array([event.trace for event in events], dtype=np.int32),
			})
			events = []

	reporter.report_finish()

def main() -> None:
	args = _parse_args()
	if not args.run:
		raise SystemExit("no runs provided; pass --run for each run to process")

	workspace = Path(args.workspace).expanduser().resolve()

	trace_reader = open_trace_reader(
		workspace=str(workspace),
		run=int(args.run),
		path="hdf5/run_<run>.h5",
		read_pad=False,
		read_si=True,
	)

	output_file = uproot.recreate(resolve_path(
		workspace=str(workspace),
		run=int(args.run),
		path="estimate/si_rise_time_<run>.root"
	))
	output_file.mktree("tree", {
		"side": "int32",
		"strip": "int32",
		"centroid": "float64",
		"amplitude": "float64",
		"peak": "float64",
		"rise_time": "float64",
		"fall_time": "float64",
		"run": "int32",
		"entry": "int32",
		"trace": "int32",
	})
	process_run(
		trace_reader=trace_reader,
		output_file=output_file,
		chunk_size=1000,
		fft_window_scale=args.baseline_window_scale,
		peak_separation=args.peak_separation,
		peak_prominence=args.peak_prominence,
		peak_max_width=args.peak_width,
		peak_threshold=args.peak_threshold,
		rel_height=args.peak_rel_height,
		min_time=args.min_time,
		max_time=args.max_time,
	)

if __name__ == "__main__":
	main()