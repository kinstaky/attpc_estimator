import numpy as np
import awkward as ak
import uproot
from dataclasses import dataclass, field

from .progress_reporter import ProgressReporter
from attpc_storage import open_trace_reader, resolve_path
from pointcloud import TraceLength, fft_filter_traces, find_trace_peaks

@dataclass
class IonChamberEvent:
	valid: bool = False
	time: float = 0.0
	energy: float = 0.0
	integral: float = 0.0
	peak_centroid: list[int] = field(default_factory=list)
	peak_amplitude: list[float] = field(default_factory=list)
	peak_integral: list[float] = field(default_factory=list)

def process_run(
	*,
	workspace: str,
	run: int,
	input_path: str,
	output_path: str,
	chunk_size: int = 1000,
	fft_window_scale: float,
	peak_separation: float,
	peak_prominence: float,
	peak_max_width: float,
	peak_threshold: float,
	rel_height: float,
	min_time: int,
	max_time: int,
	reporter: ProgressReporter,
) -> int:
	trace_reader = open_trace_reader(
		workspace=workspace,
		run=run,
		path=input_path,
		read_pad=False,
		read_ic=True,
	)
	min_event, max_event = trace_reader.range()

	output_file = uproot.recreate(resolve_path(
		workspace=workspace,
		run=run,
		path=output_path
	))
	output_file.mktree("tree", {
		"valid": "bool",
		"time": "int32",
		"energy": "float64",
		"integral": "float64",
		"peak": "var * {"
			"centroid: int32,"
			"amplitude: float64,"
			"integral: float64"
		"}",
		"run": "int32",
		"entry": "int32",
	})

	# report start
	reporter.report_start()
	last_percentage = 0
	# event list
	events: list[IonChamberEvent] = []
	runs: list[int] = []
	event_ids: list[int] = []
	for event_id in range(0, max_event+1):
		if int(event_id * 100 / max_event) > last_percentage:
			last_percentage = int(event_id * 100 / max_event)
			reporter.report_progress(current=last_percentage)
		if event_id < min_event:
			runs.append(run)
			event_ids.append(-1)
			events.append(IonChamberEvent())
			continue
		try:
			trace = trace_reader.read_event(event_id)
		except LookupError:
			runs.append(run)
			event_ids.append(-1)
			events.append(IonChamberEvent())
			continue
		meta, ic_event = trace["ic"]
		ic_data = ic_event[:]
		filtered = fft_filter_traces(
			ic_data,
			trace_length=TraceLength.TB256,
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
		order = np.argsort(peaks[:, 3], kind="stable")
		sorted_peaks = peaks[order]
		event = IonChamberEvent()
		for peak in sorted_peaks:
			if peak[3] >= min_time and peak[3] <= max_time:
				event.valid = True
				event.time = peak[3]
				event.energy = peak[1]
				event.integral = peak[2]
			event.peak_centroid.append(peak[3])
			event.peak_amplitude.append(peak[1])
			event.peak_integral.append(peak[2])
			if len(event.peak_centroid) >= 8:
				break
		runs.append(meta[1])
		event_ids.append(meta[2])
		events.append(event)

		if (event_id+1) % chunk_size == 0 or event_id == max_event:
			zip_peaks = ak.zip({
				"centroid": ak.Array([event.peak_centroid for event in events]),
				"amplitude": ak.Array([event.peak_amplitude for event in events]),
				"integral": ak.Array([event.peak_integral for event in events]),
			})
			output_file["tree"].extend({
				"valid": np.array([event.valid for event in events], dtype=np.bool_),
				"time": np.array([event.time for event in events], dtype=np.int32),
				"energy": np.array([event.energy for event in events], dtype=np.float64),
				"integral": np.array([event.integral for event in events], dtype=np.float64),
				"peak": zip_peaks,
				"run": np.array(runs, dtype=np.int32),
				"entry": np.array(event_ids, dtype=np.int32),
			})
			events = []
			runs = []
			event_ids = []

	reporter.report_finish()
	return 0