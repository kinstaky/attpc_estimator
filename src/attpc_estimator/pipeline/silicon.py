import numpy as np
import uproot
import awkward as ak
from dataclasses import dataclass, field

from .progress_reporter import ProgressReporter
from attpc_storage import open_trace_reader, resolve_path
from pointcloud import TraceLength, fft_filter_traces, find_trace_peaks

@dataclass
class SiliconEvent:
	front_strip: list[int] = field(default_factory=list)
	front_energy: list[float] = field(default_factory=list)
	front_integral: list[float] = field(default_factory=list)
	front_fix_integral: list[float] = field(default_factory=list)
	front_time: list[float] = field(default_factory=list)
	back_strip: list[int] = field(default_factory=list)
	back_energy: list[float] = field(default_factory=list)
	back_integral: list[float] = field(default_factory=list)
	back_fix_integral: list[float] = field(default_factory=list)
	back_time: list[float] = field(default_factory=list)

def SortSide(
		strip: list[int],
		energy: list[float],
		integral: list[float],
		fix_integral: list[float],
		time: list[float]
	):
	size = min(len(energy), 8)
	order = np.argsort(energy, kind="stable")[::-1]
	strip[:] = np.array(strip)[order[:size]].tolist()
	energy[:] = np.array(energy)[order[:size]].tolist()
	integral[:] = np.array(integral)[order[:size]].tolist()
	fix_integral[:] = np.array(fix_integral)[order[:size]].tolist()
	time[:] = np.array(time)[order[:size]].tolist()


def SortSiliconEvent(event: SiliconEvent):
	# sort front side by energy
	SortSide(
		event.front_strip,
		event.front_energy,
		event.front_integral,
		event.front_fix_integral,
		event.front_time,
	)
	# sort back side by energy
	SortSide(
		event.back_strip,
		event.back_energy,
		event.back_integral,
		event.back_fix_integral,
		event.back_time
	)

def process_run(
	*,
	workspace: str,
	run: int,
	input_path: str,
	output_path: list[str],
	chunk_size: int = 1000,
	fft_window_scale: float,
	peak_separation: float,
	peak_prominence: float,
	peak_max_width: float,
	peak_threshold: float,
	rel_height: float,
	min_time: float,
	max_time: float,
	reporter: ProgressReporter,
) -> int:
	trace_reader = open_trace_reader(
		workspace=workspace,
		run=run,
		path=input_path,
		read_pad=False,
		read_si=True,
	)
	min_event, max_event = trace_reader.range()

	output_file0 = uproot.recreate(resolve_path(
		workspace=workspace,
		run=run,
		path=output_path[0]
	))
	output_file0.mktree("tree", {
		"front": "var * {"
			"strip: int32,"
			"energy: float64,"
			"integral: float64,"
			"fix_integral: float64,"
			"time: float64"
		"}",
		"back": "var * {"
			"strip: int32,"
			"energy: float64,"
			"integral: float64,"
			"fix_integral: float64,"
			"time: float64"
		"}",
		"run": "int32",
		"entry": "int32",
	})
	output_file1 = uproot.recreate(resolve_path(
		workspace=workspace,
		run=run,
		path=output_path[1]
	))
	output_file1.mktree("tree", {
		"front": "var * {"
		"strip: int32,"
		"energy: float64,"
		"integral: float64,"
        "fix_integral: float64,"
		"time: float64"
		"}",
		"back": "var * {"
		"strip: int32,"
		"energy: float64,"
		"integral: float64,"
        "fix_integral: float64,"
		"time: float64"
		"}",
		"run": "int32",
		"entry": "int32",
	})

	# report start
	reporter.report_start()
	last_percentage: int = 0
	# event list
	d1: list[SiliconEvent] = []
	d2: list[SiliconEvent] = []
	runs: list[int] = []
	event_ids: list[int] = []
	for event_id in range(0, max_event + 1):
		if int(event_id * 100 / max_event) > last_percentage:
			last_percentage = int(event_id * 100 / max_event)
			reporter.report_progress(current=last_percentage)
		if event_id < min_event:
			runs.append(run)
			event_ids.append(-1)
			d1.append(SiliconEvent())
			d2.append(SiliconEvent())
			continue
		try:
			trace = trace_reader.read_event(event_id)
		except LookupError:
			runs.append(run)
			event_ids.append(-1)
			d1.append(SiliconEvent())
			d2.append(SiliconEvent())
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
		order = np.argsort(peaks[:, 1], kind="stable")
		sorted_peaks = peaks[order[::-1]]
		d1_event = SiliconEvent()
		d2_event = SiliconEvent()
		for row in sorted_peaks:
			trace_id = int(row[0])
			side = int(si_data[trace_id, 0])
			strip = int(si_data[trace_id, 1])
			amplitude = float(row[1])
			integral = float(row[2])
			time = float(row[3])
			if (time < min_time or time > max_time) and not (run == 5152 or run == 5160):
				continue
			# integral in fix range, from ATTPCROOTV2
			fix_integral = float(np.sum(filtered[trace_id, int(time)-10:int(time)+14]))

			if side == 0:
				d1_event.front_strip.append(strip)
				d1_event.front_energy.append(amplitude)
				d1_event.front_integral.append(integral)
				d1_event.front_fix_integral.append(fix_integral)
				d1_event.front_time.append(time)
			elif side == 1:
				d1_event.back_strip.append(strip)
				d1_event.back_energy.append(amplitude)
				d1_event.back_integral.append(integral)
				d1_event.back_fix_integral.append(fix_integral)
				d1_event.back_time.append(time)
			elif side == 2:
				d2_event.front_strip.append(strip)
				d2_event.front_energy.append(amplitude)
				d2_event.front_integral.append(integral)
				d2_event.front_fix_integral.append(fix_integral)
				d2_event.front_time.append(time)
			elif side == 3:
				d2_event.back_strip.append(strip)
				d2_event.back_energy.append(amplitude)
				d2_event.back_integral.append(integral)
				d2_event.back_fix_integral.append(fix_integral)
				d2_event.back_time.append(time)
		SortSiliconEvent(d1_event)
		SortSiliconEvent(d2_event)
		runs.append(meta[1])
		event_ids.append(meta[2])
		d1.append(d1_event)
		d2.append(d2_event)
		if (event_id+1) % chunk_size == 0 or event_id == max_event:
			d1_front = ak.zip({
				"strip": ak.Array([event.front_strip for event in d1]),
				"energy": ak.Array([event.front_energy for event in d1]),
				"integral": ak.Array([event.front_integral for event in d1]),
				"fix_integral": ak.Array([event.front_fix_integral for event in d1]),
				"time": ak.Array([event.front_time for event in d1]),
			})
			d1_back = ak.zip({
				"strip": ak.Array([event.back_strip for event in d1]),
				"energy": ak.Array([event.back_energy for event in d1]),
				"integral": ak.Array([event.back_integral for event in d1]),
				"fix_integral": ak.Array([event.back_fix_integral for event in d1]),
				"time": ak.Array([event.back_time for event in d1]),
			})
			output_file0["tree"].extend({
				"front": d1_front,
				"back": d1_back,
				"run": np.array(runs, dtype=np.int32),
				"entry": np.array(event_ids, dtype=np.int32),
			})
			d2_front = ak.zip({
				"strip": ak.Array([event.front_strip for event in d2]),
				"energy": ak.Array([event.front_energy for event in d2]),
				"integral": ak.Array([event.front_integral for event in d2]),
				"fix_integral": ak.Array([event.front_fix_integral for event in d2]),
				"time": ak.Array([event.front_time for event in d2]),
			})
			d2_back = ak.zip({
				"strip": ak.Array([event.back_strip for event in d2]),
				"energy": ak.Array([event.back_energy for event in d2]),
				"integral": ak.Array([event.back_integral for event in d2]),
				"fix_integral": ak.Array([event.back_fix_integral for event in d2]),
				"time": ak.Array([event.back_time for event in d2]),
			})
			output_file1["tree"].extend({
				"front": d2_front,
				"back": d2_back,
				"run": np.array(runs, dtype=np.int32),
				"entry": np.array(event_ids, dtype=np.int32),
			})
			d1 = []
			d2 = []
			runs = []
			event_ids = []

	reporter.report_finish()
	return 0
