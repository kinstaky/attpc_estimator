import numpy as np
import uproot
import awkward as ak
from scipy import signal
from dataclasses import dataclass, field
from pathlib import Path

from .progress_reporter import ProgressReporter
from attpc_storage import open_trace_reader, resolve_path
from attpc_runs import RunDatabase, RunDatabaseError
from pointcloud import TraceLength, fft_filter_traces, find_trace_peaks
from ..utils.gagg_processing import (
	GaggTraceProcessingRule,
	default_gagg_trace_rules,
	require_gagg_peak_config,
)
from ..process.gagg_fit import (
	fit_raw_gamma_pulse,
	gamma_peak_amplitude,
)
from ..process.gagg_parameters import GaggFitParameters


def _gagg_clip_value(index: int) -> float:
	return 16383.0 if index < 33 else 4095.0


def _preload_gagg_processing(
	*, rules: dict[int, GaggTraceProcessingRule], valid_indexes: tuple[int, ...]
) -> tuple[dict[int, tuple[GaggTraceProcessingRule, object]], tuple[tuple[float, object, tuple[int, ...]], ...]]:
	preloaded: dict[int, tuple[GaggTraceProcessingRule, object]] = {}
	missing: list[int] = []
	for index in valid_indexes:
		rule = rules.get(index)
		if rule is None:
			missing.append(index)
			continue
		try:
			peak_config = require_gagg_peak_config(rule, index=index)
		except ValueError:
			missing.append(index)
			continue
		preloaded[index] = (rule, peak_config)
	if missing:
		raise ValueError("GAGG valid indexes lack baseline and/or amplitude settings: " + ", ".join(map(str, missing)))
	groups: dict[tuple[float, object], list[int]] = {}
	for index, (rule, peak_config) in preloaded.items():
		groups.setdefault((rule.baseline_window_scale, peak_config), []).append(index)
	return preloaded, tuple((scale, config, tuple(indexes)) for (scale, config), indexes in groups.items())


def _peak_left_ip(trace: np.ndarray, *, peak_index: int, peak_config: object) -> float | None:
	"""Regenerate SciPy's left width crossing for one selected peak."""
	peaks, properties = signal.find_peaks(
		trace,
		distance=peak_config.peak_separation,
		prominence=peak_config.peak_prominence,
		width=(1.0, peak_config.peak_width),
		rel_height=peak_config.rel_height,
	)
	matches = np.flatnonzero(peaks == int(peak_index))
	if matches.size == 0:
		return None
	return float(properties["left_ips"][int(matches[0])])

@dataclass
class GaggEvent:
	indexes: list[int] = field(default_factory=list)
	energy: list[float] = field(default_factory=list)
	integral: list[float] = field(default_factory=list)
	time: list[float] = field(default_factory=list)
	saturation: list[bool] = field(default_factory=list)
	camp: list[float] = field(default_factory=list)
	tau: list[float] = field(default_factory=list)
	amp: list[float] = field(default_factory=list)
	base: list[float] = field(default_factory=list)
	t0: list[float] = field(default_factory=list)
	ftau: list[float] = field(default_factory=list)
	famp: list[float] = field(default_factory=list)
	fbase: list[float] = field(default_factory=list)
	ft0: list[float] = field(default_factory=list)

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
	min_time: float,
	max_time: float,
	reporter: ProgressReporter | None = None,
	gagg_trace_rules: dict[int, GaggTraceProcessingRule] | None = None,
	gagg_valid_indexes: tuple[int, ...] = tuple(range(41)),
	gagg_fit_parameters: GaggFitParameters | None = None,
	run_tags_path: str | Path | None = None,
) -> int:
	run_database = RunDatabase(
		workbook_path=Path(run_tags_path) if run_tags_path is not None else Path(workspace) / "runs.xlsx",
		database_path=Path(workspace) / "runs.sqlite",
	)
	try:
		eligibility = run_database.check(run=run, detector="gagg")
	except RunDatabaseError as exc:
		print(f"Skipping GaGG run {run}: {exc}")
		return 1
	if not eligibility.valid:
		print(f"Skipping GaGG run {run}: {eligibility.reason}")
		return 1
	try:
		execution_id = run_database.start_pipeline(pipeline="gagg", run=run)
	except RunDatabaseError as exc:
		print(f"Skipping GaGG run {run}: {exc}")
		return 1

	trace_reader = open_trace_reader(
		workspace=workspace,
		run=run,
		path=input_path,
		read_pad=False,
		read_si=False,
		read_gagg=True,
	)
	min_event, max_event = trace_reader.range()
	rules = gagg_trace_rules or default_gagg_trace_rules(
		baseline_window_scale=fft_window_scale,
	)
	preloaded, processing_groups = _preload_gagg_processing(
		rules=rules, valid_indexes=gagg_valid_indexes
	)
	if gagg_fit_parameters is None:
		raise ValueError("GAGG calibration parameters are required")
	calibration = gagg_fit_parameters

	output_file = uproot.recreate(resolve_path(
		workspace=workspace,
		run=run,
		path=output_path
	))
	output_file.mktree("tree", {
		"": "var * {"
		"index: int32,"
		"energy: float64,"
		"integral: float64,"
		"time: float64,"
		"saturation: bool,"
		"camp: float64,"
		"tau: float64,"
		"amp: float64,"
		"base: float64,"
		"t0: float64,"
		"ftau: float64,"
		"famp: float64,"
		"fbase: float64,"
		"ft0: float64"
		"}",
		"run": "int32",
		"entry": "int32",
	})

	# report start
	if reporter is not None:
		reporter.report_start()
	last_percentage: int = 0
	# event list
	events: list[GaggEvent] = []
	runs: list[int] = []
	event_ids: list[int] = []
	for event_id in range(0, max_event + 1):
		if int(event_id * 100 / max_event) > last_percentage:
			last_percentage = int(event_id * 100 / max_event)
			if reporter is not None:
				reporter.report_progress(current=last_percentage)
		if event_id < min_event:
			runs.append(run)
			event_ids.append(-1)
			events.append(GaggEvent())
			continue
		try:
			trace = trace_reader.read_event(event_id)
		except LookupError:
			runs.append(run)
			event_ids.append(-1)
			events.append(GaggEvent())
			continue
		meta, gagg_event = trace["gagg"]
		rows = np.asarray(gagg_event[:], dtype=np.float32)
		flags = rows[:, 0].astype(np.int32)
		gagg_data = rows[:, 1:]

		# Peak finding always uses the original baseline-removed waveform.  Fits
		# are attached afterwards to raw ADC samples and never replace this path.
		selected_peaks: list[tuple[int, np.ndarray, np.ndarray]] = []
		for scale, peak_config, group_indexes in processing_groups:
			trace_indexes = [index for index in group_indexes if index < gagg_data.shape[0] and not (flags[index] & 0b10)]
			if not trace_indexes:
				continue
			traces = fft_filter_traces(gagg_data[trace_indexes], trace_length=TraceLength.TB256, baseline_window_scale=scale)
			peaks = find_trace_peaks(
				traces,
				peak_separation=peak_config.peak_separation,
				peak_prominence=peak_config.peak_prominence,
				peak_max_width=peak_config.peak_width,
				peak_threshold=peak_config.peak_threshold,
				rel_height=peak_config.rel_height
			)
			for peak in peaks:
				position = int(peak[0])
				selected_peaks.append((trace_indexes[position], peak, traces[position]))
		sorted_peaks = sorted(selected_peaks, key=lambda item: float(item[1][1]), reverse=True)
		event = GaggEvent()
		for index, row, cleaned_trace in sorted_peaks:
			amplitude = float(row[1])
			integral = float(row[2])
			time = float(row[3])
			saturated = bool(flags[index] & 0b01) if index >= 16 else False
			corrected_amplitude = 0.0
			if index < 16:
				expected = calibration.p0[index] + calibration.p1[index] * integral
				if amplitude > calibration.linear_clipped and abs(expected - amplitude) > calibration.linear_window:
					saturated = True
					corrected_amplitude = expected

			fit_mask = np.ones(gagg_data.shape[1], dtype=bool)
			if index >= 16 and saturated:
				fit_mask = gagg_data[index] < _gagg_clip_value(index)
			elif index < 16 and saturated:
				left_ip = _peak_left_ip(cleaned_trace, peak_index=int(time), peak_config=preloaded[index][1])
				if left_ip is not None:
					start = int(round(left_ip))
					fit_mask[max(0, start + 5) : min(fit_mask.size, start + 25)] = False

			configured_tau = calibration.tau_for(index)
			free_fit = fit_raw_gamma_pulse(
				gagg_data[index], fit_mask, tau_bounds=(1.5, 3.5), initial_tau=configured_tau,
			)
			fixed_fit = fit_raw_gamma_pulse(gagg_data[index], fit_mask, fixed_tau=configured_tau)
			free_values = (0.0, 0.0, 0.0, 0.0)
			if free_fit.status == "ok" and free_fit.amplitude is not None and free_fit.tau is not None and free_fit.baseline is not None and free_fit.t0 is not None:
				free_values = (float(free_fit.tau), gamma_peak_amplitude(float(free_fit.amplitude), float(free_fit.tau)), float(free_fit.baseline), float(free_fit.t0))
			fixed_values = (0.0, 0.0, 0.0, 0.0)
			if fixed_fit.status == "ok" and fixed_fit.amplitude is not None and fixed_fit.tau is not None and fixed_fit.baseline is not None and fixed_fit.t0 is not None:
				fixed_values = (float(fixed_fit.tau), gamma_peak_amplitude(float(fixed_fit.amplitude), float(fixed_fit.tau)), float(fixed_fit.baseline), float(fixed_fit.t0))
			event.indexes.append(index)
			event.energy.append(amplitude)
			event.integral.append(integral)
			event.time.append(time)
			event.saturation.append(saturated)
			event.camp.append(corrected_amplitude)
			event.tau.append(free_values[0])
			event.amp.append(free_values[1])
			event.base.append(free_values[2])
			event.t0.append(free_values[3])
			event.ftau.append(fixed_values[0])
			event.famp.append(fixed_values[1])
			event.fbase.append(fixed_values[2])
			event.ft0.append(fixed_values[3])
		runs.append(meta[1])
		event_ids.append(meta[2])
		events.append(event)
		if (event_id+1) % chunk_size == 0 or event_id == max_event:
			output_file["tree"].extend({
				"": ak.zip({
					"index": ak.Array([evt.indexes for evt in events]),
					"energy": ak.Array([evt.energy for evt in events]),
					"integral": ak.Array([evt.integral for evt in events]),
					"time": ak.Array([evt.time for evt in events]),
					"saturation": ak.Array([evt.saturation for evt in events]),
					"camp": ak.Array([evt.camp for evt in events]),
					"tau": ak.Array([evt.tau for evt in events]),
					"amp": ak.Array([evt.amp for evt in events]),
					"base": ak.Array([evt.base for evt in events]),
					"t0": ak.Array([evt.t0 for evt in events]),
					"ftau": ak.Array([evt.ftau for evt in events]),
					"famp": ak.Array([evt.famp for evt in events]),
					"fbase": ak.Array([evt.fbase for evt in events]),
					"ft0": ak.Array([evt.ft0 for evt in events]),
				}),
				"run": np.array(runs, dtype=np.int32),
				"entry": np.array(event_ids, dtype=np.int32),
			})
			events = []
			runs = []
			event_ids = []

	output_file.close()
	run_database.finish_pipeline(pipeline="gagg", run=run, execution_id=execution_id)
	if reporter is not None:
		reporter.report_finish()
	return 0
