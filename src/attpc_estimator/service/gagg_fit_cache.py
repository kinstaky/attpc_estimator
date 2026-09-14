from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import numpy as np
from scipy import signal
from scipy.optimize import least_squares
from pointcloud import find_trace_peaks

from ..model.trace import TraceRecord, TraceRef
from ..process.gagg_fit import FIXED_GAMMA_N, GammaPulseFit, fit_raw_gamma_pulse, gamma_pulse, is_gagg_fit_eligible
from ..process.gagg_parameters import GaggFitParameters
from ..utils.gagg_processing import GaggTraceProcessingRule, require_gagg_peak_config


def _gagg_unclipped_mask(record: TraceRecord) -> np.ndarray | None:
    """Return a raw-ADC mask for a clipped layer-2 GAGG trace."""
    if int(record.trace_id) < 16:
        return None
    raw = np.asarray(record.raw, dtype=np.float64).reshape(-1)
    if raw.size != np.asarray(record.trace).size:
        return None
    clip_value = 16383.0 if int(record.trace_id) < 33 else 4095.0
    mask = raw < clip_value
    return mask if not bool(np.all(mask)) else None


def _low_index_saturation_mask(
    record: TraceRecord,
    *,
    parameters: GaggFitParameters | None,
    rules: dict[int, GaggTraceProcessingRule] | None,
) -> np.ndarray | None:
    """Return the onset-relative 20-sample mask for saturated index <16."""
    index = int(record.trace_id)
    if index >= 16 or parameters is None or rules is None or index not in rules:
        return None
    peak_config = require_gagg_peak_config(rules[index], index=index)
    trace = np.asarray(record.trace, dtype=np.float64).reshape(-1)
    peaks = find_trace_peaks(
        trace[np.newaxis, :], peak_separation=peak_config.peak_separation,
        peak_prominence=peak_config.peak_prominence, peak_max_width=peak_config.peak_width,
        peak_threshold=peak_config.peak_threshold, rel_height=peak_config.rel_height,
    )
    if not len(peaks):
        return None
    peak = max(peaks, key=lambda item: float(item[1]))
    expected = parameters.p0[index] + parameters.p1[index] * float(peak[2])
    if not (float(peak[1]) > parameters.linear_clipped and abs(expected - float(peak[1])) > parameters.linear_window):
        return None
    positions, properties = signal.find_peaks(
        trace, distance=peak_config.peak_separation, prominence=peak_config.peak_prominence,
        width=(1.0, peak_config.peak_width), rel_height=peak_config.rel_height,
    )
    matches = np.flatnonzero(positions == int(peak[3]))
    if matches.size == 0:
        return None
    raw = np.asarray(record.raw, dtype=np.float64).reshape(-1)
    if raw.size != trace.size:
        return None
    start = int(round(float(properties["left_ips"][int(matches[0])])))
    mask = np.ones(raw.size, dtype=bool)
    mask[max(0, start + 5) : min(mask.size, start + 25)] = False
    return mask


@dataclass(frozen=True, slots=True)
class BrowseGaggFits:
    """Raw-trace fits used only by the WebUI Browse fit view."""

    recovery: GammaPulseFit
    clipped_free: GammaPulseFit | None = None
    fixed_tau: GammaPulseFit | None = None


def _unavailable(reason: str) -> GammaPulseFit:
    return GammaPulseFit(status="unavailable", reason=reason)


def _fit_raw_gamma_legacy(
    trace: np.ndarray,
    fit_mask: np.ndarray | None = None,
    *,
    fixed_tau: float | None = None,
) -> GammaPulseFit:
    """Fit raw ADC samples with a free constant baseline plus gamma pulse."""
    values = np.asarray(trace, dtype=np.float64).reshape(-1)
    if values.size < 8 or not np.all(np.isfinite(values)):
        return _unavailable("trace is too short or contains non-finite samples")
    mask = np.ones(values.size, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool).reshape(-1)
    if mask.size != values.size or int(mask.sum()) < 8:
        return _unavailable("too few unclipped samples to fit")
    if fixed_tau is not None and fixed_tau <= 0.0:
        return _unavailable("configured fixed tau must be positive")

    samples = np.arange(values.size, dtype=np.float64)
    usable_indexes = np.flatnonzero(mask)
    baseline0 = float(np.median(values[mask]))
    signal = values - baseline0
    peak_index = int(usable_indexes[np.argmax(signal[mask])])
    peak_height = float(signal[peak_index])
    if peak_index < 2 or peak_height <= 0.0:
        return _unavailable("trace has no positive pulse above its fitted baseline")
    centered = values[mask] - float(np.mean(values[mask]))
    total_ss = float(np.dot(centered, centered))
    if total_ss <= np.finfo(np.float64).eps:
        return _unavailable("trace has no variation")

    onset_candidates = np.flatnonzero(signal[mask] >= peak_height * 0.1)
    if onset_candidates.size == 0:
        return _unavailable("trace has no pulse onset")
    onset_index = int(usable_indexes[onset_candidates[0]])
    upper_t0 = float(peak_index) - 0.05
    if upper_t0 <= 0.0:
        return _unavailable("trace peak occurs too early to fit")
    t0_candidates = (0.0, min(float(onset_index), upper_t0), max(0.0, float(peak_index) * 0.5))

    best: np.ndarray | None = None
    best_ss = np.inf
    for t0 in t0_candidates:
        tau0 = max(0.25, (float(peak_index) - t0) / FIXED_GAMMA_N) if fixed_tau is None else float(fixed_tau)
        offset = max(float(peak_index) - t0, 0.05)
        shape = offset**FIXED_GAMMA_N * np.exp(-offset / tau0)
        amplitude0 = max(peak_height / max(shape, 1e-12), 1e-12)
        try:
            if fixed_tau is None:
                result = least_squares(
                    lambda p: (p[0] + gamma_pulse(samples, p[1], FIXED_GAMMA_N, p[2], p[3]))[mask] - values[mask],
                    x0=[baseline0, amplitude0, t0, tau0],
                    bounds=([-np.inf, 0.0, 0.0, 0.25], [np.inf, np.inf, upper_t0, float(values.size)]),
                    max_nfev=4000,
                )
            else:
                result = least_squares(
                    lambda p: (p[0] + gamma_pulse(samples, p[1], FIXED_GAMMA_N, p[2], float(fixed_tau)))[mask] - values[mask],
                    x0=[baseline0, amplitude0, t0],
                    bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, upper_t0]),
                    max_nfev=4000,
                )
        except (FloatingPointError, ValueError):
            continue
        if result.success and np.all(np.isfinite(result.x)):
            ss = float(np.dot(result.fun, result.fun))
            if ss < best_ss:
                best, best_ss = result.x, ss
    if best is None:
        return _unavailable("raw gamma fit did not converge")

    if fixed_tau is None:
        baseline, amplitude, t0, tau = (float(value) for value in best)
    else:
        baseline, amplitude, t0 = (float(value) for value in best)
        tau = float(fixed_tau)
    fitted = baseline + gamma_pulse(samples, amplitude, FIXED_GAMMA_N, t0, tau)
    return GammaPulseFit(
        status="ok", fitted=fitted.astype(np.float32), amplitude=amplitude,
        n=FIXED_GAMMA_N, t0=t0, tau=tau,
        rmse=float(np.sqrt(best_ss / int(mask.sum()))),
        r_squared=float(1.0 - best_ss / total_ss), baseline=baseline,
    )


def _fit_raw_gamma(
    trace: np.ndarray,
    fit_mask: np.ndarray | None = None,
    *,
    fixed_tau: float | None = None,
    tau_bounds: tuple[float, float] | None = None,
    initial_tau: float | None = None,
) -> GammaPulseFit:
    """Shared raw baseline-plus-gamma fit used by Browse and the pipeline."""
    return fit_raw_gamma_pulse(
        trace, fit_mask, fixed_tau=fixed_tau,
        tau_bounds=tau_bounds, initial_tau=initial_tau,
    )


class GaggFitCache:
    def __init__(
        self, *, parameters: GaggFitParameters | None = None,
        gagg_trace_rules: dict[int, GaggTraceProcessingRule] | None = None,
        max_entries: int = 32,
    ) -> None:
        self._parameters = parameters
        self._gagg_trace_rules = gagg_trace_rules
        self._max_entries = max_entries
        self._results: OrderedDict[TraceRef, BrowseGaggFits] = OrderedDict()
        self._pending: dict[TraceRef, Future[BrowseGaggFits]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gagg-fit")

    def get_or_compute(self, record: TraceRecord) -> BrowseGaggFits:
        future = self._schedule(record)
        result = future.result()
        self._store(record_ref(record), result)
        return result

    def prefetch(self, records: list[TraceRecord]) -> None:
        for record in records:
            self._schedule(record)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            self._results.clear()
            self._pending.clear()

    def _schedule(self, record: TraceRecord) -> Future[BrowseGaggFits]:
        ref = record_ref(record)
        with self._lock:
            cached = self._results.get(ref)
            if cached is not None:
                self._results.move_to_end(ref)
                return completed_future(cached)
            pending = self._pending.get(ref)
            if pending is not None:
                return pending
            raw = np.array(record.raw, dtype=np.float32, copy=True)
            mask = _gagg_unclipped_mask(record)
            if mask is None:
                mask = _low_index_saturation_mask(
                    record, parameters=self._parameters, rules=self._gagg_trace_rules,
                )
            tau = self._tau_for(record.trace_id)
            future = self._executor.submit(_compute_browse_fits, raw, mask, tau)
            self._pending[ref] = future
            future.add_done_callback(lambda done, key=ref: self._complete(key, done))
            return future

    def _tau_for(self, index: int) -> float | None:
        if self._parameters is None:
            return None
        try:
            return self._parameters.tau_for(int(index))
        except ValueError:
            return None

    def _complete(self, ref: TraceRef, future: Future[BrowseGaggFits]) -> None:
        try:
            result = future.result()
        except Exception:
            with self._lock:
                self._pending.pop(ref, None)
            return
        self._store(ref, result)

    def _store(self, ref: TraceRef, result: BrowseGaggFits) -> None:
        with self._lock:
            self._pending.pop(ref, None)
            self._results[ref] = result
            self._results.move_to_end(ref)
            while len(self._results) > self._max_entries:
                self._results.popitem(last=False)


def _compute_browse_fits(raw: np.ndarray, mask: np.ndarray | None, fixed_tau: float | None) -> BrowseGaggFits:
    fit_kwargs = (
        {"tau_bounds": (1.5, 3.5), "initial_tau": fixed_tau}
        if fixed_tau is not None else {}
    )
    recovery = _fit_raw_gamma(raw, mask, **fit_kwargs)
    if mask is None:
        return BrowseGaggFits(recovery=recovery)
    clipped_free = _fit_raw_gamma(raw, **fit_kwargs)
    fixed = _fit_raw_gamma(raw, mask, fixed_tau=fixed_tau) if fixed_tau is not None else _unavailable("no configured tau for this GAGG index")
    return BrowseGaggFits(recovery=recovery, clipped_free=clipped_free, fixed_tau=fixed)


def record_ref(record: TraceRecord) -> TraceRef:
    return TraceRef(run=record.run, event_id=record.event_id, trace_id=record.trace_id)


def completed_future(result: BrowseGaggFits) -> Future[BrowseGaggFits]:
    future: Future[BrowseGaggFits] = Future()
    future.set_result(result)
    return future
