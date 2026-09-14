from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


# The GAGG front-end is modeled as a fixed fifth-order RC-CR-like shaper.
# Keeping this constant avoids the otherwise strong n/tau degeneracy.
FIXED_GAMMA_N = 5.0
GAGG_HIGH_AMPLITUDE_EXCLUDED_INDEX_LIMIT = 16
GAGG_HIGH_AMPLITUDE_EXCLUDED_THRESHOLD = 7500.0


@dataclass(frozen=True, slots=True)
class GammaPulseFit:
    status: str
    fitted: np.ndarray | None = None
    amplitude: float | None = None
    n: float | None = None
    t0: float | None = None
    tau: float | None = None
    rmse: float | None = None
    r_squared: float | None = None
    reason: str | None = None
    baseline: float | None = None


def gamma_pulse(
    samples: np.ndarray, amplitude: float, n: float, t0: float, tau: float
) -> np.ndarray:
    offsets = np.asarray(samples, dtype=np.float64) - float(t0)
    result = np.zeros_like(offsets, dtype=np.float64)
    active = offsets >= 0.0
    active_offsets = offsets[active]
    result[active] = (
        float(amplitude)
        * np.power(active_offsets, float(n))
        * np.exp(-active_offsets / float(tau))
    )
    return result


def is_gagg_fit_eligible(
    trace_id: int, trace: np.ndarray, *, high_amplitude_threshold: float = GAGG_HIGH_AMPLITUDE_EXCLUDED_THRESHOLD
) -> bool:
    """Exclude high-amplitude layer-1 traces from gamma fitting.

    ``trace`` must already have had its baseline removed.
    """
    if int(trace_id) >= GAGG_HIGH_AMPLITUDE_EXCLUDED_INDEX_LIMIT:
        return True
    values = np.asarray(trace, dtype=np.float64)
    return not bool(np.any(values > high_amplitude_threshold))


def gamma_peak_amplitude(amplitude: float, tau: float) -> float:
    """Return the continuous maximum of a fixed-n gamma pulse."""
    offset = FIXED_GAMMA_N * float(tau)
    return float(amplitude) * offset**FIXED_GAMMA_N * np.exp(-FIXED_GAMMA_N)


def fit_raw_gamma_pulse(
    trace: np.ndarray,
    fit_mask: np.ndarray | None = None,
    *,
    fixed_tau: float | None = None,
    tau_bounds: tuple[float, float] | None = None,
    initial_tau: float | None = None,
) -> GammaPulseFit:
    """Fit raw ADC samples as a constant baseline plus a fifth-order pulse."""
    values = np.asarray(trace, dtype=np.float64).reshape(-1)
    if values.size < 8 or not np.all(np.isfinite(values)):
        return GammaPulseFit(status="unavailable", reason="trace is too short or contains non-finite samples")
    mask = np.ones(values.size, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool).reshape(-1)
    if mask.size != values.size or int(mask.sum()) < 8:
        return GammaPulseFit(status="unavailable", reason="too few usable samples to fit")
    if fixed_tau is not None and fixed_tau <= 0.0:
        return GammaPulseFit(status="unavailable", reason="configured fixed tau must be positive")
    tau_min, tau_max = tau_bounds or (0.25, float(values.size))
    if fixed_tau is None and not (0.0 < tau_min <= tau_max):
        return GammaPulseFit(status="unavailable", reason="invalid tau bounds")

    samples = np.arange(values.size, dtype=np.float64)
    usable = np.flatnonzero(mask)
    baseline0 = float(np.median(values[mask]))
    signal_values = values - baseline0
    peak_index = int(usable[np.argmax(signal_values[mask])])
    peak_height = float(signal_values[peak_index])
    if peak_index < 2 or peak_height <= 0.0:
        return GammaPulseFit(status="unavailable", reason="trace has no positive pulse above baseline")
    centered = values[mask] - float(np.mean(values[mask]))
    total_ss = float(np.dot(centered, centered))
    if total_ss <= np.finfo(np.float64).eps:
        return GammaPulseFit(status="unavailable", reason="trace has no variation")
    onset = np.flatnonzero(signal_values[mask] >= peak_height * 0.1)
    if onset.size == 0:
        return GammaPulseFit(status="unavailable", reason="trace has no pulse onset")
    onset_index = int(usable[onset[0]])
    upper_t0 = float(peak_index) - 0.05
    if upper_t0 <= 0.0:
        return GammaPulseFit(status="unavailable", reason="trace peak occurs too early to fit")

    best: np.ndarray | None = None
    best_ss = np.inf
    for t0 in (0.0, min(float(onset_index), upper_t0), max(0.0, float(peak_index) * 0.5)):
        tau0 = float(fixed_tau) if fixed_tau is not None else float(initial_tau or (peak_index - t0) / FIXED_GAMMA_N)
        tau0 = min(max(tau_min, tau0), tau_max) if fixed_tau is None else tau0
        offset = max(float(peak_index) - t0, 0.05)
        shape = offset**FIXED_GAMMA_N * np.exp(-offset / tau0)
        amplitude0 = max(peak_height / max(shape, 1e-12), 1e-12)
        try:
            if fixed_tau is None:
                result = least_squares(
                    lambda p: (p[0] + gamma_pulse(samples, p[1], FIXED_GAMMA_N, p[2], p[3]))[mask] - values[mask],
                    x0=[baseline0, amplitude0, t0, tau0],
                    bounds=([-np.inf, 0.0, 0.0, tau_min], [np.inf, np.inf, upper_t0, tau_max]), max_nfev=4000,
                )
            else:
                result = least_squares(
                    lambda p: (p[0] + gamma_pulse(samples, p[1], FIXED_GAMMA_N, p[2], fixed_tau))[mask] - values[mask],
                    x0=[baseline0, amplitude0, t0],
                    bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, upper_t0]), max_nfev=4000,
                )
        except (FloatingPointError, ValueError):
            continue
        if result.success and np.all(np.isfinite(result.x)):
            ss = float(np.dot(result.fun, result.fun))
            if ss < best_ss:
                best, best_ss = result.x, ss
    if best is None:
        return GammaPulseFit(status="unavailable", reason="raw gamma fit did not converge")
    if fixed_tau is None:
        baseline, amplitude, t0, tau = (float(value) for value in best)
    else:
        baseline, amplitude, t0 = (float(value) for value in best)
        tau = float(fixed_tau)
    fitted = baseline + gamma_pulse(samples, amplitude, FIXED_GAMMA_N, t0, tau)
    return GammaPulseFit(status="ok", fitted=fitted.astype(np.float32), amplitude=amplitude,
                         n=FIXED_GAMMA_N, t0=t0, tau=tau,
                         rmse=float(np.sqrt(best_ss / int(mask.sum()))),
                         r_squared=float(1.0 - best_ss / total_ss), baseline=baseline)


def fit_gamma_pulse(
    trace: np.ndarray,
    fit_mask: np.ndarray | None = None,
    *,
    tau_bounds: tuple[float, float] | None = None,
    initial_tau: float | None = None,
) -> GammaPulseFit:
    values = np.asarray(trace, dtype=np.float64).reshape(-1)
    if values.size < 8:
        return GammaPulseFit(status="unavailable", reason="trace is too short to fit")
    if not np.all(np.isfinite(values)):
        return GammaPulseFit(status="unavailable", reason="trace contains non-finite samples")

    mask = np.ones(values.size, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool).reshape(-1)
    if mask.size != values.size or int(mask.sum()) < 8:
        return GammaPulseFit(status="unavailable", reason="too few unclipped samples to fit")

    usable_indexes = np.flatnonzero(mask)
    peak_index = int(usable_indexes[np.argmax(values[mask])])
    peak_value = float(values[peak_index])
    if peak_index < 2 or peak_value <= 0.0:
        return GammaPulseFit(status="unavailable", reason="trace has no positive pulse to fit")
    centered = values[mask] - float(np.mean(values[mask]))
    total_sum_squares = float(np.dot(centered, centered))
    if total_sum_squares <= np.finfo(np.float64).eps:
        return GammaPulseFit(status="unavailable", reason="trace has no variation to fit")

    samples = np.arange(values.size, dtype=np.float64)
    onset_index = int(usable_indexes[np.flatnonzero(values[mask] >= peak_value * 0.1)[0]])
    upper_t0 = float(peak_index) - 0.05
    tau_min, tau_max = tau_bounds or (0.25, float(values.size))
    if not (0.0 < tau_min <= tau_max):
        return GammaPulseFit(status="unavailable", reason="invalid tau bounds")
    lower_bounds = np.asarray([0.0, 0.0, tau_min], dtype=np.float64)
    upper_bounds = np.asarray([np.inf, upper_t0, tau_max], dtype=np.float64)

    candidates: list[np.ndarray] = []
    for t0_value in (0.0, float(onset_index), max(0.0, float(peak_index) * 0.5)):
        t0_value = min(t0_value, upper_t0)
        tau_value = (
            float(initial_tau)
            if initial_tau is not None
            else (float(peak_index) - t0_value) / FIXED_GAMMA_N
        )
        tau_value = min(max(tau_min, tau_value), tau_max)
        peak_offset = max(float(peak_index) - t0_value, 0.05)
        shape_at_peak = peak_offset**FIXED_GAMMA_N * np.exp(-peak_offset / tau_value)
        amplitude = max(peak_value / max(shape_at_peak, 1e-12), 1e-12)
        candidates.append(np.asarray([amplitude, t0_value, tau_value]))

    best = None
    best_sum_squares = np.inf
    for initial in candidates:
        try:
            result = least_squares(
                lambda parameters: gamma_pulse(
                    samples, parameters[0], FIXED_GAMMA_N, parameters[1], parameters[2]
                )[mask] - values[mask],
                x0=np.minimum(np.maximum(initial, lower_bounds), upper_bounds),
                bounds=(lower_bounds, upper_bounds),
                max_nfev=4000,
            )
        except (FloatingPointError, ValueError):
            continue
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        sum_squares = float(np.dot(result.fun, result.fun))
        if sum_squares < best_sum_squares:
            best = result.x
            best_sum_squares = sum_squares

    if best is None:
        return GammaPulseFit(status="unavailable", reason="gamma fit did not converge")

    fitted = gamma_pulse(samples, best[0], FIXED_GAMMA_N, best[1], best[2])
    return GammaPulseFit(
        status="ok",
        fitted=fitted.astype(np.float32),
        amplitude=float(best[0]),
        n=FIXED_GAMMA_N,
        t0=float(best[1]),
        tau=float(best[2]),
        rmse=float(np.sqrt(best_sum_squares / int(mask.sum()))),
        r_squared=float(1.0 - (best_sum_squares / total_sum_squares)),
    )


def fit_gamma_pulse_fixed_tau(
    trace: np.ndarray, tau: float, fit_mask: np.ndarray | None = None
) -> GammaPulseFit:
    """Fit amplitude and onset while keeping the shaper tau fixed."""
    values = np.asarray(trace, dtype=np.float64).reshape(-1)
    if values.size < 8 or not np.all(np.isfinite(values)) or tau <= 0.0:
        return GammaPulseFit(status="unavailable", reason="invalid fixed-tau fit input")
    peak_index = int(np.argmax(values))
    peak_value = float(values[peak_index])
    if peak_index < 2 or peak_value <= 0.0:
        return GammaPulseFit(status="unavailable", reason="trace has no positive pulse to fit")
    mask = np.ones(values.size, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool).reshape(-1)
    if mask.size != values.size or int(mask.sum()) < 8:
        return GammaPulseFit(status="unavailable", reason="too few unclipped samples to fit")
    samples = np.arange(values.size, dtype=np.float64)
    upper_t0 = float(peak_index) - 0.05
    onset = int(np.flatnonzero(values >= peak_value * 0.1)[0])
    candidates = (0.0, min(float(onset), upper_t0), max(0.0, float(peak_index) * 0.5))
    best: np.ndarray | None = None
    best_ss = np.inf
    for t0 in candidates:
        offset = max(float(peak_index) - t0, 0.05)
        shape = offset**FIXED_GAMMA_N * np.exp(-offset / tau)
        initial = [max(peak_value / max(shape, 1e-12), 1e-12), t0]
        try:
            result = least_squares(
                lambda parameters: gamma_pulse(samples, parameters[0], FIXED_GAMMA_N, parameters[1], tau)[mask] - values[mask],
                x0=initial,
                bounds=([0.0, 0.0], [np.inf, upper_t0]),
                max_nfev=4000,
            )
        except (FloatingPointError, ValueError):
            continue
        if result.success and np.all(np.isfinite(result.x)):
            ss = float(np.dot(result.fun, result.fun))
            if ss < best_ss:
                best, best_ss = result.x, ss
    if best is None:
        return GammaPulseFit(status="unavailable", reason="fixed-tau gamma fit did not converge")
    fitted = gamma_pulse(samples, best[0], FIXED_GAMMA_N, best[1], tau)
    centered = values - float(np.mean(values))
    total_ss = float(np.dot(centered, centered))
    return GammaPulseFit(
        status="ok", fitted=fitted.astype(np.float32), amplitude=float(best[0]),
        n=FIXED_GAMMA_N, t0=float(best[1]), tau=float(tau),
        rmse=float(np.sqrt(best_ss / int(mask.sum()))),
        r_squared=float(1.0 - best_ss / total_ss) if total_ss > np.finfo(float).eps else None,
    )
