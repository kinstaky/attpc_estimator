from __future__ import annotations

from typing import Any

from ...model.label import StoredLabel
from ...model.trace import TraceRecord
from ...process.bitflip import analyze_bitflip_trace
from ...process.trace_metrics import (
    analyze_trace_peaks,
    compute_first_derivative,
    pad_first_derivative,
    pad_second_derivative,
)


def serialize_trace_payload(
    record: TraceRecord,
    *,
    bitflip_baseline_threshold: float,
    label: StoredLabel | None,
    review_progress: dict[str, int] | None,
    include_run: bool,
    event_trace_count: int | None = None,
    event_id_range: dict[str, int] | None = None,
    trace_selector: dict[str, Any] | None = None,
    event_context: dict[str, Any] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    peak_separation: float = 50.0,
    peak_prominence: float = 20.0,
    peak_width: float = 50.0,
    peak_threshold: float = 0.0,
    peak_rel_height: float = 0.95,
) -> dict[str, Any]:
    bitflip_analysis = analyze_bitflip_trace(
        record.trace,
        baseline_threshold=bitflip_baseline_threshold,
    )
    padded_first_derivative = pad_first_derivative(
        compute_first_derivative(record.trace),
        int(record.trace.shape[0]),
    )
    padded_second_derivative = pad_second_derivative(
        bitflip_analysis.second_derivative,
        int(record.trace.shape[0]),
    )
    peak_analysis = analyze_trace_peaks(
        record.trace,
        peak_separation=peak_separation,
        peak_prominence=peak_prominence,
        peak_width=peak_width,
        peak_threshold=peak_threshold,
        rel_height=peak_rel_height,
    )
    payload = {
        "eventId": record.event_id,
        "traceId": record.trace_id,
        "detector": record.detector,
        "raw": record.raw.tolist(),
        "trace": record.trace.tolist(),
        "transformed": record.transformed.tolist(),
        "bitflipAnalysis": {
            "xIndices": list(range(int(record.trace.shape[0]))),
            "firstDerivative": padded_first_derivative.tolist(),
            "secondDerivative": padded_second_derivative.tolist(),
            "structures": [
                {
                    "startBaselineIndex": int(structure.start_baseline_index + 1),
                    "endBaselineIndex": int(structure.end_baseline_index + 1),
                }
                for structure in bitflip_analysis.structures
            ],
        },
        "currentLabel": serialize_label(label),
        "reviewProgress": review_progress,
        "eventTraceCount": event_trace_count,
        "eventIdRange": event_id_range,
    }
    if include_run:
        payload["run"] = int(record.run)
    if trace_selector is not None:
        payload["traceSelector"] = trace_selector
    if event_context is not None:
        payload["eventContext"] = event_context
    if trace_metadata is not None:
        payload["traceMetadata"] = trace_metadata
    if int(peak_analysis.get("peakCount", 0)) > 0:
        payload["peakAnalysis"] = peak_analysis
    return payload


def serialize_trace_metadata(record: TraceRecord) -> dict[str, Any] | None:
    hardware = record.hardware_id
    if record.detector == "ATTPC":
        return {"kind": "attpc", "padId": int(hardware[4])}
    if record.detector == "SI":
        side_code = int(hardware[0])
        return {
            "kind": "si",
            "layer": 0 if side_code < 2 else 1,
            "side": "front" if side_code % 2 == 0 else "back",
            "strip": int(hardware[1]),
        }
    if record.detector == "GAGG":
        return {
            "kind": "gagg",
            "layer": max(0, int(hardware[0]) - 1),
            "index": int(hardware[1]),
        }
    return None


def serialize_label(label: StoredLabel | None) -> dict[str, Any] | None:
    if label is None:
        return None
    return {"family": label.family, "label": label.label}
