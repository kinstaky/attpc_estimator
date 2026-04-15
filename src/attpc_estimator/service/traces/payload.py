from __future__ import annotations

from typing import Any

from ...model.label import StoredLabel
from ...model.trace import TraceRecord
from ...process.bitflip import analyze_bitflip_trace
from ...process.trace_metrics import (
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
    payload = {
        "eventId": record.event_id,
        "traceId": record.trace_id,
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
    return payload


def serialize_label(label: StoredLabel | None) -> dict[str, Any] | None:
    if label is None:
        return None
    return {"family": label.family, "label": label.label}
