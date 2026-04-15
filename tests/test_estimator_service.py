from __future__ import annotations

import random

import h5py
import numpy as np
import pytest

from attpc_estimator.model.trace import TraceRecord
from attpc_estimator.service.estimator import EstimatorService
from attpc_estimator.service.traces.payload import serialize_trace_payload
from tests.hdf5_fixtures import write_legacy_hdf5


def write_hdf5_input(path) -> None:
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["min_event"] = 1
        events.attrs["max_event"] = 2
        events.attrs["bad_events"] = np.array([], dtype=np.int64)

        event_1 = events.create_group("event_1")
        get_1 = event_1.create_group("get")
        get_1.create_dataset(
            "pads",
            data=np.array(
                [
                    [10, 11, 12, 13, 14, 1, 2, 3],
                    [20, 21, 22, 23, 24, 4, 5, 6],
                ],
                dtype=np.float32,
            ),
        )

        event_2 = events.create_group("event_2")
        get_2 = event_2.create_group("get")
        get_2.create_dataset(
            "pads",
            data=np.array(
                [
                    [30, 31, 32, 33, 34, 7, 8, 9],
                ],
                dtype=np.float32,
            ),
        )


def _trace_from_second_derivative(second_diff: list[float]) -> np.ndarray:
    values = np.asarray(second_diff, dtype=np.float32)
    trace = np.zeros(values.size + 2, dtype=np.float32)
    for index, value in enumerate(values):
        trace[index + 2] = value + (2.0 * trace[index + 1]) - trace[index]
    return trace


def test_review_mode_filters_traces_and_stops_at_bounds(tmp_path) -> None:
    trace_path = tmp_path / "run_0001.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)
    service.create_strange_label("Noise", "n")
    service.set_session(mode="label", run=1)
    service.assign_label(event_id=1, trace_id=0, family="normal", label="0")
    service.assign_label(event_id=1, trace_id=1, family="strange", label="Noise")
    service.assign_label(event_id=2, trace_id=0, family="normal", label="4")

    payload = service.set_session(mode="review", run=1, source="label_set", family="normal")
    assert payload["session"] == {
        "mode": "review",
        "run": 1,
        "source": "label_set",
        "family": "normal",
        "label": None,
        "filterFile": None,
    }
    assert payload["traceCount"] == 2
    first = payload["trace"]
    assert first == {
        "run": 1,
        "eventId": 1,
        "traceId": 0,
        "raw": [1.0, 2.0, 3.0],
        "trace": first["trace"],
        "transformed": first["transformed"],
        "bitflipAnalysis": first["bitflipAnalysis"],
        "currentLabel": {"family": "normal", "label": "0"},
        "reviewProgress": {"current": 1, "total": 2},
    }
    assert first["trace"] == payload["trace"]["trace"]
    assert first["transformed"] == payload["trace"]["transformed"]
    assert first["bitflipAnalysis"] == payload["trace"]["bitflipAnalysis"]
    assert (first["eventId"], first["traceId"]) == (1, 0)
    assert first["currentLabel"] == {"family": "normal", "label": "0"}
    assert first["reviewProgress"] == {"current": 1, "total": 2}

    second = service.next_trace()
    assert (second["eventId"], second["traceId"]) == (2, 0)
    assert second["currentLabel"] == {"family": "normal", "label": "4"}
    assert second["reviewProgress"] == {"current": 2, "total": 2}

    still_last = service.next_trace()
    assert (still_last["eventId"], still_last["traceId"]) == (2, 0)

    previous = service.previous_trace()
    assert (previous["eventId"], previous["traceId"]) == (1, 0)

    still_first = service.previous_trace()
    assert (still_first["eventId"], still_first["traceId"]) == (1, 0)

    review_strange = service.set_session(
        mode="review",
        run=1,
        source="label_set",
        family="strange",
        label="Noise",
    )
    strange = review_strange["trace"]
    assert (strange["eventId"], strange["traceId"]) == (1, 1)
    assert strange["currentLabel"] == {"family": "strange", "label": "Noise"}
    assert strange["reviewProgress"] == {"current": 1, "total": 1}


def test_label_and_review_stacks_are_independent(tmp_path) -> None:
    random.seed(7)
    trace_path = tmp_path / "run_0002.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)

    first_label_trace = service.set_session(mode="label", run=2)["trace"]
    second_label_trace = service.next_trace()
    rewound_label_trace = service.previous_trace()

    assert (rewound_label_trace["eventId"], rewound_label_trace["traceId"]) == (
        first_label_trace["eventId"],
        first_label_trace["traceId"],
    )

    service.assign_label(
        event_id=first_label_trace["eventId"],
        trace_id=first_label_trace["traceId"],
        family="normal",
        label="3",
    )

    review_mode = service.set_session(
        mode="review",
        run=2,
        source="label_set",
        family="normal",
        label="3",
    )
    assert review_mode["session"]["mode"] == "review"

    review_trace = review_mode["trace"]
    assert (review_trace["eventId"], review_trace["traceId"]) == (
        first_label_trace["eventId"],
        first_label_trace["traceId"],
    )

    resumed = service.set_session(mode="label", run=2)
    resumed_label_trace = resumed["trace"]
    assert (resumed_label_trace["eventId"], resumed_label_trace["traceId"]) == (
        first_label_trace["eventId"],
        first_label_trace["traceId"],
    )
    next_after_resume = service.next_trace()
    assert (next_after_resume["eventId"], next_after_resume["traceId"]) == (
        second_label_trace["eventId"],
        second_label_trace["traceId"],
    )


def test_label_mode_keeps_forward_stack_stable_after_relabel(tmp_path) -> None:
    random.seed(7)
    trace_path = tmp_path / "run_0009.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)

    first = service.set_session(mode="label", run=9)["trace"]
    second = service.next_trace()
    third = service.next_trace()

    rewound = service.previous_trace()
    assert (rewound["eventId"], rewound["traceId"]) == (second["eventId"], second["traceId"])

    service.assign_label(
        event_id=rewound["eventId"],
        trace_id=rewound["traceId"],
        family="normal",
        label="2",
    )

    next_after_relabel = service.next_trace()
    assert (next_after_relabel["eventId"], next_after_relabel["traceId"]) == (
        third["eventId"],
        third["traceId"],
    )


def test_review_mode_rejects_empty_selection(tmp_path) -> None:
    trace_path = tmp_path / "run_0003.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)

    with pytest.raises(LookupError, match="no traces match"):
        service.set_session(mode="review", run=3, source="label_set", family="normal", label="1")


def test_review_mode_supports_grouped_normal_filter(tmp_path) -> None:
    trace_path = tmp_path / "run_0007.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)
    service.set_session(mode="label", run=7)
    service.assign_label(event_id=1, trace_id=0, family="normal", label="4")
    service.assign_label(event_id=1, trace_id=1, family="normal", label="9")
    service.assign_label(event_id=2, trace_id=0, family="normal", label="2")

    payload = service.set_session(
        mode="review",
        run=7,
        source="label_set",
        family="normal",
        label="4+",
    )

    assert payload["session"] == {
        "mode": "review",
        "run": 7,
        "source": "label_set",
        "family": "normal",
        "label": "4+",
        "filterFile": None,
    }

    first = payload["trace"]
    second = service.next_trace()
    still_last = service.next_trace()

    assert (first["eventId"], first["traceId"]) == (1, 0)
    assert (second["eventId"], second["traceId"]) == (1, 1)
    assert (still_last["eventId"], still_last["traceId"]) == (1, 1)


def test_trace_payload_includes_transformed_trace(tmp_path) -> None:
    random.seed(11)
    trace_path = tmp_path / "run_0004.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)

    payload = service.set_session(mode="label", run=4)["trace"]

    assert "raw" in payload
    assert "trace" in payload
    assert "transformed" in payload
    assert "bitflipAnalysis" in payload
    assert len(payload["raw"]) == len(payload["trace"]) == 3
    assert len(payload["transformed"]) == 2
    assert payload["bitflipAnalysis"]["xIndices"] == [0, 1, 2]
    assert len(payload["bitflipAnalysis"]["firstDerivative"]) == 3
    np.testing.assert_allclose(
        payload["bitflipAnalysis"]["firstDerivative"],
        np.concatenate(([0.0], np.diff(payload["trace"]))),
    )
    assert len(payload["bitflipAnalysis"]["secondDerivative"]) == 3
    assert payload["bitflipAnalysis"]["secondDerivative"] == [0.0, 0.0, 0.0]
    assert payload["bitflipAnalysis"]["structures"] == []
    np.testing.assert_allclose(
        payload["transformed"],
        np.abs(np.fft.rfft(payload["trace"])),
    )


def test_trace_payload_includes_padded_second_derivative(tmp_path) -> None:
    trace_path = tmp_path / "run_0008.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_legacy_hdf5(
        trace_path,
        {
            1: np.asarray(
                [
                    [10, 11, 12, 13, 14, *([0.0] * 23), *([64.0] * 36)],
                ],
                dtype=np.float32,
            )
        },
    )

    service = EstimatorService(trace_path=trace_path, workspace=workspace)

    payload = service.set_session(mode="label", run=8)["trace"]

    assert len(payload["bitflipAnalysis"]["firstDerivative"]) == len(payload["trace"])
    assert payload["bitflipAnalysis"]["firstDerivative"][0] == 0.0
    assert len(payload["bitflipAnalysis"]["secondDerivative"]) == len(payload["trace"])
    assert payload["bitflipAnalysis"]["secondDerivative"][0] == 0.0
    assert payload["bitflipAnalysis"]["secondDerivative"][-1] == 0.0
    assert max(abs(value) for value in payload["bitflipAnalysis"]["secondDerivative"]) > 0.0


def test_serialize_trace_payload_includes_bitflip_structure_endpoints() -> None:
    trace = _trace_from_second_derivative([0.0, 61.0, -121.0, 450.0, -512.0, 0.0, 0.0, 0.0])
    record = TraceRecord(
        run=10,
        event_id=1,
        trace_id=0,
        detector="pad",
        hardware_id=np.asarray([10, 11, 12, 13, 14], dtype=np.float32),
        raw=trace.copy(),
        trace=trace.copy(),
        transformed=np.asarray([1.0, 2.0], dtype=np.float32),
        family=None,
        label=None,
    )
    payload = serialize_trace_payload(
        record,
        bitflip_baseline_threshold=1.0,
        label=None,
        review_progress=None,
        include_run=True,
    )

    assert payload["bitflipAnalysis"]["structures"] == [
        {"startBaselineIndex": 1, "endBaselineIndex": 6}
    ]


def test_label_mode_supports_legacy_trace_layout(tmp_path) -> None:
    trace_path = tmp_path / "run_0004.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_legacy_hdf5(
        trace_path,
        {
            1: np.asarray(
                [
                    [10, 11, 12, 13, 14, 1, 2, 3],
                    [20, 21, 22, 23, 24, 4, 5, 6],
                ],
                dtype=np.float32,
            )
        },
    )

    service = EstimatorService(trace_path=trace_path, workspace=workspace)
    try:
        payload = service.set_session(mode="label", run=4)["trace"]
        assert payload["run"] == 4
        assert payload["eventId"] == 1
        assert payload["traceId"] in {0, 1}
        assert len(payload["raw"]) == 3
        assert len(payload["trace"]) == 3
    finally:
        service.close()


def test_service_bootstrap_uses_requested_default_run(tmp_path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_root / "run_0004.h5")
    write_hdf5_input(trace_root / "run_0006.h5")

    service = EstimatorService(
        trace_path=trace_root,
        workspace=workspace,
        default_run=6,
    )
    try:
        bootstrap = service.bootstrap_state()
        assert bootstrap["runs"] == [4, 6]
        assert bootstrap["session"] == {
            "mode": "label",
            "run": 6,
            "source": None,
            "family": None,
            "label": None,
            "filterFile": None,
        }
    finally:
        service.close()


def test_service_rejects_missing_requested_default_run(tmp_path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_root / "run_0004.h5")

    with pytest.raises(ValueError, match="default run 6 is not available"):
        EstimatorService(
            trace_path=trace_root,
            workspace=workspace,
            default_run=6,
        )


def test_delete_strange_label_rejects_labels_with_traces(tmp_path) -> None:
    trace_path = tmp_path / "run_0005.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)
    service.create_strange_label("Noise", "n")
    service.set_session(mode="label", run=5)
    service.assign_label(event_id=1, trace_id=1, family="strange", label="Noise")

    with pytest.raises(ValueError, match='cannot delete strange label "Noise" because it has 1 labeled trace'):
        service.delete_strange_label("Noise")


def test_delete_strange_label_allows_unused_labels(tmp_path) -> None:
    trace_path = tmp_path / "run_0006.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_hdf5_input(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)
    service.create_strange_label("Noise", "n")
    service.create_strange_label("Burst", "b")

    remaining = service.delete_strange_label("Noise")

    assert remaining == [{"id": 2, "name": "Burst", "shortcutKey": "b", "count": 0}]
