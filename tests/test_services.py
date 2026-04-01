from __future__ import annotations

import random

import h5py
import numpy as np
import pytest

from attpc_estimator.label_trace.services import TraceLabelService


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


def test_review_mode_filters_traces_and_stops_at_bounds(tmp_path) -> None:
    trace_path = tmp_path / "run_0001.h5"
    db_dir = tmp_path / "db"
    write_hdf5_input(trace_path)

    service = TraceLabelService(trace_path=trace_path, db_dir=db_dir)
    service.create_strange_label("Noise", "n")
    service.save_label(event_id=1, trace_id=0, family="normal", label="0")
    service.save_label(event_id=1, trace_id=1, family="strange", label="Noise")
    service.save_label(event_id=2, trace_id=0, family="normal", label="4")

    payload = service.set_trace_mode(mode="review", family="normal")
    assert payload == {
        "mode": "review",
        "reviewFilter": {"family": "normal", "label": None},
    }

    first = service.next_trace()
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

    service.set_trace_mode(mode="review", family="strange", label="Noise")
    strange = service.next_trace()
    assert (strange["eventId"], strange["traceId"]) == (1, 1)
    assert strange["currentLabel"] == {"family": "strange", "label": "Noise"}
    assert strange["reviewProgress"] == {"current": 1, "total": 1}


def test_label_and_review_stacks_are_independent(tmp_path) -> None:
    random.seed(7)
    trace_path = tmp_path / "run_0002.h5"
    db_dir = tmp_path / "db"
    write_hdf5_input(trace_path)

    service = TraceLabelService(trace_path=trace_path, db_dir=db_dir)

    first_label_trace = service.next_trace()
    second_label_trace = service.next_trace()
    rewound_label_trace = service.previous_trace()

    assert (rewound_label_trace["eventId"], rewound_label_trace["traceId"]) == (
        first_label_trace["eventId"],
        first_label_trace["traceId"],
    )

    service.save_label(
        event_id=first_label_trace["eventId"],
        trace_id=first_label_trace["traceId"],
        family="normal",
        label="3",
    )

    review_mode = service.set_trace_mode(mode="review", family="normal", label="3")
    assert review_mode["mode"] == "review"

    review_trace = service.next_trace()
    assert (review_trace["eventId"], review_trace["traceId"]) == (
        first_label_trace["eventId"],
        first_label_trace["traceId"],
    )

    service.set_trace_mode(mode="label")
    resumed_label_trace = service.next_trace()
    assert (resumed_label_trace["eventId"], resumed_label_trace["traceId"]) == (
        second_label_trace["eventId"],
        second_label_trace["traceId"],
    )


def test_review_mode_rejects_empty_selection(tmp_path) -> None:
    trace_path = tmp_path / "run_0003.h5"
    db_dir = tmp_path / "db"
    write_hdf5_input(trace_path)

    service = TraceLabelService(trace_path=trace_path, db_dir=db_dir)

    with pytest.raises(LookupError, match="no traces match"):
        service.set_trace_mode(mode="review", family="normal", label="1")


def test_review_mode_supports_grouped_normal_filter(tmp_path) -> None:
    trace_path = tmp_path / "run_0007.h5"
    db_dir = tmp_path / "db"
    write_hdf5_input(trace_path)

    service = TraceLabelService(trace_path=trace_path, db_dir=db_dir)
    service.save_label(event_id=1, trace_id=0, family="normal", label="4")
    service.save_label(event_id=1, trace_id=1, family="normal", label="9")
    service.save_label(event_id=2, trace_id=0, family="normal", label="2")

    payload = service.set_trace_mode(mode="review", family="normal", label="4+")

    assert payload == {
        "mode": "review",
        "reviewFilter": {"family": "normal", "label": "4+"},
    }

    first = service.next_trace()
    second = service.next_trace()
    still_last = service.next_trace()

    assert (first["eventId"], first["traceId"]) == (1, 0)
    assert (second["eventId"], second["traceId"]) == (1, 1)
    assert (still_last["eventId"], still_last["traceId"]) == (1, 1)


def test_trace_payload_includes_transformed_trace(tmp_path) -> None:
    random.seed(11)
    trace_path = tmp_path / "run_0004.h5"
    db_dir = tmp_path / "db"
    write_hdf5_input(trace_path)

    service = TraceLabelService(trace_path=trace_path, db_dir=db_dir)

    payload = service.next_trace()

    assert "raw" in payload
    assert "trace" in payload
    assert "transformed" in payload
    assert len(payload["raw"]) == len(payload["trace"]) == 3
    assert len(payload["transformed"]) == 2
    np.testing.assert_allclose(
        payload["transformed"],
        np.abs(np.fft.rfft(payload["trace"])),
    )


def test_delete_strange_label_rejects_labels_with_traces(tmp_path) -> None:
    trace_path = tmp_path / "run_0005.h5"
    db_dir = tmp_path / "db"
    write_hdf5_input(trace_path)

    service = TraceLabelService(trace_path=trace_path, db_dir=db_dir)
    service.create_strange_label("Noise", "n")
    service.save_label(event_id=1, trace_id=1, family="strange", label="Noise")

    with pytest.raises(ValueError, match='cannot delete strange label "Noise" because it has 1 labeled trace'):
        service.delete_strange_label("Noise")


def test_delete_strange_label_allows_unused_labels(tmp_path) -> None:
    trace_path = tmp_path / "run_0006.h5"
    db_dir = tmp_path / "db"
    write_hdf5_input(trace_path)

    service = TraceLabelService(trace_path=trace_path, db_dir=db_dir)
    service.create_strange_label("Noise", "n")
    service.create_strange_label("Burst", "b")

    remaining = service.delete_strange_label("Noise")

    assert remaining == [{"id": 2, "name": "Burst", "shortcutKey": "b", "count": 0}]
