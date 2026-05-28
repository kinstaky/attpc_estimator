from __future__ import annotations

import logging

import numpy as np

from attpc_estimator.model.label import StoredLabel
from attpc_estimator.model.trace import TraceRef
from attpc_estimator.service.estimator import EstimatorService
from attpc_estimator.service.traces import selection
from attpc_estimator.service.traces.selection import RandomUnlabeledSelector
from tests.hdf5_fixtures import write_events_hdf5


def _rows(trace_value: int) -> np.ndarray:
    return np.asarray([[10, 20, 30, 40, 50, trace_value, trace_value + 1, trace_value + 2]], dtype=np.float32)


def test_random_selector_uses_full_scan_for_small_event_ranges(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "run_0008.h5"
    write_events_hdf5(
        trace_path,
        {event_id: _rows(event_id) for event_id in range(1, 11)},
    )

    original = selection.collect_reader_event_counts
    calls = {"count": 0}

    def wrapped(reader):
        calls["count"] += 1
        return original(reader)

    monkeypatch.setattr(selection, "collect_reader_event_counts", wrapped)

    selector = RandomUnlabeledSelector(trace_path)
    try:
        assert selector._strategy == "full_scan"
        assert calls["count"] == 1
    finally:
        selector.close()


def test_random_selector_uses_sparse_mode_for_large_event_ranges(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "run_0008.h5"
    write_events_hdf5(
        trace_path,
        {event_id: _rows(event_id) for event_id in range(1, 206)},
    )

    def fail(_reader):
        raise AssertionError("full event scan should not run for large files")

    monkeypatch.setattr(selection, "collect_reader_event_counts", fail)

    selector = RandomUnlabeledSelector(trace_path)
    try:
        assert selector._strategy == "sparse_random"
    finally:
        selector.close()


def test_random_selector_sparse_mode_falls_forward_on_collisions(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "run_0008.h5"
    write_events_hdf5(
        trace_path,
        {event_id: _rows(event_id) for event_id in range(1, 206)},
        bad_events=np.asarray([101], dtype=np.int64),
    )

    selector = RandomUnlabeledSelector(trace_path)
    try:
        monkeypatch.setattr(selection.random, "randint", lambda low, high: 101)
        monkeypatch.setattr(selection.random, "randrange", lambda count: 0)

        existing = [TraceRef(run=8, event_id=103, trace_id=0)]
        labels = {
            TraceRef(run=8, event_id=102, trace_id=0): StoredLabel(
                family="normal",
                label="0",
            )
        }

        additions = selector.ensure_forward_size(
            existing=existing,
            pointer=-1,
            target_size=2,
            labels=labels,
        )

        assert additions == [TraceRef(run=8, event_id=104, trace_id=0)]
    finally:
        selector.close()


def test_estimator_service_logs_label_mode_debug_messages(tmp_path, caplog) -> None:
    trace_path = tmp_path / "run_0008.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_events_hdf5(
        trace_path,
        {event_id: _rows(event_id) for event_id in range(1, 11)},
    )

    logger = logging.getLogger("attpc_estimator")
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    logger.handlers = []
    logger.propagate = True
    try:
        with caplog.at_level(logging.DEBUG, logger="attpc_estimator"):
            service = EstimatorService(trace_path=trace_path, workspace=workspace, verbose=True)
            try:
                service.set_session(mode="label", run=8)
            finally:
                service.close()
    finally:
        logger.handlers = original_handlers
        logger.propagate = original_propagate

    messages = [record.getMessage() for record in caplog.records]
    assert any("starting label session run=8" in message for message in messages)
    assert any("selector init run=8" in message for message in messages)
    assert any("first label trace ready run=8" in message for message in messages)
