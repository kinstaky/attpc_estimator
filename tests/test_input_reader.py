from __future__ import annotations

from pathlib import Path
import threading

import h5py
import numpy as np

from attpc_estimator.label_trace.input_reader import TraceSource


def write_hdf5_input(path: Path, trace_count: int = 20) -> None:
    rows = []
    for trace_id in range(trace_count):
        rows.append(
            [
                10 + trace_id,
                20 + trace_id,
                30 + trace_id,
                40 + trace_id,
                50 + trace_id,
                trace_id,
                trace_id + 1,
                trace_id + 2,
                trace_id + 3,
            ]
        )

    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["min_event"] = 1
        events.attrs["max_event"] = 1
        events.attrs["bad_events"] = np.array([], dtype=np.int64)

        event = events.create_group("event_1")
        get_group = event.create_group("get")
        get_group.create_dataset("pads", data=np.array(rows, dtype=np.float32))


def test_review_navigation_prefetches_next_five_traces_asynchronously(tmp_path) -> None:
    trace_path = tmp_path / "run_0010.h5"
    write_hdf5_input(trace_path)

    source = TraceSource(trace_path)
    try:
        source.set_labeled({(1, trace_id): ("normal", "0") for trace_id in range(20)})
        started = threading.Event()
        allow_prefetch = threading.Event()
        original_prefetch = source._prefetch_window

        def blocked_prefetch(file_handle, window_keys):
            started.set()
            allow_prefetch.wait(timeout=1.0)
            return original_prefetch(file_handle, window_keys)

        source._prefetch_window = blocked_prefetch

        source.set_trace_mode("review", family="normal", label=None)

        first = source.next_trace()
        assert (first.event_id, first.trace_id) == (1, 0)
        assert started.wait(timeout=1.0)
        assert list(source.trace_cache) == [(1, 0)]

        allow_prefetch.set()
        assert source._wait_for_prefetch(timeout=1.0)
        assert list(source.trace_cache) == [(1, trace_id) for trace_id in range(6)]

        second = source.next_trace()
        assert (second.event_id, second.trace_id) == (1, 1)
        rewound = source.previous_trace()
        assert (rewound.event_id, rewound.trace_id) == (1, 0)
    finally:
        source.close()


def test_trace_cache_stays_bounded_to_current_window(tmp_path) -> None:
    trace_path = tmp_path / "run_0011.h5"
    write_hdf5_input(trace_path)

    source = TraceSource(trace_path)
    try:
        source.set_labeled({(1, trace_id): ("normal", "0") for trace_id in range(20)})
        source.set_trace_mode("review", family="normal", label=None)

        current = None
        for _ in range(7):
            current = source.next_trace()
        assert source._wait_for_prefetch(timeout=1.0)

        assert current is not None
        assert current.trace_id == 6
        assert list(source.trace_cache) == [(1, trace_id) for trace_id in range(1, 12)]
        assert len(source.trace_cache) == 11
    finally:
        source.close()
