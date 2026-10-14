from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from attpc_estimator.utils.trace_data import load_trace_record


def _write_ic_run(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["version"] = "libattpc_merger:2.0"
        events.attrs["min_event"] = 1
        events.attrs["max_event"] = 2
        events.attrs["bad_events"] = np.asarray([], dtype=np.int64)

        event = events.create_group("event_1")
        event.attrs["orig_run"] = 5001
        event.attrs["orig_event"] = 7001
        frib = event.create_group("frib_physics")
        trace_matrix = np.zeros((256, 2), dtype=np.float32)
        trace_matrix[:, 0] = np.arange(256, dtype=np.float32)
        trace_matrix[:, 1] = 10_000.0
        frib.create_dataset("1903", data=trace_matrix)
        frib.create_dataset("977", data=np.asarray([2], dtype=np.uint16))


def test_ion_chamber_trace_record_uses_ic_column(tmp_path: Path) -> None:
    path = tmp_path / "run_0005.h5"
    _write_ic_run(path)

    with h5py.File(path, "r") as handle:
        record = load_trace_record(
            handle,
            run=5,
            event_id=1,
            trace_id=0,
            baseline_window_scale=10.0,
            detector="IC",
        )

    assert record.detector == "IC"
    assert record.hardware_id[0] == 1903
    assert record.raw[:4].tolist() == [0.0, 1.0, 2.0, 3.0]
