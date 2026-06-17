from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import polars as pl

from attpc_estimator.pipeline.ic import (
    IcAmplitudeConfig,
    IcBaselineConfig,
    ProgressReporter,
    process_run,
)
from attpc_estimator.storage.run_paths import ion_chamber_run_path
from attpc_estimator.utils.trace_data import load_trace_record


class NullProgress(ProgressReporter):
    def report_start(self, *, total: int, unit: str, description: str) -> None:
        pass

    def report_progress(self, current: int, *, message: str = "") -> None:
        pass

    def report_finish(self) -> None:
        pass


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


def test_ion_chamber_pipeline_writes_trigger_type_and_missing_sentinel(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    workspace = tmp_path / "workspace"
    trace_root.mkdir()
    workspace.mkdir()
    _write_ic_run(trace_root / "run_0005.h5")

    output_path = ion_chamber_run_path(workspace, 5)
    written = process_run(
        trace_path=trace_root,
        workspace=workspace,
        run=5,
        output_path=output_path,
        baseline_config=IcBaselineConfig(fft_window_scale=10.0),
        amplitude_config=IcAmplitudeConfig(
            peak_separation=50.0,
            peak_prominence=20.0,
            peak_width=500.0,
            peak_threshold=1_000_000.0,
            rel_height=0.85,
            ic_delay_time_bucket=1100.0,
            ic_multiplicity=1,
        ),
        progress=NullProgress(),
    )

    assert written == 2
    frame = pl.read_parquet(output_path)
    assert frame.select(
        "event_id",
        "orig_run",
        "orig_event",
        "trigger_type",
        "ic_multiplicity",
    ).to_dicts() == [
        {
            "event_id": 1,
            "orig_run": 5001,
            "orig_event": 7001,
            "trigger_type": 2,
            "ic_multiplicity": 0.0,
        },
        {
            "event_id": 2,
            "orig_run": 5,
            "orig_event": 2,
            "trigger_type": -1,
            "ic_multiplicity": -1.0,
        },
    ]


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
