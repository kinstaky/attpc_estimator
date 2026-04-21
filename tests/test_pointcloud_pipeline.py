from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None
    or importlib.util.find_spec("numpy") is None
    or importlib.util.find_spec("attpc_storage") is None
    or importlib.util.find_spec("pointcloud") is None,
    reason="pointcloud pipeline tests require estimator runtime dependencies",
)


def _build_pad_row(np_module, *, cobo: int, asad: int, aget: int, channel: int, pad_id: int) -> object:
    row = np_module.zeros((1, 517), dtype=np_module.float32)
    row[0, :5] = [cobo, asad, aget, channel, pad_id]
    row[0, 105:108] = [20.0, 120.0, 30.0]
    return row


def _write_v2_trace_file(path: Path, *, orig_run: int, orig_event: int) -> None:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["version"] = "libattpc_merger:2.0"
        events.attrs["min_event"] = 1
        events.attrs["max_event"] = 1
        events.attrs["bad_events"] = np.asarray([], dtype=np.int64)
        event = events.create_group("event_1")
        event.attrs["orig_run"] = orig_run
        event.attrs["orig_event"] = orig_event
        get_group = event.create_group("get")
        get_group.create_dataset(
            "pads",
            data=_build_pad_row(np, cobo=8, asad=2, aget=0, channel=65, pad_id=0),
        )


def _write_legacy_trace_file(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    with h5py.File(path, "w") as handle:
        meta_group = handle.create_group("meta")
        meta_group.create_dataset(
            "meta",
            data=np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        )
        get_group = handle.create_group("get")
        get_group.create_dataset(
            "evt1_data",
            data=_build_pad_row(np, cobo=8, asad=2, aget=0, channel=65, pad_id=0),
        )
        handle.create_group("frib").create_group("evt")


def test_process_run_writes_v2_pointcloud(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    from attpc_estimator.pipeline.pointcloud import (
        BitflipConfig,
        DriftConfig,
        FftConfig,
        PeakConfig,
        ProgressReporter,
        process_run,
    )

    class SilentProgress(ProgressReporter):
        def report_start(self, *, total: int, unit: str, description: str) -> None:
            self.started = (total, unit, description)

        def report_progress(self, current: int, *, message: str = "") -> None:
            self.current = current
            self.message = message

        def report_finish(self) -> None:
            self.finished = True

    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    input_path = trace_root / "run_0007.h5"
    output_path = workspace / "custom" / "run_0007.h5"
    _write_v2_trace_file(input_path, orig_run=70, orig_event=700)

    reporter = SilentProgress()
    written = process_run(
        trace_path=trace_root,
        workspace=workspace,
        run=7,
        output_path=output_path,
        fft_config=FftConfig(baseline_window_scale=20.0),
        bitflip_config=BitflipConfig(baseline_threshold=10.0, min_count=1),
        peak_config=PeakConfig(
            separation=10.0,
            prominence=5.0,
            max_width=12.0,
            threshold=10.0,
            rel_height=0.95,
        ),
        drift_config=DriftConfig(
            micromegas_time_bucket=10.0,
            window_time_bucket=560.0,
            detector_length=1000.0,
        ),
        progress=reporter,
    )

    assert written == 1
    with h5py.File(output_path, "r") as handle:
        assert int(handle["cloud"].attrs["min_event"]) == 1
        assert int(handle["cloud"].attrs["max_event"]) == 1
        cloud = np.asarray(handle["/cloud/cloud_1"])
        assert cloud.shape == (1, 9)
        assert cloud[0, 0] == pytest.approx(-2.8337707007520283)
        assert cloud[0, 1] == pytest.approx(269.95294)
        assert cloud[0, 2] == pytest.approx(((cloud[0, 6] - 10.0) / 550.0) * 1000.0)
        assert cloud[0, 5] == 0.0
        assert cloud[0, 8] == 0.0
        assert int(handle["/cloud/cloud_1"].attrs["orig_run"]) == 70
        assert int(handle["/cloud/cloud_1"].attrs["orig_event"]) == 700


def test_process_run_supports_legacy_trace_input(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    from attpc_estimator.pipeline.pointcloud import (
        BitflipConfig,
        DriftConfig,
        FftConfig,
        PeakConfig,
        ProgressReporter,
        process_run,
    )

    class SilentProgress(ProgressReporter):
        def report_start(self, *, total: int, unit: str, description: str) -> None:
            pass

        def report_progress(self, current: int, *, message: str = "") -> None:
            pass

        def report_finish(self) -> None:
            pass

    trace_root = tmp_path / "legacy"
    trace_root.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    input_path = trace_root / "run_0003.h5"
    output_path = workspace / "pointcloud" / "run_0003.h5"
    _write_legacy_trace_file(input_path)

    written = process_run(
        trace_path=trace_root,
        workspace=workspace,
        run=3,
        output_path=output_path,
        fft_config=FftConfig(baseline_window_scale=20.0),
        bitflip_config=BitflipConfig(baseline_threshold=10.0, min_count=1),
        peak_config=PeakConfig(
            separation=10.0,
            prominence=5.0,
            max_width=12.0,
            threshold=10.0,
            rel_height=0.95,
        ),
        drift_config=DriftConfig(
            micromegas_time_bucket=10.0,
            window_time_bucket=560.0,
            detector_length=1000.0,
        ),
        progress=SilentProgress(),
    )

    assert written == 1
    with h5py.File(output_path, "r") as handle:
        cloud = handle["/cloud/cloud_1"]
        assert cloud.shape == (1, 9)
        assert float(cloud[0, 0]) == pytest.approx(-2.8337707007520283)
        assert float(cloud[0, 1]) == pytest.approx(269.95294)
        assert int(handle["/cloud/cloud_1"].attrs["orig_run"]) == 3
        assert int(handle["/cloud/cloud_1"].attrs["orig_event"]) == 1
