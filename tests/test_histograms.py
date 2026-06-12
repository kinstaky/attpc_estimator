from __future__ import annotations

import sys
from pathlib import Path
import json

import h5py
import numpy as np
import pytest

import attpc_estimator.process.trace_scan as trace_scan
from attpc_estimator.cli.filter import main as filter_main
from attpc_estimator.process.filter_core import (
    AmplitudeFilterCore,
    CdfFilterCore,
)
from attpc_estimator.storage.labels_db import LabelRepository
from attpc_estimator.process.cdf import CDF_THRESHOLDS
from attpc_estimator.process.filter import build_filter_rows, default_output_name
from attpc_estimator.service.estimator import EstimatorService
from tests.test_estimator_service import (
    write_gagg_sparse_hdf5_input,
    write_ic_sparse_hdf5_input,
    write_si_sparse_hdf5_input,
)
from tests.hdf5_fixtures import write_events_hdf5


def _gaussian_trace(
    height: float, center: float = 120.0, width: float = 8.0
) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return (height * np.exp(-0.5 * ((x - center) / width) ** 2)).astype(np.float32)


def _sine_trace(period: float, amplitude: float = 30.0) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return (amplitude * np.sin(2.0 * np.pi * x / period)).astype(np.float32)


def _pad_rows(traces: list[np.ndarray]) -> np.ndarray:
    rows = []
    for trace_id, trace in enumerate(traces):
        hardware = np.asarray(
            [10 + trace_id, 20 + trace_id, 30 + trace_id, 40 + trace_id, 50 + trace_id],
            dtype=np.float32,
        )
        rows.append(np.concatenate([hardware, trace]).astype(np.float32))
    return np.asarray(rows, dtype=np.float32)


def write_run_file(path: Path, events: dict[int, list[np.ndarray]]) -> None:
    with h5py.File(path, "w") as handle:
        event_ids = sorted(events)
        group = handle.create_group("events")
        group.attrs["version"] = "libattpc_merger:2.0"
        group.attrs["min_event"] = min(event_ids)
        group.attrs["max_event"] = max(event_ids)
        group.attrs["bad_events"] = np.asarray([], dtype=np.int64)
        for event_id in event_ids:
            event_group = group.create_group(f"event_{event_id}")
            get_group = event_group.create_group("get")
            get_group.create_dataset("pads", data=_pad_rows(events[event_id]))


def write_ic_peak_hdf5_input(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["version"] = "libattpc_merger:2.0"
        events.attrs["min_event"] = 1
        events.attrs["max_event"] = 2
        events.attrs["bad_events"] = np.array([], dtype=np.int64)

        for event_id, center in ((1, 80.0), (2, 140.0)):
            event = events.create_group(f"event_{event_id}")
            frib = event.create_group("frib_physics")
            x = np.arange(256, dtype=np.float32)
            peak = 2000.0 * np.exp(-0.5 * ((x - center) / 6.0) ** 2)
            trace_matrix = np.zeros((256, 2), dtype=np.float32)
            trace_matrix[:, 0] = peak
            trace_matrix[:, 1] = 10_000.0
            frib.create_dataset("1903", data=trace_matrix)
            frib.create_dataset("1904", data=np.zeros((256, 1), dtype=np.float32))
            frib.create_dataset("1905", data=np.zeros((256, 1), dtype=np.float32))
            frib.create_dataset("1906", data=np.zeros((257, 1), dtype=np.float32))
            frib.create_dataset("977", data=np.asarray([2], dtype=np.uint16))


def _si_peak_matrix(count: int, *, center: float, strip_offset: int = 0) -> np.ndarray:
    data = np.zeros((count, 517), dtype=np.float32)
    if count <= 0:
        return data
    for row in range(count):
        data[row, 4] = 100 + strip_offset + row
        samples = np.zeros(512, dtype=np.float32)
        peak_index = int(np.clip(center, 0, 511))
        samples[peak_index] = 400.0 + row * 25.0
        data[row, 5:] = samples
    return data


def write_si_peak_hdf5_input(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.attrs["version"] = "libattpc_merger:2.0"
        events.attrs["min_event"] = 1
        events.attrs["max_event"] = 2
        events.attrs["bad_events"] = np.array([], dtype=np.int64)

        event_1 = events.create_group("event_1")
        get_1 = event_1.create_group("get")
        get_1.create_dataset("si_upstream_front", data=_si_peak_matrix(2, center=80.0))
        get_1.create_dataset("si_upstream_back", data=_si_peak_matrix(1, center=120.0, strip_offset=10))
        get_1.create_dataset("si_downstream_front", data=_si_peak_matrix(1, center=200.0, strip_offset=20))
        get_1.create_dataset("si_downstream_back", data=_si_peak_matrix(0, center=260.0, strip_offset=30))

        event_2 = events.create_group("event_2")
        get_2 = event_2.create_group("get")
        get_2.create_dataset("si_upstream_front", data=_si_peak_matrix(1, center=300.0, strip_offset=40))
        get_2.create_dataset("si_upstream_back", data=_si_peak_matrix(0, center=340.0, strip_offset=50))
        get_2.create_dataset("si_downstream_front", data=_si_peak_matrix(0, center=380.0, strip_offset=60))
        get_2.create_dataset("si_downstream_back", data=_si_peak_matrix(1, center=420.0, strip_offset=70))


def seed_workspace(workspace: Path) -> None:
    repository = LabelRepository(workspace / "labels.db")
    repository.initialize()
    repository.create_strange_label("Noise", "n")
    repository.save_label(8, 1, 0, "pad", 1, 1, 1, 1, 1, "normal", "0")
    repository.save_label(8, 1, 1, "pad", 2, 2, 2, 2, 2, "strange", "Noise")
    repository.connection.close()


def make_trace_root(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {
            1: [_gaussian_trace(20.0), _gaussian_trace(50.0), _gaussian_trace(80.0)],
            2: [_gaussian_trace(40.0)],
        },
    )
    seed_workspace(workspace)
    return workspace, trace_root


def test_build_filter_rows_amplitude_is_inclusive_and_ordered(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    _ = workspace

    rows = build_filter_rows(
        trace_path=trace_root,
        run=8,
        filter_cores=[
            AmplitudeFilterCore(
                min_amplitude=20.0,
                max_amplitude=50.0,
                peak_separation=20.0,
                peak_prominence=5.0,
                peak_width=40.0,
            )
        ],
    )

    assert rows.dtype == np.int64
    assert rows.shape == (2, 3)
    assert rows.tolist() == [
        [8, 1, 1],
        [8, 2, 0],
    ]


def test_build_filter_rows_cdf_uses_f60_cutoff(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _ = workspace
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {
            1: [
                _sine_trace(period=4.0),
                _sine_trace(period=64.0),
                _gaussian_trace(40.0),
            ],
        },
    )

    rows = build_filter_rows(
        trace_path=trace_root,
        run=8,
        filter_cores=[CdfFilterCore()],
        baseline_window_scale=10.0,
    )

    assert rows.dtype == np.int64
    assert rows.tolist() == [[8, 1, 0]]


def test_build_filter_rows_skips_bad_events_in_v2_layout(tmp_path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_events_hdf5(
        trace_root / "run_0008.h5",
        {
            1: _pad_rows([_gaussian_trace(20.0), _gaussian_trace(50.0)]),
            2: _pad_rows([_gaussian_trace(40.0)]),
        },
        bad_events=np.asarray([2], dtype=np.int64),
    )

    rows = build_filter_rows(
        trace_path=trace_root,
        run=8,
        filter_cores=[
            AmplitudeFilterCore(
                min_amplitude=20.0,
                max_amplitude=50.0,
                peak_separation=20.0,
                peak_prominence=5.0,
                peak_width=40.0,
            )
        ],
    )

    assert rows.tolist() == [[8, 1, 1]]


def test_build_filter_rows_stops_scanning_after_reaching_limit(
    tmp_path, monkeypatch
) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {
            1: [_gaussian_trace(40.0)],
            2: [_gaussian_trace(40.0)],
        },
    )

    seen_events: list[int] = []
    original = trace_scan.load_pad_traces

    def wrapped_load_pad_traces(*args, **kwargs) -> np.ndarray:
        seen_events.append(int(kwargs["event_id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(trace_scan, "load_pad_traces", wrapped_load_pad_traces)

    rows = build_filter_rows(
        trace_path=trace_root,
        run=8,
        filter_cores=[
            AmplitudeFilterCore(
                min_amplitude=20.0,
                max_amplitude=50.0,
                peak_separation=20.0,
                peak_prominence=5.0,
                peak_width=40.0,
            )
        ],
        limit=1,
    )

    assert rows.tolist() == [[8, 1, 0]]
    assert seen_events == [1]


def test_build_filter_rows_reports_match_progress_when_limited(tmp_path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {
            1: [_gaussian_trace(40.0)],
            2: [_gaussian_trace(80.0)],
        },
    )
    progress_updates = []

    rows = build_filter_rows(
        trace_path=trace_root,
        run=8,
        filter_cores=[
            AmplitudeFilterCore(
                min_amplitude=20.0,
                max_amplitude=50.0,
                peak_separation=20.0,
                peak_prominence=5.0,
                peak_width=40.0,
            )
        ],
        limit=1,
        progress=progress_updates.append,
    )

    assert rows.tolist() == [[8, 1, 0]]
    assert [update.current for update in progress_updates] == [0, 1]
    assert all(update.total == 1 for update in progress_updates)
    assert all(update.unit == "trace" for update in progress_updates)


def test_build_filter_rows_keeps_only_traces_matching_all_filter_cores(tmp_path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    write_run_file(
        trace_root / "run_0008.h5",
        {
            1: [
                _sine_trace(period=4.0, amplitude=30.0),
                _sine_trace(period=4.0, amplitude=80.0),
                _gaussian_trace(40.0),
            ],
        },
    )

    rows = build_filter_rows(
        trace_path=trace_root,
        run=8,
        filter_cores=[
            CdfFilterCore(),
            AmplitudeFilterCore(
                min_amplitude=20.0,
                max_amplitude=50.0,
                peak_separation=20.0,
                peak_prominence=5.0,
                peak_width=40.0,
            ),
        ],
        baseline_window_scale=10.0,
    )

    assert rows.tolist() == [[8, 1, 0]]


def test_default_output_name_uses_filter_core_tokens_in_order() -> None:
    assert default_output_name(
        "0008",
        [
            CdfFilterCore(),
            AmplitudeFilterCore(
                min_amplitude=20.0,
                max_amplitude=50.0,
            ),
        ],
    ) == "filter_run_0008_cdf_amp_20_50.npy"


def test_filter_main_zero_pads_integer_run_from_config_file(tmp_path, monkeypatch) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    config_path = tmp_path / "filter.toml"
    config_path.write_text(
        "\n".join(
            [
                f'trace_path = "{trace_root}"',
                f'workspace = "{workspace}"',
                "run = 8",
                "",
                "[filter]",
                "use_amplitude = true",
                "use_cdf = false",
                "use_bitflip = false",
                "use_saturation = false",
                "min_amplitude = 20",
                "max_amplitude = 50",
                "baseline_window_scale = 10.0",
                "limit = 1000",
                "",
                "[attpc.amplitude]",
                "peak_separation = 50.0",
                "peak_prominence = 20.0",
                "peak_width = 50.0",
                "peak_threshold = 0.0",
                "rel_height = 0.95",
                "",
                "[attpc.bitflip]",
                "baseline = 10.0",
                "min_count = 1",
                "",
                "[attpc.saturation]",
                "threshold = 2000.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["filter", "-c", str(config_path)])
    filter_main()

    output_path = workspace / "filter" / "filter_run_0008_amp_20_50.npy"
    rows = np.load(output_path)

    assert output_path.is_file()
    assert rows.tolist() == [[8, 1, 1], [8, 2, 0]]


def test_estimator_service_bootstrap_and_histograms(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    (workspace / "filter").mkdir()
    (workspace / "histograms").mkdir()
    np.save(
        workspace / "filter" / "filter_amp_20_50.npy",
        np.asarray([[8, 1, 0], [8, 1, 1]], dtype=np.int64),
    )
    np.save(
        workspace / "histograms" / "run_0008_cdf.npy",
        np.ones((len(CDF_THRESHOLDS), 100), dtype=np.int64),
    )
    np.savez(
        workspace / "histograms" / "run_0008_labeled_cdf.npz",
        run_id=np.int64(8),
        label_keys=np.asarray(["normal:0", "strange:Noise"], dtype=np.str_),
        label_titles=np.asarray(["0 peak", "Noise"], dtype=np.str_),
        trace_counts=np.asarray([1, 1], dtype=np.int64),
        histograms=np.ones((2, len(CDF_THRESHOLDS), 100), dtype=np.int64),
    )
    np.save(workspace / "histograms" / "run_0008_amp.npy", np.arange(16, dtype=np.int64))
    np.savez(
        workspace / "histograms" / "run_0008_labeled_amp.npz",
        run_id=np.int64(8),
        label_keys=np.asarray(["normal:0", "normal:4", "normal:9", "strange:Noise"], dtype=np.str_),
        trace_counts=np.asarray([1, 2, 3, 1], dtype=np.int64),
        histograms=np.asarray(
            [
                np.arange(16, dtype=np.int64),
                np.ones(16, dtype=np.int64),
                np.full(16, 2, dtype=np.int64),
                np.arange(16, dtype=np.int64),
            ],
            dtype=np.int64,
        ),
    )

    service = EstimatorService(trace_path=trace_root, workspace=workspace)
    try:
        bootstrap = service.bootstrap_state()
        assert bootstrap["appType"] == "merged"
        assert bootstrap["runs"] == [8]
        assert bootstrap["filterFiles"] == [{"name": "filter_amp_20_50.npy"}]
        assert bootstrap["histogramAvailability"]["8"]["cdf"]["all"] is True
        assert bootstrap["histogramAvailability"]["8"]["amplitude"]["labeled"] is True
        assert bootstrap["histogramAvailability"]["8"]["cdf"]["filtered"] is True

        cdf_payload = service.get_histogram(metric="cdf", mode="labeled", run=8)
        assert cdf_payload["metric"] == "cdf"
        assert [series["title"] for series in cdf_payload["series"]] == [
            "0 peak",
            "Noise",
        ]

        amp_payload = service.get_histogram(metric="amplitude", mode="labeled", run=8)
        assert amp_payload["metric"] == "amplitude"
        assert [series["title"] for series in amp_payload["series"]] == [
            "0 peak",
            "4+ peaks",
            "Noise",
        ]
        assert amp_payload["series"][1]["traceCount"] == 5
        assert amp_payload["series"][1]["histogram"][0] == 3

        filtered_cdf = service.get_histogram(
            metric="cdf",
            mode="filtered",
            run=8,
            filter_file="filter_amp_20_50.npy",
        )
        assert filtered_cdf["mode"] == "filtered"
        assert filtered_cdf["filterFile"] == "filter_amp_20_50.npy"
        assert filtered_cdf["veto"] is False
        assert [series["traceCount"] for series in filtered_cdf["series"]] == [2]
    finally:
        service.close()


def test_estimator_service_builds_veto_filtered_histogram(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    (workspace / "filter").mkdir()
    np.save(
        workspace / "filter" / "filter_amp_20_50.npy",
        np.asarray([[8, 1, 0], [8, 1, 1]], dtype=np.int64),
    )

    service = EstimatorService(trace_path=trace_root, workspace=workspace)
    try:
        filtered_cdf = service.get_histogram(
            metric="cdf",
            mode="filtered",
            run=8,
            filter_file="filter_amp_20_50.npy",
            veto=True,
        )
        assert filtered_cdf["mode"] == "filtered"
        assert filtered_cdf["filterFile"] == "filter_amp_20_50.npy"
        assert filtered_cdf["veto"] is True
        assert [series["traceCount"] for series in filtered_cdf["series"]] == [2]
        assert filtered_cdf["series"][0]["title"] == "Vetoed traces · filter_amp_20_50"
    finally:
        service.close()


def test_estimator_service_veto_filtered_histogram_uses_all_run_traces_when_run_missing_from_file(
    tmp_path,
) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    (workspace / "filter").mkdir()
    np.save(
        workspace / "filter" / "filter_other_run.npy",
        np.asarray([[9, 1, 0]], dtype=np.int64),
    )

    service = EstimatorService(trace_path=trace_root, workspace=workspace)
    try:
        filtered_cdf = service.get_histogram(
            metric="cdf",
            mode="filtered",
            run=8,
            filter_file="filter_other_run.npy",
            veto=True,
        )
        assert filtered_cdf["veto"] is True
        assert [series["traceCount"] for series in filtered_cdf["series"]] == [4]
    finally:
        service.close()


def test_histogram_service_reports_filtered_histogram_progress(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    (workspace / "filter").mkdir()
    np.save(
        workspace / "filter" / "filter_amp_20_50.npy",
        np.asarray([[8, 1, 0], [8, 2, 0]], dtype=np.int64),
    )
    service = EstimatorService(trace_path=trace_root, workspace=workspace)
    progress_updates = []
    try:
        payload = service.histograms.get_histogram(
            metric="cdf",
            mode="filtered",
            run=8,
            filter_file="filter_amp_20_50.npy",
            progress=progress_updates.append,
        )
        assert payload["mode"] == "filtered"
        assert [update.current for update in progress_updates] == [0, 1, 2]
        assert all(update.total == 2 for update in progress_updates)
        assert all(update.unit == "trace" for update in progress_updates)
    finally:
        service.close()


def test_histogram_service_reports_veto_filtered_histogram_progress(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    (workspace / "filter").mkdir()
    np.save(
        workspace / "filter" / "filter_amp_20_50.npy",
        np.asarray([[8, 1, 0], [8, 1, 1]], dtype=np.int64),
    )
    service = EstimatorService(trace_path=trace_root, workspace=workspace)
    progress_updates = []
    try:
        payload = service.histograms.get_histogram(
            metric="cdf",
            mode="filtered",
            run=8,
            filter_file="filter_amp_20_50.npy",
            veto=True,
            progress=progress_updates.append,
        )
        assert payload["mode"] == "filtered"
        assert payload["veto"] is True
        assert [update.current for update in progress_updates] == [0, 1, 2]
        assert all(update.total == 2 for update in progress_updates)
        assert all(update.unit == "trace" for update in progress_updates)
    finally:
        service.close()


def test_estimator_service_selects_filter_and_navigates(tmp_path) -> None:
    workspace, trace_root = make_trace_root(tmp_path)
    (workspace / "filter").mkdir()
    np.save(
        workspace / "filter" / "filter_amp_20_80.npy",
        np.asarray([[8, 1, 0], [8, 1, 1]], dtype=np.int64),
    )

    service = EstimatorService(trace_path=trace_root, workspace=workspace)
    try:
        selected = service.set_session(
            mode="review",
            source="filter_file",
            filter_file="filter_amp_20_80.npy",
        )
        assert selected["session"] == {
            "mode": "review",
            "run": None,
            "detector": "ATTPC",
            "source": "filter_file",
            "family": None,
            "label": None,
            "filterFile": "filter_amp_20_80.npy",
            "eventId": None,
            "traceId": None,
        }
        assert selected["traceCount"] == 2
        assert selected["trace"]["run"] == 8
        assert selected["trace"]["currentLabel"] == {"family": "normal", "label": "0"}

        next_trace = service.next_trace()
        assert (next_trace["run"], next_trace["eventId"], next_trace["traceId"]) == (
            8,
            1,
            1,
        )
        assert next_trace["currentLabel"] == {"family": "strange", "label": "Noise"}

        previous_trace = service.previous_trace()
        assert (
            previous_trace["run"],
            previous_trace["eventId"],
            previous_trace["traceId"],
        ) == (8, 1, 0)
    finally:
        service.close()


@pytest.mark.parametrize(
    ("detector", "metric", "writer", "expected_name"),
    [
        (
            "IC",
            "baseline",
            write_ic_sparse_hdf5_input,
            "run_0016_detector_ic_metric_baseline_variant_none.npz",
        ),
        (
            "IC",
            "time",
            write_ic_sparse_hdf5_input,
            "run_0016_detector_ic_metric_time_variant_none.npz",
        ),
        (
            "SI",
            "baseline",
            write_si_sparse_hdf5_input,
            "run_0017_detector_si_metric_baseline_variant_none.npz",
        ),
        (
            "SI",
            "time",
            write_si_peak_hdf5_input,
            "run_0017_detector_si_metric_time_variant_none.npz",
        ),
        (
            "GAGG",
            "baseline",
            write_gagg_sparse_hdf5_input,
            "run_0018_detector_gagg_metric_baseline_variant_none.npz",
        ),
    ],
)
def test_detector_histograms_write_artifacts_and_reuse_cache(
    tmp_path,
    monkeypatch,
    detector: str,
    metric: str,
    writer,
    expected_name: str,
) -> None:
    run = int(expected_name.split("_")[1])
    trace_path = tmp_path / f"run_{run:04d}.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    writer(trace_path)

    service = EstimatorService(trace_path=trace_path, workspace=workspace)
    try:
        payload = service.get_histogram(metric=metric, mode="all", run=run, detector=detector)
        assert payload["detector"] == detector
        artifact_path = workspace / "histograms" / expected_name
        assert artifact_path.exists()

        def fail_build(**_kwargs):
            raise AssertionError("detector histogram should have been loaded from cache")

        monkeypatch.setattr(service.histograms, "_build_detector_artifact_payload", fail_build)
        cached = service.get_histogram(metric=metric, mode="all", run=run, detector=detector)
        assert cached == payload
    finally:
        service.close()


def test_detector_histogram_cache_recomputes_when_config_changes(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "run_0016.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_ic_sparse_hdf5_input(trace_path)

    first = EstimatorService(trace_path=trace_path, workspace=workspace)
    artifact_path = (
        workspace / "histograms" / "run_0016_detector_ic_metric_baseline_variant_none.npz"
    )
    try:
        first.get_histogram(metric="baseline", mode="all", run=16, detector="IC")
        assert artifact_path.exists()
    finally:
        first.close()

    second = EstimatorService(
        trace_path=trace_path,
        workspace=workspace,
        ic_baseline_window_scale=5.0,
    )
    build_calls = 0
    original = second.histograms._build_detector_artifact_payload

    def wrapped_build(**kwargs):
        nonlocal build_calls
        build_calls += 1
        return original(**kwargs)

    try:
        monkeypatch.setattr(second.histograms, "_build_detector_artifact_payload", wrapped_build)
        second.get_histogram(metric="baseline", mode="all", run=16, detector="IC")
        assert build_calls == 1
        payload = np.load(artifact_path)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        assert metadata["baseline_window_scale"] == 5.0
    finally:
        second.close()


def test_ic_time_histogram_returns_peak_bucket_counts(tmp_path) -> None:
    trace_path = tmp_path / "run_0016.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_ic_peak_hdf5_input(trace_path)

    service = EstimatorService(
        trace_path=trace_path,
        workspace=workspace,
        detector_peak_configs={
            "SI": {
                "peak_separation": 10.0,
                "peak_prominence": 5.0,
                "peak_width": 20.0,
                "peak_threshold": 1.0,
                "peak_rel_height": 0.95,
            }
        },
    )
    try:
        payload = service.get_histogram(metric="time", mode="all", run=16, detector="IC")
        assert payload["metric"] == "time"
        assert payload["detector"] == "IC"
        assert payload["binCount"] == 256
        assert payload["binLabel"] == "Time bucket"
        assert payload["countLabel"] == "Peak count"
        assert len(payload["series"]) == 1
        assert payload["series"][0]["traceCount"] == 2
        assert sum(payload["series"][0]["histogram"]) >= 2
    finally:
        service.close()


def test_si_time_histogram_returns_side_grouped_peak_bucket_counts(tmp_path) -> None:
    trace_path = tmp_path / "run_0017.h5"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_si_peak_hdf5_input(trace_path)

    service = EstimatorService(
        trace_path=trace_path,
        workspace=workspace,
        detector_peak_configs={
            "SI": {
                "peak_separation": 10.0,
                "peak_prominence": 5.0,
                "peak_width": 20.0,
                "peak_threshold": 1.0,
                "peak_rel_height": 0.95,
            }
        },
    )
    try:
        payload = service.get_histogram(metric="time", mode="all", run=17, detector="SI")
        assert payload["metric"] == "time"
        assert payload["detector"] == "SI"
        assert payload["binCount"] == 512
        assert payload["binLabel"] == "Time bucket"
        assert payload["countLabel"] == "Peak count"
        assert len(payload["series"]) == 4
        assert [series["labelKey"] for series in payload["series"]] == [
            "upstream_front",
            "upstream_back",
            "downstream_front",
            "downstream_back",
        ]
        assert [series["traceCount"] for series in payload["series"]] == [3, 1, 1, 1]
        assert max(sum(series["histogram"]) for series in payload["series"]) >= 1
    finally:
        service.close()
