from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from ..model.label import NORMAL_BUCKETS, StoredLabel
from ..model.trace import TraceRef
from ..process.bitflip import BITFLIP_BASELINE_DEFAULT
from ..process.line_pipeline import MergeConfig, RansacConfig
from ..storage.labels_db import LabelRepository
from ..storage.run_paths import (
    collect_run_files,
    filter_dir,
    labels_db_path,
    webui_state_path,
)
from ..storage.webui_state import WebUiStateStore
from ..utils.trace_data import (
    DETECTOR_ATTPC,
    DETECTOR_GAGG,
    DETECTOR_IC,
    DETECTOR_SI,
    describe_trace_events,
    normalize_detector,
)
from .histograms import HistogramService
from .labeling import (
    RESERVED_SHORTCUTS,
    labels_snapshot,
    normal_summary,
    normalize_shortcut,
    pointcloud_summary,
)
from .pointcloud_browse import PointcloudBrowseSource
from .pointcloud_label import PointcloudLabelSource
from .pointcloud import PointcloudService
from .traces import DirectTraceSource, TraceSource
from .traces.payload import serialize_trace_metadata, serialize_trace_payload

ReviewSource = Literal["label_set", "filter_file", "event_trace"]
SessionMode = Literal[
    "label",
    "label_review",
    "review",
    "pointcloud_label",
    "pointcloud_label_review",
    "pointcloud",
]
logger = logging.getLogger("attpc_estimator.estimator")
PEAK_CONFIG_KEYS = (
    "peak_separation",
    "peak_prominence",
    "peak_width",
    "peak_threshold",
    "peak_rel_height",
)


@dataclass(slots=True)
class SessionState:
    mode: SessionMode
    run: int | None
    detector: str = DETECTOR_ATTPC
    source: str | None = None
    family: str | None = None
    label: str | None = None
    filter_file: str | None = None
    event_id: int | None = None
    trace_id: int | None = None
    filter_item: str | None = None
    filter_value: float | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "run": self.run,
            "detector": self.detector,
            "source": self.source,
            "family": self.family,
            "label": self.label,
            "filterFile": self.filter_file,
            "eventId": self.event_id,
            "traceId": self.trace_id,
        }
        if self.source == "event_trace" and (
            self.filter_item not in {None, "none"} or self.filter_value is not None
        ):
            payload["filterItem"] = self.filter_item
            payload["filterValue"] = self.filter_value
        return payload


def default_ui_state() -> dict[str, Any]:
    return {
        "route": "/",
        "shell": {"selectedRun": None},
        "label": {"visualMode": "raw"},
        "review": {
            "source": "label_set",
            "detector": DETECTOR_ATTPC,
            "run": None,
            "family": "normal",
            "label": "",
            "filterFile": "",
            "eventId": None,
            "traceId": None,
            "filterItem": "none",
            "filterValue": None,
            "visualMode": "cdf",
        },
        "histograms": {
            "selectedDetector": DETECTOR_ATTPC,
            "selectedRun": None,
            "selectedPhase": "phase1",
            "selectedMetric": "cdf",
            "selectedMode": "all",
            "selectedBitflipVariant": "baseline",
            "selectedSaturationVariant": "drop",
            "selectedHistogramFilter": "",
            "selectedHistogramVeto": False,
            "cdfScaleMode": "linear",
            "amplitudeScaleMode": "linear",
            "cdfRenderMode": "2d",
            "cdfProjectionBin": 60,
            "labeledSeriesOrder": {},
        },
        "mapping": {
            "selectedLayer": "Pads",
            "selectedView": "Upstream",
            "rules": [],
        },
        "pointcloud": {
            "source": "event_id",
            "selectedRun": None,
            "selectedEventId": None,
            "selectedLabel": "",
            "layoutMode": "1x1",
            "panelTypes": [
                "hits-3d-amplitude",
                "pads-z",
                "hits-2d-amplitude",
                "traces",
            ],
            "selectedTraceIds": [],
        },
        "pointcloudLabel": {
            "visualMode": "basic",
        },
    }


class EstimatorService:
    def __init__(
        self,
        trace_path: Path,
        workspace: Path,
        baseline_window_scale: float = 10.0,
        ic_baseline_window_scale: float | None = None,
        detector_baseline_window_scales: dict[str, float] | None = None,
        peak_separation: float = 50.0,
        peak_prominence: float = 20.0,
        peak_width: float = 50.0,
        peak_threshold: float = 0.0,
        peak_rel_height: float = 0.95,
        detector_peak_configs: dict[str, dict[str, float]] | None = None,
        bitflip_baseline_threshold: float = BITFLIP_BASELINE_DEFAULT,
        saturation_threshold: float = 2000.0,
        saturation_drop_threshold: float = 10.0,
        saturation_window_radius: int = 16,
        default_run: int | None = None,
        pointcloud_micromegas_time_bucket: float | None = None,
        pointcloud_window_time_bucket: float | None = None,
        pointcloud_detector_length: float | None = None,
        ransac_config: RansacConfig = RansacConfig(),
        merge_config: MergeConfig = MergeConfig(),
        verbose: bool = False,
    ) -> None:
        self.trace_path = trace_path
        self.workspace = workspace
        self.baseline_window_scale = baseline_window_scale
        self.ic_baseline_window_scale = (
            baseline_window_scale
            if ic_baseline_window_scale is None
            else float(ic_baseline_window_scale)
        )
        self.detector_baseline_window_scales = self._build_detector_baseline_window_scales(
            baseline_window_scale=baseline_window_scale,
            ic_baseline_window_scale=self.ic_baseline_window_scale,
            detector_baseline_window_scales=detector_baseline_window_scales,
        )
        self.peak_separation = peak_separation
        self.peak_prominence = peak_prominence
        self.peak_width = peak_width
        self.peak_threshold = peak_threshold
        self.peak_rel_height = peak_rel_height
        self.detector_peak_configs = self._build_detector_peak_configs(
            peak_separation=peak_separation,
            peak_prominence=peak_prominence,
            peak_width=peak_width,
            peak_threshold=peak_threshold,
            peak_rel_height=peak_rel_height,
            detector_peak_configs=detector_peak_configs,
        )
        self.bitflip_baseline_threshold = bitflip_baseline_threshold
        self.ransac_config = ransac_config
        self.merge_config = merge_config
        self.verbose = verbose
        self.run_files = collect_run_files(trace_path)
        self.run_event_ranges = self._collect_run_event_ranges()
        self.repository = LabelRepository(labels_db_path(workspace))
        self.repository.initialize()
        self.histograms = HistogramService(
            trace_path=trace_path,
            workspace=workspace,
            baseline_window_scale=baseline_window_scale,
            peak_separation=peak_separation,
            peak_prominence=peak_prominence,
            peak_width=peak_width,
            peak_threshold=peak_threshold,
            peak_rel_height=peak_rel_height,
            detector_baseline_window_scales=self.detector_baseline_window_scales,
            detector_peak_configs=self.detector_peak_configs,
            bitflip_baseline_threshold=bitflip_baseline_threshold,
            saturation_threshold=saturation_threshold,
            saturation_drop_threshold=saturation_drop_threshold,
            saturation_window_radius=saturation_window_radius,
            ic_baseline_window_scale=self.ic_baseline_window_scale,
        )
        self.pointcloud = PointcloudService(
            trace_path=trace_path,
            workspace=workspace,
            baseline_window_scale=baseline_window_scale,
            micromegas_time_bucket=pointcloud_micromegas_time_bucket,
            window_time_bucket=pointcloud_window_time_bucket,
            detector_length=pointcloud_detector_length,
        )
        self.pointcloud.validate_processing_configs()
        self.ui_state_store = WebUiStateStore(webui_state_path(workspace))
        resolved_default_run = self._resolve_initial_run(default_run)
        self.session = SessionState(mode="label", run=resolved_default_run)
        self.ui_state = default_ui_state()
        self._sources: dict[tuple[Any, ...], TraceSource | DirectTraceSource] = {}
        self._pointcloud_label_sources: dict[
            tuple[Any, ...], PointcloudLabelSource | PointcloudBrowseSource
        ] = {}
        self._pointcloud_sources: dict[tuple[Any, ...], PointcloudBrowseSource] = {}
        self._active_source_key: tuple[Any, ...] | None = None
        self._active_pointcloud_label_key: tuple[Any, ...] | None = None
        self._active_pointcloud_key: tuple[Any, ...] | None = None
        self._restore_saved_state()

    def close(self) -> None:
        for source in self._sources.values():
            source.close()
        self._sources.clear()
        self._pointcloud_label_sources.clear()
        self._pointcloud_sources.clear()
        self.pointcloud.close()
        self.repository.connection.close()

    def bootstrap_state(self) -> dict[str, Any]:
        histogram_bootstrap = self.histograms.bootstrap_state()
        pointcloud_bootstrap = self.pointcloud.bootstrap_state()
        return {
            "appType": "merged",
            "workspace": str(self.workspace),
            "tracePath": str(self.trace_path),
            "databaseFile": str(labels_db_path(self.workspace)),
            "runs": histogram_bootstrap["runs"],
            "eventRanges": {
                str(run): {"min": event_range[0], "max": event_range[1]}
                for run, event_range in self.run_event_ranges.items()
            },
            "filterFiles": histogram_bootstrap["filterFiles"],
            "histogramAvailability": histogram_bootstrap["histogramAvailability"],
            "detectorHistogramAvailability": histogram_bootstrap[
                "detectorHistogramAvailability"
            ],
            "pointcloudRuns": pointcloud_bootstrap["runs"],
            "pointcloudEventRanges": pointcloud_bootstrap["eventRanges"],
            "normalSummary": normal_summary(self.repository),
            "pointcloudSummary": pointcloud_summary(self.repository),
            "strangeSummary": self.repository.get_strange_counts(),
            "strangeLabels": self.repository.list_strange_labels(),
            "session": self.session.as_payload(),
            "uiState": self.ui_state,
        }

    def set_session(
        self,
        *,
        mode: str,
        run: int | None = None,
        detector: str | None = None,
        source: str | None = None,
        family: str | None = None,
        label: str | None = None,
        filter_file: str | None = None,
        event_id: int | None = None,
        trace_id: int | None = None,
        si_side: str | None = None,
        si_index: int | None = None,
        gagg_layer: int | None = None,
        gagg_index: int | None = None,
        filter_item: str | None = None,
        filter_value: float | None = None,
    ) -> dict[str, Any]:
        resolved_detector = normalize_detector(detector)
        if mode == "pointcloud_label":
            resolved_run = self._resolve_pointcloud_run(run)
            source_key = ("label", resolved_run)
            label_source = self._get_or_create_pointcloud_label_source(source_key)
            assert isinstance(label_source, PointcloudLabelSource)
            ref = label_source.current_ref() or label_source.next_ref()
            self._active_source_key = None
            self._active_pointcloud_label_key = source_key
            self._active_pointcloud_key = None
            self.session = SessionState(
                mode="pointcloud_label",
                run=resolved_run,
                detector=DETECTOR_ATTPC,
            )
            return {
                "session": self.session.as_payload(),
                "event": self._serialize_pointcloud_label_event(ref.run, ref.event_id),
            }

        if mode == "pointcloud_label_review":
            resolved_run = self._resolve_pointcloud_run(run)
            source_key = ("review", resolved_run, label)
            label_source = self._get_or_create_pointcloud_label_source(source_key)
            ref = label_source.current_ref() or label_source.next_ref()
            self._active_source_key = None
            self._active_pointcloud_label_key = source_key
            self._active_pointcloud_key = None
            self.session = SessionState(
                mode="pointcloud_label_review",
                run=resolved_run,
                detector=DETECTOR_ATTPC,
                source="label_set",
                label=label,
            )
            return {
                "session": self.session.as_payload(),
                "event": self._serialize_pointcloud_label_event(ref.run, ref.event_id),
            }

        if mode == "pointcloud":
            resolved_run = self._resolve_pointcloud_run(run)
            if source not in {"event_id", "label_set"}:
                raise ValueError("pointcloud source must be 'event_id' or 'label_set'")
            source_key = ("browse", source, resolved_run, label)
            browse_source = self._get_or_create_pointcloud_source(source_key)
            if source == "event_id":
                if event_id is None:
                    raise ValueError("eventId is required for pointcloud direct browse")
                ref = browse_source.set_current(int(event_id))
            else:
                ref = browse_source.current_ref() or browse_source.next_ref()
            self._active_source_key = None
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = source_key
            self.session = SessionState(
                mode="pointcloud",
                run=resolved_run,
                detector=DETECTOR_ATTPC,
                source=source,
                label=label,
                event_id=ref.event_id,
            )
            return {
                "session": self.session.as_payload(),
                "event": self._serialize_pointcloud_event(ref.run, ref.event_id),
            }

        if mode == "label":
            if resolved_detector != DETECTOR_ATTPC:
                raise ValueError(f"{resolved_detector} trace labeling is not available")
            resolved_run = self._resolve_run(run)
            source_key = ("label", resolved_run)
            started = time.perf_counter()
            self._debug("starting label session run=%s", resolved_run)
            label_source = self._get_or_create_source(source_key)
            initialized = time.perf_counter()
            self._debug(
                "label source ready run=%s took=%.3fs",
                resolved_run,
                initialized - started,
            )
            self._active_source_key = source_key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            self.session = SessionState(
                mode="label", run=resolved_run, detector=DETECTOR_ATTPC
            )
            trace_started = time.perf_counter()
            trace_record = label_source.current_trace() or label_source.next_trace()
            self._debug(
                "first label trace ready run=%s event=%s trace=%s took=%.3fs total=%.3fs",
                trace_record.run,
                trace_record.event_id,
                trace_record.trace_id,
                time.perf_counter() - trace_started,
                time.perf_counter() - started,
            )
            return {
                "session": self.session.as_payload(),
                "trace": self._serialize_source_trace(trace_record),
            }

        if mode == "label_review":
            if resolved_detector != DETECTOR_ATTPC:
                raise ValueError(f"{resolved_detector} label review is not available")
            resolved_run = self._resolve_run(run)
            if family not in {"normal", "strange"}:
                raise ValueError("review family must be 'normal' or 'strange'")
            if family == "normal" and label is not None:
                if label not in {str(bucket) for bucket in NORMAL_BUCKETS} | {"4+"}:
                    raise ValueError("normal review label must be one of 0-9 or 4+")
            if family == "strange" and label is not None:
                if not self.repository.has_strange_label_name(label):
                    raise ValueError("selected strange label does not exist")
            source_key = ("review", "label_set", resolved_run, family, label)
            label_source = self._get_or_create_source(source_key)
            trace_count = label_source.trace_count()
            if trace_count == 0:
                raise LookupError("no traces match the selected review filter")
            self._active_source_key = source_key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            self.session = SessionState(
                mode="label_review",
                run=resolved_run,
                detector=DETECTOR_ATTPC,
                source="label_set",
                family=family,
                label=label,
            )
            return {
                "session": self.session.as_payload(),
                "traceCount": trace_count,
                "trace": self._serialize_source_trace(
                    label_source.current_trace() or label_source.next_trace()
                ),
            }

        if mode != "review":
            raise ValueError("session mode must be 'label', 'label_review', or 'review'")
        if source not in {"label_set", "filter_file", "event_trace"}:
            raise ValueError(
                "review session source must be 'label_set', 'filter_file', or 'event_trace'"
            )

        if source == "label_set":
            if resolved_detector != DETECTOR_ATTPC:
                raise ValueError(f"{resolved_detector} labeled review is not available")
            resolved_run = self._resolve_run(run)
            if family not in {"normal", "strange"}:
                raise ValueError("review family must be 'normal' or 'strange'")
            if family == "normal" and label is not None:
                if label not in {str(bucket) for bucket in NORMAL_BUCKETS} | {"4+"}:
                    raise ValueError("normal review label must be one of 0-9 or 4+")
            if family == "strange" and label is not None:
                if not self.repository.has_strange_label_name(label):
                    raise ValueError("selected strange label does not exist")
            source_key = ("review", "label_set", resolved_run, family, label)
            label_source = self._get_or_create_source(source_key)
            trace_count = label_source.trace_count()
            if trace_count == 0:
                raise LookupError("no traces match the selected review filter")
            self._active_source_key = source_key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            self.session = SessionState(
                mode="review",
                run=resolved_run,
                detector=DETECTOR_ATTPC,
                source="label_set",
                family=family,
                label=label,
            )
            return {
                "session": self.session.as_payload(),
                "traceCount": trace_count,
                "trace": self._serialize_source_trace(
                    label_source.current_trace() or label_source.next_trace()
                ),
            }

        if source == "filter_file":
            if resolved_detector != DETECTOR_ATTPC:
                raise ValueError(f"{resolved_detector} filter-file review is not available")
            if filter_file is None:
                raise ValueError("filterFile is required for filter-file review")
            source_key = ("review", "filter_file", filter_file)
            filter_source = self._get_or_create_source(source_key)
            trace_count = filter_source.trace_count()
            self._active_source_key = source_key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            self.session = SessionState(
                mode="review",
                run=None,
                detector=DETECTOR_ATTPC,
                source="filter_file",
                filter_file=filter_file,
            )
            payload: dict[str, Any] = {
                "session": self.session.as_payload(),
                "traceCount": trace_count,
            }
            payload["trace"] = (
                self._serialize_source_trace(
                    filter_source.current_trace() or filter_source.next_trace()
                )
                if trace_count > 0
                else None
            )
            return payload

        resolved_run = self._resolve_run(run)
        if event_id is None:
            raise ValueError("eventId is required for direct event review")
        resolved_trace_id = (
            0
            if resolved_detector == DETECTOR_IC
            else int(trace_id) if trace_id is not None else None
        )
        if resolved_detector == DETECTOR_ATTPC and resolved_trace_id is None:
            raise ValueError("traceId is required for ATTPC direct event review")
        resolved_filter_item = self._normalize_direct_filter_item(filter_item)
        resolved_filter_value = (
            float(filter_value)
            if resolved_filter_item == "max" and filter_value is not None
            else None
        )
        source_key = (
            "review",
            "event_trace",
            resolved_run,
            resolved_detector,
            resolved_filter_item,
            resolved_filter_value,
        )
        direct_source = self._get_or_create_source(source_key)
        assert isinstance(direct_source, DirectTraceSource)
        record = direct_source.set_position(
            event_id=int(event_id),
            trace_id=resolved_trace_id,
            si_side=si_side,
            si_index=si_index,
            gagg_layer=gagg_layer,
            gagg_index=gagg_index,
        )
        self._active_source_key = source_key
        self._active_pointcloud_label_key = None
        self._active_pointcloud_key = None
        self.session = SessionState(
            mode="review",
            run=resolved_run,
            detector=resolved_detector,
            source="event_trace",
            event_id=int(event_id),
            trace_id=int(record.trace_id),
            filter_item=resolved_filter_item,
            filter_value=resolved_filter_value,
        )
        return {
            "session": self.session.as_payload(),
            "trace": self._serialize_source_trace(record),
        }

    def current_trace(self) -> dict[str, Any]:
        return self._serialize_source_trace(
            self._current_source().current_trace_or_raise()
        )

    def next_trace(self) -> dict[str, Any]:
        return self._serialize_source_trace(self._current_source().next_trace())

    def previous_trace(self) -> dict[str, Any]:
        return self._serialize_source_trace(self._current_source().previous_trace())

    def next_event(self) -> dict[str, Any]:
        source = self._current_source()
        if not isinstance(source, DirectTraceSource):
            raise LookupError(
                "event navigation is only available for direct event review"
            )
        return self._serialize_source_trace(source.next_event())

    def previous_event(self) -> dict[str, Any]:
        source = self._current_source()
        if not isinstance(source, DirectTraceSource):
            raise LookupError(
                "event navigation is only available for direct event review"
            )
        return self._serialize_source_trace(source.previous_event())

    def assign_label(
        self,
        *,
        event_id: int,
        trace_id: int,
        family: str,
        label: str,
    ) -> dict[str, Any]:
        if family not in {"normal", "strange"}:
            raise ValueError("label family must be 'normal' or 'strange'")
        if self.session.detector != DETECTOR_ATTPC:
            raise ValueError(f"{self.session.detector} trace labeling is not available")
        if self.session.run is None:
            raise ValueError("label assignment requires an active run-backed session")

        ref = TraceRef(run=self.session.run, event_id=event_id, trace_id=trace_id)
        active_source = self._current_source()
        record = active_source.get_trace(ref)
        if record is None:
            raise ValueError("selected trace is not available")

        self.repository.save_label(
            record.run,
            event_id,
            trace_id,
            record.detector,
            record.hardware_id[0],
            record.hardware_id[1],
            record.hardware_id[2],
            record.hardware_id[3],
            record.hardware_id[4],
            family,
            label,
        )
        active_source.apply_label(ref, family, label)
        labels = self._labels_snapshot()
        for key, source in list(self._sources.items()):
            if source is active_source:
                continue
            if self._is_labeled_review_source(key, run=ref.run):
                source.close()
                del self._sources[key]
                continue
            source.replace_labels(labels)
        return {
            "labeledCount": self.repository.total_labeled(),
            "normalSummary": normal_summary(self.repository),
            "strangeSummary": self.repository.get_strange_counts(),
            "currentLabel": {"family": family, "label": label},
        }

    def current_pointcloud_label_event(self) -> dict[str, Any]:
        source = self._current_pointcloud_label_source()
        ref = source.current_ref_or_raise()
        return self._serialize_pointcloud_label_event(ref.run, ref.event_id)

    def next_pointcloud_label_event(self) -> dict[str, Any]:
        ref = self._current_pointcloud_label_source().next_ref()
        return self._serialize_pointcloud_label_event(ref.run, ref.event_id)

    def previous_pointcloud_label_event(self) -> dict[str, Any]:
        ref = self._current_pointcloud_label_source().previous_ref()
        return self._serialize_pointcloud_label_event(ref.run, ref.event_id)

    def assign_pointcloud_label(self, *, event_id: int, label: str) -> dict[str, Any]:
        if self.session.mode not in {"pointcloud_label", "pointcloud_label_review"}:
            raise ValueError("pointcloud label assignment requires an active pointcloud label session")
        if self.session.run is None:
            raise ValueError("pointcloud label assignment requires an active run-backed session")
        self.repository.save_pointcloud_label(self.session.run, int(event_id), label)
        self._refresh_pointcloud_label_sources(run=self.session.run)
        self._refresh_pointcloud_sources(run=self.session.run)
        return {
            "pointcloudSummary": pointcloud_summary(self.repository),
            "currentLabel": label,
        }

    def get_strange_labels(self) -> dict[str, Any]:
        return {"strangeLabels": self.repository.list_strange_labels()}

    def create_strange_label(self, name: str, shortcut_key: str) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("label name cannot be empty")
        normalized_shortcut = normalize_shortcut(shortcut_key)
        if len(normalized_shortcut) != 1:
            raise ValueError("shortcut key must be a single key")
        if normalized_shortcut in RESERVED_SHORTCUTS:
            raise ValueError("shortcut key is reserved")
        if self.repository.has_strange_label_name(clean_name):
            raise ValueError("label name already exists")
        if self.repository.has_shortcut(normalized_shortcut):
            raise ValueError("shortcut key already exists")
        return self.repository.create_strange_label(clean_name, normalized_shortcut)

    def delete_strange_label(self, strange_label_name: str) -> list[dict[str, Any]]:
        self.repository.delete_strange_label(strange_label_name)
        return self.repository.get_strange_counts()

    def get_histogram(
        self,
        *,
        metric: str,
        mode: str,
        run: int,
        detector: str | None = None,
        variant: str | None = None,
        filter_file: str | None = None,
        veto: bool = False,
    ) -> dict[str, Any]:
        return self.histograms.get_histogram(
            metric=metric,
            mode=mode,
            run=run,
            detector=detector,
            variant=variant,
            filter_file=filter_file,
            veto=veto,
        )

    def create_histogram_job(
        self,
        *,
        metric: str,
        mode: str,
        run: int,
        detector: str | None = None,
        variant: str | None = None,
        filter_file: str | None = None,
        veto: bool = False,
    ) -> dict[str, str]:
        return {
            "jobId": self.histograms.create_histogram_job(
                metric=metric,
                mode=mode,
                run=run,
                detector=detector,
                variant=variant,
                filter_file=filter_file,
                veto=veto,
            )
        }

    def next_histogram_job_message(
        self,
        *,
        job_id: str,
        after_index: int,
    ) -> tuple[int, dict] | None:
        return self.histograms.next_job_message(job_id, after_index)

    def get_pointcloud_event(self, *, run: int, event_id: int) -> dict[str, Any]:
        return self._serialize_pointcloud_event(run, event_id)

    def current_pointcloud_event(self) -> dict[str, Any]:
        source = self._current_pointcloud_source()
        ref = source.current_ref_or_raise()
        return self._serialize_pointcloud_event(ref.run, ref.event_id)

    def next_pointcloud_event(self) -> dict[str, Any]:
        ref = self._current_pointcloud_source().next_ref()
        return self._serialize_pointcloud_event(ref.run, ref.event_id)

    def previous_pointcloud_event(self) -> dict[str, Any]:
        ref = self._current_pointcloud_source().previous_ref()
        return self._serialize_pointcloud_event(ref.run, ref.event_id)

    def get_pointcloud_traces(
        self,
        *,
        run: int,
        event_id: int,
        trace_ids: list[int],
    ) -> dict[str, Any]:
        return self.pointcloud.get_traces(run=run, event_id=event_id, trace_ids=trace_ids)

    def update_ui_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ui_state = self._normalize_ui_state(payload)
        self._persist_saved_state()
        return self.ui_state

    def _resolve_run(self, run: int | None) -> int:
        if run is not None:
            if run not in self.run_files:
                raise ValueError(f"run {run} is not available")
            return run
        if self.session.run is not None and self.session.run in self.run_files:
            return self.session.run
        if self.run_files:
            return sorted(self.run_files)[0]
        raise ValueError("no runs are available")

    def _resolve_pointcloud_run(self, run: int | None) -> int:
        available_runs = self.pointcloud.bootstrap_state()["runs"]
        if run is not None:
            if run not in available_runs:
                raise ValueError(f"pointcloud run {run} is not available")
            return run
        if self.session.run is not None and self.session.run in available_runs:
            return self.session.run
        if available_runs:
            return int(sorted(available_runs)[0])
        raise ValueError("no pointcloud runs are available")

    def _resolve_initial_run(self, run: int | None) -> int | None:
        if run is None:
            return sorted(self.run_files)[0] if self.run_files else None
        if run not in self.run_files:
            raise ValueError(f"default run {run} is not available")
        return run

    @staticmethod
    def _normalize_direct_filter_item(filter_item: str | None) -> str:
        token = str(filter_item or "none").strip().lower()
        if token not in {"none", "max"}:
            raise ValueError("filterItem must be 'none' or 'max'")
        return token

    def _current_source(self) -> TraceSource | DirectTraceSource:
        if self._active_source_key is None:
            raise LookupError("no active trace source is available")
        return self._get_or_create_source(self._active_source_key)

    def _current_pointcloud_label_source(self) -> PointcloudLabelSource | PointcloudBrowseSource:
        if self._active_pointcloud_label_key is None:
            raise LookupError("no active pointcloud label source is available")
        source = self._pointcloud_label_sources.get(self._active_pointcloud_label_key)
        if source is None:
            source = self._get_or_create_pointcloud_label_source(self._active_pointcloud_label_key)
        return source

    def _current_pointcloud_source(self) -> PointcloudBrowseSource:
        if self._active_pointcloud_key is None:
            raise LookupError("no active pointcloud source is available")
        return self._get_or_create_pointcloud_source(self._active_pointcloud_key)

    def _serialize_source_trace(self, record) -> dict[str, Any]:
        if self.session.source == "event_trace":
            self.session.event_id = int(record.event_id)
            self.session.trace_id = int(record.trace_id)
        self._persist_saved_state()
        label = (
            None
            if record.detector != DETECTOR_ATTPC
            else self.repository.get_label(record.run, record.event_id, record.trace_id)
        )
        source = self._current_source()
        event_trace_count = (
            source.current_event_trace_count()
            if isinstance(source, DirectTraceSource)
            else None
        )
        event_id_range = (
            source.event_id_range() if isinstance(source, DirectTraceSource) else None
        )
        peak_config = self._peak_config_for(record.detector)
        return serialize_trace_payload(
            record,
            bitflip_baseline_threshold=self.bitflip_baseline_threshold,
            label=label,
            review_progress=source.get_progress(),
            include_run=True,
            peak_separation=peak_config["peak_separation"],
            peak_prominence=peak_config["peak_prominence"],
            peak_width=peak_config["peak_width"],
            peak_threshold=peak_config["peak_threshold"],
            peak_rel_height=peak_config["peak_rel_height"],
            event_trace_count=event_trace_count,
            event_id_range=event_id_range,
            trace_selector=(
                source.current_trace_selector()
                if isinstance(source, DirectTraceSource)
                else None
            ),
            event_context=(
                source.event_context() if isinstance(source, DirectTraceSource) else None
            ),
            trace_metadata=serialize_trace_metadata(record),
        )

    def _get_or_create_source(
        self,
        key: tuple[Any, ...],
    ) -> TraceSource | DirectTraceSource:
        source = self._sources.get(key)
        labels = self._labels_snapshot()
        if source is None:
            source = self._build_source(key, labels)
            self._sources[key] = source
            return source
        source.replace_labels(labels)
        return source

    def _get_or_create_pointcloud_label_source(
        self,
        key: tuple[Any, ...],
    ) -> PointcloudLabelSource | PointcloudBrowseSource:
        source = self._pointcloud_label_sources.get(key)
        if source is None:
            run = int(key[1])
            if key[0] == "label":
                source = PointcloudLabelSource(
                    event_ranges=self.pointcloud._event_ranges,
                    run=run,
                    labeled_event_ids=self.repository.list_labeled_pointcloud_event_ids(run),
                )
            elif key[0] == "review":
                source = PointcloudBrowseSource(
                    event_ranges=self.pointcloud._event_ranges,
                    run=run,
                    source="label_set",
                    labeled_event_ids=[
                        event_id
                        for event_id, _ in self.repository.list_labeled_pointcloud_events(
                            run,
                            label=key[2] if len(key) > 2 else None,
                        )
                    ],
                )
            else:
                raise LookupError(f"unsupported pointcloud label source key: {key!r}")
            self._pointcloud_label_sources[key] = source
        else:
            self._refresh_pointcloud_label_source(key, source)
        return source

    def _get_or_create_pointcloud_source(
        self,
        key: tuple[Any, ...],
    ) -> PointcloudBrowseSource:
        source = self._pointcloud_sources.get(key)
        if source is None:
            run = int(key[2])
            source = PointcloudBrowseSource(
                event_ranges=self.pointcloud._event_ranges,
                run=run,
                source=str(key[1]),
                labeled_event_ids=[
                    event_id
                    for event_id, _ in self.repository.list_labeled_pointcloud_events(
                        run,
                        label=key[3] if len(key) > 3 else None,
                    )
                ],
            )
            self._pointcloud_sources[key] = source
        elif key[1] == "label_set":
            source.update_labeled_event_ids(
                [
                    event_id
                    for event_id, _ in self.repository.list_labeled_pointcloud_events(
                        int(key[2]),
                        label=key[3] if len(key) > 3 else None,
                    )
                ]
            )
        return source

    def _build_source(
        self,
        key: tuple[Any, ...],
        labels: dict[TraceRef, StoredLabel],
    ) -> TraceSource | DirectTraceSource:
        if key[0] == "label":
            run = int(key[1])
            return TraceSource.for_label_mode(
                self.run_files[run],
                labels=labels,
                baseline_window_scale=self.baseline_window_scale,
                verbose=self.verbose,
            )
        if key[0] == "review" and key[1] == "label_set":
            run = int(key[2])
            family = str(key[3])
            label = key[4]
            return TraceSource.for_review_mode(
                self.run_files[run],
                family=family,
                label=label,
                labels=labels,
                baseline_window_scale=self.baseline_window_scale,
                verbose=self.verbose,
            )
        if key[0] == "review" and key[1] == "filter_file":
            filter_file = str(key[2])
            filter_path = filter_dir(self.workspace) / filter_file
            available = {
                path.name for path in filter_dir(self.workspace).glob("filter_*.npy")
            }
            if filter_file not in available:
                raise ValueError(f"filter file not found: {filter_file}")
            rows = np.load(filter_path)
            return TraceSource.for_filter_rows(
                self.run_files,
                rows,
                labels=labels,
                baseline_window_scale=self.baseline_window_scale,
                verbose=self.verbose,
            )
        if key[0] == "review" and key[1] == "event_trace":
            run = int(key[2])
            detector = (
                normalize_detector(str(key[3]))
                if len(key) > 3
                else DETECTOR_ATTPC
            )
            filter_item = str(key[4]) if len(key) > 4 else "none"
            filter_value = float(key[5]) if len(key) > 5 and key[5] is not None else None
            return DirectTraceSource(
                self.run_files[run],
                run=run,
                labels=labels,
                baseline_window_scale=self._baseline_window_scale_for(detector),
                detector=detector,
                filter_item=filter_item,
                filter_value=filter_value,
            )
        raise ValueError(f"unsupported source key: {key!r}")

    @staticmethod
    def _build_detector_baseline_window_scales(
        *,
        baseline_window_scale: float,
        ic_baseline_window_scale: float,
        detector_baseline_window_scales: dict[str, float] | None,
    ) -> dict[str, float]:
        scales = {
            DETECTOR_ATTPC: float(baseline_window_scale),
            DETECTOR_IC: float(ic_baseline_window_scale),
            DETECTOR_SI: float(baseline_window_scale),
            DETECTOR_GAGG: float(baseline_window_scale),
        }
        for detector, value in (detector_baseline_window_scales or {}).items():
            scales[normalize_detector(detector)] = float(value)
        return scales

    @staticmethod
    def _build_detector_peak_configs(
        *,
        peak_separation: float,
        peak_prominence: float,
        peak_width: float,
        peak_threshold: float,
        peak_rel_height: float,
        detector_peak_configs: dict[str, dict[str, float]] | None,
    ) -> dict[str, dict[str, float]]:
        default_config = {
            "peak_separation": float(peak_separation),
            "peak_prominence": float(peak_prominence),
            "peak_width": float(peak_width),
            "peak_threshold": float(peak_threshold),
            "peak_rel_height": float(peak_rel_height),
        }
        configs = {
            detector: dict(default_config)
            for detector in (
                DETECTOR_ATTPC,
                DETECTOR_IC,
                DETECTOR_SI,
                DETECTOR_GAGG,
            )
        }
        for detector, values in (detector_peak_configs or {}).items():
            resolved = normalize_detector(detector)
            for key in PEAK_CONFIG_KEYS:
                if key in values:
                    configs[resolved][key] = float(values[key])
        return configs

    def _baseline_window_scale_for(self, detector: str | None) -> float:
        return float(
            self.detector_baseline_window_scales[
                normalize_detector(detector)
            ]
        )

    def _peak_config_for(self, detector: str | None) -> dict[str, float]:
        return self.detector_peak_configs[normalize_detector(detector)]

    def _restore_saved_state(self) -> None:
        payload = self.ui_state_store.load()
        self.ui_state = self._normalize_ui_state(payload.get("uiState"))
        runtime_session = payload.get("runtimeSession")
        if not isinstance(runtime_session, dict):
            self._persist_saved_state()
            return
        try:
            self._restore_runtime_session(runtime_session)
        except (LookupError, ValueError):
            self._active_source_key = None
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            self.session = SessionState(
                mode="label",
                run=self._resolve_initial_run(self.session.run),
            )
        self._persist_saved_state()

    def _restore_runtime_session(self, payload: dict[str, Any]) -> None:
        session_payload = payload.get("session")
        source_payload = payload.get("source")
        if not isinstance(session_payload, dict) or not isinstance(
            source_payload, dict
        ):
            return
        mode = session_payload.get("mode")
        run = session_payload.get("run")
        source_name = session_payload.get("source")
        family = session_payload.get("family")
        label = session_payload.get("label")
        filter_file = session_payload.get("filterFile")
        event_id = session_payload.get("eventId")
        trace_id = session_payload.get("traceId")
        filter_item = session_payload.get("filterItem")
        filter_value = session_payload.get("filterValue")

        if mode == "label":
            resolved_run = self._resolve_run(run if isinstance(run, int) else None)
            key = ("label", resolved_run)
            source = self._get_or_create_source(key)
            assert isinstance(source, TraceSource)
            source.restore_state(source_payload)
            if source.current_trace() is None:
                source.next_trace()
            self.session = SessionState(mode="label", run=resolved_run)
            self._active_source_key = key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            return

        if mode == "label_review":
            resolved_run = self._resolve_run(run if isinstance(run, int) else None)
            key = ("review", "label_set", resolved_run, family, label)
            source = self._get_or_create_source(key)
            assert isinstance(source, TraceSource)
            source.restore_state(source_payload)
            if source.current_trace() is None:
                if source.trace_count() == 0:
                    raise LookupError("no traces match the selected review filter")
                source.next_trace()
            self.session = SessionState(
                mode="label_review",
                run=resolved_run,
                source="label_set",
                family=family if isinstance(family, str) else None,
                label=label if isinstance(label, str) else None,
            )
            self._active_source_key = key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            return

        if mode == "pointcloud_label":
            resolved_run = self._resolve_pointcloud_run(run if isinstance(run, int) else None)
            key = ("label", resolved_run)
            source = self._get_or_create_pointcloud_label_source(key)
            source.restore_state(source_payload)
            if source.current_ref() is None:
                source.next_ref()
            self.session = SessionState(
                mode="pointcloud_label",
                run=resolved_run,
                detector=DETECTOR_ATTPC,
            )
            self._active_source_key = None
            self._active_pointcloud_label_key = key
            self._active_pointcloud_key = None
            return

        if mode == "pointcloud_label_review":
            resolved_run = self._resolve_pointcloud_run(run if isinstance(run, int) else None)
            key = ("review", resolved_run, label if isinstance(label, str) else None)
            source = self._get_or_create_pointcloud_label_source(key)
            source.restore_state(source_payload)
            if source.current_ref() is None:
                source.next_ref()
            self.session = SessionState(
                mode="pointcloud_label_review",
                run=resolved_run,
                detector=DETECTOR_ATTPC,
                source="label_set",
                label=label if isinstance(label, str) else None,
            )
            self._active_source_key = None
            self._active_pointcloud_label_key = key
            self._active_pointcloud_key = None
            return

        if mode == "pointcloud":
            resolved_run = self._resolve_pointcloud_run(run if isinstance(run, int) else None)
            if source_name not in {"event_id", "label_set"}:
                raise ValueError("pointcloud source must be 'event_id' or 'label_set'")
            key = ("browse", source_name, resolved_run, label if isinstance(label, str) else None)
            source = self._get_or_create_pointcloud_source(key)
            source.restore_state(source_payload)
            if source.current_ref() is None:
                if source_name == "event_id" and isinstance(event_id, int):
                    source.set_current(event_id)
                else:
                    source.next_ref()
            current = source.current_ref_or_raise()
            self.session = SessionState(
                mode="pointcloud",
                run=resolved_run,
                detector=DETECTOR_ATTPC,
                source=source_name,
                label=label if isinstance(label, str) else None,
                event_id=current.event_id,
            )
            self._active_source_key = None
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = key
            return

        if mode != "review":
            return
        if source_name == "label_set":
            resolved_run = self._resolve_run(run if isinstance(run, int) else None)
            key = ("review", "label_set", resolved_run, family, label)
            source = self._get_or_create_source(key)
            assert isinstance(source, TraceSource)
            source.restore_state(source_payload)
            if source.current_trace() is None:
                if source.trace_count() == 0:
                    raise LookupError("no traces match the selected review filter")
                source.next_trace()
            self.session = SessionState(
                mode="review",
                run=resolved_run,
                source="label_set",
                family=family if isinstance(family, str) else None,
                label=label if isinstance(label, str) else None,
            )
            self._active_source_key = key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            return
        if source_name == "filter_file" and isinstance(filter_file, str):
            key = ("review", "filter_file", filter_file)
            source = self._get_or_create_source(key)
            assert isinstance(source, TraceSource)
            source.restore_state(source_payload)
            current = source.current_trace()
            if current is None and source.trace_count() > 0:
                source.next_trace()
            self.session = SessionState(
                mode="review",
                run=None,
                source="filter_file",
                filter_file=filter_file,
            )
            self._active_source_key = key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None
            return
        if (
            source_name == "event_trace"
            and isinstance(run, int)
            and isinstance(event_id, int)
        ):
            detector_name = session_payload.get("detector")
            resolved_detector = normalize_detector(
                detector_name if isinstance(detector_name, str) else DETECTOR_ATTPC
            )
            resolved_run = self._resolve_run(run)
            resolved_filter_item = self._normalize_direct_filter_item(
                filter_item if isinstance(filter_item, str) else None
            )
            resolved_filter_value = (
                float(filter_value)
                if resolved_filter_item == "max"
                and isinstance(filter_value, (int, float))
                else None
            )
            key = (
                "review",
                "event_trace",
                resolved_run,
                resolved_detector,
                resolved_filter_item,
                resolved_filter_value,
            )
            source = self._get_or_create_source(key)
            assert isinstance(source, DirectTraceSource)
            source.restore_state(source_payload)
            resolved_trace_id = (
                0
                if resolved_detector == DETECTOR_IC
                else trace_id if isinstance(trace_id, int) else None
            )
            if resolved_trace_id is None:
                raise ValueError(
                    f"traceId is required for {resolved_detector} direct event review"
                )
            self.session = SessionState(
                mode="review",
                run=resolved_run,
                detector=resolved_detector,
                source="event_trace",
                event_id=event_id,
                trace_id=resolved_trace_id,
                filter_item=resolved_filter_item,
                filter_value=resolved_filter_value,
            )
            self._active_source_key = key
            self._active_pointcloud_label_key = None
            self._active_pointcloud_key = None

    def _runtime_session_snapshot(self) -> dict[str, Any] | None:
        source = None
        if self._active_source_key is not None:
            source = self._sources.get(self._active_source_key)
        elif self._active_pointcloud_label_key is not None:
            source = self._pointcloud_label_sources.get(self._active_pointcloud_label_key)
        elif self._active_pointcloud_key is not None:
            source = self._pointcloud_sources.get(self._active_pointcloud_key)
        if source is None:
            return None
        return {
            "session": self.session.as_payload(),
            "source": source.snapshot_state(),
        }

    def _persist_saved_state(self) -> None:
        self.ui_state_store.save(
            ui_state=self.ui_state,
            runtime_session=self._runtime_session_snapshot(),
        )

    def _normalize_ui_state(self, payload: object) -> dict[str, Any]:
        state = default_ui_state()
        if not isinstance(payload, dict):
            state["shell"]["selectedRun"] = self._resolve_initial_run(self.session.run)
            state["histograms"]["selectedRun"] = self._resolve_initial_run(
                self.session.run
            )
            state["pointcloud"]["selectedRun"] = (
                sorted(self.pointcloud.pointcloud_files)[0]
                if self.pointcloud.pointcloud_files
                else None
            )
            return state

        route = payload.get("route")
        if isinstance(route, str) and self._is_supported_route(route):
            state["route"] = route

        runs = sorted(self.run_files)
        run_values = set(runs)
        pointcloud_runs = sorted(self.pointcloud.pointcloud_files)
        pointcloud_run_values = set(pointcloud_runs)
        filter_files = {
            path.name for path in filter_dir(self.workspace).glob("filter_*.npy")
        }

        shell_payload = payload.get("shell")
        if isinstance(shell_payload, dict):
            selected_run = shell_payload.get("selectedRun")
            if isinstance(selected_run, int) and selected_run in run_values:
                state["shell"]["selectedRun"] = selected_run

        label_payload = payload.get("label")
        if isinstance(label_payload, dict):
            visual_mode = label_payload.get("visualMode")
            if visual_mode in {"raw", "cdf", "curvature", "peak"}:
                state["label"]["visualMode"] = visual_mode

        review_payload = payload.get("review")
        if isinstance(review_payload, dict):
            state["review"]["detector"] = DETECTOR_ATTPC
            source_name = review_payload.get("source")
            if source_name in {"label_set", "filter_file", "event_trace"}:
                state["review"]["source"] = source_name
            if (
                state["review"]["source"] == "event_trace"
                and review_payload.get("detector")
                in {DETECTOR_ATTPC, DETECTOR_IC, DETECTOR_SI, DETECTOR_GAGG}
            ):
                state["review"]["detector"] = review_payload["detector"]
            review_run = review_payload.get("run")
            if isinstance(review_run, int) and review_run in run_values:
                state["review"]["run"] = review_run
            family = review_payload.get("family")
            if family in {"normal", "strange"}:
                state["review"]["family"] = family
            for key in {"label", "filterFile"}:
                value = review_payload.get(key)
                if isinstance(value, str):
                    state["review"][key] = value
            if state["review"]["filterFile"] not in filter_files:
                state["review"]["filterFile"] = ""
            for key in {"eventId", "traceId"}:
                value = review_payload.get(key)
                if isinstance(value, int):
                    state["review"][key] = value
            filter_item = review_payload.get("filterItem")
            if filter_item in {"none", "max"}:
                state["review"]["filterItem"] = filter_item
            filter_value = review_payload.get("filterValue")
            if isinstance(filter_value, (int, float)):
                state["review"]["filterValue"] = float(filter_value)
            if (
                state["review"]["source"] == "event_trace"
                and state["review"]["detector"] == DETECTOR_IC
            ):
                state["review"]["traceId"] = 0
            visual_mode = review_payload.get("visualMode")
            if visual_mode in {"raw", "cdf", "curvature", "peak"}:
                state["review"]["visualMode"] = visual_mode

        histogram_payload = payload.get("histograms")
        if isinstance(histogram_payload, dict):
            if histogram_payload.get("selectedDetector") in {
                DETECTOR_ATTPC,
                DETECTOR_IC,
                DETECTOR_SI,
                DETECTOR_GAGG,
            }:
                state["histograms"]["selectedDetector"] = histogram_payload[
                    "selectedDetector"
                ]
            histogram_run = histogram_payload.get("selectedRun")
            if isinstance(histogram_run, int) and histogram_run in run_values:
                state["histograms"]["selectedRun"] = histogram_run
            for key, values in {
                "selectedPhase": {"phase1", "phase2"},
                "selectedMetric": {
                    "cdf",
                    "amplitude",
                    "baseline",
                    "bitflip",
                    "saturation",
                    "line_distance",
                    "line_property",
                    "coplanar",
                },
                "selectedMode": {"all", "labeled", "filtered"},
                "selectedBitflipVariant": {"baseline", "value", "length", "count"},
                "selectedSaturationVariant": {"drop", "length"},
                "cdfScaleMode": {"linear", "log"},
                "amplitudeScaleMode": {"linear", "log"},
                "cdfRenderMode": {"2d", "projection"},
            }.items():
                value = histogram_payload.get(key)
                if value in values:
                    state["histograms"][key] = value
            histogram_filter = histogram_payload.get("selectedHistogramFilter")
            if isinstance(histogram_filter, str) and histogram_filter in filter_files:
                state["histograms"]["selectedHistogramFilter"] = histogram_filter
            state["histograms"]["selectedHistogramVeto"] = bool(
                histogram_payload.get("selectedHistogramVeto", False)
            )
            cdf_projection_bin = histogram_payload.get("cdfProjectionBin")
            if isinstance(cdf_projection_bin, int):
                state["histograms"]["cdfProjectionBin"] = min(
                    150, max(1, cdf_projection_bin)
                )
            labeled_order = histogram_payload.get("labeledSeriesOrder")
            if isinstance(labeled_order, dict):
                normalized_order: dict[str, list[str]] = {}
                for key, values in labeled_order.items():
                    if not isinstance(key, str) or not isinstance(values, list):
                        continue
                    normalized_order[key] = [
                        value for value in values if isinstance(value, str)
                    ]
                state["histograms"]["labeledSeriesOrder"] = normalized_order

        mapping_payload = payload.get("mapping")
        if isinstance(mapping_payload, dict):
            selected_layer = mapping_payload.get("selectedLayer")
            legacy_layers = {"Si-0": "T0D1", "Si-1": "T0D2"}
            selected_layer = legacy_layers.get(selected_layer, selected_layer)
            if selected_layer in {"Pads", "T0D1", "T0D2"}:
                state["mapping"]["selectedLayer"] = selected_layer
            if mapping_payload.get("selectedView") in {"Upstream", "Downstream"}:
                state["mapping"]["selectedView"] = mapping_payload["selectedView"]
            rules = mapping_payload.get("rules")
            if isinstance(rules, list):
                state["mapping"]["rules"] = [
                    {
                        "cobo": str(rule.get("cobo", "")),
                        "asad": str(rule.get("asad", "")),
                        "aget": str(rule.get("aget", "")),
                        "channel": str(rule.get("channel", "")),
                        "color": str(rule.get("color", "")),
                    }
                    for rule in rules
                    if isinstance(rule, dict)
                ]

        pointcloud_payload = payload.get("pointcloud")
        if isinstance(pointcloud_payload, dict):
            source_name = pointcloud_payload.get("source")
            if source_name in {"event_id", "label_set"}:
                state["pointcloud"]["source"] = source_name
            selected_run = pointcloud_payload.get("selectedRun")
            if isinstance(selected_run, int) and selected_run in pointcloud_run_values:
                state["pointcloud"]["selectedRun"] = selected_run
            selected_event_id = pointcloud_payload.get("selectedEventId")
            if isinstance(selected_event_id, int):
                state["pointcloud"]["selectedEventId"] = selected_event_id
            selected_label = pointcloud_payload.get("selectedLabel")
            if isinstance(selected_label, str):
                state["pointcloud"]["selectedLabel"] = selected_label
            layout_mode = pointcloud_payload.get("layoutMode")
            if layout_mode in {"1x1", "2x2"}:
                state["pointcloud"]["layoutMode"] = layout_mode
            panel_types = pointcloud_payload.get("panelTypes")
            if isinstance(panel_types, list):
                state["pointcloud"]["panelTypes"] = [
                    value for value in panel_types if isinstance(value, str)
                ]
            selected_trace_ids = pointcloud_payload.get("selectedTraceIds")
            if isinstance(selected_trace_ids, list):
                state["pointcloud"]["selectedTraceIds"] = [
                    int(value) for value in selected_trace_ids if isinstance(value, int)
                ]

        pointcloud_label_payload = payload.get("pointcloudLabel")
        if isinstance(pointcloud_label_payload, dict):
            visual_mode = pointcloud_label_payload.get("visualMode")
            if visual_mode in {"basic", "detail"}:
                state["pointcloudLabel"]["visualMode"] = visual_mode

        if state["shell"]["selectedRun"] is None and runs:
            state["shell"]["selectedRun"] = runs[0]
        if state["histograms"]["selectedRun"] is None:
            state["histograms"]["selectedRun"] = state["shell"]["selectedRun"]

        return state

    def _labels_snapshot(self) -> dict[TraceRef, StoredLabel]:
        return labels_snapshot(self.repository)

    def _debug(self, message: str, *args: object) -> None:
        if self.verbose:
            logger.debug(message, *args)

    def _collect_run_event_ranges(self) -> dict[int, tuple[int, int]]:
        ranges: dict[int, tuple[int, int]] = {}
        for run, path in self.run_files.items():
            with h5py.File(path, "r") as handle:
                metadata = describe_trace_events(handle)
            ranges[int(run)] = (metadata.min_event, metadata.max_event)
        return ranges

    @staticmethod
    def _is_labeled_review_source(key: tuple[Any, ...], *, run: int) -> bool:
        return (
            len(key) >= 3
            and key[0] == "review"
            and key[1] == "label_set"
            and int(key[2]) == run
        )

    def _is_supported_route(self, route: str) -> bool:
        path = route.split("?", 1)[0].split("#", 1)[0]
        return path in {
            "/",
            "/label",
            "/label/trace",
            "/label/pointcloud",
            "/browse",
            "/browse/trace",
            "/browse/pointcloud",
            "/mapping",
            "/histograms",
        }

    def _serialize_pointcloud_event(self, run: int, event_id: int) -> dict[str, Any]:
        payload = self.pointcloud.get_event(run=run, event_id=event_id)
        if self.session.mode == "pointcloud":
            self.session.run = int(run)
            self.session.event_id = int(event_id)
        self._persist_saved_state()
        return payload

    def _serialize_pointcloud_label_event(self, run: int, event_id: int) -> dict[str, Any]:
        payload = self.pointcloud.get_label_event(
            run=run,
            event_id=event_id,
            ransac_config=self.ransac_config,
            merge_config=self.merge_config,
        )
        if self.session.mode in {"pointcloud_label", "pointcloud_label_review"}:
            self.session.run = int(run)
            self.session.event_id = int(event_id)
        payload["currentLabel"] = self.repository.get_pointcloud_label(run, event_id)
        self._persist_saved_state()
        return payload

    def _refresh_pointcloud_label_source(
        self,
        key: tuple[Any, ...],
        source: PointcloudLabelSource | PointcloudBrowseSource,
    ) -> None:
        run = int(key[1])
        if key[0] == "label":
            assert isinstance(source, PointcloudLabelSource)
            source.update_labeled_event_ids(self.repository.list_labeled_pointcloud_event_ids(run))
            return
        source.update_labeled_event_ids(
            [
                event_id
                for event_id, _ in self.repository.list_labeled_pointcloud_events(
                    run,
                    label=key[2] if len(key) > 2 else None,
                )
            ]
        )

    def _refresh_pointcloud_label_sources(self, *, run: int) -> None:
        for key, source in self._pointcloud_label_sources.items():
            if int(key[1]) == int(run):
                self._refresh_pointcloud_label_source(key, source)

    def _refresh_pointcloud_sources(self, *, run: int) -> None:
        for key, source in self._pointcloud_sources.items():
            if key[1] != "label_set" or int(key[2]) != int(run):
                continue
            source.update_labeled_event_ids(
                [
                    event_id
                    for event_id, _ in self.repository.list_labeled_pointcloud_events(
                        int(run),
                        label=key[3] if len(key) > 3 else None,
                    )
                ]
            )
