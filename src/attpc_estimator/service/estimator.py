from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from ..model.label import NORMAL_BUCKETS, StoredLabel
from ..model.trace import TraceRef
from ..process.bitflip import BITFLIP_BASELINE_DEFAULT
from ..storage.labels_db import LabelRepository
from ..storage.run_paths import collect_run_files, labels_db_path
from .histograms import HistogramService
from .labeling import (
    RESERVED_SHORTCUTS,
    labels_snapshot,
    normalize_shortcut,
    normal_summary,
)
from .traces import TraceSource
from .traces.payload import serialize_trace_payload

ReviewSource = Literal["label_set", "filter_file"]
SessionMode = Literal["label", "review"]
logger = logging.getLogger("attpc_estimator.estimator")


@dataclass(slots=True)
class SessionState:
    mode: SessionMode
    run: int | None
    source: ReviewSource | None = None
    family: str | None = None
    label: str | None = None
    filter_file: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "run": self.run,
            "source": self.source,
            "family": self.family,
            "label": self.label,
            "filterFile": self.filter_file,
        }


class EstimatorService:
    def __init__(
        self,
        trace_path: Path,
        workspace: Path,
        baseline_window_scale: float = 10.0,
        bitflip_baseline_threshold: float = BITFLIP_BASELINE_DEFAULT,
        saturation_threshold: float = 2000.0,
        saturation_drop_threshold: float = 10.0,
        saturation_window_radius: int = 16,
        default_run: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.trace_path = trace_path
        self.workspace = workspace
        self.baseline_window_scale = baseline_window_scale
        self.bitflip_baseline_threshold = bitflip_baseline_threshold
        self.verbose = verbose
        self.run_files = collect_run_files(trace_path)
        self.repository = LabelRepository(labels_db_path(workspace))
        self.repository.initialize()
        self.histograms = HistogramService(
            trace_path=trace_path,
            workspace=workspace,
            baseline_window_scale=baseline_window_scale,
            bitflip_baseline_threshold=bitflip_baseline_threshold,
            saturation_threshold=saturation_threshold,
            saturation_drop_threshold=saturation_drop_threshold,
            saturation_window_radius=saturation_window_radius,
        )
        resolved_default_run = self._resolve_initial_run(default_run)
        self.session = SessionState(mode="label", run=resolved_default_run)
        self._sources: dict[tuple[Any, ...], TraceSource] = {}
        self._active_source_key: tuple[Any, ...] | None = None

    def close(self) -> None:
        for source in self._sources.values():
            source.close()
        self._sources.clear()
        self.repository.connection.close()

    def bootstrap_state(self) -> dict[str, Any]:
        histogram_bootstrap = self.histograms.bootstrap_state()
        return {
            "appType": "merged",
            "workspace": str(self.workspace),
            "tracePath": str(self.trace_path),
            "databaseFile": str(labels_db_path(self.workspace)),
            "runs": histogram_bootstrap["runs"],
            "filterFiles": histogram_bootstrap["filterFiles"],
            "histogramAvailability": histogram_bootstrap["histogramAvailability"],
            "normalSummary": normal_summary(self.repository),
            "strangeSummary": self.repository.get_strange_counts(),
            "strangeLabels": self.repository.list_strange_labels(),
            "session": self.session.as_payload(),
        }

    def set_session(
        self,
        *,
        mode: str,
        run: int | None = None,
        source: str | None = None,
        family: str | None = None,
        label: str | None = None,
        filter_file: str | None = None,
    ) -> dict[str, Any]:
        if mode == "label":
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
            self.session = SessionState(mode="label", run=resolved_run)
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

        if mode != "review":
            raise ValueError("session mode must be 'label' or 'review'")
        if source not in {"label_set", "filter_file"}:
            raise ValueError(
                "review session source must be 'label_set' or 'filter_file'"
            )

        if source == "label_set":
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
            self.session = SessionState(
                mode="review",
                run=resolved_run,
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

        if filter_file is None:
            raise ValueError("filterFile is required for filter-file review")
        source_key = ("review", "filter_file", filter_file)
        filter_source = self._get_or_create_source(source_key)
        trace_count = filter_source.trace_count()
        self._active_source_key = source_key
        self.session = SessionState(
            mode="review",
            run=None,
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

    def next_trace(self) -> dict[str, Any]:
        return self._serialize_source_trace(self._current_source().next_trace())

    def previous_trace(self) -> dict[str, Any]:
        return self._serialize_source_trace(self._current_source().previous_trace())

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
        variant: str | None = None,
        filter_file: str | None = None,
        veto: bool = False,
    ) -> dict[str, Any]:
        return self.histograms.get_histogram(
            metric=metric,
            mode=mode,
            run=run,
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
        variant: str | None = None,
        filter_file: str | None = None,
        veto: bool = False,
    ) -> dict[str, str]:
        return {
            "jobId": self.histograms.create_histogram_job(
                metric=metric,
                mode=mode,
                run=run,
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

    def _resolve_initial_run(self, run: int | None) -> int | None:
        if run is None:
            return sorted(self.run_files)[0] if self.run_files else None
        if run not in self.run_files:
            raise ValueError(f"default run {run} is not available")
        return run

    def _current_source(self) -> TraceSource:
        if self._active_source_key is None:
            raise LookupError("no active trace source is available")
        return self._get_or_create_source(self._active_source_key)

    def _serialize_source_trace(self, record) -> dict[str, Any]:
        label = self.repository.get_label(record.run, record.event_id, record.trace_id)
        return serialize_trace_payload(
            record,
            bitflip_baseline_threshold=self.bitflip_baseline_threshold,
            label=label,
            review_progress=self._current_source().get_progress(),
            include_run=True,
        )

    def _get_or_create_source(self, key: tuple[Any, ...]) -> TraceSource:
        source = self._sources.get(key)
        labels = self._labels_snapshot()
        if source is None:
            source = self._build_source(key, labels)
            self._sources[key] = source
            return source
        source.replace_labels(labels)
        return source

    def _build_source(
        self,
        key: tuple[Any, ...],
        labels: dict[TraceRef, StoredLabel],
    ) -> TraceSource:
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
            filter_path = self.workspace / filter_file
            available = {path.name for path in self.workspace.glob("filter_*.npy")}
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
        raise ValueError(f"unsupported source key: {key!r}")

    def _labels_snapshot(self) -> dict[TraceRef, StoredLabel]:
        return labels_snapshot(self.repository)

    def _debug(self, message: str, *args: object) -> None:
        if self.verbose:
            logger.debug(message, *args)

    @staticmethod
    def _is_labeled_review_source(key: tuple[Any, ...], *, run: int) -> bool:
        return (
            len(key) >= 3
            and key[0] == "review"
            and key[1] == "label_set"
            and int(key[2]) == run
        )
