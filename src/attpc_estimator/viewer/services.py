from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

from ..db import TraceLabelRepository
from ..models import StoredLabel, TraceRecord
from .amplitude import AMPLITUDE_BIN_COUNT, _accumulate_peak_histogram
from .cdf import _accumulate_cdf_histogram_numba
from .trace_review import FilteredTraceSource
from .utils import (
    CDF_THRESHOLDS,
    CDF_VALUE_BINS,
    PAD_TRACE_OFFSET,
    collect_run_files,
    compute_frequency_distribution,
    db_file,
    label_title_from_key,
    preprocess_traces,
    sample_cdf_points,
)

ARTIFACT_SUFFIXES = {
    ("cdf", "all"): "_cdf.npy",
    ("cdf", "labeled"): "_labeled_cdf.npy",
    ("amplitude", "all"): "_amp.npy",
    ("amplitude", "labeled"): "_labeled_amp.npy",
}


class ViewerService:
    def __init__(self, trace_path: Path, workspace: Path, baseline_window_scale: float = 10.0) -> None:
        self.trace_path = trace_path
        self.workspace = workspace
        self.run_files = collect_run_files(trace_path)
        self.source = FilteredTraceSource(trace_path=trace_path, baseline_window_scale=baseline_window_scale)
        self.repository = TraceLabelRepository(db_file(workspace))
        self.repository.initialize()

    def close(self) -> None:
        self.source.close()
        self.repository.connection.close()

    def bootstrap_state(self) -> dict[str, Any]:
        run_ids = sorted(self.run_files)
        filter_files = self._filter_files()
        return {
            "appType": "viewer",
            "workspace": str(self.workspace),
            "tracePath": str(self.trace_path),
            "runs": run_ids,
            "filterFiles": [{"name": path.name} for path in filter_files],
            "histogramAvailability": {
                str(run_id): {
                    metric: {
                        mode: (
                            bool(filter_files)
                            if mode == "filtered"
                            else self._artifact_path(metric=metric, mode=mode, run=run_id) is not None
                        )
                        for mode in ("all", "labeled", "filtered")
                    }
                    for metric in ("cdf", "amplitude")
                }
                for run_id in run_ids
            },
        }

    def get_histogram(
        self,
        metric: str,
        mode: str,
        run: int,
        filter_file: str | None = None,
    ) -> dict[str, Any]:
        if metric not in {"cdf", "amplitude"}:
            raise ValueError("metric must be 'cdf' or 'amplitude'")
        if mode not in {"all", "labeled", "filtered"}:
            raise ValueError("mode must be 'all', 'labeled', or 'filtered'")
        if run not in self.run_files:
            raise ValueError(f"run {run} is not available")
        if mode == "filtered":
            return self._build_filtered_histogram(
                metric=metric,
                run=run,
                filter_file=filter_file,
            )

        artifact_path = self._artifact_path(metric=metric, mode=mode, run=run)
        if artifact_path is None:
            raise LookupError(f"no {metric} histogram artifact found for run {run} in {mode} mode")

        payload = np.load(artifact_path, allow_pickle=mode == "labeled")
        if metric == "cdf":
            return self._normalize_cdf_payload(run=run, mode=mode, payload=payload)
        return self._normalize_amplitude_payload(run=run, mode=mode, payload=payload)

    def select_filter_file(self, name: str) -> dict[str, Any]:
        filter_path = self.workspace / name
        if filter_path not in self._filter_files():
            raise ValueError(f"filter file not found: {name}")
        rows = np.load(filter_path)
        row_count = self.source.set_filter_rows(rows)
        trace_payload = self._serialize_trace_payload(self.source.next_trace()) if row_count > 0 else None
        return {
            "filterFile": name,
            "trace": trace_payload,
            "traceCount": row_count,
        }

    def next_trace(self) -> dict[str, Any]:
        return self._serialize_trace_payload(self.source.next_trace())

    def previous_trace(self) -> dict[str, Any]:
        return self._serialize_trace_payload(self.source.previous_trace())

    def _filter_files(self) -> list[Path]:
        return sorted(self.workspace.glob("filter_*.npy"))

    def _filter_path(self, name: str | None) -> Path:
        if not name:
            raise ValueError("filterFile is required when mode is 'filtered'")
        filter_path = self.workspace / name
        if filter_path not in self._filter_files():
            raise ValueError(f"filter file not found: {name}")
        return filter_path

    def _artifact_path(self, metric: str, mode: str, run: int) -> Path | None:
        suffix = ARTIFACT_SUFFIXES[(metric, mode)]
        pattern = re.compile(rf"^run_(\d+){re.escape(suffix)}$")
        for candidate in sorted(self.workspace.glob(f"run_*{suffix}")):
            match = pattern.match(candidate.name)
            if match is not None and int(match.group(1)) == run:
                return candidate
        return None

    def _normalize_cdf_payload(
        self,
        run: int,
        mode: str,
        payload: np.ndarray,
        *,
        title: str = "All traces",
        filter_file: str | None = None,
        trace_count: int | None = None,
    ) -> dict[str, Any]:
        if mode in {"all", "filtered"}:
            histogram = np.asarray(payload, dtype=np.int64)
            resolved_trace_count = (
                int(trace_count)
                if trace_count is not None
                else int(histogram.sum() // len(CDF_THRESHOLDS)) if histogram.size else 0
            )
            series = [
                {
                    "labelKey": "all" if mode == "all" else "filtered",
                    "title": title,
                    "traceCount": resolved_trace_count,
                    "histogram": histogram.tolist(),
                }
            ]
        else:
            loaded = payload.item()
            label_keys = loaded["label_keys"].tolist()
            label_titles = loaded["label_titles"].tolist()
            trace_counts = loaded["trace_counts"].astype(np.int64)
            histograms = loaded["histograms"].astype(np.int64)
            series = []
            for index, label_key in enumerate(label_keys):
                if int(trace_counts[index]) == 0:
                    continue
                series.append(
                    {
                        "labelKey": str(label_key),
                        "title": str(label_titles[index]),
                        "traceCount": int(trace_counts[index]),
                        "histogram": histograms[index].tolist(),
                    }
                )
        return {
            "metric": "cdf",
            "mode": mode,
            "run": run,
            "filterFile": filter_file,
            "thresholds": CDF_THRESHOLDS.tolist(),
            "valueBinCount": CDF_VALUE_BINS,
            "series": series,
        }

    def _normalize_amplitude_payload(
        self,
        run: int,
        mode: str,
        payload: np.ndarray,
        *,
        title: str = "All traces",
        filter_file: str | None = None,
        trace_count: int | None = None,
    ) -> dict[str, Any]:
        if mode in {"all", "filtered"}:
            histogram = np.asarray(payload, dtype=np.int64)
            series = [
                {
                    "labelKey": "all" if mode == "all" else "filtered",
                    "title": title,
                    "traceCount": trace_count,
                    "histogram": histogram.tolist(),
                }
            ]
        else:
            loaded = payload.item()
            label_keys = [str(value) for value in loaded["label_keys"].tolist()]
            trace_counts = loaded["trace_counts"].astype(np.int64)
            grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
            for index, label_key in enumerate(label_keys):
                if int(trace_counts[index]) == 0:
                    continue
                grouped_key = _amplitude_group_key(label_key)
                histogram = np.asarray(loaded[label_key], dtype=np.int64)
                if grouped_key not in grouped:
                    grouped[grouped_key] = {
                        "labelKey": grouped_key,
                        "title": label_title_from_key(grouped_key),
                        "traceCount": 0,
                        "histogram": np.zeros_like(histogram),
                    }
                grouped[grouped_key]["traceCount"] += int(trace_counts[index])
                grouped[grouped_key]["histogram"] += histogram
            series = [
                {
                    **entry,
                    "histogram": entry["histogram"].tolist(),
                }
                for entry in grouped.values()
            ]
        return {
            "metric": "amplitude",
            "mode": mode,
            "run": run,
            "filterFile": filter_file,
            "binCount": AMPLITUDE_BIN_COUNT,
            "series": series,
        }

    def _build_filtered_histogram(
        self,
        *,
        metric: str,
        run: int,
        filter_file: str | None,
    ) -> dict[str, Any]:
        filter_path = self._filter_path(filter_file)
        rows = np.asarray(np.load(filter_path), dtype=np.int64)
        if rows.size == 0:
            return self._empty_filtered_payload(metric=metric, run=run, filter_file=filter_path.name)
        if rows.ndim != 2 or rows.shape[1] != 3:
            raise ValueError(f"filter file must contain an Nx3 integer array: {filter_path.name}")
        run_rows = rows[rows[:, 0] == run]
        if run_rows.size == 0:
            return self._empty_filtered_payload(metric=metric, run=run, filter_file=filter_path.name)

        title = f"Filtered traces · {filter_path.stem}"
        if metric == "cdf":
            histogram = self._build_filtered_cdf_histogram(run=run, rows=run_rows)
            return self._normalize_cdf_payload(
                run=run,
                mode="filtered",
                payload=histogram,
                title=title,
                filter_file=filter_path.name,
                trace_count=int(run_rows.shape[0]),
            )
        histogram = self._build_filtered_amplitude_histogram(run=run, rows=run_rows)
        return self._normalize_amplitude_payload(
            run=run,
            mode="filtered",
            payload=histogram,
            title=title,
            filter_file=filter_path.name,
            trace_count=int(run_rows.shape[0]),
        )

    def _empty_filtered_payload(
        self,
        *,
        metric: str,
        run: int,
        filter_file: str,
    ) -> dict[str, Any]:
        if metric == "cdf":
            return {
                "metric": "cdf",
                "mode": "filtered",
                "run": run,
                "filterFile": filter_file,
                "thresholds": CDF_THRESHOLDS.tolist(),
                "valueBinCount": CDF_VALUE_BINS,
                "series": [],
            }
        return {
            "metric": "amplitude",
            "mode": "filtered",
            "run": run,
            "filterFile": filter_file,
            "binCount": AMPLITUDE_BIN_COUNT,
            "series": [],
        }

    def _build_filtered_cdf_histogram(self, *, run: int, rows: np.ndarray) -> np.ndarray:
        grouped = _group_filter_rows_by_event(rows)
        histogram = np.zeros((len(CDF_THRESHOLDS), CDF_VALUE_BINS), dtype=np.int64)
        with h5py.File(self.run_files[run], "r") as handle:
            events = handle["events"]
            for event_id in sorted(grouped):
                trace_ids = np.asarray(grouped[event_id], dtype=np.int64)
                pads = events[f"event_{event_id}"]["get"]["pads"]
                event_traces = np.asarray(pads[:, PAD_TRACE_OFFSET:], dtype=np.float32)
                traces = np.asarray(event_traces[trace_ids], dtype=np.float32)
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=self.source.baseline_window_scale
                )
                spectrum = compute_frequency_distribution(cleaned)
                samples = sample_cdf_points(spectrum, thresholds=CDF_THRESHOLDS)
                _accumulate_cdf_histogram_numba(samples, histogram)
        return histogram

    def _build_filtered_amplitude_histogram(self, *, run: int, rows: np.ndarray) -> np.ndarray:
        grouped = _group_filter_rows_by_event(rows)
        histogram = np.zeros(AMPLITUDE_BIN_COUNT, dtype=np.int64)
        with h5py.File(self.run_files[run], "r") as handle:
            events = handle["events"]
            for event_id in sorted(grouped):
                trace_ids = np.asarray(grouped[event_id], dtype=np.int64)
                pads = events[f"event_{event_id}"]["get"]["pads"]
                event_traces = np.asarray(pads[:, PAD_TRACE_OFFSET:], dtype=np.float32)
                traces = np.asarray(event_traces[trace_ids], dtype=np.float32)
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=self.source.baseline_window_scale
                )
                for row in cleaned:
                    _accumulate_peak_histogram(
                        row=row,
                        histogram=histogram,
                        peak_separation=50.0,
                        peak_prominence=20.0,
                        peak_width=50.0,
                    )
        return histogram

    def _serialize_trace_payload(self, record: TraceRecord) -> dict[str, Any]:
        run = int(record.run)
        label = self.repository.get_label(run, record.event_id, record.trace_id)
        return {
            "run": run,
            "eventId": record.event_id,
            "traceId": record.trace_id,
            "raw": record.raw.tolist(),
            "trace": record.trace.tolist(),
            "transformed": record.transformed.tolist(),
            "currentLabel": self._serialize_label(label),
            "reviewProgress": self.source.get_review_progress(),
        }

    @staticmethod
    def _serialize_label(label: StoredLabel | None) -> dict[str, Any] | None:
        if label is None:
            return None
        return {"family": label.family, "label": label.label}


def _group_filter_rows_by_event(rows: np.ndarray) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for _, event_id, trace_id in rows.tolist():
        grouped.setdefault(int(event_id), []).append(int(trace_id))
    return grouped


def _amplitude_group_key(label_key: str) -> str:
    family, _, label = label_key.partition(":")
    if family == "normal" and label in {"4", "5", "6", "7", "8", "9"}:
        return "normal:4+"
    return label_key
