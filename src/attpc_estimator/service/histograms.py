from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

from ..process.amplitude import AMPLITUDE_BIN_COUNT, _accumulate_peak_histogram
from ..process.baseline import (
    BASELINE_BIN_CENTERS,
    BASELINE_BIN_COUNT,
    BASELINE_BIN_LABEL,
    BASELINE_COUNT_LABEL,
    accumulate_baseline_histogram,
)
from ..process.cdf import _accumulate_cdf_histogram_numba
from ..process.line_distance import serialize_line_distance_payload
from ..process.bitflip import (
    BITFLIP_BASELINE_DEFAULT,
    BITFLIP_BIN_COUNTS,
    BITFLIP_BIN_LABELS,
    BITFLIP_COUNT_LABELS,
    BITFLIP_VARIANTS,
    accumulate_bitflip_histograms,
)
from ..process.progress import ProgressReporter, emit_progress
from ..process.saturation import (
    SATURATION_BIN_COUNTS,
    SATURATION_BIN_LABELS,
    SATURATION_VARIANTS,
    accumulate_saturation_histograms,
)
from ..storage.run_paths import collect_run_files, filter_dir, histogram_dir
from ..utils.label_keys import label_title_from_key
from ..utils.trace_data import (
    CDF_THRESHOLDS,
    CDF_VALUE_BINS,
    collect_event_counts,
    compute_frequency_distribution,
    load_pad_traces,
    preprocess_traces,
    sample_cdf_points,
)
from .histogram_jobs import HistogramJobManager

SUPPORTED_METRICS = (
    "cdf",
    "amplitude",
    "baseline",
    "bitflip",
    "saturation",
    "line_distance",
    "coplanar",
)
DEFAULT_VARIANTS = {
    "bitflip": "baseline",
    "saturation": "drop",
}
METRIC_VARIANTS = {
    "bitflip": BITFLIP_VARIANTS,
    "saturation": SATURATION_VARIANTS,
}
ARTIFACT_SUFFIXES = {
    ("cdf", "all"): ("_cdf.npy",),
    ("cdf", "labeled"): ("_labeled_cdf.npz", "_labeled_cdf.npy"),
    ("amplitude", "all"): ("_amp.npy",),
    ("amplitude", "labeled"): ("_labeled_amp.npz", "_labeled_amp.npy"),
    ("baseline", "all"): ("_baseline.npz",),
    ("baseline", "labeled"): ("_labeled_baseline.npz",),
    ("bitflip", "all"): ("_bitflip.npz",),
    ("bitflip", "labeled"): ("_labeled_bitflip.npz",),
    ("saturation", "all"): ("_saturation.npz",),
    ("saturation", "labeled"): ("_labeled_saturation.npz",),
    ("line_distance", "all"): ("_line_distance.npz",),
    ("coplanar", "all"): ("_coplanar.npz",),
}
ONE_D_METRIC_METADATA = {
    "amplitude": {
        None: {
            "bin_count": AMPLITUDE_BIN_COUNT,
            "bin_label": "Amplitude",
            "count_label": "Peak count",
            "all_key": None,
            "labeled_key": "histograms",
        }
    },
    "baseline": {
        None: {
            "bin_count": BASELINE_BIN_COUNT,
            "bin_label": BASELINE_BIN_LABEL,
            "count_label": BASELINE_COUNT_LABEL,
            "all_key": "histogram",
            "labeled_key": "histograms",
            "bin_centers_key": "bin_centers",
        }
    },
    "bitflip": {
        "baseline": {
            "bin_count": BITFLIP_BIN_COUNTS["baseline"],
            "bin_label": BITFLIP_BIN_LABELS["baseline"],
            "count_label": BITFLIP_COUNT_LABELS["baseline"],
            "all_key": "baseline_histogram",
            "labeled_key": "baseline_histograms",
            "bin_centers": BASELINE_BIN_CENTERS.tolist(),
        },
        "value": {
            "bin_count": BITFLIP_BIN_COUNTS["value"],
            "bin_label": BITFLIP_BIN_LABELS["value"],
            "count_label": BITFLIP_COUNT_LABELS["value"],
            "all_key": "value_histogram",
            "labeled_key": "value_histograms",
        },
        "length": {
            "bin_count": BITFLIP_BIN_COUNTS["length"],
            "bin_label": BITFLIP_BIN_LABELS["length"],
            "count_label": BITFLIP_COUNT_LABELS["length"],
            "all_key": "length_histogram",
            "labeled_key": "length_histograms",
        },
        "count": {
            "bin_count": BITFLIP_BIN_COUNTS["count"],
            "bin_label": BITFLIP_BIN_LABELS["count"],
            "count_label": BITFLIP_COUNT_LABELS["count"],
            "all_key": "count_histogram",
            "labeled_key": "count_histograms",
        },
    },
    "saturation": {
        "drop": {
            "bin_count": SATURATION_BIN_COUNTS["drop"],
            "bin_label": SATURATION_BIN_LABELS["drop"],
            "count_label": "Count",
            "all_key": "drop_histogram",
            "labeled_key": "drop_histograms",
        },
        "length": {
            "bin_count": SATURATION_BIN_COUNTS["length"],
            "bin_label": SATURATION_BIN_LABELS["length"],
            "count_label": "Trace count",
            "all_key": "length_histogram",
            "labeled_key": "length_histograms",
        },
    },
    "coplanar": {
        None: {
            "bin_label": "λ₃/λ₁",
            "count_label": "Event count",
            "all_key": "histogram",
            "labeled_key": "histograms",
            "bin_edges_key": "bin_edges",
        }
    },
}


class HistogramService:
    def __init__(
        self,
        trace_path: Path,
        workspace: Path,
        baseline_window_scale: float = 10.0,
        bitflip_baseline_threshold: float = BITFLIP_BASELINE_DEFAULT,
        saturation_threshold: float = 2000.0,
        saturation_drop_threshold: float = 10.0,
        saturation_window_radius: int = 16,
    ) -> None:
        self.trace_path = trace_path
        self.workspace = workspace
        self.baseline_window_scale = baseline_window_scale
        self.bitflip_baseline_threshold = bitflip_baseline_threshold
        self.saturation_threshold = saturation_threshold
        self.saturation_drop_threshold = saturation_drop_threshold
        self.saturation_window_radius = saturation_window_radius
        self.run_files = collect_run_files(trace_path)
        self.jobs = HistogramJobManager()

    def bootstrap_state(self) -> dict[str, Any]:
        run_ids = sorted(self.run_files)
        filter_files = self._filter_files()
        return {
            "runs": run_ids,
            "filterFiles": [{"name": path.name} for path in filter_files],
            "histogramAvailability": {
                str(run_id): {
                    metric: self._availability_for_metric(run_id, metric, bool(filter_files))
                    for metric in SUPPORTED_METRICS
                }
                for run_id in run_ids
            },
        }

    def get_histogram(
        self,
        *,
        metric: str,
        mode: str,
        run: int,
        variant: str | None = None,
        filter_file: str | None = None,
        veto: bool = False,
        progress: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        resolved_variant = self._validate_histogram_request(
            metric=metric,
            mode=mode,
            run=run,
            variant=variant,
            filter_file=filter_file,
        )
        resolved_veto = veto if mode == "filtered" else False
        if mode == "filtered":
            return self._build_filtered_histogram(
                metric=metric,
                variant=resolved_variant,
                run=run,
                filter_file=filter_file,
                veto=resolved_veto,
                progress=progress,
            )

        artifact_path = self._artifact_path(metric=metric, mode=mode, run=run)
        if artifact_path is None:
            raise LookupError(
                f"no {metric} histogram artifact found for run {run} in {mode} mode"
            )

        payload = _load_artifact_payload(artifact_path, allow_pickle=mode == "labeled")
        if metric == "cdf":
            return self._normalize_cdf_payload(
                run=run,
                mode=mode,
                payload=payload,
                veto=resolved_veto,
            )
        if metric == "amplitude":
            return self._normalize_amplitude_payload(
                run=run,
                mode=mode,
                payload=payload,
                veto=resolved_veto,
            )
        if metric == "line_distance":
            return serialize_line_distance_payload(run, _mapping_payload(payload))
        return self._normalize_generic_1d_payload(
            metric=metric,
            variant=resolved_variant,
            run=run,
            mode=mode,
            payload=payload,
            veto=resolved_veto,
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
    ) -> str:
        if mode != "filtered":
            raise ValueError("histogram jobs are only available for filtered mode")
        resolved_variant = self._validate_histogram_request(
            metric=metric,
            mode=mode,
            run=run,
            variant=variant,
            filter_file=filter_file,
        )
        return self.jobs.create_job(
            lambda progress: self.get_histogram(
                metric=metric,
                mode=mode,
                run=run,
                variant=resolved_variant,
                filter_file=filter_file,
                veto=veto,
                progress=progress,
            )
        )

    def next_job_message(
        self,
        job_id: str,
        after_index: int,
    ) -> tuple[int, dict] | None:
        return self.jobs.next_message(job_id, after_index)

    def _filter_files(self) -> list[Path]:
        return sorted(filter_dir(self.workspace).glob("filter_*.npy"))

    def _availability_for_metric(
        self,
        run: int,
        metric: str,
        has_filter_files: bool,
    ) -> dict[str, bool]:
        if metric in {"line_distance", "coplanar"}:
            return {
                "all": self._artifact_path(metric=metric, mode="all", run=run) is not None,
                "labeled": False,
                "filtered": False,
            }
        return {
            mode: (
                has_filter_files
                if mode == "filtered"
                else self._artifact_path(metric=metric, mode=mode, run=run) is not None
            )
            for mode in ("all", "labeled", "filtered")
        }

    def _validate_histogram_request(
        self,
        *,
        metric: str,
        mode: str,
        run: int,
        variant: str | None,
        filter_file: str | None,
    ) -> str | None:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                "metric must be 'cdf', 'amplitude', 'baseline', 'bitflip', 'saturation', 'line_distance', or 'coplanar'"
            )
        if mode not in {"all", "labeled", "filtered"}:
            raise ValueError("mode must be 'all', 'labeled', or 'filtered'")
        if run not in self.run_files:
            raise ValueError(f"run {run} is not available")
        if metric in {"line_distance", "coplanar"} and mode != "all":
            raise ValueError(f"metric '{metric}' only supports mode 'all'")
        if mode == "filtered":
            self._filter_path(filter_file)
        return self._resolve_variant(metric=metric, variant=variant)

    def _resolve_variant(self, *, metric: str, variant: str | None) -> str | None:
        allowed_variants = METRIC_VARIANTS.get(metric)
        if allowed_variants is None:
            if variant not in {None, ""}:
                raise ValueError(f"metric '{metric}' does not support variants")
            return None
        if variant is None or variant == "":
            return DEFAULT_VARIANTS[metric]
        if variant not in allowed_variants:
            raise ValueError(
                f"variant must be one of {', '.join(allowed_variants)} for metric '{metric}'"
            )
        return variant

    def _filter_path(self, name: str | None) -> Path:
        if not name:
            raise ValueError("filterFile is required when mode is 'filtered'")
        filter_path = filter_dir(self.workspace) / name
        if filter_path not in self._filter_files():
            raise ValueError(f"filter file not found: {name}")
        return filter_path

    def _artifact_path(self, metric: str, mode: str, run: int) -> Path | None:
        suffixes = ARTIFACT_SUFFIXES.get((metric, mode))
        if suffixes is None:
            return None
        for suffix in suffixes:
            pattern = re.compile(rf"^run_(\d+){re.escape(suffix)}$")
            for candidate in sorted(histogram_dir(self.workspace).glob(f"run_*{suffix}")):
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
        veto: bool = False,
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
            loaded = _mapping_payload(payload)
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
            "veto": veto,
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
        veto: bool = False,
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
            loaded = _mapping_payload(payload)
            label_keys = [str(value) for value in loaded["label_keys"].tolist()]
            trace_counts = loaded["trace_counts"].astype(np.int64)
            histogram_matrix = loaded.get("histograms")
            grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
            for index, label_key in enumerate(label_keys):
                if int(trace_counts[index]) == 0:
                    continue
                grouped_key = _amplitude_group_key(label_key)
                if histogram_matrix is not None:
                    histogram = np.asarray(histogram_matrix[index], dtype=np.int64)
                else:
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
            "veto": veto,
            "binCount": AMPLITUDE_BIN_COUNT,
            "binLabel": "Amplitude",
            "countLabel": "Peak count",
            "series": series,
        }

    def _normalize_generic_1d_payload(
        self,
        *,
        metric: str,
        variant: str | None,
        run: int,
        mode: str,
        payload: Any,
        title: str = "All traces",
        filter_file: str | None = None,
        veto: bool = False,
        trace_count: int | None = None,
    ) -> dict[str, Any]:
        metadata = ONE_D_METRIC_METADATA[metric][variant]
        loaded = _mapping_payload(payload)
        if mode in {"all", "filtered"}:
            histogram = (
                np.asarray(payload, dtype=np.int64)
                if isinstance(payload, np.ndarray)
                else np.asarray(loaded[metadata["all_key"]], dtype=np.int64)
            )
            resolved_trace_count = (
                int(trace_count)
                if trace_count is not None
                else int(np.asarray(loaded.get("trace_count", 0)).item())
                if isinstance(loaded, dict)
                else None
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
            label_keys = loaded["label_keys"].tolist()
            label_titles = loaded["label_titles"].tolist()
            trace_counts = loaded["trace_counts"].astype(np.int64)
            histograms = np.asarray(loaded[metadata["labeled_key"]], dtype=np.int64)
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
        bin_count = metadata.get("bin_count", int(histogram.shape[-1]) if mode in {"all", "filtered"} else 0)
        result = {
            "metric": metric,
            "mode": mode,
            "run": run,
            "filterFile": filter_file,
            "veto": veto,
            "binCount": bin_count,
            "binLabel": metadata["bin_label"],
            "countLabel": metadata["count_label"],
            "series": series,
        }
        if "bin_edges_key" in metadata and mode in {"all", "filtered"}:
            bin_edges = np.asarray(loaded[metadata["bin_edges_key"]], dtype=np.float64)
            result["binCenters"] = _bin_centers(bin_edges).tolist()
            result["binCount"] = max(0, int(bin_edges.shape[0]) - 1)
        elif "bin_centers_key" in metadata and metadata["bin_centers_key"] in loaded:
            bin_centers = np.asarray(loaded[metadata["bin_centers_key"]], dtype=np.float64)
            result["binCenters"] = bin_centers.tolist()
            result["binCount"] = int(bin_centers.shape[0])
        elif "bin_centers" in metadata:
            result["binCenters"] = metadata["bin_centers"]
        if variant is not None:
            result["variant"] = variant
        return result

    def _build_filtered_histogram(
        self,
        *,
        metric: str,
        variant: str | None,
        run: int,
        filter_file: str | None,
        veto: bool,
        progress: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        filter_path = self._filter_path(filter_file)
        rows = np.asarray(np.load(filter_path), dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != 3:
            raise ValueError(
                f"filter file must contain an Nx3 integer array: {filter_path.name}"
            )
        grouped_trace_ids, trace_count = self._resolve_filtered_trace_ids(
            run=run,
            rows=rows,
            veto=veto,
        )
        if trace_count == 0:
            return self._empty_filtered_payload(
                metric=metric,
                variant=variant,
                run=run,
                filter_file=filter_path.name,
                veto=veto,
            )

        title_prefix = "Vetoed traces" if veto else "Filtered traces"
        title = f"{title_prefix} · {filter_path.stem}"
        if metric == "cdf":
            histogram = self._build_filtered_cdf_histogram(
                run=run,
                grouped_trace_ids=grouped_trace_ids,
                total_traces=trace_count,
                progress=progress,
            )
            return self._normalize_cdf_payload(
                run=run,
                mode="filtered",
                payload=histogram,
                title=title,
                filter_file=filter_path.name,
                veto=veto,
                trace_count=trace_count,
            )
        if metric == "amplitude":
            histogram = self._build_filtered_amplitude_histogram(
                run=run,
                grouped_trace_ids=grouped_trace_ids,
                total_traces=trace_count,
                progress=progress,
            )
            return self._normalize_amplitude_payload(
                run=run,
                mode="filtered",
                payload=histogram,
                title=title,
                filter_file=filter_path.name,
                veto=veto,
                trace_count=trace_count,
            )
        if metric == "baseline":
            histogram = self._build_filtered_baseline_histogram(
                run=run,
                grouped_trace_ids=grouped_trace_ids,
                total_traces=trace_count,
                progress=progress,
            )
            return self._normalize_generic_1d_payload(
                metric="baseline",
                variant=None,
                run=run,
                mode="filtered",
                payload={
                    "trace_count": np.int64(trace_count),
                    "histogram": histogram,
                },
                title=title,
                filter_file=filter_path.name,
                veto=veto,
                trace_count=trace_count,
            )
        if metric == "bitflip":
            histograms = self._build_filtered_bitflip_histograms(
                run=run,
                grouped_trace_ids=grouped_trace_ids,
                total_traces=trace_count,
                progress=progress,
            )
            return self._normalize_generic_1d_payload(
                metric="bitflip",
                variant=variant,
                run=run,
                mode="filtered",
                payload={
                    "trace_count": np.int64(trace_count),
                    "baseline_histogram": histograms["baseline_histogram"],
                    "value_histogram": histograms["value_histogram"],
                    "length_histogram": histograms["length_histogram"],
                    "count_histogram": histograms["count_histogram"],
                },
                title=title,
                filter_file=filter_path.name,
                veto=veto,
                trace_count=trace_count,
            )
        histograms = self._build_filtered_saturation_histograms(
            run=run,
            grouped_trace_ids=grouped_trace_ids,
            total_traces=trace_count,
            progress=progress,
        )
        return self._normalize_generic_1d_payload(
            metric="saturation",
            variant=variant,
            run=run,
            mode="filtered",
            payload={
                "trace_count": np.int64(trace_count),
                "drop_histogram": histograms["drop_histogram"],
                "length_histogram": histograms["length_histogram"],
            },
            title=title,
            filter_file=filter_path.name,
            veto=veto,
            trace_count=trace_count,
        )

    def _empty_filtered_payload(
        self,
        *,
        metric: str,
        variant: str | None,
        run: int,
        filter_file: str,
        veto: bool,
    ) -> dict[str, Any]:
        if metric == "cdf":
            return {
                "metric": "cdf",
                "mode": "filtered",
                "run": run,
                "filterFile": filter_file,
                "veto": veto,
                "thresholds": CDF_THRESHOLDS.tolist(),
                "valueBinCount": CDF_VALUE_BINS,
                "series": [],
            }
        if metric == "amplitude":
            return {
                "metric": "amplitude",
                "mode": "filtered",
                "run": run,
                "filterFile": filter_file,
                "veto": veto,
                "binCount": AMPLITUDE_BIN_COUNT,
                "binLabel": "Amplitude",
                "countLabel": "Peak count",
                "series": [],
            }
        if metric == "baseline":
            return {
                "metric": "baseline",
                "mode": "filtered",
                "run": run,
                "filterFile": filter_file,
                "veto": veto,
                "binCount": BASELINE_BIN_COUNT,
                "binLabel": BASELINE_BIN_LABEL,
                "countLabel": BASELINE_COUNT_LABEL,
                "binCenters": BASELINE_BIN_CENTERS.tolist(),
                "series": [],
            }
        metadata = ONE_D_METRIC_METADATA[metric][variant]
        payload = {
            "metric": metric,
            "mode": "filtered",
            "run": run,
            "filterFile": filter_file,
            "veto": veto,
            "binCount": metadata["bin_count"],
            "binLabel": metadata["bin_label"],
            "countLabel": metadata["count_label"],
            "series": [],
        }
        if "bin_centers" in metadata:
            payload["binCenters"] = metadata["bin_centers"]
        if variant is not None:
            payload["variant"] = variant
        return payload

    def _build_filtered_cdf_histogram(
        self,
        *,
        run: int,
        grouped_trace_ids: dict[int, np.ndarray],
        total_traces: int,
        progress: ProgressReporter | None = None,
    ) -> np.ndarray:
        histogram = np.zeros((len(CDF_THRESHOLDS), CDF_VALUE_BINS), dtype=np.int64)
        processed_traces = 0
        emit_progress(progress, current=0, total=total_traces, unit="trace")
        with h5py.File(self.run_files[run], "r") as handle:
            for event_id in sorted(grouped_trace_ids):
                trace_ids = grouped_trace_ids[event_id]
                batch_size = int(trace_ids.shape[0])
                try:
                    traces = load_pad_traces(
                        handle, run=run, event_id=event_id, trace_ids=trace_ids
                    )
                except LookupError:
                    processed_traces += batch_size
                    emit_progress(
                        progress,
                        current=processed_traces,
                        total=total_traces,
                        unit="trace",
                        message=f"event={event_id}",
                    )
                    continue
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=self.baseline_window_scale
                )
                spectrum = compute_frequency_distribution(cleaned)
                samples = sample_cdf_points(spectrum, thresholds=CDF_THRESHOLDS)
                _accumulate_cdf_histogram_numba(samples, histogram)
                processed_traces += batch_size
                emit_progress(
                    progress,
                    current=processed_traces,
                    total=total_traces,
                    unit="trace",
                    message=f"event={event_id}",
                )
        return histogram

    def _build_filtered_amplitude_histogram(
        self,
        *,
        run: int,
        grouped_trace_ids: dict[int, np.ndarray],
        total_traces: int,
        progress: ProgressReporter | None = None,
    ) -> np.ndarray:
        histogram = np.zeros(AMPLITUDE_BIN_COUNT, dtype=np.int64)
        processed_traces = 0
        emit_progress(progress, current=0, total=total_traces, unit="trace")
        with h5py.File(self.run_files[run], "r") as handle:
            for event_id in sorted(grouped_trace_ids):
                trace_ids = grouped_trace_ids[event_id]
                batch_size = int(trace_ids.shape[0])
                try:
                    traces = load_pad_traces(
                        handle, run=run, event_id=event_id, trace_ids=trace_ids
                    )
                except LookupError:
                    processed_traces += batch_size
                    emit_progress(
                        progress,
                        current=processed_traces,
                        total=total_traces,
                        unit="trace",
                        message=f"event={event_id}",
                    )
                    continue
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=self.baseline_window_scale
                )
                for row in cleaned:
                    _accumulate_peak_histogram(
                        row=row,
                        histogram=histogram,
                        peak_separation=50.0,
                        peak_prominence=20.0,
                        peak_width=50.0,
                    )
                processed_traces += batch_size
                emit_progress(
                    progress,
                    current=processed_traces,
                    total=total_traces,
                    unit="trace",
                    message=f"event={event_id}",
                )
        return histogram

    def _build_filtered_baseline_histogram(
        self,
        *,
        run: int,
        grouped_trace_ids: dict[int, np.ndarray],
        total_traces: int,
        progress: ProgressReporter | None = None,
    ) -> np.ndarray:
        histogram = np.zeros(BASELINE_BIN_COUNT, dtype=np.int64)
        processed_traces = 0
        emit_progress(progress, current=0, total=total_traces, unit="trace")
        with h5py.File(self.run_files[run], "r") as handle:
            for event_id in sorted(grouped_trace_ids):
                trace_ids = grouped_trace_ids[event_id]
                batch_size = int(trace_ids.shape[0])
                try:
                    traces = load_pad_traces(
                        handle, run=run, event_id=event_id, trace_ids=trace_ids
                    )
                except LookupError:
                    processed_traces += batch_size
                    emit_progress(
                        progress,
                        current=processed_traces,
                        total=total_traces,
                        unit="trace",
                        message=f"event={event_id}",
                    )
                    continue
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=self.baseline_window_scale
                )
                accumulate_baseline_histogram(cleaned, histogram=histogram)
                processed_traces += batch_size
                emit_progress(
                    progress,
                    current=processed_traces,
                    total=total_traces,
                    unit="trace",
                    message=f"event={event_id}",
                )
        return histogram

    def _build_filtered_bitflip_histograms(
        self,
        *,
        run: int,
        grouped_trace_ids: dict[int, np.ndarray],
        total_traces: int,
        progress: ProgressReporter | None = None,
    ) -> dict[str, np.ndarray]:
        baseline_histogram = np.zeros(BITFLIP_BIN_COUNTS["baseline"], dtype=np.int64)
        value_histogram = np.zeros(BITFLIP_BIN_COUNTS["value"], dtype=np.int64)
        length_histogram = np.zeros(BITFLIP_BIN_COUNTS["length"], dtype=np.int64)
        count_histogram = np.zeros(BITFLIP_BIN_COUNTS["count"], dtype=np.int64)
        processed_traces = 0
        emit_progress(progress, current=0, total=total_traces, unit="trace")
        with h5py.File(self.run_files[run], "r") as handle:
            for event_id in sorted(grouped_trace_ids):
                trace_ids = grouped_trace_ids[event_id]
                batch_size = int(trace_ids.shape[0])
                try:
                    traces = load_pad_traces(
                        handle, run=run, event_id=event_id, trace_ids=trace_ids
                    )
                except LookupError:
                    processed_traces += batch_size
                    emit_progress(
                        progress,
                        current=processed_traces,
                        total=total_traces,
                        unit="trace",
                        message=f"event={event_id}",
                    )
                    continue
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=self.baseline_window_scale
                )
                accumulate_bitflip_histograms(
                    cleaned,
                    baseline_histogram=baseline_histogram,
                    value_histogram=value_histogram,
                    length_histogram=length_histogram,
                    count_histogram=count_histogram,
                    baseline_threshold=self.bitflip_baseline_threshold,
                )
                processed_traces += batch_size
                emit_progress(
                    progress,
                    current=processed_traces,
                    total=total_traces,
                    unit="trace",
                    message=f"event={event_id}",
                )
        return {
            "baseline_histogram": baseline_histogram,
            "value_histogram": value_histogram,
            "length_histogram": length_histogram,
            "count_histogram": count_histogram,
        }

    def _build_filtered_saturation_histograms(
        self,
        *,
        run: int,
        grouped_trace_ids: dict[int, np.ndarray],
        total_traces: int,
        progress: ProgressReporter | None = None,
    ) -> dict[str, np.ndarray]:
        drop_histogram = np.zeros(SATURATION_BIN_COUNTS["drop"], dtype=np.int64)
        length_histogram = np.zeros(SATURATION_BIN_COUNTS["length"], dtype=np.int64)
        processed_traces = 0
        emit_progress(progress, current=0, total=total_traces, unit="trace")
        with h5py.File(self.run_files[run], "r") as handle:
            for event_id in sorted(grouped_trace_ids):
                trace_ids = grouped_trace_ids[event_id]
                batch_size = int(trace_ids.shape[0])
                try:
                    traces = load_pad_traces(
                        handle, run=run, event_id=event_id, trace_ids=trace_ids
                    )
                except LookupError:
                    processed_traces += batch_size
                    emit_progress(
                        progress,
                        current=processed_traces,
                        total=total_traces,
                        unit="trace",
                        message=f"event={event_id}",
                    )
                    continue
                cleaned = preprocess_traces(
                    traces, baseline_window_scale=self.baseline_window_scale
                )
                accumulate_saturation_histograms(
                    cleaned,
                    drop_histogram=drop_histogram,
                    length_histogram=length_histogram,
                    threshold=self.saturation_threshold,
                    drop_threshold=self.saturation_drop_threshold,
                    window_radius=self.saturation_window_radius,
                )
                processed_traces += batch_size
                emit_progress(
                    progress,
                    current=processed_traces,
                    total=total_traces,
                    unit="trace",
                    message=f"event={event_id}",
                )
        return {
            "drop_histogram": drop_histogram,
            "length_histogram": length_histogram,
        }

    def _resolve_filtered_trace_ids(
        self,
        *,
        run: int,
        rows: np.ndarray,
        veto: bool,
    ) -> tuple[dict[int, np.ndarray], int]:
        run_rows = rows[rows[:, 0] == run]
        grouped = _group_filter_rows_by_event(run_rows)
        if not veto:
            total_traces = int(run_rows.shape[0]) if run_rows.size else 0
            return {
                event_id: np.asarray(trace_ids, dtype=np.int64)
                for event_id, trace_ids in grouped.items()
            }, total_traces

        selected_by_event = {
            event_id: np.asarray(sorted(set(trace_ids)), dtype=np.int64)
            for event_id, trace_ids in grouped.items()
        }
        veto_grouped: dict[int, np.ndarray] = {}
        total_traces = 0
        with h5py.File(self.run_files[run], "r") as handle:
            for event_id, trace_count in collect_event_counts(handle):
                selected_trace_ids = selected_by_event.get(event_id)
                if selected_trace_ids is None or selected_trace_ids.size == 0:
                    veto_trace_ids = np.arange(trace_count, dtype=np.int64)
                else:
                    valid_selected = selected_trace_ids[
                        (selected_trace_ids >= 0) & (selected_trace_ids < trace_count)
                    ]
                    keep_mask = np.ones(trace_count, dtype=bool)
                    keep_mask[valid_selected] = False
                    veto_trace_ids = np.flatnonzero(keep_mask).astype(
                        np.int64, copy=False
                    )
                if veto_trace_ids.size == 0:
                    continue
                veto_grouped[int(event_id)] = veto_trace_ids
                total_traces += int(veto_trace_ids.size)
        return veto_grouped, total_traces


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


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        return np.empty(0, dtype=np.float64)
    return (values[:-1] + values[1:]) / 2.0


def _load_artifact_payload(path: Path, *, allow_pickle: bool) -> Any:
    return np.load(path, allow_pickle=allow_pickle)


def _mapping_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, np.lib.npyio.NpzFile):
        return {key: payload[key] for key in payload.files}
    if isinstance(payload, np.ndarray):
        return payload.item()
    return payload
