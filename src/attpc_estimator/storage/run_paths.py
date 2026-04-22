from __future__ import annotations

from pathlib import Path

DEFAULT_LABELS_DB_FILENAME = "labels.db"
DEFAULT_POINTCLOUD_DIRNAME = "pointcloud"
DEFAULT_HISTOGRAM_DIRNAME = "histograms"
DEFAULT_FILTER_DIRNAME = "filter"


def format_run_id(run: int | str) -> str:
    return f"{int(run):04d}"


def resolve_run_file(trace_path: Path, run: int | str) -> Path:
    normalized_run = int(run)
    if trace_path.is_file():
        resolved = trace_path.resolve()
        actual_run = extract_run_id(resolved)
        if actual_run != normalized_run:
            raise ValueError(
                f"trace file {resolved} is for run {actual_run}, not requested run {normalized_run}"
            )
        return resolved

    direct_candidate = trace_path / f"run_{format_run_id(normalized_run)}.h5"
    if direct_candidate.is_file():
        return direct_candidate.resolve()

    resolved: Path | None = None
    for candidate in sorted(trace_path.glob("run_*.h5")):
        if extract_run_id(candidate) != normalized_run:
            continue
        candidate_resolved = candidate.resolve()
        if resolved is not None:
            raise ValueError(
                f"multiple workspace files resolve to run {normalized_run}: {resolved} and {candidate_resolved}"
            )
        resolved = candidate_resolved

    if resolved is not None:
        return resolved
    raise ValueError(f"trace file not found for run {normalized_run}: {direct_candidate}")


def collect_run_files(trace_path: Path) -> dict[int, Path]:
    if trace_path.is_file():
        resolved = trace_path.resolve()
        return {extract_run_id(resolved): resolved}
    run_files: dict[int, Path] = {}
    for candidate in sorted(trace_path.glob("run_*.h5")):
        run_id = extract_run_id(candidate)
        if run_id in run_files:
            raise ValueError(
                f"multiple workspace files resolve to run {run_id}: {run_files[run_id]} and {candidate}"
            )
        run_files[run_id] = candidate.resolve()
    return run_files


def extract_run_id(trace_file_path: Path) -> int:
    stem = trace_file_path.stem
    prefix, _, run_token = stem.partition("_")
    if prefix != "run" or not run_token.isdigit():
        raise ValueError(
            f"expected input filename like run_<run>.h5, got {trace_file_path.name}"
        )
    return int(run_token)


def labels_db_path(workspace: Path) -> Path:
    return workspace / DEFAULT_LABELS_DB_FILENAME


def histogram_dir(workspace: Path) -> Path:
    return workspace / DEFAULT_HISTOGRAM_DIRNAME


def filter_dir(workspace: Path) -> Path:
    return workspace / DEFAULT_FILTER_DIRNAME


def pointcloud_dir(workspace: Path) -> Path:
    return workspace / DEFAULT_POINTCLOUD_DIRNAME


def pointcloud_run_path(workspace: Path, run: int | str) -> Path:
    return pointcloud_dir(workspace) / f"run_{format_run_id(run)}.h5"
