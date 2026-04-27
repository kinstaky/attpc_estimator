from __future__ import annotations

from ..model.label import NORMAL_BUCKETS, StoredLabel
from ..model.trace import TraceRef
from ..storage.labels_db import LabelRepository, POINTCLOUD_LABEL_BUCKETS

RESERVED_SHORTCUTS = {
    "arrowleft",
    "arrowright",
    "arrowup",
    "arrowdown",
    "h",
    "j",
    "k",
    "l",
    "q",
    "escape",
    "space",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
}


def bucket_title(bucket: int) -> str:
    if bucket == 0:
        return "0 peak"
    if bucket == 1:
        return "1 peak"
    if bucket == 9:
        return "9+ peaks"
    return f"{bucket} peaks"


def normalize_shortcut(value: str) -> str:
    lowered = value.strip().lower()
    if lowered == "space":
        return " "
    if lowered == "esc":
        return "escape"
    return lowered


def normal_summary(repository: LabelRepository) -> list[dict[str, int | str]]:
    counts = repository.get_normal_counts()
    return [
        {
            "bucket": bucket,
            "title": bucket_title(bucket),
            "count": counts[bucket],
        }
        for bucket in NORMAL_BUCKETS
    ]


def pointcloud_bucket_title(bucket: str) -> str:
    if bucket == "1":
        return "1 line"
    if bucket == "6+":
        return "6+ lines"
    return f"{bucket} lines"


def pointcloud_summary(repository: LabelRepository) -> list[dict[str, int | str]]:
    counts = repository.get_pointcloud_counts()
    return [
        {
            "bucket": bucket,
            "title": pointcloud_bucket_title(bucket),
            "count": counts[bucket],
        }
        for bucket in POINTCLOUD_LABEL_BUCKETS
    ]


def labels_snapshot(repository: LabelRepository) -> dict[TraceRef, StoredLabel]:
    return {
        TraceRef(run=run, event_id=event_id, trace_id=trace_id): StoredLabel(
            family=family,
            label=label,
        )
        for run, event_id, trace_id, family, label in repository.list_labeled_traces()
    }
