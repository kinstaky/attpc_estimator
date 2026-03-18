from __future__ import annotations

from pathlib import Path
from typing import Any

from .db import TraceLabelRepository
from .input_reader import TraceSource
from .models import NORMAL_BUCKETS, RESERVED_SHORTCUTS, StoredLabel, TraceRecord, bucket_title, normalize_shortcut


class TraceLabelService:
    def __init__(self, input_path: Path, db_dir: Path) -> None:
        self.input_path = input_path
        self.db_path = db_dir / "trace_label.sqlite3"
        self.source = TraceSource(input_path)
        self.repository = TraceLabelRepository(self.db_path)
        self.repository.initialize()
        run = self.source.get_run()
        self.source.set_labeled(self.repository.list_labeled_trace_keys(run))

    def bootstrap_state(self) -> dict[str, Any]:
        return {
            "inputFile": str(self.input_path),
            "databaseFile": str(self.db_path),
            "normalSummary": self._normal_summary(),
            "strangeSummary": self.repository.get_strange_counts(),
        }

    def next_trace(self) -> dict[str, Any]:
        return self._serialize_trace_payload(self.source.next_trace())

    def previous_trace(self) -> dict[str, Any]:
        return self._serialize_trace_payload(self.source.previous_trace())

    def set_trace_mode(
        self,
        mode: str,
        family: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        if mode == "label":
            self.source.set_trace_mode("label")
            return {"mode": "label", "reviewFilter": None}
        if mode != "review":
            raise ValueError("trace mode must be 'label' or 'review'")
        if family not in {"normal", "strange"}:
            raise ValueError("review family must be 'normal' or 'strange'")
        if family == "normal" and label is not None:
            if label not in {str(bucket) for bucket in NORMAL_BUCKETS}:
                raise ValueError("normal review label must be one of 0-9")
        if family == "strange" and label is not None:
            if not self.repository.has_strange_label_name(label):
                raise ValueError("selected strange label does not exist")

        match_count = self.source.set_trace_mode("review", family=family, label=label)
        if match_count == 0:
            raise LookupError("no traces match the selected review filter")
        return {
            "mode": "review",
            "reviewFilter": {
                "family": family,
                "label": label,
            },
        }

    def save_label(
        self,
        event_id: int,
        trace_id: int,
        family: str,
        label: str
    ) -> dict[str, Any]:
        if family not in {"normal", "strange"}:
            raise ValueError("label family must be 'normal' or 'strange'")

        record = self.source.get_trace(event_id, trace_id)
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
        self.source.label_trace(event_id, trace_id, family, label)
        return {
            "labeledCount": self.repository.total_labeled(),
            "normalSummary": self._normal_summary(),
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
        created = self.repository.create_strange_label(clean_name, normalized_shortcut)
        return created

    def delete_strange_label(self, strange_label_name: str) -> dict[str, Any]:
        self.repository.delete_strange_label(strange_label_name)
        return self.repository.list_strange_labels()

    def _normal_summary(self) -> list[dict[str, Any]]:
        counts = self.repository.get_normal_counts()
        return [
            {
                "bucket": bucket,
                "title": bucket_title(bucket),
                "count": counts[bucket],
            }
            for bucket in NORMAL_BUCKETS
        ]

    def _serialize_trace_payload(self, record: TraceRecord) -> dict[str, Any]:
        label = self.repository.get_label(record.run, record.event_id, record.trace_id)
        return {
            "eventId": record.event_id,
            "traceId": record.trace_id,
            "trace": record.trace.tolist(),
            "currentLabel": self._serialize_label(label),
            "reviewProgress": self.source.get_review_progress(),
        }

    @staticmethod
    def _serialize_label(label: StoredLabel | None) -> dict[str, Any] | None:
        if label is None:
            return None
        return {"family": label.family, "label": label.label}
