from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import NORMAL_BUCKETS, StoredLabel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TraceLabelRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS strange_labels (
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL UNIQUE,
              shortcut_key TEXT NOT NULL UNIQUE,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trace_labels (
              id INTEGER PRIMARY KEY,
              run INTEGER NOT NULL,
              event_id INTEGER NOT NULL,
              trace_id INTEGER NOT NULL,
              detector TEXT NOT NULL,
              cobo INTEGER NOT NULL,
              asad INTEGER NOT NULL,
              aget INTEGER NOT NULL,
              channel INTEGER NOT NULL,
              pad INTEGER NOT NULL,
              family TEXT NOT NULL CHECK(family IN ('normal', 'strange')),
              label TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(run, event_id, trace_id)
            );
            """
        )
        self.connection.commit()

    def list_labeled_trace_keys(self, run) -> dict[tuple[int, int], tuple[str, str]]:
        rows = self.connection.execute(
            """
            SELECT event_id, trace_id, family, label
            FROM trace_labels
            WHERE run = ?
            """,
            (run,),
        ).fetchall()
        return {
            (int(row["event_id"]), int(row["trace_id"])): (row["family"], row["label"])
            for row in rows
        }

    def list_labeled_traces(self, run: int | None = None) -> list[tuple[int, int, int, str, str]]:
        query = """
            SELECT run, event_id, trace_id, family, label
            FROM trace_labels
        """
        params: tuple[int, ...] = ()
        if run is not None:
            query += " WHERE run = ?"
            params = (run,)
        query += " ORDER BY run, event_id, trace_id"
        rows = self.connection.execute(query, params).fetchall()
        return [
            (int(row["run"]), int(row["event_id"]), int(row["trace_id"]), row["family"], row["label"])
            for row in rows
        ]

    def create_strange_label(self, name: str, shortcut_key: str) -> dict[str, Any]:
        now = utc_now()
        cursor = self.connection.execute(
            "INSERT INTO strange_labels(name, shortcut_key, created_at) VALUES(?, ?, ?)",
            (name, shortcut_key, now),
        )
        self.connection.commit()
        return {"id": cursor.lastrowid, "name": name, "shortcutKey": shortcut_key}

    def list_strange_labels(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT name, shortcut_key FROM strange_labels ORDER BY name COLLATE NOCASE"
        ).fetchall()
        return [{"name": row["name"], "shortcutKey": row["shortcut_key"]} for row in rows]

    def get_label(self, run: str, event_id: int, trace_id: int) -> StoredLabel | None:
        row = self.connection.execute(
            """
            SELECT family, label
            FROM trace_labels
            WHERE run = ? AND event_id = ? AND trace_id = ?
            """,
            (run, event_id, trace_id),
        ).fetchone()
        if row is None:
            return None
        return StoredLabel(family=row["family"], label=row["label"])

    def has_label(self, run: str, event_id: int, trace_id: int) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            FROM trace_labels
            WHERE run = ? AND event_id = ? AND trace_id = ?
            """,
            (run, event_id, trace_id),
        ).fetchone()
        return row is not None

    def save_label(
            self,
            run: str,
            event_id: int,
            trace_id: int,
            detector: str,
            cobo: int,
            asad: int,
            aget: int,
            channel: int,
            pad: int,
            family: str,
            label: str
        ) -> None:
        row = self.connection.execute(
            """
            SELECT created_at
            FROM trace_labels
            WHERE run = ? AND event_id = ? AND trace_id = ?
            """,
            (run, event_id, trace_id),
        ).fetchone()
        now = utc_now()
        if row is None:
            self.connection.execute(
                """
                INSERT INTO trace_labels(
                    run, event_id, trace_id,
                    detector,
                    cobo, asad, aget, channel, pad,
                    family, label,
                    created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run, event_id, trace_id,
                    detector,
                    cobo, asad, aget, channel, pad,
                    family, label,
                    now, now
                ),
            )
        else:
            self.connection.execute(
                """
                UPDATE trace_labels
                SET detector = ?,
                    cobo = ?,
                    asad = ?,
                    aget = ?,
                    channel = ?,
                    pad = ?,
                    family = ?,
                    label = ?,
                    updated_at = ?
                WHERE run = ? AND event_id = ? AND trace_id = ?
                """,
                (
                    detector,
                    cobo, asad, aget, channel, pad,
                    family, label,
                    now,
                    run, event_id, trace_id,
                ),
            )
        self.connection.commit()

    def total_labeled(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM trace_labels").fetchone()
        return int(row["count"])

    def get_normal_counts(self) -> dict[int, int]:
        counts = {bucket: 0 for bucket in NORMAL_BUCKETS}
        rows = self.connection.execute(
            """
            SELECT label, COUNT(*) AS count
            FROM trace_labels
            WHERE family = 'normal'
            GROUP BY label
            """
        ).fetchall()
        for row in rows:
            bucket = int(row["label"])
            if bucket in counts:
                counts[bucket] = int(row["count"])
        return counts

    def get_strange_counts(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT strange_labels.id, strange_labels.name, strange_labels.shortcut_key, COUNT(trace_labels.id) AS count
            FROM strange_labels
            LEFT JOIN trace_labels
              ON trace_labels.family = 'strange' AND trace_labels.label = strange_labels.name
            GROUP BY strange_labels.id, strange_labels.name, strange_labels.shortcut_key
            ORDER BY strange_labels.name COLLATE NOCASE
            """
        ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "shortcutKey": row["shortcut_key"],
                "count": int(row["count"]),
            }
            for row in rows
        ]

    def delete_strange_label(self, strange_label_name: str) -> None:
        row = self.connection.execute(
            "SELECT 1 FROM strange_labels WHERE name = ?",
            (strange_label_name,),
        ).fetchone()
        if row is None:
            raise ValueError("strange label not found")

        usage_row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM trace_labels
            WHERE family = 'strange' AND label = ?
            """,
            (strange_label_name,),
        ).fetchone()
        usage_count = int(usage_row["count"])
        if usage_count > 0:
            noun = "trace" if usage_count == 1 else "traces"
            raise ValueError(
                f'cannot delete strange label "{strange_label_name}" because it has {usage_count} labeled {noun}'
            )

        self.connection.execute(
            "DELETE FROM strange_labels WHERE name = ?",
            (strange_label_name,),
        )
        self.connection.commit()

    def strange_label_exists(self, strange_label_id: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM strange_labels WHERE id = ?",
            (strange_label_id,),
        ).fetchone()
        return row is not None

    def get_strange_label(self, strange_label_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT id, name, shortcut_key FROM strange_labels WHERE id = ?",
            (strange_label_id,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "name": row["name"], "shortcutKey": row["shortcut_key"]}

    def has_shortcut(self, shortcut_key: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM strange_labels WHERE shortcut_key = ?",
            (shortcut_key,),
        ).fetchone()
        return row is not None

    def has_strange_label_name(self, name: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM strange_labels WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
        return row is not None
