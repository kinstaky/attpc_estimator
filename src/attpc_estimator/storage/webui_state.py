from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_VERSION = 1


class WebUiStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if payload.get("version") != STATE_VERSION:
            return {}
        return payload

    def save(
        self,
        *,
        ui_state: dict[str, Any],
        runtime_session: dict[str, Any] | None,
    ) -> None:
        payload = {
            "version": STATE_VERSION,
            "uiState": ui_state,
            "runtimeSession": runtime_session,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
