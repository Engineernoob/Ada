"""Session-level state tracking for Ada conversations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, MutableMapping


STATE_PATH = Path(__file__).resolve().parents[1] / "logs" / "session_state.json"


def get_context(limit: int = 5) -> List[str]:
    if not STATE_PATH.exists():
        return []
    try:
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    subset = data[-limit:]
    return [
        f"User: {turn.get('prompt', '')}\nAda: {turn.get('response', '')}".strip()
        for turn in subset
        if isinstance(turn, MutableMapping)
    ]


def save_turn(prompt: str, response: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: List[MutableMapping[str, str]] = []
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, list):
                data = [item for item in existing if isinstance(item, MutableMapping)]
        except (json.JSONDecodeError, OSError):
            data = []

    data.append({"prompt": prompt, "response": response})
    trimmed = data[-10:]

    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(trimmed, handle, indent=2)
