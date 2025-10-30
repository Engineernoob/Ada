"""SQLite-backed storage for Ada conversations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "conversations.db"


@dataclass
class ConversationRecord:
    id: int
    timestamp: str
    user_input: str
    ada_response: str
    reward: float


class ConversationStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    ada_response TEXT NOT NULL,
                    reward REAL DEFAULT 0
                )
                """
            )
            connection.commit()

    def log_interaction(self, user_input: str, ada_response: str, reward: float) -> int:
        timestamp = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO conversations (timestamp, user_input, ada_response, reward) "
                "VALUES (?, ?, ?, ?)",
                (timestamp, user_input, ada_response, reward),
            )
            connection.commit()
            record_id = cursor.lastrowid
            return int(record_id) if record_id is not None else 0

    def fetch_recent(self, limit: int = 5) -> Iterable[ConversationRecord]:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT id, timestamp, user_input, ada_response, reward "
                "FROM conversations ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        for row in rows:
            yield ConversationRecord(*row)

    def as_context(self, limit: int = 5) -> str:
        records = list(self.fetch_recent(limit))
        if not records:
            return ""
        lines = [
            f"You: {record.user_input}\nAda: {record.ada_response}"
            for record in reversed(records)
        ]
        return "\n".join(lines)

    def update_reward(self, record_id: int, reward: float) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE conversations SET reward = ? WHERE id = ?",
                (reward, record_id),
            )
            connection.commit()
