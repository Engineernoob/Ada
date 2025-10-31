"""SQLite-backed storage for Ada conversations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Iterable, List, Optional


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


_store_lock = RLock()
_store: ConversationStore | None = None

try:
    from neural.encoder import TextEncoder
    try:
        from neural.encoder import LanguageEncoder
    except ImportError:  # sentence-transformers may be optional
        LanguageEncoder = None  # type: ignore
except ImportError:  # pragma: no cover - fallback when neural module missing
    TextEncoder = None  # type: ignore
    LanguageEncoder = None  # type: ignore

try:
    from memory.episodic_store import EpisodicStore
except Exception:  # pragma: no cover - episodic memory optional
    EpisodicStore = None  # type: ignore

_encoder_lock = RLock()
_encoder = None
_episodic_store = None


def _get_store() -> ConversationStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ConversationStore()
        return _store


def _get_encoder():  # type: ignore[return-type]
    global _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        if LanguageEncoder is not None:
            try:
                _encoder = LanguageEncoder()
                return _encoder
            except Exception:
                _encoder = None
        if TextEncoder is not None:
            try:
                _encoder = TextEncoder()
            except Exception:
                _encoder = None
        return _encoder


def _get_episodic_store():  # type: ignore[return-type]
    global _episodic_store
    if _episodic_store is not None or EpisodicStore is None:
        return _episodic_store
    try:
        _episodic_store = EpisodicStore()
    except Exception:
        _episodic_store = None
    return _episodic_store


def _encode_text(text: str):  # type: ignore[return-type]
    encoder = _get_encoder()
    if encoder is None:
        return None
    try:
        embedding = encoder.encode(text)
        try:
            import torch

            if isinstance(embedding, torch.Tensor):
                return embedding.detach().cpu()
            return torch.tensor(embedding, dtype=torch.float32)
        except Exception:
            return embedding
    except Exception:
        return None


def add_memory(user_input: str, ada_response: str = "", reward: float = 0.0) -> int:
    store = _get_store()
    record_id = store.log_interaction(user_input, ada_response, reward)

    episodic = _get_episodic_store()
    if episodic is not None:
        embedding = _encode_text(user_input)
        if embedding is not None:
            try:
                episodic.store(user_input, ada_response, reward, embedding)
            except Exception:
                pass

    return record_id


def recall(query: str, limit: int = 5) -> List[str]:
    contexts: List[str] = []
    episodic = _get_episodic_store()
    if episodic is not None:
        embedding = _encode_text(query)
        if embedding is not None:
            try:
                matches = episodic.retrieve(embedding, top_k=limit)
                contexts.extend(
                    f"You: {user_input}\nAda: {ada_response}"
                    for _, user_input, ada_response, _ in matches
                )
            except Exception:
                contexts = []

    if contexts:
        return contexts

    recent = list(_get_store().fetch_recent(limit=limit))
    return [
        f"You: {record.user_input}\nAda: {record.ada_response}"
        for record in reversed(recent)
    ]
