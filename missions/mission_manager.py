"""Mission persistence and lifecycle management."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

from core.settings import get_setting

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "missions.db"


@dataclass(slots=True)
class MissionStep:
    """Represents a single actionable step inside a mission."""

    description: str
    tool: str | None = None
    parameters: dict[str, Any] | None = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "tool": self.tool,
            "parameters": self.parameters or {},
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MissionStep":
        return cls(
            description=payload.get("description", ""),
            tool=payload.get("tool"),
            parameters=payload.get("parameters") or {},
            status=payload.get("status", "pending"),
        )


@dataclass(slots=True)
class Mission:
    """Domain object persisted in the mission database."""

    id: str
    goal: str
    steps: list[MissionStep]
    reward: float | None
    status: str
    created_at: datetime
    updated_at: datetime
    last_run_at: datetime | None = None

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.id,
            self.goal,
            json.dumps([step.to_dict() for step in self.steps], ensure_ascii=False),
            self.reward,
            self.status,
            self.created_at.isoformat(),
            self.updated_at.isoformat(),
            self.last_run_at.isoformat() if self.last_run_at else None,
        )


class MissionManager:
    """Provides mission CRUD and bookkeeping utilities backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        path = Path(db_path) if db_path else DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._ensure_schema()

    # -- Public API -----------------------------------------------------------------
    def create_mission(self, goal: str, steps: Iterable[MissionStep] | None = None) -> Mission:
        mission_id = self._generate_id(goal)
        now = datetime.now(timezone.utc)
        mission = Mission(
            id=mission_id,
            goal=goal,
            steps=list(steps or []),
            reward=None,
            status="pending",
            created_at=now,
            updated_at=now,
            last_run_at=None,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO missions (id, goal, steps, reward, status, created_at, updated_at, last_run_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                mission.as_tuple(),
            )
        return mission

    def list_missions(self, limit: int | None = None) -> list[Mission]:
        query = "SELECT id, goal, steps, reward, status, created_at, updated_at, last_run_at FROM missions ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            rows = self._execute(query, (limit,))
        else:
            rows = self._execute(query)
        return [self._row_to_mission(row) for row in rows]

    def get_mission(self, mission_id: str) -> Mission | None:
        rows = self._execute(
            "SELECT id, goal, steps, reward, status, created_at, updated_at, last_run_at FROM missions WHERE id = ?",
            (mission_id,),
        )
        if not rows:
            return None
        return self._row_to_mission(rows[0])

    def update_status(self, mission_id: str, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE missions SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, mission_id),
            )

    def record_reward(self, mission_id: str, reward: float | None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE missions SET reward = ?, updated_at = ? WHERE id = ?",
                (reward, now, mission_id),
            )

    def update_steps(self, mission_id: str, steps: Sequence[MissionStep]) -> None:
        payload = json.dumps([step.to_dict() for step in steps], ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE missions SET steps = ?, updated_at = ? WHERE id = ?",
                (payload, now, mission_id),
            )

    def record_run(self, mission_id: str, status: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE missions SET last_run_at = ?, status = ?, updated_at = ? WHERE id = ?",
                (now_iso, status, now_iso, mission_id),
            )

    # -- Internal helpers ------------------------------------------------------------
    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS missions (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    reward REAL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_run_at TEXT
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _execute(self, query: str, params: Sequence[Any] | None = None) -> list[sqlite3.Row]:
        with self._connect() as conn:
            cursor = conn.execute(query, params or [])
            rows = cursor.fetchall()
        return rows

    def _row_to_mission(self, row: sqlite3.Row) -> Mission:
        steps_payload = json.loads(row["steps"]) if row["steps"] else []
        steps = [MissionStep.from_dict(item) for item in steps_payload]
        created_at = datetime.fromisoformat(row["created_at"])
        updated_at = datetime.fromisoformat(row["updated_at"])
        last_run_at = datetime.fromisoformat(row["last_run_at"]) if row["last_run_at"] else None
        return Mission(
            id=row["id"],
            goal=row["goal"],
            steps=steps,
            reward=row["reward"],
            status=row["status"],
            created_at=created_at,
            updated_at=updated_at,
            last_run_at=last_run_at,
        )

    def _generate_id(self, goal: str) -> str:
        prefix = goal.lower().strip().replace(" ", "_")[:12] or "mission"
        mission_id = f"{prefix}_{uuid4().hex[:6]}"
        return mission_id


def load_default_manager() -> MissionManager:
    """Factory that respects configuration overrides if present."""

    db_path = get_setting("missions", "database", default=str(DEFAULT_DB_PATH))
    return MissionManager(Path(db_path))
