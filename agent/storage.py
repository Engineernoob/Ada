"""Storage utilities for Ada's autonomous action system."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from contextlib import contextmanager


@dataclass
class PlanRecord:
    id: str
    goal: str
    category: str
    priority: float
    confidence: float
    status: str
    created_at: str
    updated_at: str


@dataclass
class ActionRecord:
    id: int
    plan_id: str
    step_number: int
    tool: str
    description: str
    input_data: str
    output_data: str
    success: bool
    reward: float
    duration_seconds: float
    timestamp: str


class ActionLogger:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = (
            db_path or Path(__file__).resolve().parents[1] / "storage" / "actions.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority REAL,
                    confidence REAL,
                    status TEXT DEFAULT 'created',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    tool TEXT NOT NULL,
                    description TEXT NOT NULL,
                    input_data TEXT,
                    output_data TEXT,
                    success BOOLEAN NOT NULL,
                    reward REAL,
                    duration_seconds REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (plan_id) REFERENCES plans (id)
                )
            """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal TEXT NOT NULL,
                    result_summary TEXT,
                    context TEXT,
                    base_score REAL,
                    context_modifier REAL,
                    quality_modifier REAL,
                    final_score REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # For plan-level evaluations
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    overall_success BOOLEAN NOT NULL,
                    completion_rate REAL,
                    base_reward REAL,
                    consistency_score REAL,
                    adjusted_reward REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (plan_id) REFERENCES plans (id)
                )
            """
            )

            connection.commit()

    def log_plan_creation(
        self, plan_id: str, goal: str, category: str, priority: float, confidence: float
    ) -> None:
        timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO plans (id, goal, category, priority, confidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (plan_id, goal, category, priority, confidence, timestamp, timestamp),
            )
            connection.commit()

    def log_plan_execution(self, execution_data: Dict[str, Any]) -> None:
        plan_id = execution_data.get("plan_id")
        if not plan_id:
            return

        # Update plan status
        status = "completed" if execution_data.get("success") else "failed"
        timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE plans SET status = ?, updated_at = ? WHERE id = ?",
                (status, timestamp, plan_id),
            )

            # Log individual actions
            for step in execution_data.get("steps", []):
                self._log_action_step(plan_id, step)

            connection.commit()

    def _log_action_step(self, plan_id: str, step_data: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO actions (plan_id, step_number, tool, description, input_data, "
                "output_data, success, reward, duration_seconds, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    plan_id,
                    step_data.get("step_number", 0),
                    step_data.get("tool", "unknown"),
                    step_data.get("description", ""),
                    str(step_data.get("input", "")),
                    str(step_data.get("output", "")),
                    step_data.get("success", False),
                    step_data.get("reward", 0.0),
                    step_data.get("duration_seconds", 0.0),
                    step_data.get("timestamp", datetime.now().isoformat()),
                ),
            )

    def log_evaluation(self, evaluation_data: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO evaluations (goal, result_summary, context, base_score, "
                "context_modifier, quality_modifier, final_score, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evaluation_data.get("goal", ""),
                    evaluation_data.get("result_summary", ""),
                    evaluation_data.get("context", ""),
                    evaluation_data.get("base_score", 0.0),
                    evaluation_data.get("context_modifier", 0.0),
                    evaluation_data.get("quality_modifier", 0.0),
                    evaluation_data.get("final_score", 0.0),
                    evaluation_data.get("timestamp", datetime.now().isoformat()),
                ),
            )
            connection.commit()

    def log_plan_evaluation(self, evaluation_data: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO plan_evaluations (plan_id, goal, overall_success, completion_rate, "
                "base_reward, consistency_score, adjusted_reward, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evaluation_data.get("plan_id", ""),
                    evaluation_data.get("goal", ""),
                    evaluation_data.get("overall_success", False),
                    evaluation_data.get("completion_rate", 0.0),
                    evaluation_data.get("base_reward", 0.0),
                    evaluation_data.get("consistency_score", 0.0),
                    evaluation_data.get("adjusted_reward", 0.0),
                    evaluation_data.get("timestamp", datetime.now().isoformat()),
                ),
            )
            connection.commit()

    def get_plans(
        self, limit: int = 10, status: str | None = None
    ) -> Iterable[PlanRecord]:
        with sqlite3.connect(self.db_path) as connection:
            query = "SELECT id, goal, category, priority, confidence, status, created_at, updated_at FROM plans"
            params = []

            if status:
                query += " WHERE status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = connection.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                yield PlanRecord(*row)

    def get_plan_actions(self, plan_id: str) -> Iterable[ActionRecord]:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT id, plan_id, step_number, tool, description, input_data, "
                "output_data, success, reward, duration_seconds, timestamp "
                "FROM actions WHERE plan_id = ? ORDER BY step_number",
                (plan_id,),
            )
            rows = cursor.fetchall()

            for row in rows:
                yield ActionRecord(*row)

    def get_recent_evaluations(self, limit: int = 10) -> Iterable[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT * FROM evaluations ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            columns = [description[0] for description in cursor.description]

            for row in cursor.fetchall():
                yield dict(zip(columns, row))

    def get_performance_summary(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as connection:
            # Plan statistics
            cursor = connection.execute(
                """
                SELECT status, COUNT(*) as count FROM plans GROUP BY status
            """
            )
            plan_stats = {row[0]: row[1] for row in cursor.fetchall()}

            # Action statistics
            cursor = connection.execute(
                """
                SELECT 
                    AVG(CASE WHEN success THEN 1 ELSE 0 END) as success_rate,
                    AVG(reward) as avg_reward,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(*) as total_actions
                FROM actions
            """
            )
            action_stats = cursor.fetchone()

            # Tool usage statistics
            cursor = connection.execute(
                """
                SELECT tool, COUNT(*) as usage_count, 
                       AVG(CASE WHEN success THEN 1 ELSE 0 END) as success_rate
                FROM actions GROUP BY tool ORDER BY usage_count DESC
            """
            )
            tool_stats = [
                {"tool": row[0], "usage_count": row[1], "success_rate": row[2]}
                for row in cursor.fetchall()
            ]

            # Recent performance trend
            cursor = connection.execute(
                """
                SELECT DATE(timestamp) as date, AVG(final_score) as avg_score
                FROM evaluations 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY DATE(timestamp) ORDER BY date
            """
            )
            performance_trend = [
                {"date": row[0], "avg_score": row[1]} for row in cursor.fetchall()
            ]

        return {
            "plan_statistics": plan_stats,
            "action_statistics": {
                "success_rate": action_stats[0] or 0.0,
                "average_reward": action_stats[1] or 0.0,
                "average_duration": action_stats[2] or 0.0,
                "total_actions": action_stats[3] or 0,
            },
            "tool_usage": tool_stats,
            "performance_trend": performance_trend,
        }

    def cleanup_old_records(self, days: int = 30) -> None:
        """Clean up records older than specified days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "DELETE FROM evaluations WHERE timestamp < ?", (cutoff_date,)
            )
            connection.execute(
                "DELETE FROM plan_evaluations WHERE timestamp < ?", (cutoff_date,)
            )
            connection.commit()

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
