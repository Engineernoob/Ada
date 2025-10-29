"""Metrics tracking utilities for Ada's optimizer."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import ROOT_PATH

LOGGER = logging.getLogger("ada.optimizer.metrics")
LOG_PATH = ROOT_PATH / "storage" / "logs" / "optimizer.log"
METRICS_DB = ROOT_PATH / "storage" / "optimizer" / "metrics.db"


@dataclass(slots=True)
class MetricsSnapshot:
    """Captured metrics for a single optimization cycle."""

    timestamp: datetime
    reward_avg: float
    loss: float
    grad_norm: float
    cpu_usage: float
    gpu_usage: float | None
    latency_ms: float

    def as_tuple(self) -> tuple[str, float, float, float, float, float | None, float]:
        return (
            self.timestamp.isoformat(),
            self.reward_avg,
            self.loss,
            self.grad_norm,
            self.cpu_usage,
            self.gpu_usage if self.gpu_usage is not None else -1.0,
            self.latency_ms,
        )


class MetricsTracker:
    """Persists and logs runtime metrics for optimization cycles."""

    def __init__(self, db_path: Path | None = None, log_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else METRICS_DB
        self.log_path = Path(log_path) if log_path else LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._configure_logger()
        self._ensure_schema()

    # ------------------------------------------------------------------
    def record(
        self,
        *,
        reward_avg: float,
        loss: float,
        grad_norm: float,
        cpu_usage: float,
        gpu_usage: float | None,
        latency_ms: float,
    ) -> MetricsSnapshot:
        """Persist a metrics snapshot and emit a log line."""

        snapshot = MetricsSnapshot(
            timestamp=datetime.now(timezone.utc),
            reward_avg=reward_avg,
            loss=loss,
            grad_norm=grad_norm,
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage,
            latency_ms=latency_ms,
        )
        self._insert(snapshot)
        LOGGER.info(
            "Metrics | reward_avg=%.3f loss=%.4f grad_norm=%.3f cpu=%.1f gpu=%s latency=%.1fms",
            snapshot.reward_avg,
            snapshot.loss,
            snapshot.grad_norm,
            snapshot.cpu_usage,
            "%.1f" % snapshot.gpu_usage if snapshot.gpu_usage is not None else "na",
            snapshot.latency_ms,
        )
        return snapshot

    def get_recent(self, limit: int = 10) -> list[MetricsSnapshot]:
        """Return the most recent metrics snapshots."""

        query = (
            "SELECT timestamp, reward_avg, loss, grad_norm, cpu_usage, gpu_usage, latency_ms "
            "FROM optimizer_metrics ORDER BY timestamp DESC LIMIT ?"
        )
        rows = self._execute(query, (limit,))
        return [self._row_to_snapshot(row) for row in rows]

    def latest(self) -> MetricsSnapshot | None:
        """Return the latest metrics snapshot if available."""

        rows = self.get_recent(limit=1)
        return rows[0] if rows else None

    def reward_trend(self, window: int = 5) -> float:
        """Compute the reward delta over the given window."""

        snapshots = self.get_recent(limit=window)
        if len(snapshots) < 2:
            return 0.0
        return snapshots[0].reward_avg - snapshots[-1].reward_avg

    # ------------------------------------------------------------------
    def _configure_logger(self) -> None:
        LOGGER.setLevel(logging.INFO)
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(self.log_path) for handler in LOGGER.handlers):
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            LOGGER.addHandler(handler)
        LOGGER.propagate = False

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimizer_metrics (
                    timestamp TEXT PRIMARY KEY,
                    reward_avg REAL NOT NULL,
                    loss REAL NOT NULL,
                    grad_norm REAL NOT NULL,
                    cpu_usage REAL NOT NULL,
                    gpu_usage REAL,
                    latency_ms REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _insert(self, snapshot: MetricsSnapshot) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO optimizer_metrics
                (timestamp, reward_avg, loss, grad_norm, cpu_usage, gpu_usage, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                snapshot.as_tuple(),
            )
            conn.commit()

    def _execute(self, query: str, params: Iterable[object] | None = None) -> list[sqlite3.Row]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params or [])
            rows = cursor.fetchall()
        return rows

    def _row_to_snapshot(self, row: sqlite3.Row) -> MetricsSnapshot:
        gpu_usage = row["gpu_usage"]
        gpu_val = None if gpu_usage is None or float(gpu_usage) < 0 else float(gpu_usage)
        return MetricsSnapshot(
            timestamp=datetime.fromisoformat(row["timestamp"]),
            reward_avg=float(row["reward_avg"]),
            loss=float(row["loss"]),
            grad_norm=float(row["grad_norm"]),
            cpu_usage=float(row["cpu_usage"]),
            gpu_usage=gpu_val,
            latency_ms=float(row["latency_ms"]),
        )
