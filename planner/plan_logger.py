"""Plan logging system for storing plan history and execution results."""

from __future__ import annotations

import sqlite3
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

import yaml


def get_setting(*keys: str, default=None):
    """Local copy of get_setting to avoid circular imports."""
    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    try:
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)

        node = settings
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node
    except (FileNotFoundError, yaml.YAMLError):
        return default


class PlanLogger:
    """Database logger for plans, executions, and results."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = Path(
                get_setting("paths", "conversations_db", default="storage/plans.db")
            )

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self) -> None:
        """Create database tables for plan logging."""
        # Plans table
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            category TEXT NOT NULL,
            priority REAL NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            started_at REAL,
            completed_at REAL,
            success_rate REAL DEFAULT 0.0,
            details TEXT
        )
        """
        )

        # Plan steps table
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS plan_steps (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL,
            tool TEXT NOT NULL,
            action TEXT NOT NULL,
            parameters TEXT,
            dependencies TEXT,
            completed BOOLEAN DEFAULT FALSE,
            result TEXT,
            error TEXT,
            timestamp REAL NOT NULL,
            FOREIGN KEY (plan_id) REFERENCES plans (id)
        )
        """
        )

        # Executions table
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT NOT NULL,
            step_id TEXT,
            success BOOLEAN NOT NULL,
            data TEXT,
            error TEXT,
            timestamp REAL NOT NULL,
            execution_time REAL NOT NULL,
            reward REAL,
            FOREIGN KEY (plan_id) REFERENCES plans (id)
        )
        """
        )

        # Goals table (for inferred goals)
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal TEXT NOT NULL,
            category TEXT NOT NULL,
            priority REAL NOT NULL,
            confidence REAL NOT NULL,
            source TEXT,
            timestamp REAL NOT NULL,
            status TEXT DEFAULT 'pending',
            plan_id TEXT,
            FOREIGN KEY (plan_id) REFERENCES plans (id)
        )
        """
        )

        # Create indexes for better performance
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_plans_created_at ON plans(created_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_plan_steps_plan_id ON plan_steps(plan_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_executions_plan_id ON executions(plan_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)"
        )

        self.conn.commit()

    def log_plan(self, plan) -> None:
        """Log a plan to the database."""
        self.conn.execute(
            """
        INSERT OR REPLACE INTO plans 
        (id, goal, category, priority, status, created_at, started_at, completed_at, success_rate, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                plan.id,
                plan.goal,
                plan.category,
                plan.priority,
                plan.status,
                plan.created_at,
                plan.started_at,
                plan.completed_at,
                plan.success_rate,
                str(plan.to_dict()),
            ),
        )
        self.conn.commit()

    def log_plan_steps(self, plan) -> None:
        """Log all steps for a plan."""
        # Clear existing steps for this plan
        self.conn.execute("DELETE FROM plan_steps WHERE plan_id = ?", (plan.id,))

        for step in plan.steps:
            self.conn.execute(
                """
            INSERT INTO plan_steps 
            (id, plan_id, tool, action, parameters, dependencies, completed, result, error, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    step.step_id,
                    plan.id,
                    step.tool,
                    step.action,
                    str(step.parameters),
                    str(step.dependencies),
                    step.completed,
                    step.result,
                    step.error,
                    step.timestamp,
                ),
            )

        self.conn.commit()

    def log_execution(
        self, execution_result, plan_id: str, reward: Optional[float] = None
    ) -> None:
        """Log an execution result."""
        self.conn.execute(
            """
        INSERT INTO executions
        (plan_id, step_id, success, data, error, timestamp, execution_time, reward)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                plan_id,
                execution_result.step_id,
                execution_result.success,
                str(execution_result.data) if execution_result.data else None,
                execution_result.error,
                execution_result.timestamp,
                execution_result.execution_time,
                reward,
            ),
        )
        self.conn.commit()

    def log_goal(self, goal) -> None:
        """Log an inferred goal."""
        self.conn.execute(
            """
        INSERT INTO goals
        (goal, category, priority, confidence, source, timestamp, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                goal.goal,
                goal.category,
                goal.priority,
                goal.confidence,
                goal.source,
                goal.timestamp,
                "pending",
            ),
        )
        self.conn.commit()

    def update_plan_status(self, plan_id: str, status: str) -> None:
        """Update the status of a plan."""
        self.conn.execute(
            """
        UPDATE plans SET status = ?, completed_at = ? WHERE id = ?
        """,
            (
                status,
                time.time() if status in ["completed", "failed", "aborted"] else None,
                plan_id,
            ),
        )
        self.conn.commit()

    def update_goal_status(
        self, goal_id: int, status: str, plan_id: Optional[str] = None
    ) -> None:
        """Update the status of a goal."""
        update_fields = ["status = ?"]
        params = [status]

        if plan_id:
            update_fields.append("plan_id = ?")
            params.append(plan_id)

        params.append(goal_id)

        query = f"UPDATE goals SET {', '.join(update_fields)} WHERE id = ?"
        self.conn.execute(query, params)
        self.conn.commit()

    def get_plan_history(
        self, limit: int = 50, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get plan history from the database."""
        query = "SELECT * FROM plans"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()

        plans = []
        for row in rows:
            plan_dict = {
                "id": row[0],
                "goal": row[1],
                "category": row[2],
                "priority": row[3],
                "status": row[4],
                "created_at": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "success_rate": row[8],
                "details": row[9],
            }
            plans.append(plan_dict)

        return plans

    def get_plan_details(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific plan."""
        # Get plan info
        cursor = self.conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,))
        plan_row = cursor.fetchone()

        if not plan_row:
            return None

        # Get plan steps
        cursor = self.conn.execute(
            "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY timestamp", (plan_id,)
        )
        step_rows = cursor.fetchall()

        # Get executions
        cursor = self.conn.execute(
            "SELECT * FROM executions WHERE plan_id = ? ORDER BY timestamp", (plan_id,)
        )
        execution_rows = cursor.fetchall()

        # Assemble detailed plan
        plan_details = {
            "plan": {
                "id": plan_row[0],
                "goal": plan_row[1],
                "category": plan_row[2],
                "priority": plan_row[3],
                "status": plan_row[4],
                "created_at": plan_row[5],
                "started_at": plan_row[6],
                "completed_at": plan_row[7],
                "success_rate": plan_row[8],
                "details": plan_row[9],
            },
            "steps": [],
            "executions": [],
        }

        # Add steps
        for step_row in step_rows:
            step = {
                "id": step_row[0],
                "plan_id": step_row[1],
                "tool": step_row[2],
                "action": step_row[3],
                "parameters": step_row[4],
                "dependencies": step_row[5],
                "completed": bool(step_row[6]),
                "result": step_row[7],
                "error": step_row[8],
                "timestamp": step_row[9],
            }
            plan_details["steps"].append(step)

        # Add executions
        for exec_row in execution_rows:
            execution = {
                "id": exec_row[0],
                "plan_id": exec_row[1],
                "step_id": exec_row[2],
                "success": bool(exec_row[3]),
                "data": exec_row[4],
                "error": exec_row[5],
                "timestamp": exec_row[6],
                "execution_time": exec_row[7],
                "reward": exec_row[8],
            }
            plan_details["executions"].append(execution)

        return plan_details

    def get_pending_goals(self) -> List[Dict[str, Any]]:
        """Get all pending goals."""
        cursor = self.conn.execute(
            """
        SELECT * FROM goals WHERE status = 'pending' 
        ORDER BY priority DESC, timestamp DESC
        """
        )
        rows = cursor.fetchall()

        goals = []
        for row in rows:
            goal = {
                "id": row[0],
                "goal": row[1],
                "category": row[2],
                "priority": row[3],
                "confidence": row[4],
                "source": row[5],
                "timestamp": row[6],
                "status": row[7],
                "plan_id": row[8],
            }
            goals.append(goal)

        return goals

    def get_planning_stats(self) -> Dict[str, Any]:
        """Get statistics about planning and execution."""
        # Plan stats
        cursor = self.conn.execute("SELECT status, COUNT(*) FROM plans GROUP BY status")
        plan_status_counts = dict(cursor.fetchall())

        # Execution stats
        cursor = self.conn.execute(
            "SELECT AVG(execution_time), COUNT(*) FROM executions"
        )
        exec_stats = cursor.fetchone()
        avg_execution_time = exec_stats[0] or 0.0
        total_executions = exec_stats[1] or 0

        # Success rates
        cursor = self.conn.execute(
            "SELECT AVG(success), COUNT(*) FROM executions WHERE success = 1"
        )
        success_stats = cursor.fetchone()
        success_rate = (success_stats[0] or 0.0) if success_stats[1] else 0.0

        # Goal stats
        cursor = self.conn.execute("SELECT status, COUNT(*) FROM goals GROUP BY status")
        goal_status_counts = dict(cursor.fetchall())

        # Recent activity
        cursor = self.conn.execute(
            """
        SELECT COUNT(*) FROM plans WHERE created_at > ?
        """,
            (time.time() - 24 * 60 * 60,),
        )  # Last 24 hours
        recent_plans = cursor.fetchone()[0]

        return {
            "plans": {
                "total": sum(plan_status_counts.values()),
                "by_status": plan_status_counts,
                "recent": recent_plans,
            },
            "executions": {
                "total": total_executions,
                "average_time": avg_execution_time,
                "success_rate": success_rate,
            },
            "goals": {
                "total": sum(goal_status_counts.values()),
                "by_status": goal_status_counts,
            },
        }

    def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old records to manage database size."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        # Delete old completed/failed plans and their related data
        self.conn.execute(
            """
        DELETE FROM executions 
        WHERE plan_id IN (
            SELECT id FROM plans 
            WHERE status IN ('completed', 'failed', 'aborted') AND created_at < ?
        )
        """,
            (cutoff_time,),
        )

        deleted_executions = self.conn.total_changes

        self.conn.execute(
            """
        DELETE FROM plan_steps 
        WHERE plan_id IN (
            SELECT id FROM plans 
            WHERE status IN ('completed', 'failed', 'aborted') AND created_at < ?
        )
        """,
            (cutoff_time,),
        )

        self.conn.execute(
            """
        DELETE FROM plans 
        WHERE status IN ('completed', 'failed', 'aborted') AND created_at < ?
        """,
            (cutoff_time,),
        )

        # Delete old resolved goals
        self.conn.execute(
            """
        DELETE FROM goals 
        WHERE status != 'pending' AND timestamp < ?
        """,
            (cutoff_time,),
        )

        deleted_total = self.conn.total_changes - deleted_executions
        self.conn.commit()

        return deleted_total

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __del__(self):
        """Ensure connection is closed on deletion."""
        if hasattr(self, "conn"):
            self.close()
