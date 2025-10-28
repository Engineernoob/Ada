"""Mission audit and checkpoint comparison utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .mission_manager import Mission

LOGGER = logging.getLogger("ada.missions.auditor")

AUDIT_STATE_PATH = Path(__file__).resolve().parents[1] / "storage" / "checkpoints" / "mission_audit.json"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "storage" / "checkpoints" / "ada_core.pt"


@dataclass(slots=True)
class AuditReport:
    """Summarizes an audit comparison."""

    mission_id: str | None
    checkpoint_path: Path
    previous_size: int
    current_size: int
    reward_delta: float
    drift: float
    notes: str


class MissionAuditor:
    """Computes lightweight metrics on checkpoints to ensure progress tracking."""

    def __init__(self, state_path: Path | None = None, default_checkpoint: Path | None = None) -> None:
        self.state_path = state_path or AUDIT_STATE_PATH
        self.default_checkpoint = default_checkpoint or DEFAULT_CHECKPOINT
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def audit_checkpoint(self, mission: Mission | None, checkpoint_path: Path | None = None) -> AuditReport:
        target = checkpoint_path or self.default_checkpoint
        target.parent.mkdir(parents=True, exist_ok=True)

        previous = self._state.get("checkpoints", {}).get(str(target), {})
        previous_size = int(previous.get("size", 0))
        current_size = target.stat().st_size if target.exists() else 0

        reward_delta = self._compute_reward_delta(previous_size, current_size)
        drift = abs(reward_delta) * 0.4

        mission_id = mission.id if mission else None
        notes = "Checkpoint evaluated"
        report = AuditReport(
            mission_id=mission_id,
            checkpoint_path=target,
            previous_size=previous_size,
            current_size=current_size,
            reward_delta=reward_delta,
            drift=drift,
            notes=notes,
        )

        self._state.setdefault("checkpoints", {})[str(target)] = {
            "size": current_size,
            "updated_at": datetime.utcnow().isoformat(),
            "mission_id": mission_id,
            "reward_delta": reward_delta,
            "drift": drift,
        }
        self._save_state()

        LOGGER.info(
            "Audit complete for %s | Δreward=%.3f drift=%.3f",
            target,
            reward_delta,
            drift,
        )
        return report

    def audit_latest(self) -> AuditReport:
        return self.audit_checkpoint(None, self.default_checkpoint)

    def _compute_reward_delta(self, previous_size: int, current_size: int) -> float:
        if previous_size == 0:
            return 0.0 if current_size == 0 else 0.1
        delta = current_size - previous_size
        return max(min(delta / max(previous_size, 1), 1.0), -1.0)

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return {}

    def _save_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2)
