"""Curriculum learning helpers for the mission daemon."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Optional

from .mission_manager import Mission

LOGGER = logging.getLogger("ada.missions.curriculum")


@dataclass(slots=True)
class CurriculumResult:
    """Outcome metadata from running a mission curriculum."""

    mission_id: str
    checkpoint_path: Path | None
    notes: str = ""


class CurriculumTrainer:
    """Coordinates task-specific training logic for missions."""

    async def run(self, mission: Mission) -> CurriculumResult:
        """Dispatch mission goal to the appropriate trainer."""

        goal = mission.goal.lower()
        if "voice" in goal or "persona" in goal:
            return await self._run_voice_curriculum(mission)
        return await self._run_core_curriculum(mission)

    async def _run_core_curriculum(self, mission: Mission) -> CurriculumResult:
        LOGGER.info("Starting reinforcement curriculum for %s", mission.id)
        checkpoint_path: Path | None = None
        notes = ""
        try:
            from neural import trainer as neural_trainer

            loop = asyncio.get_running_loop()
            checkpoint_path = await loop.run_in_executor(None, neural_trainer.train_reinforcement)
            notes = "Reinforcement fine-tuning complete"
        except Exception as exc:  # noqa: BLE001 - best effort logging for daemon
            notes = f"Core curriculum failed: {exc}"
            LOGGER.exception("Reinforcement curriculum failed for %s", mission.id)
        return CurriculumResult(mission_id=mission.id, checkpoint_path=checkpoint_path, notes=notes)

    async def _run_voice_curriculum(self, mission: Mission) -> CurriculumResult:
        LOGGER.info("Starting voice curriculum for %s", mission.id)
        trainer_callable = self._resolve_voice_trainer()
        checkpoint_path: Path | None = None
        notes = ""
        if trainer_callable is None:
            notes = "Voice training module unavailable"
            LOGGER.warning("Voice training not available; skipping mission %s", mission.id)
            return CurriculumResult(mission_id=mission.id, checkpoint_path=None, notes=notes)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, trainer_callable)
            if isinstance(result, Path):
                checkpoint_path = result
            notes = "Voice fine-tuning executed"
        except Exception as exc:  # noqa: BLE001 - best effort logging
            notes = f"Voice curriculum failed: {exc}"
            LOGGER.exception("Voice curriculum failed for %s", mission.id)
        return CurriculumResult(mission_id=mission.id, checkpoint_path=checkpoint_path, notes=notes)

    def _resolve_voice_trainer(self) -> Optional[Callable[[], Path | None]]:
        """Return the best available voice fine-tuning callable."""

        module_name_candidates = [
            "voice_training.finetune_xtts",
            "voice_persona.controller",
        ]
        for module_name in module_name_candidates:
            if find_spec(module_name) is None:
                continue
            module = import_module(module_name)
            for attr in ("fine_tune", "train", "run", "main"):
                candidate = getattr(module, attr, None)
                if callable(candidate):
                    return candidate  # type: ignore[return-value]
        return None
