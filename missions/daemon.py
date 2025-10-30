"""Asynchronous mission daemon for Ada."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING

from core.settings import get_setting

from .auditor import AuditReport, MissionAuditor
from .curriculum_trainer import CurriculumResult, CurriculumTrainer
from .mission_manager import Mission, MissionManager

if TYPE_CHECKING:
    from optimizer.auto_tuner import AutoTuner

LOGGER = logging.getLogger("ada.missions.daemon")
LOG_PATH = Path(__file__).resolve().parents[1] / "storage" / "logs" / "mission.log"


@dataclass(slots=True)
class MissionSettings:
    """Runtime configuration for the daemon."""

    enabled: bool = True
    check_interval_minutes: int = 60
    curriculum_learning: bool = True
    auto_audit: bool = True

    @property
    def check_interval(self) -> timedelta:
        return timedelta(minutes=max(self.check_interval_minutes, 1))

    @classmethod
    def from_settings(cls) -> "MissionSettings":
        node = get_setting("missions", default={}) or {}
        return cls(
            enabled=bool(node.get("enabled", True)),
            check_interval_minutes=int(node.get("check_interval_minutes", 60)),
            curriculum_learning=bool(node.get("curriculum_learning", True)),
            auto_audit=bool(node.get("auto_audit", True)),
        )


@dataclass(slots=True)
class MissionCycleResult:
    """Aggregated information for a completed mission run."""

    mission: Mission
    curriculum: CurriculumResult | None
    audit: AuditReport | None
    success: bool
    message: str


class MissionDaemon:
    """Coordinates background execution of missions."""

    def __init__(
        self,
        manager: MissionManager,
        trainer: CurriculumTrainer | None = None,
        auditor: MissionAuditor | None = None,
        settings: MissionSettings | None = None,
        log_path: Path | None = None,
        auto_tuner: "AutoTuner" | None = None,
    ) -> None:
        self.manager = manager
        self.trainer = trainer or CurriculumTrainer()
        self.auditor = auditor or MissionAuditor()
        self.settings = settings or MissionSettings.from_settings()
        self.log_path = log_path or LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._configure_logger()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._background_loop: asyncio.AbstractEventLoop | None = None
        self._cycle = 0
        self.auto_tuner = auto_tuner

    # -- Public API -----------------------------------------------------------------
    def start_background(self) -> None:
        """Launch the daemon loop in a background thread."""

        if not self.settings.enabled:
            LOGGER.info("Mission daemon is disabled via settings.")
            return
        if self._thread and self._thread.is_alive():
            LOGGER.info("Mission daemon already running.")
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run_background_loop, name="mission-daemon", daemon=True)
        self._thread.start()
        LOGGER.info("Mission daemon thread started.")

    def stop_background(self) -> None:
        """Signal the background loop to stop."""

        self._stop_event.set()
        if self._background_loop and self._background_loop.is_running():
            self._background_loop.call_soon_threadsafe(lambda: None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        LOGGER.info("Mission daemon stopped.")

    @property
    def is_running(self) -> bool:
        """Return True when the daemon thread is active."""

        return self._thread is not None and self._thread.is_alive()

    async def run_pending_once(self) -> list[MissionCycleResult]:
        """Run all pending missions a single time asynchronously."""

        results: list[MissionCycleResult] = []
        missions = [mission for mission in self.manager.list_missions() if mission.status in {"pending", "scheduled"}]
        if not missions:
            LOGGER.info("No pending missions detected.")
            return results
        for mission in missions:
            result = await self._execute_mission(mission)
            results.append(result)
        return results

    async def execute_mission(self, mission_id: str) -> MissionCycleResult:
        """Execute a single mission asynchronously."""

        mission = self.manager.get_mission(mission_id)
        if not mission:
            raise ValueError(f"Mission {mission_id} not found")
        return await self._execute_mission(mission)

    def run_mission_blocking(self, mission_id: str) -> MissionCycleResult:
        """Execute a mission synchronously for CLI usage."""

        LOGGER.info("Manually running mission %s", mission_id)
        return asyncio.run(self.execute_mission(mission_id))

    def run_audit_blocking(self) -> AuditReport:
        """Trigger a manual audit on the latest checkpoint."""

        return self.auditor.audit_latest()

    # -- Internal helpers ------------------------------------------------------------
    def _run_background_loop(self) -> None:
        self._background_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._background_loop)
        try:
            self._background_loop.run_until_complete(self._mission_loop())
        finally:
            self._background_loop.close()
            self._background_loop = None

    async def _mission_loop(self) -> None:
        LOGGER.info("Mission daemon loop started. Interval=%s", self.settings.check_interval)
        while not self._stop_event.is_set():
            self._cycle += 1
            LOGGER.info("🗓 Mission daemon cycle #%s", self._cycle)
            try:
                await self.run_pending_once()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Mission daemon cycle failed")
            await asyncio.sleep(self.settings.check_interval.total_seconds())
        LOGGER.info("Mission daemon loop exiting.")

    async def _execute_mission(self, mission: Mission) -> MissionCycleResult:
        LOGGER.info("Starting mission %s (%s)", mission.id, mission.goal)
        self.manager.update_status(mission.id, "running")
        audit_report: AuditReport | None = None
        curriculum_result: CurriculumResult | None = None
        success = True
        message = "Mission completed"
        try:
            if self.settings.curriculum_learning:
                curriculum_result = await self.trainer.run(mission)
            self.manager.record_run(mission.id, "completed")
            if self.settings.auto_audit:
                audit_report = self.auditor.audit_checkpoint(
                    mission,
                    curriculum_result.checkpoint_path if curriculum_result else None,
                )
            if self.auto_tuner and self._should_optimize(mission):
                try:
                    await self.auto_tuner.run_cycle()
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Auto-tuner cycle failed after mission %s", mission.id)
                audit_report = self.auditor.audit_checkpoint(mission, curriculum_result.checkpoint_path if curriculum_result else None)
        except Exception as exc:  # noqa: BLE001
            success = False
            message = f"Mission failed: {exc}"
            LOGGER.exception("Mission %s failed", mission.id)
            self.manager.record_run(mission.id, "failed")
        else:
            LOGGER.info(
                "✅ Mission %s completed. %s",
                mission.id,
                curriculum_result.notes if curriculum_result else "No curriculum run",
            )
            if audit_report:
                self.manager.record_reward(mission.id, audit_report.reward_delta)
                LOGGER.info(
                    "📈 Reward Δ %+.2f Drift %.2f | %s",
                    audit_report.reward_delta,
                    audit_report.drift,
                    audit_report.notes,
                )
        return MissionCycleResult(
            mission=mission,
            curriculum=curriculum_result,
            audit=audit_report,
            success=success,
            message=message,
        )

    def _configure_logger(self) -> None:
        LOGGER.setLevel(logging.INFO)
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(self.log_path) for handler in LOGGER.handlers):
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            LOGGER.addHandler(handler)
        LOGGER.propagate = False

    def _should_optimize(self, mission: Mission) -> bool:
        goal = mission.goal.lower()
        return any(keyword in goal for keyword in ("optimize", "train", "tune", "voice", "persona"))


def build_default_daemon(manager: MissionManager | None = None, auto_tuner: "AutoTuner" | None = None) -> MissionDaemon:
    manager = manager or MissionManager()
    settings = MissionSettings.from_settings()
    trainer = CurriculumTrainer()
    auditor = MissionAuditor()
    return MissionDaemon(manager=manager, trainer=trainer, auditor=auditor, settings=settings, auto_tuner=auto_tuner)
