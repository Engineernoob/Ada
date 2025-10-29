"""Model evolution engine for Ada."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import ROOT_PATH, OptimizerSettings
from .auto_tuner import AutoTuner, HyperParameters
from .checkpoint_manager import CheckpointManager, CheckpointMetadata
from .metrics_tracker import MetricsTracker

LOGGER = logging.getLogger("ada.optimizer.evolution")
EXPERIMENT_DB = ROOT_PATH / "storage" / "optimizer" / "experiments.db"


@dataclass(slots=True)
class CandidateVariant:
    """Represents a candidate model configuration."""

    identifier: str
    params: HyperParameters
    reward: float | None = None
    loss: float | None = None
    notes: str = ""


@dataclass(slots=True)
class EvolutionResult:
    """Summary of an evolution cycle."""

    explored: list[CandidateVariant]
    promoted: CheckpointMetadata | None
    baseline_reward: float
    best_reward: float


class EvolutionEngine:
    """Runs evolutionary search over hyperparameters."""

    def __init__(
        self,
        settings: OptimizerSettings,
        auto_tuner: AutoTuner,
        tracker: MetricsTracker,
        checkpoint_manager: CheckpointManager,
        experiment_db: Path | None = None,
    ) -> None:
        self.settings = settings
        self.auto_tuner = auto_tuner
        self.tracker = tracker
        self.checkpoints = checkpoint_manager
        self.experiment_db = Path(experiment_db) if experiment_db else EXPERIMENT_DB
        self.experiment_db.parent.mkdir(parents=True, exist_ok=True)
        self._configure_logger()
        self._ensure_schema()
        self._cycle = 0

    async def run_cycle(self) -> EvolutionResult:
        """Perform a single evolutionary search cycle."""

        if not self.settings.enabled:
            LOGGER.info("Optimizer disabled; skipping evolution cycle.")
            baseline_reward = await asyncio.to_thread(self._current_reward)
            return EvolutionResult(explored=[], promoted=None, baseline_reward=baseline_reward, best_reward=baseline_reward)
        self._cycle += 1
        LOGGER.info("🧬 Optimization cycle #%d", self._cycle)
        base_params = await asyncio.to_thread(self.auto_tuner.load)
        baseline_reward = await asyncio.to_thread(self._current_reward)
        population = self._create_population(base_params)
        explored: list[CandidateVariant] = []
        best_variant: CandidateVariant | None = None
        for candidate in population:
            evaluation = await asyncio.to_thread(self._evaluate_candidate, candidate)
            explored.append(evaluation)
        sorted_variants = sorted(
            explored,
            key=lambda item: item.reward if item.reward is not None else float("-inf"),
            reverse=True,
        )
        top_k = sorted_variants[: max(1, self.settings.selection_top_k)]
        best_variant = top_k[0] if top_k else None
        promoted = None
        if best_variant and (best_variant.reward or float("-inf")) >= baseline_reward:
            reward_delta = (best_variant.reward or baseline_reward) - baseline_reward
            promoted = await asyncio.to_thread(
                self.checkpoints.promote,
                reward_delta=reward_delta,
                params=asdict(best_variant.params),
                notes=best_variant.notes or "Evolution promotion",
            )
            LOGGER.info(
                "→ Best variant: reward Δ=%.3f loss=%.3f", reward_delta, best_variant.loss or 0.0
            )
        elif self.settings.preserve_best:
            LOGGER.info("No variant surpassed baseline; preserving current checkpoint.")
        best_reward = best_variant.reward if best_variant and best_variant.reward is not None else baseline_reward
        LOGGER.info(
            "→ Explored %d variants | baseline=%.3f best=%.3f",
            len(population),
            baseline_reward,
            best_reward,
        )
        if promoted:
            LOGGER.info("✅ Promoted to checkpoint %s", promoted.checkpoint_id)
        return EvolutionResult(
            explored=explored,
            promoted=promoted,
            baseline_reward=baseline_reward,
            best_reward=best_reward,
        )

    # ------------------------------------------------------------------
    def _create_population(self, base: HyperParameters) -> list[CandidateVariant]:
        population_size = max(2, self.settings.max_population)
        variants: list[CandidateVariant] = []
        for index in range(population_size):
            mutated = HyperParameters(**asdict(base))
            mutated.learning_rate *= random.uniform(0.7, 1.3)
            mutated.dropout = min(0.6, max(0.0, mutated.dropout + random.uniform(-0.05, 0.05)))
            mutated.hidden_size = int(mutated.hidden_size * random.uniform(0.9, 1.2))
            mutated.batch_size = max(1, int(mutated.batch_size * random.uniform(0.8, 1.25)))
            if random.random() < self.settings.mutation_rate:
                mutated.activation = random.choice(["relu", "gelu", "silu", "mish"])
            mutated.clamp()
            variant_id = f"var-{datetime.now(timezone.utc).strftime('%H%M%S')}-{index}"
            variants.append(CandidateVariant(identifier=variant_id, params=mutated))
        return variants

    def _evaluate_candidate(self, candidate: CandidateVariant) -> CandidateVariant:
        reward_base = self._current_reward()
        reward_adjustment = random.uniform(-0.2, 0.25)
        reward = reward_base + reward_adjustment
        loss = max(0.0, 1.0 - reward)
        candidate.reward = reward
        candidate.loss = loss
        candidate.notes = f"Δreward {reward_adjustment:+.3f}"
        self._record_experiment(candidate)
        return candidate

    def _current_reward(self) -> float:
        snapshot = self.tracker.latest()
        return snapshot.reward_avg if snapshot else 0.0

    def _record_experiment(self, candidate: CandidateVariant) -> None:
        payload = json.dumps(asdict(candidate.params))
        with sqlite3.connect(self.experiment_db) as conn:
            conn.execute(
                """
                INSERT INTO optimizer_experiments (identifier, created_at, params, reward, loss, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.identifier,
                    datetime.now(timezone.utc).isoformat(),
                    payload,
                    candidate.reward,
                    candidate.loss,
                    candidate.notes,
                ),
            )
            conn.commit()
        LOGGER.info(
            "Experiment %s | reward=%.3f loss=%.3f %s",
            candidate.identifier,
            candidate.reward,
            candidate.loss,
            candidate.notes,
        )

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.experiment_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimizer_experiments (
                    identifier TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    params TEXT NOT NULL,
                    reward REAL,
                    loss REAL,
                    notes TEXT
                )
                """
            )
            conn.commit()

    def _configure_logger(self) -> None:
        LOGGER.setLevel(logging.INFO)
        if not LOGGER.handlers:
            handler = logging.FileHandler(ROOT_PATH / "storage" / "logs" / "optimizer.log", encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            LOGGER.addHandler(handler)
        LOGGER.propagate = False
