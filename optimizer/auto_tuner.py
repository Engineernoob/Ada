"""Automated hyperparameter tuning for Ada."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import ROOT_PATH, OptimizerSettings
from .metrics_tracker import MetricsSnapshot, MetricsTracker

LOGGER = logging.getLogger("ada.optimizer.auto_tuner")
PARAMS_PATH = ROOT_PATH / "storage" / "optimizer" / "hyperparams.json"


@dataclass(slots=True)
class HyperParameters:
    """Container for tunable hyperparameters."""

    learning_rate: float = 1e-4
    hidden_size: int = 768
    dropout: float = 0.1
    activation: str = "gelu"
    batch_size: int = 8

    def clamp(self) -> None:
        """Ensure values remain within safe ranges."""

        self.learning_rate = max(1e-6, min(self.learning_rate, 5e-3))
        self.hidden_size = max(128, min(self.hidden_size, 4096))
        self.dropout = max(0.0, min(self.dropout, 0.6))
        self.batch_size = max(1, min(self.batch_size, 64))
        if self.activation not in {"relu", "gelu", "silu", "mish"}:
            self.activation = "gelu"


class AutoTuner:
    """Analyzes metrics to adapt hyperparameters."""

    def __init__(
        self,
        tracker: MetricsTracker,
        settings: OptimizerSettings,
        apply_callback: Callable[[HyperParameters], None] | None = None,
        params_path: Path | None = None,
    ) -> None:
        self.tracker = tracker
        self.settings = settings
        self.apply_callback = apply_callback
        self.params_path = Path(params_path) if params_path else PARAMS_PATH
        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        self._configure_logger()

    async def run_cycle(self) -> HyperParameters:
        """Run a tuning pass asynchronously."""

        if not self.settings.enabled:
            LOGGER.info("Optimizer disabled; skipping tuning cycle.")
            return await asyncio.to_thread(self.load)
        params = await asyncio.to_thread(self.load)
        metrics = await asyncio.to_thread(self.tracker.get_recent, 5)
        if not metrics:
            LOGGER.info("No metrics available for tuning; keeping parameters unchanged.")
            return params
        updated = self._adjust(params, metrics)
        await asyncio.to_thread(self.save, updated)
        if self.apply_callback:
            await asyncio.to_thread(self.apply_callback, updated)
        LOGGER.info(
            "Tuning applied | lr=%.5f hidden=%d dropout=%.2f activation=%s batch=%d",
            updated.learning_rate,
            updated.hidden_size,
            updated.dropout,
            updated.activation,
            updated.batch_size,
        )
        return updated

    # ------------------------------------------------------------------
    def load(self) -> HyperParameters:
        """Load hyperparameters from disk or defaults."""

        if not self.params_path.exists():
            params = HyperParameters()
            params.clamp()
            self.save(params)
            return params
        payload = json.loads(self.params_path.read_text())
        params = HyperParameters(**payload)
        params.clamp()
        return params

    def save(self, params: HyperParameters) -> None:
        """Persist hyperparameters to disk."""

        params.clamp()
        self.params_path.write_text(json.dumps(asdict(params), indent=2))

    def _adjust(self, params: HyperParameters, metrics: list[MetricsSnapshot]) -> HyperParameters:
        latest = metrics[0]
        reward_trend = self.tracker.reward_trend(window=min(len(metrics), 5))
        avg_loss = sum(item.loss for item in metrics) / len(metrics)
        avg_grad = sum(item.grad_norm for item in metrics) / len(metrics)
        avg_latency = sum(item.latency_ms for item in metrics) / len(metrics)

        adjusted = HyperParameters(**asdict(params))

        if reward_trend < 0:
            adjusted.learning_rate *= 0.9
        else:
            adjusted.learning_rate *= 1.05

        if avg_loss > latest.loss * 1.05:
            adjusted.hidden_size = int(adjusted.hidden_size * 1.1)
        elif reward_trend < 0 and adjusted.hidden_size > 256:
            adjusted.hidden_size = int(adjusted.hidden_size * 0.9)

        if avg_grad > 1.0:
            adjusted.dropout = min(adjusted.dropout + 0.05, 0.6)
        elif reward_trend > 0.1:
            adjusted.dropout = max(adjusted.dropout - 0.02, 0.0)

        if avg_latency > 500 and adjusted.batch_size > 4:
            adjusted.batch_size = max(4, int(adjusted.batch_size * 0.8))
        elif reward_trend > 0.05:
            adjusted.batch_size = min(64, int(adjusted.batch_size * 1.1))

        if reward_trend < -0.1 or avg_loss > latest.loss:
            adjusted.activation = random.choice(["relu", "gelu", "silu", "mish"])

        adjusted.clamp()
        return adjusted

    def _configure_logger(self) -> None:
        LOGGER.setLevel(logging.INFO)
        if not LOGGER.handlers:
            handler = logging.FileHandler(ROOT_PATH / "storage" / "logs" / "optimizer.log", encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            LOGGER.addHandler(handler)
        LOGGER.propagate = False
