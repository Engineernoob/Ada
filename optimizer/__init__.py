"""Optimization and self-evolution utilities for Ada."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.settings import get_setting

__all__ = [
    "OptimizerSettings",
]


@dataclass(slots=True)
class OptimizerSettings:
    """Runtime configuration for the optimizer subsystem."""

    enabled: bool = True
    evaluation_interval_hours: int = 6
    max_population: int = 5
    mutation_rate: float = 0.15
    selection_top_k: int = 2
    preserve_best: bool = True
    rollback_safe: bool = True

    @classmethod
    def from_settings(cls) -> "OptimizerSettings":
        node = get_setting("optimizer", default={}) or {}
        return cls(
            enabled=bool(node.get("enabled", True)),
            evaluation_interval_hours=int(node.get("evaluation_interval_hours", 6)),
            max_population=int(node.get("max_population", 5)),
            mutation_rate=float(node.get("mutation_rate", 0.15)),
            selection_top_k=int(node.get("selection_top_k", 2)),
            preserve_best=bool(node.get("preserve_best", True)),
            rollback_safe=bool(node.get("rollback_safe", True)),
        )


ROOT_PATH = Path(__file__).resolve().parents[1]
"""Base path for optimizer resources."""
