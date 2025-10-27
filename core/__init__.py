"""Core reasoning modules for Ada.

This package historically exposed the reasoning engine classes at import time.
Those modules depend on optional third-party libraries (``numpy``/``torch``),
which aren't always installed in lightweight environments.  Importing them
eagerly caused unrelated functionality—such as the planner CLI—to crash before
it even started because simply doing ``import core`` tried to import the heavy
reasoning stack.

To make the package robust we now expose the reasoning symbols lazily.  The
lightweight modules remain available immediately, but ``ReasoningEngine`` and
its companions are only imported the first time they're accessed.  This keeps
existing import sites working while avoiding the optional dependency failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .memory import ConversationStore
from .context_manager import ContextManager
from .settings import get_setting, load_settings
from .autonomous_planner import AutonomousPlanner

if TYPE_CHECKING:  # pragma: no cover - only used for type checkers
    from .reasoning import GenerationResult, ReasoningEngine, RewardEngine

__all__ = [
    "ConversationStore",
    "ContextManager",
    "ReasoningEngine",
    "GenerationResult",
    "RewardEngine",
    "get_setting",
    "load_settings",
    "AutonomousPlanner",
]


def __getattr__(name: str):
    """Lazily expose reasoning module symbols when requested.

    Importing the reasoning stack is expensive and may fail if optional
    dependencies are missing.  Deferring the import keeps light-weight tools
    usable without pulling those dependencies, while still supporting the
    public API ``from core import ReasoningEngine`` when needed.
    """

    if name in {"GenerationResult", "ReasoningEngine", "RewardEngine"}:
        from .reasoning import GenerationResult, ReasoningEngine, RewardEngine

        mapping = {
            "GenerationResult": GenerationResult,
            "ReasoningEngine": ReasoningEngine,
            "RewardEngine": RewardEngine,
        }
        return mapping[name]
    raise AttributeError(f"module 'core' has no attribute {name!r}")
