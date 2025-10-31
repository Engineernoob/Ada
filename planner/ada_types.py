"""Core types for the planner module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Intent:
    goal: str
    priority: float
    category: str
    confidence: float
