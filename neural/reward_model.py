"""Reward shaping utilities for Ada's RL loop."""

from __future__ import annotations

import math
from typing import Iterable


def simple_reward(user_input: str, response: str) -> float:
    coverage = _coverage_score(user_input, response)
    coherence = _coherence_score(response)
    return 0.6 * coverage + 0.4 * coherence


def _coverage_score(question: str, answer: str) -> float:
    question_tokens = set(question.lower().split())
    answer_tokens = set(answer.lower().split())
    if not question_tokens:
        return 0.0
    overlap = len(question_tokens & answer_tokens)
    return overlap / len(question_tokens)


def _coherence_score(text: str) -> float:
    length_penalty = math.exp(-abs(len(text) - 30) / 30)
    token_penalty = math.exp(-max(0, text.count("??")))
    return 0.5 * length_penalty + 0.5 * token_penalty


def batch_reward(samples: Iterable[tuple[str, str]]) -> list[float]:
    return [simple_reward(inp, out) for inp, out in samples]
