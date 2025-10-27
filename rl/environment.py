"""Dialogue environment abstractions supporting Ada's RL loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import torch

from neural.encoder import TextEncoder


@dataclass
class DialogueState:
    history: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        return "\n".join(self.history).strip()


class DialogueEnvironment:
    def __init__(self, max_history: int = 4, encoder: TextEncoder | None = None) -> None:
        self.max_history = max_history * 2  # user/ada pairs
        self.encoder = encoder or TextEncoder()
        self._history: Deque[str] = deque(maxlen=self.max_history)
        self._state = DialogueState()

    @property
    def state(self) -> DialogueState:
        return self._state

    def reset(self) -> DialogueState:
        self._history.clear()
        self._state = DialogueState()
        return self._state

    def observe(self, user_input: str, ada_response: str) -> DialogueState:
        self._history.append(f"You: {user_input}")
        self._history.append(f"Ada: {ada_response}")
        self._state = DialogueState(history=list(self._history))
        return self._state

    def encode(self, state: DialogueState | None = None) -> torch.Tensor:
        target_state = state or self._state
        text = target_state.as_text()
        if not text:
            return torch.zeros(self.encoder.dim, dtype=torch.float32)
        return torch.tensor(self.encoder.encode(text), dtype=torch.float32)
