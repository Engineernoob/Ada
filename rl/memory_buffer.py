"""Experience replay memory for Ada's reinforcement learning loop."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List

import torch


@dataclass
class Transition:
    state: torch.Tensor
    action_index: int
    reward: float
    next_state: torch.Tensor

    def to(self, device: torch.device) -> "Transition":
        return Transition(
            state=self.state.to(device),
            action_index=self.action_index,
            reward=self.reward,
            next_state=self.next_state.to(device),
        )


@dataclass
class ExperienceBuffer:
    capacity: int = 512

    def __post_init__(self) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def append(self, state: torch.Tensor, action_index: int, reward: float, next_state: torch.Tensor) -> None:
        state_cpu = state.detach().to(torch.float32).cpu()
        next_state_cpu = next_state.detach().to(torch.float32).cpu()
        transition = Transition(state=state_cpu, action_index=action_index, reward=reward, next_state=next_state_cpu)
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        batch_size = min(batch_size, len(self._buffer))
        if batch_size == 0:
            return []
        if batch_size == len(self._buffer):
            return list(self._buffer)
        return random.sample(list(self._buffer), k=batch_size)

    def latest(self) -> Iterable[Transition]:
        if not self._buffer:
            return []
        return [self._buffer[-1]]

    def __len__(self) -> int:
        return len(self._buffer)
