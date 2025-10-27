"""Reinforcement learning agent driving Ada's adaptive behaviour."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from neural.policy_network import AdaCore, get_device, save_model

from .memory_buffer import ExperienceBuffer, Transition


@dataclass
class AdaAgent:
    model: AdaCore
    memory: ExperienceBuffer
    action_space: int
    lr: float = 5e-5
    gamma: float = 0.95
    checkpoint_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "storage" / "checkpoints" / "ada_core.pt"
    )
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        self.device = get_device()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state)
        action_logits = logits[0, : self.action_space]
        probabilities = F.softmax(action_logits, dim=-1)
        action_index = int(torch.argmax(probabilities).item())
        return action_index, probabilities.detach().cpu()

    def update_policy(self, state: torch.Tensor, action_index: int, reward: float, next_state: torch.Tensor) -> Optional[Tuple[float, float]]:
        self.memory.append(state, action_index, reward, next_state)
        transitions = list(self.memory.latest())
        if not transitions:
            return None
        return self._optimize(transitions)

    def train_on_batch(self, batch_size: int = 4) -> Optional[Tuple[float, float]]:
        transitions = self.memory.sample(batch_size)
        if not transitions:
            return None
        return self._optimize(transitions)

    def save(self) -> None:
        save_model(self.model.to(torch.device("cpu")), self.checkpoint_path)
        self.model.to(self.device)

    def _optimize(self, transitions: Iterable[Transition]) -> Tuple[float, float]:
        self.model.train()
        states = torch.stack([t.state for t in transitions]).to(self.device)
        actions = torch.tensor([t.action_index for t in transitions], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)

        logits = self.model(states)
        action_logits = logits[:, : self.action_space]
        log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = -(rewards * selected_log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()

        return loss.item(), rewards.mean().item()
