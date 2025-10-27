"""Training utilities for AdaCore."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core import ConversationStore, ReasoningEngine
from rl import AdaAgent, ExperienceBuffer

from .encoder import TextEncoder, LanguageEncoder
from .policy_network import AdaCore, get_device, save_model

CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "storage" / "checkpoints" / "ada_core.pt"
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
RL_BATCH_SIZE = 8
RL_EPISODES = 20


def _synthetic_corpus() -> Iterable[tuple[str, str]]:
    return [
        ("hello ada", "greeting detected"),
        ("how are you", "status acknowledgement"),
        ("describe your goals", "assist and learn"),
        ("thank you", "gratitude returned"),
    ]


def _build_dataset(encoder: LanguageEncoder) -> TensorDataset:
    inputs, targets = [], []
    for question, answer in _synthetic_corpus():
        inputs.append(encoder.encode(question))
        targets.append(encoder.encode(answer))
    # LanguageEncoder already returns tensors, TextEncoder returns numpy arrays
    if hasattr(encoder, 'model'):
        # LanguageEncoder - already tensors
        inputs_tensor = torch.stack(inputs, dim=0)
        targets_tensor = torch.stack(targets, dim=0)
    else:
        # TextEncoder - numpy arrays
        inputs_tensor = torch.tensor(np.array(inputs, dtype=np.float32), dtype=torch.float32)
        targets_tensor = torch.tensor(np.array(targets, dtype=np.float32), dtype=torch.float32)
    return TensorDataset(inputs_tensor, targets_tensor)


def train() -> Path:
    device = get_device()
    encoder = LanguageEncoder()
    dataset = _build_dataset(encoder)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AdaCore().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    save_model(model.to(torch.device("cpu")), CHECKPOINT_PATH)
    return CHECKPOINT_PATH


def _build_replay_from_store(store: ConversationStore, encoder: LanguageEncoder, engine: ReasoningEngine) -> Sequence[tuple[torch.Tensor, int, float, torch.Tensor]]:
    records = list(store.fetch_recent(limit=200))
    if not records:
        return []
    records = list(reversed(records))
    transitions = []
    for idx, record in enumerate(records):
        state_text = record.user_input or ""
        next_state_text = records[idx + 1].user_input if idx + 1 < len(records) else ""
        # LanguageEncoder already gives tensors, TextEncoder gives numpy arrays
        if hasattr(encoder, 'model'):
            state_vec = encoder.encode(state_text)
            next_state_vec = encoder.encode(next_state_text)
        else:
            state_vec = torch.tensor(encoder.encode(state_text), dtype=torch.float32)
            next_state_vec = torch.tensor(encoder.encode(next_state_text), dtype=torch.float32)
        action_index = engine.text_to_action(record.ada_response)
        transitions.append((state_vec, action_index, float(record.reward), next_state_vec))
    return transitions


def train_reinforcement(episodes: int = RL_EPISODES, batch_size: int = RL_BATCH_SIZE) -> Path:
    store = ConversationStore()
    encoder = LanguageEncoder()
    engine = ReasoningEngine(checkpoint_path=CHECKPOINT_PATH)
    buffer = ExperienceBuffer()

    transitions = _build_replay_from_store(store, encoder, engine)
    if not transitions:
        print("No conversation data available for RL training. Collect feedback first.")
        return CHECKPOINT_PATH

    for state_vec, action_index, reward, next_state_vec in transitions:
        buffer.append(state_vec, action_index, reward, next_state_vec)

    agent = AdaAgent(model=engine.model, memory=buffer, action_space=engine.action_space, checkpoint_path=engine.checkpoint_path, lr=1e-5)

    print(f"Loaded {len(buffer)} transitions for RL fine-tuning.")
    for episode in range(1, episodes + 1):
        metrics = agent.train_on_batch(batch_size=batch_size)
        if not metrics:
            print("Insufficient transitions for a full batch; stopping early.")
            break
        loss, reward = metrics
        print(f"[Episode {episode}] Reward Avg: {reward:+.2f} | Loss: {loss:.4f}")

    agent.save()
    return engine.checkpoint_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ada training utilities")
    parser.add_argument("--mode", choices=["pretrain", "rl"], default="pretrain")
    parser.add_argument("--episodes", type=int, default=RL_EPISODES, help="Episodes for RL fine-tuning")
    parser.add_argument("--batch-size", type=int, default=RL_BATCH_SIZE, help="Batch size for RL fine-tuning")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "rl":
        path = train_reinforcement(episodes=args.episodes, batch_size=args.batch_size)
    else:
        path = train()
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
