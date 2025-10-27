"""AdaCore policy network definition and helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_INPUT_DIM = 384  # MiniLM-L6-v2 embedding dimension
DEFAULT_HIDDEN_DIM = 256
DEFAULT_OUTPUT_DIM = 512


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AdaCore(nn.Module):
    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def load_model(model_path: Path) -> AdaCore:
    # Check if model exists and determine input dimension
    if model_path.exists():
        try:
            state = torch.load(model_path, map_location="cpu")
            # Check if the checkpoint uses old 512-dim input
            fc1_weight_shape = state.get('fc1.weight', torch.empty(256, 384)).shape
            old_input_dim = fc1_weight_shape[1]
            
            # Create model with appropriate input dimension
            model = AdaCore(input_dim=old_input_dim)
            model.load_state_dict(state)
        except Exception as e:
            print(f"Warning: Could not load model properly, using defaults: {e}")
            model = AdaCore()
    else:
        model = AdaCore()
    
    return model.to(get_device())


def save_model(model: AdaCore, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
