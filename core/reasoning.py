"""Reasoning engine orchestrating Ada's response flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import numpy as np
import torch
import torch.nn.functional as F

from memory.episodic_store import EpisodicStore
from .settings import get_setting

# Import persona module locally to avoid circular imports
try:
    from persona.meta_persona import MetaPersona

    PERSONA_AVAILABLE = True
except ImportError:
    PERSONA_AVAILABLE = False
    MetaPersona = None

# Disable neural modules completely due to transformers issues
NEURAL_AVAILABLE = False
print("Note: Using fallback mode due to transformers dependency issues")

# Create minimal fallbacks with consistent dimensions
DEFAULT_embedding_DIM = 384  # Match expected input dimension


class TextEncoder:
    def __init__(self, dim=None):
        # Use consistent dimension to match fallback model expectations
        self.dim = dim or DEFAULT_embedding_DIM

    def encode(self, text):
        import hashlib
        import numpy as np

        tokens = text.lower().split()
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).hexdigest()
            idx = int(digest, 16) % self.dim
            vector[idx] += 1.0
        # Convert to list to maintain consistency
        return vector.tolist()


LanguageEncoder = None
AdaCore = None


# Safe fallback functions that avoid model loading
def get_device():
    import torch

    return torch.device("cpu")


def load_model(path):
    # Always return None to force fallback model usage
    return None


def simple_reward(a, b):
    return 0.5


PHRASES = [
    "neural core initialized and ready to learn.",
    "processing your ideas with curiosity.",
    "reflecting on our shared objectives.",
    "analyzing pathways for improvement.",
]


@dataclass
class GenerationResult:
    text: str
    action_index: int
    probabilities: torch.Tensor
    confidence: float
    entropy: float
    logits: torch.Tensor
    memory_id: int = 0


class RewardEngine:
    positive_tokens = {"good", "+1", "up", "yes", "positive"}
    negative_tokens = {"bad", "-1", "down", "no", "negative"}

    def compute(
        self, user_feedback: Optional[str], user_input: str, ada_output: str
    ) -> float:
        if user_feedback:
            token = user_feedback.strip().lower()
            if token in self.positive_tokens:
                return 1.0
            if token in self.negative_tokens:
                return -1.0
        return float(simple_reward(user_input, ada_output))


class ReasoningEngine:
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        use_language_encoder: bool = True,
        use_memory: bool = True,
        use_persona: bool = True,
    ) -> None:
        # Use language encoder by default for better semantic understanding
        if use_language_encoder and LanguageEncoder is not None:
            self.encoder = LanguageEncoder()
        else:
            self.encoder = TextEncoder()

        self.device = get_device()
        default_path = (
            Path(__file__).resolve().parents[1]
            / "storage"
            / "checkpoints"
            / "ada_core.pt"
        )
        self.checkpoint_path = checkpoint_path or default_path

        # Handle model loading with fallback
        if AdaCore is not None:
            try:
                self.model: AdaCore = load_model(self.checkpoint_path)
                self.model.eval()
            except (FileNotFoundError, OSError, RuntimeError) as e:
                print(f"Warning: Could not load model file {self.checkpoint_path}: {e}")
                print("Using fallback model instead.")
                self.model = self._create_fallback_model()
        else:
            print("Warning: Using fallback model due to neural module unavailability")
            self.model = self._create_fallback_model()

        self.phrases = PHRASES

        # Initialize episodic memory
        self.use_memory = use_memory and get_setting(
            "memory", "episodic_enabled", default=True
        )
        if self.use_memory:
            self.memory_store = EpisodicStore()
        else:
            self.memory_store = None

        # Initialize persona system
        self.use_persona = use_persona and get_setting(
            "persona", "enabled", default=True
        )
        if self.use_persona and PERSONA_AVAILABLE:
            self.persona = MetaPersona()
            self.persona.load()
        else:
            self.persona = None

    @property
    def action_space(self) -> int:
        return len(self.phrases)

    def generate(self, user_input: str) -> str:
        return self.generate_with_metrics(user_input).text

    def generate_with_metrics(self, user_input: str) -> GenerationResult:
        # Get base embedding for storage (original input, not context)
        if hasattr(self.encoder, "model"):  # LanguageEncoder
            original_embedding = self.encoder.encode(user_input)
        else:  # TextEncoder
            embedding_array = self.encoder.encode(user_input)
            original_embedding = torch.tensor(embedding_array, dtype=torch.float32)

        # Retrieve relevant memories if enabled
        context_input = user_input
        if self.use_memory and self.memory_store:
            context_text = self.memory_store.get_context_string(
                original_embedding,
                max_tokens=get_setting("memory", "max_context_tokens", default=100),
            )
            if context_text:
                context_input = f"{context_text}\n\nYou: {user_input}"
                # Re-encode with context
                if hasattr(self.encoder, "model"):  # LanguageEncoder
                    embedding = self.encoder.encode(context_input).to(self.device)
                else:  # TextEncoder
                    embedding = torch.tensor(
                        self.encoder.encode(context_input),
                        dtype=torch.float32,
                        device=self.device,
                    )
            else:
                embedding = original_embedding.to(self.device)
        else:
            # Ensure the embedding is properly shaped
            if original_embedding.dim() == 1:
                embedding = original_embedding.to(self.device)
            else:
                embedding = original_embedding.flatten().to(self.device)

        # Apply persona bias if enabled
        if self.use_persona and self.persona:
            embedding = self.persona.apply_bias(embedding)

        with torch.no_grad():
            logits = self.model(embedding).squeeze(0)
        action_logits = logits[: self.action_space]
        probabilities = F.softmax(action_logits, dim=-1)
        action_index = int(torch.argmax(probabilities).item())
        entropy = self._entropy(probabilities)
        confidence = 1.0 - entropy / max(np.log(self.action_space), 1e-9)
        text = self.phrases[action_index]

        # Store the conversation in memory
        memory_id = 0
        if self.use_memory and self.memory_store:
            memory_id = self.memory_store.store(
                user_input, text, 0.0, original_embedding
            )

        return GenerationResult(
            text=text,
            action_index=action_index,
            probabilities=probabilities.detach().cpu(),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            entropy=entropy,
            logits=logits.detach().cpu(),
            memory_id=memory_id,
        )

    def reward(self, user_input: str, response: str) -> float:
        return float(simple_reward(user_input, response))

    def text_to_action(self, text: str) -> int:
        try:
            return self.phrases.index(text)
        except ValueError:
            return 0

    def _entropy(self, probabilities: torch.Tensor) -> float:
        probs = probabilities.clamp_min(1e-9)
        entropy = -(probs * probs.log()).sum().item()
        return float(entropy)

    def get_memory_stats(self) -> dict:
        """Get episodic memory statistics."""
        if not self.use_memory or not self.memory_store:
            return {"memory_enabled": False}

        stats = self.memory_store.get_stats()
        stats["memory_enabled"] = True
        return stats

    def update_memory_reward(self, memory_id: int, reward: float) -> bool:
        """Update reward for a specific memory record."""
        if not self.use_memory or not self.memory_store:
            return False

        try:
            self.memory_store.update_reward(memory_id, reward)
            return True
        except Exception:
            return False

    def cleanup_memory(self, days: int = 30) -> int:
        """Clean up old memories."""
        if not self.use_memory or not self.memory_store:
            return 0

        return self.memory_store.cleanup_old_memories(days)

    def get_persona_stats(self) -> dict:
        """Get persona system statistics."""
        if not self.use_persona or not self.persona:
            return {"persona_enabled": False}

        persona_stats = self.persona.analyze_persona()
        return {
            "persona_enabled": True,
            "tone": persona_stats.tone,
            "phrasing": persona_stats.phrasing,
            "drift": persona_stats.drift,
            "bias_weight": self.persona.bias_weight,
            "last_update": (
                time.ctime(persona_stats.update_time)
                if persona_stats.update_time > 0
                else "Never"
            ),
        }

    def update_persona_from_memories(self, limit: int = 50) -> bool:
        """Update persona from recent conversation memories."""
        if not self.use_persona or not self.persona or not self.memory_store:
            return False

        recent_memories = self.memory_store.get_recent(limit)
        if not recent_memories:
            return False

        # Extract Ada's responses from recent memories
        ada_responses = [memory.ada_response for memory in recent_memories]

        # Build new persona vector
        self.persona.build_from_transcripts(ada_responses)
        return True

    def reset_persona(self) -> bool:
        """Reset persona to neutral state."""
        if not self.use_persona or not self.persona:
            return False

        self.persona.reset_persona()
        return True

    def _create_fallback_model(self):
        """Create a minimal fallback model when neural models are unavailable."""
        
        import numpy as np

        class FallbackModel:
            def __init__(self):
                self.parameters = None
                self.training = False

            def __call__(self, x):
                # Simple rule-based responses when neural model unavailable
                # We don't actually use the input for generation in fallback mode
                # Just generate based on the input's existence
                logits = np.random.random(len(PHRASES))

                # Add some smarter bias based on input content
                if hasattr(x, "numel"):
                    input_size = x.numel()
                else:
                    input_size = len(x) if hasattr(x, "__len__") else 512

                # Bias response based on input size and content
                if input_size > 500:
                    logits[0] += 0.3  # Favor first phrase for longer inputs
                elif input_size > 200:
                    logits[1] += 0.2
                else:
                    logits[2] += 0.1

                return torch.tensor(logits, dtype=torch.float32)

            def eval(self):
                self.training = False

            def to(self, device):
                # Fallback model works on any device
                return self

            def train(self, mode=True):
                self.training = mode

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict, strict=True):
                pass

        return FallbackModel()
