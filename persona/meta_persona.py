"""MetaPersona module for Ada's persona formation and identity consolidation."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

import yaml

def get_setting(*keys: str, default=None):
    """Local copy of get_setting to avoid circular imports."""
    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    try:
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        
        node = settings
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node
    except (FileNotFoundError, yaml.YAMLError):
        return default

# Optional import for sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError):
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class PersonaStats:
    """Statistics about Ada's current persona."""
    
    def __init__(self, tone: str, phrasing: str, drift: float, update_time: float):
        self.tone = tone
        self.phrasing = phrasing
        self.drift = drift
        self.update_time = update_time
    
    def to_dict(self) -> dict:
        return {
            "tone": self.tone,
            "phrasing": self.phrasing,
            "drift": self.drift,
            "update_time": self.update_time,
            "last_update": time.ctime(self.update_time)
        }


class MetaPersona:
    """Manages Ada's persona through embedding aggregation and style biasing."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persona_path: Optional[Path] = None) -> None:
        if persona_path is None:
            persona_path = Path(get_setting("paths", "checkpoints", default="storage/checkpoints")) / "persona_vector.pt"
        
        self.persona_path = persona_path
        self.persona_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer(model_name)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            # Fallback to simpler embedding if sentence transformers not available
            self.encoder = None
            self.embedding_dim = 384  # Default dimension
        
        self.vector: Optional[torch.Tensor] = None
        self.previous_vector: Optional[torch.Tensor] = None
        self.last_update_time: float = 0
        self.drift_threshold: float = get_setting("persona", "drift_threshold", default=0.3)
        self.bias_weight: float = get_setting("persona", "bias_weight", default=0.2)
        
        # Persona analysis cache
        self._persona_stats: Optional[PersonaStats] = None
        
    def load(self) -> bool:
        """Load existing persona vector from disk."""
        try:
            self.vector = torch.load(self.persona_path, weights_only=True)
            self.last_update_time = self.persona_path.stat().st_mtime
            return True
        except (FileNotFoundError, RuntimeError):
            # Initialize to zeros if no existing persona
            if self.encoder is not None:
                self.vector = torch.zeros(self.embedding_dim, dtype=torch.float32)
            else:
                self.vector = torch.zeros(384, dtype=torch.float32)
            return False
    
    def save(self) -> None:
        """Save current persona vector to disk."""
        if self.vector is not None:
            torch.save(self.vector, self.persona_path)
            self.last_update_time = time.time()
            self._persona_stats = None  # Clear cache
    
    def build_from_transcripts(self, ada_responses: List[str]) -> torch.Tensor:
        """Build persona vector from Ada's response texts."""
        if not ada_responses:
            return self.vector or torch.zeros(self.embedding_dim, dtype=torch.float32)
        
        # Store previous vector for drift calculation
        self.previous_vector = self.vector.clone() if self.vector is not None else None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.encoder is not None:
            # Use sentence transformers for semantic embeddings
            embeddings = self.encoder.encode(ada_responses, normalize_embeddings=True)
            new_vector = torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float32)
        else:
            # Fallback to simple text hashing approach
            try:
                from ..neural.encoder import TextEncoder
                text_encoder = TextEncoder(dim=self.embedding_dim)
                embeddings = [text_encoder.encode(response) for response in ada_responses]
                new_vector = torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float32)
            except ImportError:
                # Even further fallback - simple hashing
                import hashlib
                embeddings = []
                for response in ada_responses:
                    vector = np.zeros(self.embedding_dim, dtype=np.float32)
                    for token in response.lower().split():
                        digest = hashlib.blake2b(token.encode('utf-8'), digest_size=4).hexdigest()
                        idx = int(digest, 16) % self.embedding_dim
                        vector[idx] += 1.0
                    # Normalize
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                    embeddings.append(vector)
                new_vector = torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float32)
        
        # Normalize the vector
        new_vector = new_vector / (torch.norm(new_vector) + 1e-8)
        self.vector = new_vector
        self.save()
        
        return self.vector
    
    def apply_bias(self, embedding: torch.Tensor, weight: Optional[float] = None) -> torch.Tensor:
        """Apply persona bias to an embedding."""
        if self.vector is None:
            self.load()
        
        if self.vector is None:
            return embedding
        
        bias_weight = weight if weight is not None else self.bias_weight
        
        # Blend the embedding with persona vector
        biased_embedding = (1 - bias_weight) * embedding + bias_weight * self.vector
        return biased_embedding / (torch.norm(biased_embedding) + 1e-8)
    
    def calculate_drift(self) -> float:
        """Calculate drift between current and previous persona vectors."""
        if self.previous_vector is None or self.vector is None:
            return 0.0
        
        # Calculate cosine similarity and convert to drift
        similarity = cosine_similarity(self.vector.unsqueeze(0), self.previous_vector.unsqueeze(0)).item()
        drift = 1.0 - similarity
        return float(drift)
    
    def needs_update(self, dialog_count: int) -> bool:
        """Check if persona should be updated based on dialog count."""
        update_interval = get_setting("persona", "update_interval", default=20)
        return dialog_count >= update_interval and get_setting("persona", "enabled", default=True)
    
    def analyze_persona(self) -> PersonaStats:
        """Analyze current persona to generate human-readable description."""
        if self._persona_stats is not None:
            return self._persona_stats
        
        # Determine tone based on sentiment patterns
        tone = self._analyze_tone()
        
        # Determine phrasing style
        phrasing = self._analyze_phrasing()
        
        # Calculate drift
        drift = self.calculate_drift()
        
        # Create stats object
        self._persona_stats = PersonaStats(
            tone=tone,
            phrasing=phrasing,
            drift=drift,
            update_time=self.last_update_time
        )
        
        return self._persona_stats
    
    def _analyze_tone(self) -> str:
        """Analyze tone patterns from persona vector components."""
        if self.vector is None:
            return "neutral, developing"
        
        # Simple heuristic analysis based on vector characteristics
        norm = torch.norm(self.vector).item()
        
        # Check for patterns that might indicate tone (simplified heuristic)
        if norm > 0.8:
            if self.vector.mean().item() > 0.3:
                return "warm, confident"
            else:
                return "analytical, measured"
        elif norm < 0.5:
            return "cautious, developing"
        else:
            return "balanced, adaptive"
    
    def _analyze_phrasing(self) -> str:
        """Analyze phrasing style from persona vector components."""
        if self.vector is None:
            return "simple, direct"
        
        # Simple heuristic analysis of vector variance
        variance = torch.var(self.vector).item()
        
        if variance > 0.1:
            return "varied, expressive"
        elif variance < 0.05:
            return "concise, consistent"
        else:
            return "moderated, thoughtful"
    
    def get_persona_summary(self) -> str:
        """Get a human-readable summary of Ada's current persona."""
        stats = self.analyze_persona()
        
        summary_lines = [
            "Ada Persona Summary:",
            f"Tone: {stats.tone}",
            f"Phrasing: {stats.phrasing}",
            f"Drift: {stats.drift:.3f} since last update",
        ]
        
        # Add drift warning if needed
        if stats.drift > self.drift_threshold:
            summary_lines.append("⚠️  High drift detected - persona may need recalibration")
        
        return "\n".join(summary_lines)
    
    def reset_persona(self) -> None:
        """Reset persona to neutral state."""
        if self.encoder is not None:
            self.vector = torch.zeros(self.embedding_dim, dtype=torch.float32)
        else:
            self.vector = torch.zeros(384, dtype=torch.float32)
        self.previous_vector = None
        self._persona_stats = None
        self.save()
