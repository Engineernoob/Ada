"""Text encoding utilities for Ada."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import torch

# Optional import for sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError):
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TextEncoder:
    """Converts text into fixed-size embeddings using a hashing trick."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            vector[self._token_to_index(token)] += 1.0
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm

    def batch_encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = [self.encode(text) for text in texts]
        return np.stack(embeddings, axis=0)

    def _tokenize(self, text: str) -> list[str]:
        return [token.strip() for token in text.lower().split() if token.strip()]

    def _token_to_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).hexdigest()
        return int(digest, 16) % self.dim


class LanguageEncoder:
    """Converts text into semantic embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available. Please install sentence-transformers package.")
        
        self.model_name = model_name
        self._model = None
        self._embedding_dim = None
    
    @property
    def model(self):
        if self._model is None:
            print(f"Loading SentenceTransformer model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            print("✅ Model loaded successfully")
        return self._model
    
    @property 
    def embedding_dim(self):
        if self._embedding_dim is None:
            # Access model property to trigger loading
            _ = self.model
        return self._embedding_dim
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text string into a tensor embedding."""
        embedding = self.model.encode([text], normalize_embeddings=True)
        return torch.tensor(embedding, dtype=torch.float32)
    
    def batch_encode(self, texts: Iterable[str]) -> torch.Tensor:
        """Encode multiple texts into tensor embeddings."""
        text_list = list(texts)
        embeddings = self.model.encode(text_list, normalize_embeddings=True)
        return torch.tensor(embeddings, dtype=torch.float32)
