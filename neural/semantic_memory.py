"""FAISS-based semantic memory for Ada's contextual recall system."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS or sentence-transformers not available. Semantic memory disabled.")

logger = logging.getLogger(__name__)


class SemanticMemory:
    """FAISS-based semantic memory for contextual recall."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        storage_path: Optional[Path] = None,
        max_memories: int = 10000
    ):
        """Initialize semantic memory system.
        
        Args:
            model_name: Sentence transformer model name
            embedding_dim: Dimension of embeddings
            storage_path: Path to persist memory index
            max_memories: Maximum number of memories to store
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS and sentence-transformers required for semantic memory")
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.storage_path = storage_path or Path("storage/memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Memory store (parallel to FAISS index)
        self.memory_store: List[str] = []
        
        # Load existing memories if available
        self._load_memories()
    
    def add_memory(self, text: str) -> None:
        """Add text to semantic memory.
        
        Args:
            text: Text to store in memory
        """
        if not text or len(text.strip()) < 3:
            return
        
        # Check if we've exceeded max memories
        if len(self.memory_store) >= self.max_memories:
            logger.warning(f"Memory limit reached ({self.max_memories}). Removing oldest memories.")
            self._remove_oldest_memories(count=100)
        
        # Encode text
        embedding = self.encoder.encode([text], convert_to_numpy=True)
        
        # Add to FAISS index
        self.index.add(embedding.astype('float32'))
        
        # Add to memory store
        self.memory_store.append(text)
        
        logger.debug(f"Added memory: {text[:50]}...")
    
    def recall(self, query: str, k: int = 3) -> List[str]:
        """Recall similar memories based on query.
        
        Args:
            query: Query text to search for similar memories
            k: Number of top memories to return
            
        Returns:
            List of similar memory texts
        """
        if not self.memory_store:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        k = min(k, len(self.memory_store))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Retrieve memories
        recalled = []
        for idx in indices[0]:
            if 0 <= idx < len(self.memory_store):
                recalled.append(self.memory_store[idx])
        
        logger.debug(f"Recalled {len(recalled)} memories for query: {query[:50]}...")
        return recalled
    
    def recall_with_scores(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Recall memories with similarity scores.
        
        Args:
            query: Query text to search for similar memories
            k: Number of top memories to return
            
        Returns:
            List of tuples (memory_text, similarity_score)
        """
        if not self.memory_store:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        k = min(k, len(self.memory_store))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Retrieve memories with scores (convert L2 distance to similarity)
        recalled = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.memory_store):
                # Convert L2 distance to similarity score (0-1 range)
                similarity = 1.0 / (1.0 + dist)
                recalled.append((self.memory_store[idx], float(similarity)))
        
        return recalled
    
    def clear_memory(self) -> None:
        """Clear all memories."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.memory_store = []
        logger.info("Cleared all semantic memories")
    
    def _remove_oldest_memories(self, count: int = 100) -> None:
        """Remove oldest memories to free up space.
        
        Args:
            count: Number of memories to remove
        """
        if count >= len(self.memory_store):
            self.clear_memory()
            return
        
        # Remove from memory store
        self.memory_store = self.memory_store[count:]
        
        # Rebuild FAISS index (FAISS doesn't support efficient deletion)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        if self.memory_store:
            embeddings = self.encoder.encode(self.memory_store, convert_to_numpy=True)
            self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Removed {count} oldest memories")
    
    def save(self) -> None:
        """Persist memories to disk."""
        try:
            # Save FAISS index
            index_path = self.storage_path / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save memory store
            store_path = self.storage_path / "memory_store.pkl"
            with open(store_path, 'wb') as f:
                pickle.dump(self.memory_store, f)
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'memory_count': len(self.memory_store)
            }
            metadata_path = self.storage_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {len(self.memory_store)} memories to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def _load_memories(self) -> None:
        """Load memories from disk."""
        try:
            index_path = self.storage_path / "faiss.index"
            store_path = self.storage_path / "memory_store.pkl"
            metadata_path = self.storage_path / "metadata.json"
            
            if not index_path.exists() or not store_path.exists():
                logger.info("No existing memories found. Starting fresh.")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load memory store
            with open(store_path, 'rb') as f:
                self.memory_store = pickle.load(f)
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded {metadata.get('memory_count', 0)} memories")
            
        except Exception as e:
            logger.warning(f"Failed to load memories: {e}. Starting fresh.")
            self.clear_memory()
    
    def get_stats(self) -> dict:
        """Get memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        return {
            'total_memories': len(self.memory_store),
            'max_memories': self.max_memories,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'storage_path': str(self.storage_path)
        }


# Global memory instance (lazy initialization)
_global_memory: Optional[SemanticMemory] = None


def get_global_memory() -> SemanticMemory:
    """Get or create global semantic memory instance.
    
    Returns:
        Global semantic memory instance
    """
    global _global_memory
    if _global_memory is None:
        _global_memory = SemanticMemory()
    return _global_memory


def add_memory(text: str) -> None:
    """Add text to global semantic memory.
    
    Args:
        text: Text to store in memory
    """
    memory = get_global_memory()
    memory.add_memory(text)


def recall(query: str, k: int = 3) -> List[str]:
    """Recall similar memories from global memory.
    
    Args:
        query: Query text to search for similar memories
        k: Number of top memories to return
        
    Returns:
        List of similar memory texts
    """
    memory = get_global_memory()
    return memory.recall(query, k)


def save_memory() -> None:
    """Save global memory to disk."""
    if _global_memory is not None:
        _global_memory.save()


# Auto-save on module exit
import atexit
atexit.register(save_memory)
