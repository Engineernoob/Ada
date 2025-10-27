"""Episodic memory store for Ada's conversational recall."""

from __future__ import annotations

import sqlite3
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

# Suppress the non-writable tensor warning
warnings.filterwarnings("ignore", message=".*not writeable.*does not support non-writable tensors.*", category=UserWarning)

import yaml

def get_setting(*keys: str, default=None):
    """Local copy of get_setting to avoid circular imports."""
    from pathlib import Path
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


class MemoryRecord:
    """Represents a single memory record."""
    
    def __init__(self, id: int, user_input: str, ada_response: str, reward: float, 
                 embedding: torch.Tensor, timestamp: float):
        self.id = id
        self.user_input = user_input
        self.ada_response = ada_response
        self.reward = reward
        self.embedding = embedding
        self.timestamp = timestamp


class EpisodicStore:
    """Stores and retrieves conversational episodes using semantic similarity."""
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = Path(get_setting("memory", "db_path", default="storage/conversations.db"))
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()
        
    def create_table(self) -> None:
        """Create the memory table if it doesn't exist."""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            ada_response TEXT NOT NULL,
            reward REAL DEFAULT 0.0,
            embedding BLOB NOT NULL,
            timestamp REAL NOT NULL,
            session_id TEXT
        )
        """)
        
        # Create indexes for better performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodic_memory(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_reward ON episodic_memory(reward)")
        self.conn.commit()
    
    def store(self, user_input: str, ada_response: str, reward: float, 
              embedding, session_id: Optional[str] = None) -> int:
        """
        Store a conversational exchange with its embedding.
        
        Args:
            user_input: What the user said
            ada_response: What Ada replied  
            reward: Reward value for the exchange
            embedding: Language embedding of user input (torch.Tensor or list/array)
            session_id: Optional session identifier
            
        Returns:
            The ID of the stored record
        """
        # Convert to tensor if needed
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Convert tensor to bytes for storage
        emb_bytes = embedding.cpu().numpy().astype(np.float32).tobytes()
        timestamp = time.time()
        
        cursor = self.conn.execute(
            "INSERT INTO episodic_memory (user_input, ada_response, reward, embedding, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_input, ada_response, reward, emb_bytes, timestamp, session_id)
        )
        self.conn.commit()
        return cursor.lastrowid or 0
    
    def retrieve(self, query_embedding, top_k: int = 3, 
                min_similarity: float = 0.1) -> List[Tuple[float, str, str, int]]:
        """
        Retrieve semantically similar memories.
        
        Args:
            query_embedding: Embedding to search for (tensor, list, or array)
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (similarity, user_input, ada_response, id) tuples
        """
        # Convert to tensor if needed
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        
        cursor = self.conn.execute(
            "SELECT id, user_input, ada_response, embedding, reward "
            "FROM episodic_memory ORDER BY timestamp DESC"
        )
        rows = cursor.fetchall()
        
        if not rows:
            return []
        
        similarities = []
        for row_id, user_input, ada_response, emb_blob, reward in rows:
            # Convert bytes back to tensor (make writable to avoid warning)
            emb_array = np.frombuffer(emb_blob, dtype=np.float32).copy()
            stored_emb = torch.from_numpy(emb_array)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, stored_emb.unsqueeze(0)).item()
            
            if similarity >= min_similarity:
                similarities.append((similarity, user_input, ada_response, row_id))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    
    def get_recent(self, limit: int = 5) -> List[MemoryRecord]:
        """
        Get the most recent conversation exchanges.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of MemoryRecord objects
        """
        cursor = self.conn.execute(
            "SELECT id, user_input, ada_response, reward, embedding, timestamp "
            "FROM episodic_memory ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        
        records = []
        for row_id, user_input, ada_response, reward, emb_blob, timestamp in rows:
            emb_array = np.frombuffer(emb_blob, dtype=np.float32).copy()
            embedding = torch.from_numpy(emb_array)
            records.append(MemoryRecord(
                id=row_id,
                user_input=user_input,
                ada_response=ada_response,
                reward=reward,
                embedding=embedding,
                timestamp=timestamp
            ))
        
        return records
    
    def get_context_string(self, query_embedding: torch.Tensor, max_tokens: int = 100) -> str:
        """
        Get a context string built from relevant memories.
        
        Args:
            query_embedding: Embedding to search for
            max_tokens: Maximum tokens to include in context
            
        Returns:
            Formatted context string
        """
        similarities = self.retrieve(query_embedding, top_k=3, min_similarity=0.1)
        
        if not similarities:
            return ""
        
        context_parts = []
        for similarity, user_input, ada_response, _ in similarities:
            context_parts.append(f"You: {user_input}")
            context_parts.append(f"Ada: {ada_response}")
        
        return "\n".join(context_parts)
    
    def get_best_memory(self, query_embedding) -> Optional[MemoryRecord]:
        """Get the single most similar memory."""
        similarities = self.retrieve(query_embedding, top_k=1, min_similarity=0.1)
        
        if not similarities:
            return None
            
        similarity, user_input, ada_response, memory_id = similarities[0]
        
        # Get the full record
        cursor = self.conn.execute(
            "SELECT id, user_input, ada_response, embedding, timestamp "
            "FROM episodic_memory WHERE id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
            
        emb_blob = row[3]
        emb_array = np.frombuffer(emb_blob, dtype=np.float32).copy()
        embedding = torch.from_numpy(emb_array)
        
        return MemoryRecord(
            id=row[0],
            user_input=row[1],
            ada_response=row[2],
            embedding=embedding,
            timestamp=row[4]
        )
    
    def update_reward(self, memory_id: int, reward: float) -> None:
        """Update the reward for a specific memory."""
        self.conn.execute(
            "UPDATE episodic_memory SET reward = ? WHERE id = ?",
            (reward, memory_id)
        )
        self.conn.commit()
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM episodic_memory")
        total = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT AVG(reward), MAX(reward), MIN(reward) FROM episodic_memory")
        stats = cursor.fetchone()
        avg_reward, max_reward, min_reward = stats if stats[0] else (0, 0, 0)
        
        return {
            "total_memories": total,
            "average_reward": avg_reward or 0,
            "max_reward": max_reward or 0,
            "min_reward": min_reward or 0
        }
    
    def cleanup_old_memories(self, days: int = 30) -> int:
        """Remove memories older than specified days."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        cursor = self.conn.execute(
            "DELETE FROM episodic_memory WHERE timestamp < ?",
            (cutoff_time,)
        )
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        return deleted_count
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
    
    def __del__(self):
        """Ensure connection is closed on deletion."""
        if hasattr(self, 'conn'):
            self.close()
