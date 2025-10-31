"""Enhanced AdaCore policy network with advanced architecture."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy_network import get_device

try:
    from .semantic_memory import add_memory, recall
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    add_memory = lambda x: None
    recall = lambda x, k=3: []


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for better context understanding."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(output)


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence processing."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                          (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class EnhancedAdaCore(nn.Module):
    """Enhanced AdaCore with transformer-based architecture and better capabilities."""
    
    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        output_dim: int = 512,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Custom attention for global context
        self.global_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Output layers
        self.norm1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        
        # Multi-head output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with enhanced architecture.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (seq_len, batch_size, input_dim)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Handle single sequence input
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        if x.dim() == 2:
            # Add sequence dimension if missing
            x = x.unsqueeze(0)  # (1, batch_size, input_dim)
            single_sequence = True
        else:
            single_sequence = False
            
        # Input projection
        x = self.input_projection(x)  # (seq_len, batch_size, d_model)
        
        # Add positional encoding
        if not single_sequence:
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global attention for context awareness
        if not single_sequence:
            attended = self.global_attention(x, mask)
            x = self.norm1(x + attended)  # Residual connection + layer norm
        
        # Take mean across sequence dimension for final representation
        if not single_sequence:
            x = torch.mean(x, dim=0)
        
        # Final output projection
        output = self.norm2(x)
        output = self.output_projection(output)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        x = self.input_projection(x)
        if x.size(0) > 1:
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Extract attention weights from the last transformer layer
        with torch.no_grad():
            x = self.transformer(x)
            attention_weights = self.global_attention(x)
            
        return attention_weights


class AdaCoreWithMemory(nn.Module):
    """AdaCore with integrated memory mechanism for better context retention."""
    
    def __init__(
        self,
        base_model: EnhancedAdaCore,
        memory_size: int = 1000,
        memory_dim: int = 512,
        top_k: int = 5
    ):
        super().__init__()
        
        self.base_model = base_model
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.top_k = top_k
        
        # Memory bank
        self.register_buffer('memory_bank', torch.zeros(memory_size, memory_dim))
        self.register_buffer('memory_pointer', torch.tensor(0))
        
        # Memory attention
        self.memory_attention = nn.MultiheadAttention(memory_dim, num_heads=8)
        self.memory_gate = nn.GRU(memory_dim, memory_dim, batch_first=True)
        
    def forward(self, x: torch.Tensor, use_memory: bool = True) -> torch.Tensor:
        # Get base model output
        base_output = self.base_model(x)
        
        if not use_memory:
            return base_output
            
        # Query memory bank
        memory_output = self._query_memory(base_output)
        
        # Combine with base output
        combined, _ = self.memory_gate(base_output.unsqueeze(0).unsqueeze(0), 
                                     memory_output.unsqueeze(0).unsqueeze(0))
        
        return combined.squeeze()
    
    def _query_memory(self, query: torch.Tensor) -> torch.Tensor:
        """Query memory bank for similar contexts."""
        # Calculate similarity with all memories
        similarities = torch.cosine_similarity(
            query.unsqueeze(0), 
            self.memory_bank, 
            dim=1
        )
        
        # Get top-k most similar memories
        _, indices = torch.topk(similarities, min(self.top_k, similarities.size(0)))
        
        # Retrieve and average relevant memories
        relevant_memories = self.memory_bank[indices]
        return torch.mean(relevant_memories, dim=0)
    
    def update_memory(self, new_memory: torch.Tensor):
        """Update memory bank with new information."""
        idx = self.memory_pointer.item() % self.memory_size
        self.memory_bank[idx] = new_memory.detach()
        self.memory_pointer += 1


def create_enhanced_model(config: dict = None) -> EnhancedAdaCore:
    """Factory function to create enhanced model with configurable parameters."""
    default_config = {
        'input_dim': 384,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 4,  # Reduced for faster training
        'output_dim': 512,
        'dropout': 0.1,
        'use_layer_norm': True,
        'use_residual': True
    }
    
    if config:
        default_config.update(config)
    
    model = EnhancedAdaCore(**default_config)
    
    # Move to appropriate device
    device = get_device()
    model = model.to(device)
    
    return model


def save_enhanced_model(model: nn.Module, model_path: Path) -> None:
    """Save enhanced model with metadata."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save state dict and metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': getattr(model, 'input_dim', 384),
            'd_model': getattr(model, 'd_model', 512),
            'output_dim': 512,
        }
    }
    
    torch.save(checkpoint, model_path)


def load_enhanced_model(model_path: Path) -> nn.Module:
    """Load enhanced model with automatic configuration."""
    if not model_path.exists():
        print(f"⚠️  Model not found at {model_path}, creating new enhanced model")
        return create_enhanced_model()
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('model_config', {})
    
    model = create_enhanced_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


# Adaptive learning utilities
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_FEEDBACK_LOG = LOG_DIR / "training_feedback.jsonl"


def log_feedback(prompt: str, response: str, reward: float) -> None:
    """Log training feedback for adaptive retraining.
    
    Args:
        prompt: User prompt/input
        response: Ada's response
        reward: Reward value (typically -1 to 1)
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response,
        "reward": reward
    }
    
    try:
        with open(TRAINING_FEEDBACK_LOG, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"⚠️  Failed to log feedback: {e}")


class AdaCoreWithAdaptiveLearning:
    """Wrapper for EnhancedAdaCore with adaptive learning capabilities."""
    
    def __init__(
        self, 
        model: Optional[EnhancedAdaCore] = None,
        enable_memory: bool = True,
        enable_feedback: bool = True,
        learning_rate: float = 0.0005
    ):
        """Initialize adaptive learning wrapper.
        
        Args:
            model: EnhancedAdaCore model (creates new if None)
            enable_memory: Enable semantic memory for context
            enable_feedback: Enable reward-based feedback
            learning_rate: Learning rate for adaptive updates
        """
        self.model = model or create_enhanced_model()
        self.enable_memory = enable_memory and MEMORY_AVAILABLE
        self.enable_feedback = enable_feedback
        self.learning_rate = learning_rate
        self.device = next(self.model.parameters()).device
        
        # Training mode tracking
        self._optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
    
    def infer(self, prompt: str, prompt_embedding: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Run inference with memory-enhanced context.
        
        Args:
            prompt: Text prompt
            prompt_embedding: Embedding of the prompt
            
        Returns:
            Tuple of (response_embedding, context_string)
        """
        context_str = ""
        
        # Recall similar memories if enabled
        if self.enable_memory:
            recalled_memories = recall(prompt, k=3)
            if recalled_memories:
                context_str = " ".join(recalled_memories)
            
            # Add current prompt to memory
            add_memory(prompt)
        
        # Generate response
        self.model.eval()
        with torch.no_grad():
            response_embedding = self.model(prompt_embedding)
        
        return response_embedding, context_str
    
    def apply_feedback(
        self, 
        prompt: str,
        prompt_embedding: torch.Tensor,
        response: str,
        response_embedding: torch.Tensor,
        reward: float
    ) -> None:
        """Apply reinforcement learning feedback to update model.
        
        Args:
            prompt: User prompt
            prompt_embedding: Embedding of prompt
            response: Ada's response
            response_embedding: Embedding of response (target)
            reward: Reward value (-1 to 1, where 1 is best)
        """
        if not self.enable_feedback:
            return
        
        # Log feedback for batch retraining
        log_feedback(prompt, response, reward)
        
        # Immediate weight adjustment (RLHF-like)
        if abs(reward) > 0.1:  # Only update if reward is significant
            self.model.train()
            
            # Zero gradients
            self._optimizer.zero_grad()
            
            # Forward pass
            predicted = self.model(prompt_embedding)
            
            # Compute loss weighted by reward
            # Positive reward: minimize distance to response
            # Negative reward: maximize distance (or minimize inverse similarity)
            if reward > 0:
                # Good response - move closer
                loss = F.mse_loss(predicted, response_embedding) * (1 - reward)
            else:
                # Bad response - apply penalty
                loss = F.mse_loss(predicted, response_embedding) * (1 + abs(reward))
            
            # Backward pass
            loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self._optimizer.step()
            
            print(f"🧩 Ada updated. Reward={reward:.3f}, Loss={loss.item():.4f}")
    
    def save(self, path: Path) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        save_enhanced_model(self.model, path)
    
    def load(self, path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        self.model = load_enhanced_model(path)
        self.device = next(self.model.parameters()).device
        
        # Recreate optimizer
        self._optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
