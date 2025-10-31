"""Enhanced training system for Ada with advanced techniques."""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

try:
    from core import ConversationStore, ReasoningEngine
    from rl import AdaAgent, ExperienceBuffer
    from .encoder import TextEncoder, LanguageEncoder
    from .enhanced_policy_network import EnhancedAdaCore, create_enhanced_model, save_enhanced_model, load_enhanced_model
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}")
    # Create dummy imports for when modules aren't available
    ConversationStore = None
    ReasoningEngine = None
    AdaAgent = None
    ExperienceBuffer = None
    TextEncoder = None
    LanguageEncoder = None
    EnhancedAdaCore = None
    create_enhanced_model = None
    save_enhanced_model = None
    load_enhanced_model = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    """Enhanced dataset for conversation training with better preprocessing."""
    
    def __init__(
        self, 
        conversations: List[Tuple[str, str, float]], 
        encoder: LanguageEncoder,
        max_sequence_length: int = 512
    ):
        self.conversations = conversations
        self.encoder = encoder
        self.max_length = max_sequence_length
        
        # Pre-process all conversations
        logger.info(f"Processing {len(conversations)} conversation pairs...")
        self.processed_data = self._preprocess_conversations()
        
    def _preprocess_conversations(self) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Pre-process conversations into embeddings."""
        processed = []
        
        for user_input, ada_response, reward in self.conversations:
            try:
                # Encode both sides
                user_embedding = self.encoder.encode(user_input.strip())
                ada_embedding = self.encoder.encode(ada_response.strip())
                
                # Filter out very short or empty responses
                if user_embedding.norm() > 0.1 and ada_embedding.norm() > 0.1:
                    processed.append((user_embedding, ada_embedding, reward))
                    
            except Exception as e:
                logger.warning(f"Failed to process conversation: {e}")
                continue
                
        logger.info(f"Successfully processed {len(processed)} conversation pairs")
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]


class AdvancedTrainer:
    """Enhanced trainer with modern optimization techniques."""
    
    def __init__(
        self,
        model: EnhancedAdaCore,
        encoder: LanguageEncoder,
        device: torch.device = None
    ):
        self.model = model
        self.encoder = encoder
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training configuration
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
        
        # Enhanced loss functions
        self.contrastive_loss = nn.InfoNCELoss()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def setup_optimizer(
        self, 
        lr: float = 2e-4, 
        weight_decay: float = 1e-4,
        use_cosine_annealing: bool = True
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Setup optimizer with advanced features."""
        
        # Use AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        if use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=lr * 0.1
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
        return optimizer, scheduler
    
    def contrastive_loss_fn(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negatives: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Enhanced contrastive loss for better semantic understanding."""
        
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)
        
        # Compute logits
        logits_pos = torch.matmul(anchor.unsqueeze(1), positive.unsqueeze(-1)).squeeze(-1) / temperature
        logits_neg = torch.matmul(anchor.unsqueeze(1), negatives.transpose(-2, -1)) / temperature
        
        # Combine
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)
    
    def train_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        epoch: int,
        use_contrastive: bool = True
    ) -> float:
        """Train one epoch with enhanced techniques."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                user_embeddings, ada_embeddings, rewards = batch
            else:
                user_embeddings, ada_embeddings = batch[:2]
                rewards = torch.ones(user_embeddings.size(0))
            
            # Move to device
            user_embeddings = user_embeddings.to(self.device)
            ada_embeddings = ada_embeddings.to(self.device)
            rewards = rewards.to(self.device)
            
            optimizer.zero_grad()
            
            # Use mixed precision if available
            if self.scaler:
                with autocast():
                    loss = self._compute_loss(
                        user_embeddings, 
                        ada_embeddings, 
                        rewards, 
                        use_contrastive
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(
                    user_embeddings, 
                    ada_embeddings, 
                    rewards, 
                    use_contrastive
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}, LR: {current_lr:.6f}")
        
        avg_loss = total_loss / max(1, num_batches)
        self.history['train_loss'].append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        self.history['learning_rates'].append(current_lr)
        
        return avg_loss
    
    def _compute_loss(
        self, 
        user_embeddings: torch.Tensor, 
        ada_embeddings: torch.Tensor, 
        rewards: torch.Tensor,
        use_contrastive: bool = True
    ) -> torch.Tensor:
        """Compute multi-task loss."""
        
        losses = []
        
        # Direct reconstruction loss
        predictions = self.model(user_embeddings)
        recon_loss = self.mse_loss(predictions, ada_embeddings)
        losses.append(recon_loss)
        
        # Contrastive loss for better semantic matching
        if use_contrastive and len(user_embeddings) > 1:
            # Create negative samples by shuffling
            negatives_indices = torch.randperm(len(ada_embeddings))
            negatives = ada_embeddings[negatives_indices]
            
            contrastive_loss = self.contrastive_loss_fn(user_embeddings, ada_embeddings, negatives)
            losses.append(0.3 * contrastive_loss)  # Weighted
        
        # Cosine similarity loss for better alignment
        target = torch.ones(user_embeddings.size(0), device=self.device)
        cos_loss = self.cosine_loss(
            F.normalize(predictions, p=2, dim=-1),
            F.normalize(ada_embeddings, p=2, dim=-1),
            target
        )
        losses.append(0.2 * cos_loss)
        
        # Reward-weighted loss
        if rewards.abs().sum() > 0:
            reward_weights = (rewards + 1.0) / 2.0  # Normalize rewards to [0, 1]
            reward_loss = (recon_loss * reward_weights).mean()
            losses.append(0.1 * reward_loss)
        
        # Combine all losses
        total_loss = sum(losses)
        
        return total_loss
    
    def validate(self, val_dataloader: DataLoader) -> float:
        """Validate the model."""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                if len(batch) == 2:
                    user_embeddings, ada_embeddings = batch
                else:
                    user_embeddings, ada_embeddings = batch[:2]
                
                user_embeddings = user_embeddings.to(self.device)
                ada_embeddings = ada_embeddings.to(self.device)
                
                predictions = self.model(user_embeddings)
                loss = self.mse_loss(predictions, ada_embeddings)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train_with_early_stopping(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 50,
        patience: int = 10,
        min_delta: float = 1e-4,
        save_path: Path = None
    ) -> Dict[str, Any]:
        """Train with early stopping and best model saving."""
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            
            # Validation
            val_loss = train_loss  # Default to train loss if no validation set
            if val_loader:
                val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Check for improvement
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                if save_path:
                    save_enhanced_model(self.model, save_path)
                    logger.info(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {epoch_time:.2f}s | "
                f"Best: Epoch {best_epoch} ({best_val_loss:.4f})"
            )
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}")
                break
        
        # Load best model if saved
        if save_path and save_path.exists():
            self.model = load_enhanced_model(save_path)
            self.model.to(self.device)
        
        return {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_epochs': epoch,
            'history': self.history
        }


def load_conversation_data(store: ConversationStore) -> List[Tuple[str, str, float]]:
    """Load and preprocess conversation data from the conversation store."""
    
    logger.info("Loading conversation data...")
    
    # Fetch recent conversations
    records = list(store.fetch_recent(limit=500))
    if not records:
        logger.warning("No conversation data found!")
        return []
    
    conversations = []
    
    for record in records:
        user_input = record.user_input or ""
        ada_response = record.ada_response or ""
        reward = float(record.reward) if record.reward is not None else 0.0
        
        # Filter quality conversations
        if (len(user_input.strip()) > 5 and 
            len(ada_response.strip()) > 10 and
            not user_input.startswith('/') and  # Skip commands
            not ada_response.startswith('❌')):  # Skip errors
            
            conversations.append((user_input, ada_response, reward))
    
    logger.info(f"Loaded {len(conversations)} quality conversations")
    return conversations


def main():
    """Main training pipeline."""
    
    # Check dependencies
    if not ConversationStore or not LanguageEncoder or not create_enhanced_model:
        logger.error("Required modules not available. Using fallback training.")
        # Fall back to original trainer
        from . import trainer
        trainer.main()
        return
    
    # Setup paths
    checkpoint_path = Path(__file__).resolve().parents[1] / "storage" / "checkpoints" / "enhanced_ada_core.pt"
    
    # Initialize components
    logger.info("Initializing enhanced trainer components...")
    
    # Load conversation data
    try:
        store = ConversationStore()
        conversation_data = load_conversation_data(store)
        
        if not conversation_data:
            logger.error("No conversation data available for training!")
            return
    except Exception as e:
        logger.error(f"Failed to load conversation data: {e}")
        return
    
    # Create encoder and dataset
    try:
        encoder = LanguageEncoder()
        dataset = ConversationDataset(conversation_data, encoder)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Create enhanced model
        model = create_enhanced_model({
            'd_model': 256,  # Smaller for faster training
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1
        })
        
        # Create trainer
        trainer = AdvancedTrainer(model, encoder)
        
        # Train the model
        results = trainer.train_with_early_stopping(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=30,
            patience=8,
            save_path=checkpoint_path
        )
        
        logger.info(f"Training completed! Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Model saved to: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {e}")
        # Fall back to original trainer
        from . import trainer
        trainer.main()


if __name__ == "__main__":
    main()
