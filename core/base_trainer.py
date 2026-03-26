import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
from pathlib import Path


class BaseTrainer(ABC):
    """Abstract base trainer class for all SSL methods"""

    def __init__(self, model, train_loader, config, device='cuda', checkpoint_dir=None):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def train_step(self, batch, optimizer, epoch, total_epochs):
        """Single training step - must be implemented by subclasses"""
        pass

    @abstractmethod
    def get_optimizer(self):
        """Get optimizer - must be implemented by subclasses"""
        pass

    def train(self, epochs=None):
        """Main training loop"""
        if epochs is None:
            epochs = self.config.epochs

        self.model = self.model.to(self.device)
        optimizer = self.get_optimizer()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                loss = self.train_step(batch, optimizer, epoch, epochs)
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": epoch_loss / num_batches})

            avg_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

        return self.model

    def save_checkpoint(self, path=None):
        """Save model checkpoint"""
        if path is None:
            path = self.checkpoint_dir / f"{self.__class__.__name__.lower()}_model.pth"

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f"Model loaded from {path}")
