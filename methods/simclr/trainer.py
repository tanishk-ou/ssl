import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
import logging

from core.base_trainer import BaseTrainer
from methods.simclr.loss import simclr_loss


class SimCLRTrainer(BaseTrainer):
    """Trainer for SimCLR pretraining"""

    def __init__(self, model, train_loader, config, device='cuda', checkpoint_dir=None):
        super().__init__(model, train_loader, config, device, checkpoint_dir)
        self.scaler = GradScaler()
        self.method_config = config.SimCLR

    def get_optimizer(self):
        """Create optimizer for SimCLR training"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.method_config.lr,
            weight_decay=self.method_config.weight_decay
        )

    def train_step(self, batch, optimizer, epoch, total_epochs):
        """Single training step for SimCLR"""
        x1, x2 = batch
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            z1 = self.model(x1)
            z2 = self.model(x2)
            loss = simclr_loss(z1, z2, temperature=self.method_config.temperature)

        # Backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        return loss
