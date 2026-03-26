import torch
import torch.optim as optim
import logging

from core.base_trainer import BaseTrainer
from methods.mae.loss import mae_loss


class MAETrainer(BaseTrainer):
    """Trainer for MAE pretraining"""

    def __init__(self, model, train_loader, config, device='cuda', checkpoint_dir=None):
        super().__init__(model, train_loader, config, device, checkpoint_dir)
        self.method_config = config.MAE

    def get_optimizer(self):
        """Create optimizer for MAE training"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.method_config.lr_base,
            weight_decay=self.method_config.weight_decay
        )

    def train_step(self, batch, optimizer, epoch, total_epochs):
        """Single training step for MAE"""
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        x = x.to(self.device)

        optimizer.zero_grad()

        # Forward pass
        reconstructed, mask_idx = self.model(x)

        # Compute loss on masked patches only
        num_patches = self.model.num_patches
        mask = torch.zeros(x.size(0), num_patches, dtype=torch.bool, device=self.device)
        mask[:, mask_idx] = True

        original_patches = self.model.patchify(x)
        loss = mae_loss(reconstructed, original_patches, mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss
