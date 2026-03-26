import torch
import torch.optim as optim
import logging

from core.base_trainer import BaseTrainer
from core.transforms import CosineScheduler, LinearScheduler
from methods.dino.loss import dino_loss


class DINOTrainer(BaseTrainer):
    """Trainer for DINO pretraining"""

    def __init__(self, model, train_loader, val_loader, config, device='cuda', checkpoint_dir=None):
        super().__init__(model, train_loader, val_loader, config, device, checkpoint_dir)
        self.method_config = config.DINO

    def get_optimizer(self):
        """Create optimizer for DINO training"""
        return optim.SGD(
            self.model.student_encoder.parameters(),
            lr=self.method_config.lr,
            momentum=0.9,
            weight_decay=1e-4
        )

    def train_step(self, batch, optimizer, epoch=0, total_epochs=1000):
        """Single training step for DINO"""
        if isinstance(batch, (tuple, list)):
            crops = batch[0]  # Get global and local crops
        else:
            crops = batch

        # Global crops
        x1 = crops[0].to(self.device) if isinstance(crops, (list, tuple)) else batch[0].to(self.device)
        x2 = crops[1].to(self.device) if len(crops) > 1 else batch[1].to(self.device) if len(batch) > 1 else x1

        optimizer.zero_grad()

        # Forward pass
        student_logits_1, teacher_logits_1 = self.model(x1, x1)
        student_logits_2, teacher_logits_2 = self.model(x2, x2)

        # Compute loss
        loss = dino_loss(student_logits_1, teacher_logits_2,
                         self.method_config.student_temp,
                         self.method_config.teacher_temp_start) + \
               dino_loss(student_logits_2, teacher_logits_1,
                         self.method_config.student_temp,
                         self.method_config.teacher_temp_start)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update teacher with EMA
        self.model.update_teacher()

        return loss / 2

    def train(self, epochs=None):
        """Main training loop for DINO"""
        if epochs is None:
            epochs = self.method_config.epochs

        self.model = self.model.to(self.device)
        optimizer = self.get_optimizer()

        # Schedulers
        momentum_schedule = LinearScheduler(self.method_config.momentum_start, self.method_config.momentum_end, epochs)
        weight_decay_schedule = CosineScheduler(self.method_config.weight_decay_start,
                                               self.method_config.weight_decay_end, epochs)

        self.logger.info(f"Starting DINO training for {epochs} epochs")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                loss = self.train_step(batch, optimizer, epoch, epochs)
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

            # Update schedules
            momentum_schedule.step()
            weight_decay_schedule.step()

        self.logger.info("DINO training completed")
        return self.model
