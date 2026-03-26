#!/usr/bin/env python
"""
Main training script for SSL methods

Usage:
    python main.py --method simclr --epochs 100 --batch-size 128
    python main.py --method mae --epochs 600 --batch-size 256
    python main.py --method dino --epochs 1000 --batch-size 96
"""

import argparse
import logging
import torch
from pathlib import Path

from core.config import config
from core.datasets import get_train_dataloader
from methods.simclr.model import SimCLRModel
from methods.simclr.trainer import SimCLRTrainer
from methods.mae.model import MAEModel
from methods.mae.trainer import MAETrainer
from methods.dino.model import DINOModel
from methods.dino.trainer import DINOTrainer


def setup_logging(method_name):
    """Setup logging configuration"""
    log_dir = Path(config.checkpoint_path) / f"{method_name}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'{method_name}.log'),
            logging.StreamHandler()
        ]
    )


def train_simclr(args):
    """Train SimCLR model"""
    setup_logging('simclr')
    logger = logging.getLogger('SimCLR')

    logger.info("Initializing SimCLR training...")

    # DataLoader
    train_loader = get_train_dataloader('simclr', batch_size=args.batch_size or config.SimCLR.batch_size)

    # Model
    model = SimCLRModel(
        embed_dim=config.embed_dim,
        projection_dim=config.SimCLR.projection_dim,
        temperature=config.SimCLR.temperature
    )

    # Trainer
    trainer = SimCLRTrainer(
        model,
        train_loader,
        None,  # TODO: Add val_loader
        config,
        device=str(config.device),
        checkpoint_dir=config.checkpoint_path
    )

    # Train
    trainer.train(epochs=args.epochs or config.SimCLR.epochs)
    trainer.save_checkpoint(Path(config.checkpoint_path) / 'simclr' / 'model.pth')

    logger.info("SimCLR training completed successfully!")


def train_mae(args):
    """Train MAE model"""
    setup_logging('mae')
    logger = logging.getLogger('MAE')

    logger.info("Initializing MAE training...")

    # DataLoader
    train_loader = get_train_dataloader('mae', batch_size=args.batch_size or config.MAE.batch_size)

    # Model
    model = MAEModel(
        mask_ratio=config.MAE.mask_ratio,
        embed_dim=config.embed_dim
    )

    # Trainer
    trainer = MAETrainer(
        model,
        train_loader,
        None,
        config,
        device=str(config.device),
        checkpoint_dir=config.checkpoint_path
    )

    # Train
    trainer.train(epochs=args.epochs or config.MAE.epochs)
    trainer.save_checkpoint(Path(config.checkpoint_path) / 'mae' / 'model.pth')

    logger.info("MAE training completed successfully!")


def train_dino(args):
    """Train DINO model"""
    setup_logging('dino')
    logger = logging.getLogger('DINO')

    logger.info("Initializing DINO training...")

    # DataLoader
    train_loader = get_train_dataloader('dino', batch_size=args.batch_size or config.DINO.batch_size)

    # Model
    model = DINOModel(
        embed_dim=config.embed_dim,
        num_prototypes=config.DINO.prototypes,
        student_temp=config.DINO.student_temp,
        teacher_temp_start=config.DINO.teacher_temp_start,
        teacher_temp_end=config.DINO.teacher_temp_end
    )

    # Trainer
    trainer = DINOTrainer(
        model,
        train_loader,
        None,
        config,
        device=str(config.device),
        checkpoint_dir=config.checkpoint_path
    )

    # Train
    trainer.train(epochs=args.epochs or config.DINO.epochs)
    trainer.save_checkpoint(Path(config.checkpoint_path) / 'dino' / 'model.pth')

    logger.info("DINO training completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Train Self-Supervised Learning models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --method simclr
  python main.py --method mae --epochs 300 --batch-size 128
  python main.py --method dino --epochs 500
        """
    )

    parser.add_argument('--method', type=str, choices=['simclr', 'mae', 'dino'],
                        default='simclr',
                        help='Which SSL method to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (uses config default if not specified)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (uses config default if not specified)')
    parser.add_argument('--device', type=str, default=str(config.device),
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    logger = logging.getLogger('Main')
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Training {args.method.upper()} on {args.device}")
    logger.info(f"Config file loaded from: config.yaml")

    if args.method == 'simclr':
        train_simclr(args)
    elif args.method == 'mae':
        train_mae(args)
    elif args.method == 'dino':
        train_dino(args)
    else:
        logger.error(f"Unknown method: {args.method}")


if __name__ == '__main__':
    main()
