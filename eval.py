#!/usr/bin/env python
"""
Evaluation script for SSL models with k-NN and Linear Probe evaluation

Usage:
    python eval.py --method simclr --checkpoint checkpoints/simclr/model.pth
    python eval.py --method dino --visualize --generate-results
"""

import argparse
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json

from core.config import config
from core.datasets import get_linear_eval_dataloader, get_val_dataloader
from methods.simclr.model import SimCLRModel
from methods.mae.model import MAEModel
from methods.dino.model import DINOModel


def setup_logging(method_name):
    """Setup logging configuration"""
    log_dir = Path(config.checkpoint_path) / f"{method_name}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'{method_name}_eval.log'),
            logging.StreamHandler()
        ]
    )


def get_model(method, checkpoint_path, device):
    """Load model from checkpoint"""
    if method == 'simclr':
        model = SimCLRModel()
    elif method == 'mae':
        model = MAEModel()
    elif method == 'dino':
        model = DINOModel()
    else:
        raise ValueError(f"Unknown method: {method}")

    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.getLogger('Eval').info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logging.getLogger('Eval').warning(f"No checkpoint found at {checkpoint_path}, using untrained model")

    return model.to(device).eval()


def extract_features(model, dataloader, device, method='simclr'):
    """Extract features from model"""
    features = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, batch_labels = batch
            images = images.to(device)

            # Get encoder output
            if method == 'simclr':
                features_batch = model.get_encoder()(pixel_values=images, interpolate_pos_encoding=True)
                features_batch = features_batch['last_hidden_state'][:, 0, :]  # CLS token
            elif method == 'mae':
                features_batch = model.encoder(pixel_values=images, interpolate_pos_encoding=True)
                features_batch = features_batch['last_hidden_state'][:, 0, :]  # CLS token
            elif method == 'dino':
                features_batch = model.student_encoder(pixel_values=images, interpolate_pos_encoding=True)
                features_batch = features_batch['last_hidden_state'][:, 0, :]  # CLS token
            else:
                raise ValueError(f"Unknown method: {method}")

            features.append(features_batch.cpu().numpy())
            labels.append(batch_labels.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def knn_evaluation(model, train_loader, val_loader, device, k=20, method='simclr'):
    """Perform k-NN evaluation"""
    logger = logging.getLogger('Eval')
    logger.info(f"Extracting training features...")

    # Extract training features
    train_features, train_labels = extract_features(model, train_loader, device, method)

    logger.info(f"Extracting validation features...")
    # Extract validation features
    val_features, val_labels = extract_features(model, val_loader, device, method)

    logger.info(f"Running k-NN with k={k}...")
    # k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=8)
    knn.fit(train_features, train_labels)

    # Evaluate
    pred_labels = knn.predict(val_features)
    accuracy = accuracy_score(val_labels, pred_labels)

    logger.info(f"k-NN Accuracy (k={k}): {accuracy:.4f} ({accuracy*100:.2f}%)")

    return accuracy


def linear_probe_evaluation(model, train_loader, val_loader, device, method='simclr', epochs=20):
    """Perform linear probe evaluation"""
    logger = logging.getLogger('Eval')
    logger.info(f"Starting linear probe evaluation for {epochs} epochs...")

    # Extract features once
    train_features, train_labels = extract_features(model, train_loader, device, method)
    val_features, val_labels = extract_features(model, val_loader, device, method)

    # Linear layer on top of frozen encoder
    import torch.nn as nn
    linear_head = nn.Linear(model.embed_dim, config.num_classes).to(device)
    optimizer = torch.optim.SGD(linear_head.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train linear head
    train_features_tensor = torch.from_numpy(train_features).float().to(device)
    train_labels_tensor = torch.from_numpy(train_labels).long().to(device)

    for epoch in range(epochs):
        linear_head.train()
        logits = linear_head(train_features_tensor)
        loss = criterion(logits, train_labels_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            logger.info(f"Linear probe epoch {epoch + 1}/{epochs}: Loss = {loss.item():.4f}")

    # Evaluate
    linear_head.eval()
    with torch.no_grad():
        val_features_tensor = torch.from_numpy(val_features).float().to(device)
        val_logits = linear_head(val_features_tensor)
        val_preds = val_logits.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(val_labels, val_preds)
    logger.info(f"Linear Probe Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return accuracy


def generate_results_json(method, knn_acc, linear_acc, checkpoint_path):
    """Generate results JSON file"""
    results = {
        "method": method,
        "k_nn_accuracy": float(knn_acc),
        "linear_probe_accuracy": float(linear_acc),
        "checkpoint_path": str(checkpoint_path),
        "evaluation_date": str(Path(checkpoint_path).stat().st_mtime if Path(checkpoint_path).exists() else "N/A")
    }

    results_file = Path(config.results_path) / method / "evaluation_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.getLogger('Eval').info(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SSL models')
    parser.add_argument('--method', type=str, choices=['simclr', 'mae', 'dino'],
                        default='simclr', help='Which method to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--k', type=int, default=20,
                        help='k for k-NN evaluation')
    parser.add_argument('--linear-epochs', type=int, default=20,
                        help='Epochs for linear probe')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run full evaluation (k-NN + Linear Probe)')
    parser.add_argument('--generate-results', action='store_true',
                        help='Generate results JSON file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations (t-SNE, UMAP)')
    parser.add_argument('--device', type=str, default=str(config.device),
                        help='Device to use')

    args = parser.parse_args()

    setup_logging(args.method)
    logger = logging.getLogger('Eval')

    device = torch.device(args.device)

    # Set default checkpoint if not provided
    if args.checkpoint is None:
        args.checkpoint = str(Path(config.checkpoint_path) / args.method / 'model.pth')

    logger.info(f"Evaluating {args.method.upper()} model")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Load model
    model = get_model(args.method, args.checkpoint, device)

    # Get dataloaders
    train_loader = get_linear_eval_dataloader(batch_size=256)
    val_loader = get_val_dataloader(batch_size=256)

    results = {}

    # k-NN evaluation
    try:
        knn_acc = knn_evaluation(model, train_loader, val_loader, device, k=args.k, method=args.method)
        results['knn_accuracy'] = knn_acc
    except Exception as e:
        logger.error(f"k-NN evaluation failed: {e}")

    # Linear probe evaluation
    try:
        linear_acc = linear_probe_evaluation(model, train_loader, val_loader, device,
                                            method=args.method, epochs=args.linear_epochs)
        results['linear_accuracy'] = linear_acc
    except Exception as e:
        logger.error(f"Linear probe evaluation failed: {e}")

    # Generate results JSON
    if args.generate_results and 'knn_accuracy' in results and 'linear_accuracy' in results:
        generate_results_json(args.method, results['knn_accuracy'], results['linear_accuracy'],
                             args.checkpoint)

    logger.info(f"Evaluation completed!")


if __name__ == '__main__':
    import numpy as np
    main()
