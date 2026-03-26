import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from pathlib import Path


def generate_tsne(features, labels, save_path=None, title="t-SNE Visualization"):
    """
    Generate t-SNE visualization of features

    Args:
        features: Feature vectors (N, D)
        labels: Class labels (N,)
        save_path: Path to save the figure
        title: Title for the plot
    """
    print(f"Generating t-SNE visualization ({features.shape[0]} samples)...")

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    # Create plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels, cmap='tab20', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()


def generate_umap(features, labels, save_path=None, title="UMAP Visualization"):
    """
    Generate UMAP visualization of features

    Args:
        features: Feature vectors (N, D)
        labels: Class labels (N,)
        save_path: Path to save the figure
        title: Title for the plot
    """
    print(f"Generating UMAP visualization ({features.shape[0]} samples)...")

    # Compute UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    features_2d = reducer.fit_transform(features)

    # Create plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels, cmap='tab20', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()


def plot_attention_maps(images, attention_maps, save_path=None, title="Attention Maps"):
    """
    Plot attention maps overlaid on images (for DINO)

    Args:
        images: Input images (N, C, H, W) or (N, H, W, C)
        attention_maps: Attention maps (N, H, W)
        save_path: Path to save the figure
        title: Title for the plot
    """
    n_samples = min(4, len(images))
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        img = images[i]
        if img.shape[2] == 3:  # Assuming channels last
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)

        attn = attention_maps[i]
        attn = (attn - attn.min()) / (attn.max() -attn.min() + 1e-5)

        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i+1}")
        axes[i, 0].axis('off')

        # Attention overlay
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(attn, alpha=0.5, cmap='hot')
        axes[i, 1].set_title(f"Attention Overlay {i+1}")
        axes[i, 1].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention maps to {save_path}")

    plt.close()


def plot_mae_reconstruction(original_images, reconstructed_images, save_path=None,
                           title="MAE Reconstruction"):
    """
    Plot original vs reconstructed images (for MAE)

    Args:
        original_images: Original images (N, C, H, W)
        reconstructed_images: Reconstructed images (N, C, H, W)
        save_path: Path to save the figure
        title: Title for the plot
    """
    n_samples = min(4, len(original_images))
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        orig = original_images[i]
        recon = reconstructed_images[i]

        # Normalize for visualization
        orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-5)
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-5)

        # Permute if needed (channels last for matplotlib)
        if orig.shape[0] == 3:
            orig = orig.permute(1, 2, 0).numpy()
            recon = recon.permute(1, 2, 0).numpy()

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(recon)
        axes[i, 1].set_title(f"Reconstructed {i+1}")
        axes[i, 1].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstructions to {save_path}")

    plt.close()


def plot_training_curves(train_losses, val_losses=None, save_path=None,
                         title="Training Curves", xlabel="Epoch", ylabel="Loss"):
    """
    Plot training and validation curves

    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs (optional)
        save_path: Path to save the figure
        title: Title for the plot
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.close()
