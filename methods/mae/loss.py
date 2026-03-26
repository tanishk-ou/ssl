import torch
import torch.nn.functional as F


def mae_loss(reconstructed, original, mask):
    """
    MAE loss - MSE on masked patches only

    Args:
        reconstructed: Model output (batch_size, num_patches, patch_dim)
        original: Original image patches (batch_size, num_patches, patch_dim)
        mask: Boolean mask of masked patches (batch_size, num_patches)

    Returns:
        Loss value
    """
    return F.mse_loss(reconstructed[mask], original[mask])
