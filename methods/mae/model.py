import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import numpy as np


class MAEModel(nn.Module):
    """Masked Autoencoder - MAE model"""

    def __init__(self, mask_ratio=0.75, embed_dim=768):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.patch_size = 16
        self.patch_dim = (self.patch_size ** 2) * 3
        self.image_size = 224
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Vision Transformer encoder
        vit_config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_hidden_layers=12,
            hidden_size=embed_dim,
            num_attention_heads=12,
            intermediate_size=3072,
            drop_path_rate=0.1
        )
        self.encoder = ViTModel(vit_config)

        # Mask token and positional embeddings for decoder
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8)

        # Reconstruction head
        self.reconstruction_head = nn.Linear(embed_dim, self.patch_dim)

    def patchify(self, x):
        """Convert image to patches"""
        batch_size, channels, height, width = x.shape
        patches = x.reshape(
            batch_size,
            channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(batch_size, -1, channels * self.patch_size * self.patch_size)
        return patches

    def unpatchify(self, patches):
        """Convert patches back to image"""
        batch_size = patches.shape[0]
        x = patches.reshape(
            batch_size,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
            self.patch_size,
            self.patch_size,
            3
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch_size, 3, self.image_size, self.image_size)
        return x

    def forward(self, x):
        """Forward pass"""
        # Patchify
        patches = self.patchify(x)

        # Random masking
        num_patches = self.num_patches
        mask_len = int(np.ceil(num_patches * self.mask_ratio))
        mask_idx = torch.randperm(num_patches)[:mask_len]

        # Create masked patches
        masked_patches = patches.clone()
        masked_patches[torch.arange(patches.size(0)).unsqueeze(1), mask_idx] = 0

        # Encode
        encoder_output = self.encoder(pixel_values=x, interpolate_pos_encoding=True)
        encoded = encoder_output['last_hidden_state']

        # Create full sequence for decoder
        decoder_input = encoded.clone()
        decoder_input[torch.arange(decoder_input.size(0)).unsqueeze(1), mask_idx] = self.mask_token

        # Decode
        decoded = self.decoder(decoder_input + self.decoder_pos_embed)

        # Reconstruct
        reconstructed = self.reconstruction_head(decoded)

        return reconstructed, mask_idx

    def loss(self, reconstructed, original, mask_idx):
        """MAE loss - MSE on masked patches"""
        original_patches = self.patchify(original)

        # Loss only on masked patches
        loss = 0.0
        for i, idx in enumerate(mask_idx):
            loss += F.mse_loss(reconstructed[i, idx], original_patches[i, idx])

        return loss / len(mask_idx)
