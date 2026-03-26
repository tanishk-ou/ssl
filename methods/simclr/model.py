import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig


class SimCLRModel(nn.Module):
    """SimCLR model with Vision Transformer encoder and projection head"""

    def __init__(self, embed_dim=768, projection_dim=256, temperature=0.5):
        super().__init__()
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

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, projection_dim)
        )
        self.temperature = temperature

    def forward(self, x):
        """Forward pass: return normalized projections"""
        # Get encoder output (use CLS token which is first in sequence)
        encoder_output = self.encoder(pixel_values=x, interpolate_pos_encoding=True)
        cls_embedding = encoder_output['last_hidden_state'][:, 0, :]  # Take CLS token

        # Project to projection dimension
        proj = self.projection(cls_embedding)

        # Normalize projections
        return F.normalize(proj, dim=1)

    def get_encoder(self):
        """Get encoder without projection head"""
        return self.encoder
