import torch
import torch.nn as nn
from models.decoder import MAEDecoder
from config import config

class SimpleViTEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=6, heads=8, mlp_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=embed_dim * mlp_ratio,
                activation="gelu",
                norm_first=True,
                batch_first=True
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MAE(nn.Module):
    def __init__(self, mask_ratio=0.75, embed_dim=512):
        super(MAE, self).__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = config.patch_size
        self.patch_dim = (config.patch_size**2) * 3
        self.H = config.image_size
        self.W = self.H
        self.num_patches = (self.H // self.patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.linear_embed = nn.Linear(self.patch_dim, embed_dim)
        self.mask_tokens = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dec_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.encoder = SimpleViTEncoder(embed_dim=embed_dim)
        self.decoder = MAEDecoder(embed_dim=embed_dim, output_dim=self.patch_dim)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        patch = x.reshape(B, C, H//p, p, W//p, p).permute(0, 2, 4, 3, 5, 1)
        patch = patch.reshape(B, (H//p)*(W//p), p*p*C)
        return patch

    def unpatchify(self, x):
        B, N, D = x.shape
        p = self.patch_size
        H = W = self.H
        C = D // (p * p)
        x = x.reshape(B, H//p, W//p, p, p, C).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, H, W)
        return x

    def forward(self, x):
        patch = self.patchify(x.to(config.device))
        patches = self.linear_embed(patch)
        B, N, D = patches.shape
        patches = patches + self.pos_embed

        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=config.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        ids_mask = ids_shuffle[:, keep:]

        x_keep = torch.gather(patches, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        e = self.encoder(x_keep)

        mask_token = self.mask_tokens.repeat(B, N - x_keep.size(1), 1)
        z = torch.cat([e, mask_token], dim=1)
        z = torch.gather(z, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))
        z = z + self.dec_pos_embed

        output = self.decoder(z)
        return output, patch, ids_mask
