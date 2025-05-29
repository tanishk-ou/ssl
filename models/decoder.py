import torch.nn as nn

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=512, output_dim=768, num_layers=2, num_heads=4, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                norm_first=True,
                activation="gelu"
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)
