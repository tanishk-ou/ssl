import torch.nn as nn

class SimCLR(nn.Module):
    def __init__(self, encoder, embed_dim=512, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        return x
