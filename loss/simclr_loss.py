import torch
import torch.nn.functional as F

def simclr_loss(z_i, z_j, temperature=0.5):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    r = torch.cat([z_i, z_j], dim=0)
    sim = torch.matmul(r, r.T)
    mask = torch.eye(r.size(0), dtype=torch.bool).to(z_i.device)
    sim.masked_fill_(mask, float('-inf'))

    logits = F.log_softmax(sim / temperature, dim=1)

    N = z_i.size(0)
    pos_idx = torch.arange(N)
    pos_idx = torch.cat([pos_idx + N, pos_idx], dim=0).to(z_i.device)
    targets = pos_idx

    loss = -logits[torch.arange(2*N), targets].mean()
    return loss
