import torch
import torch.nn.functional as F


def simclr_loss(z_i, z_j, temperature=0.5):
    """
    SimCLR contrastive loss function

    Args:
        z_i: Embeddings from first view (batch_size, projection_dim)
        z_j: Embeddings from second view (batch_size, projection_dim)
        temperature: Temperature parameter for scaling

    Returns:
        Contrastive loss
    """
    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Concatenate both views
    representations = torch.cat([z_i, z_j], dim=0)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(representations, representations.T)

    # Mask out diagonal (self-similarities)
    mask = torch.eye(representations.size(0), dtype=torch.bool).to(z_i.device)
    similarity_matrix.masked_fill_(mask, float('-inf'))

    # Apply temperature and compute logits
    logits = similarity_matrix / temperature

    # Compute targets (positive pairs are at i and i+N)
    batch_size = z_i.size(0)
    device = z_i.device

    # Positive pairs for each view
    pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = True
        pos_mask[i + batch_size, i] = True

    # Negative log-sum-exp formula
    exp_logits = torch.exp(logits)
    exp_logits[mask] = 0

    # Positive log-probs
    pos_logits = logits[pos_mask].view(2 * batch_size, 1)

    # Full loss
    all_logits = torch.cat([pos_logits, logits[~mask].view(2 * batch_size, -1)], dim=1)
    loss = -torch.logsumexp(all_logits / temperature, dim=1).mean()

    return loss
