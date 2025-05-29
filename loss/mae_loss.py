import torch.nn.functional as F

def mae_loss(output, patches, ids_mask):
    B, N, D = patches.shape
    output_masked = torch.gather(output, 1, ids_mask.unsqueeze(-1).repeat(1, 1, D))
    patches_masked = torch.gather(patches, 1, ids_mask.unsqueeze(-1).repeat(1, 1, D))
    return F.mse_loss(output_masked, patches_masked, reduction='mean')
