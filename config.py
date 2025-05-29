import torch

class config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = 224
    patch_size = 16
    B_simclr = 128
    B_mae = 64
    mask_ratio = 0.75
    epochs_pretrain = 100
    epochs_linear_eval = 5
    dataset_path = './ssl_dataset'
    num_classes = 100
    simclr_lr = 0.001
    mae_lr = 0.0001
    embed_dim = 512
