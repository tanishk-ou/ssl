import torch
from config import config
from models.mae import MAE
from utils.dataset_loader import train_dataloader
from loss.mae_loss import mae_loss
from train.pretrain_mae import train_mae

train_loader = train_dataloader(sim=False)

model = MAE().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.mae_lr)

losses = train_mae(model, train_loader, optimizer)

torch.save(model.encoder.state_dict(), 'checkpoints/mae_encoder.pth')
