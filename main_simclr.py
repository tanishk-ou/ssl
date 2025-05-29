import torch
from torchvision import models
from config import config
from models.simclr import SimCLR
from utils.dataset_loader import train_dataloader
from loss.simclr_loss import simclr_loss
from train.pretrain_simclr import train_simclr

train_loader = train_dataloader(sim=True)

resnet = models.resnet18(pretrained=False)
resnet.fc = torch.nn.Identity()

model = SimCLR(encoder=resnet).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.simclr_lr)

losses = train_simclr(model, train_loader, optimizer)

torch.save(model.encoder.state_dict(), 'checkpoints/simclr_encoder.pth')
