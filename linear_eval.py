from train.linear_eval_simclr import train_linear_eval_simclr
from train.linear_eval_mae import train_linear_eval_mae
from utils.dataset_loader import linear_dataloader
from models.simclr import SimCLR
from models.mae import MAE
from config import config
from torchvision import models
import torch.nn as nn
import torch

def evaluate_simclr():
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Identity()
    model = nn.Sequential(resnet, nn.Linear(config.embed_dim, 100)).to(config.device)
    optimizer = torch.optim.Adam(model[-1].parameters(), lr=0.001)
    dataloader = linear_dataloader()
    train_linear_eval_simclr(model, dataloader, optimizer)

def evaluate_mae():
    mae_model = MAE().to(config.device)
    encoder = nn.Sequential(
        mae_model.linear_embed,
        mae_model.encoder,
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten()
    )
    model = nn.Sequential(encoder, nn.Linear(config.embed_dim, 100)).to(config.device)
    optimizer = torch.optim.Adam(model[-1].parameters(), lr=0.001)
    dataloader = linear_dataloader()
    train_linear_eval_mae(model, dataloader, optimizer)
