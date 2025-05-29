import torch
from models.simclr import SimCLR
from models.mae import MAE
from utils.dataset_loader import val_dataloader
from utils.eval_utils import evaluate_model
from torchvision import models
import torch.nn as nn

from config import config

val_loader = val_dataloader()

# SimCLR
resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Identity()
sim_model = nn.Sequential(resnet, nn.Linear(config.embed_dim, 100)).to(config.device)
sim_model.load_state_dict(torch.load("checkpoints/sim_linear.pth"))
acc, f1 = evaluate_model(sim_model, val_loader, config.device)
print("SimCLR → Accuracy:", acc * 100, "F1 Score:", f1)

# MAE
mae_model = MAE().to(config.device)
encoder = nn.Sequential(
    mae_model.linear_embed,
    mae_model.encoder,
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten()
)
mae_head = nn.Linear(config.embed_dim, 100)
mae_model_final = nn.Sequential(encoder, mae_head).to(config.device)
mae_model_final.load_state_dict(torch.load("checkpoints/mae_linear.pth"))
acc, f1 = evaluate_model(mae_model_final, val_loader, config.device)
print("MAE → Accuracy:", acc * 100, "F1 Score:", f1)
