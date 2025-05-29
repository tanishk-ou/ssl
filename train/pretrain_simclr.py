import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from config import config

def train_simclr(model, train_loader, optimizer):
    scaler = GradScaler()
    model.train()
    losses = []

    for epoch in range(config.epochs_pretrain):
        for x1, x2 in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x1, x2 = x1.to(config.device), x2.to(config.device)

            with autocast():
                z1, z2 = model(x1), model(x2)
                loss = simclr_loss(z1, z2)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        print(f"Epoch {epoch+1} Loss: {sum(losses[-len(train_loader):])/len(train_loader):.4f}")

    return losses
