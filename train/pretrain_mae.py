import torch
from tqdm import tqdm
from config import config

def train_mae(model, train_loader, optimizer):
    model.train()
    losses = []

    for epoch in range(config.epochs_pretrain):
        for imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs = imgs.to(config.device)
            output, patches, ids_mask = model(imgs)
            loss = mae_loss(output, patches, ids_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1} Loss: {sum(losses[-len(train_loader):])/len(train_loader):.4f}")

    return losses
