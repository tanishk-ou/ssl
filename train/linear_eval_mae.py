import torch
from tqdm import tqdm
import torch.nn as nn

def train_linear_eval_mae(model, dataloader, optimizer, epochs=5):
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for imgs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss / len(dataloader))
        print(f"Epoch {epoch+1} Loss: {losses[-1]:.4f}")
    return losses
