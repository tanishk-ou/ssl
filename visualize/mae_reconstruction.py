import matplotlib.pyplot as plt
import torch
import random
import os
from PIL import Image
from config import config
from utils.transforms import get_transform

def visualize_reconstruction(model, dataset_path):
    val_classes = os.listdir(os.path.join(dataset_path, 'val'))
    cls = random.choice(val_classes)
    image_name = random.choice(os.listdir(os.path.join(dataset_path, 'val', cls)))

    image = Image.open(os.path.join(dataset_path, 'val', cls, image_name)).convert('RGB')
    image = get_transform(False)(image)

    model.eval()
    output, _, _ = model(image.unsqueeze(0).to(config.device))
    output = model.unpatchify(output)[0]

    image = torch.clamp(image.cpu().permute(1, 2, 0), 0, 1).numpy()
    output = torch.clamp(output.detach().cpu().permute(1, 2, 0), 0, 1).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title("Reconstructed")
    plt.axis('off')

    plt.show()
