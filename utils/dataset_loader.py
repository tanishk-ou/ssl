import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
from config import config
from utils.transforms import get_transform

class simclr_dataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.image_filenames = os.listdir(path)
        self.folder_path = path
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        while True:
            try:
                image = Image.open(os.path.join(self.folder_path, self.image_filenames[index])).convert('RGB')
                return self.transform(image), self.transform(image)
            except (UnidentifiedImageError, OSError):
                index = (index + 1) % len(self.image_filenames)

class mae_dataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.image_filenames = os.listdir(path)
        self.folder_path = path
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        while True:
            try:
                image = Image.open(os.path.join(self.folder_path, self.image_filenames[index])).convert('RGB')
                return self.transform(image)
            except (UnidentifiedImageError, OSError):
                index = (index + 1) % len(self.image_filenames)

def train_dataloader(sim=True):
    batch_size = config.B_simclr if sim else config.B_mae
    train_folder = os.path.join(config.dataset_path, "train_unlabeled")
    transform = get_transform(sim=True if sim else False)
    dataset = simclr_dataset(train_folder, transform) if sim else mae_dataset(train_folder, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def linear_dataloader():
    train_folder = os.path.join(config.dataset_path, "train_labeled")
    transform = get_transform(sim=False)
    dataset = ImageFolder(train_folder, transform=transform)
    return DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

def val_dataloader():
    val_folder = os.path.join(config.dataset_path, "val")
    transform = get_transform(sim=False)
    dataset = ImageFolder(val_folder, transform=transform)
    return DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
