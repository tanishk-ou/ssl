import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
import torch


class SimCLRDataset(Dataset):
    """Dataset for SimCLR - returns two augmented views of the same image"""

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
                image = Image.open(
                    os.path.join(self.folder_path, self.image_filenames[index])
                ).convert('RGB')
                return self.transform(image), self.transform(image)
            except (UnidentifiedImageError, OSError):
                index = (index + 1) % len(self.image_filenames)


class MAEDataset(Dataset):
    """Dataset for MAE - returns single image with masking during training"""

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
                image = Image.open(
                    os.path.join(self.folder_path, self.image_filenames[index])
                ).convert('RGB')
                return self.transform(image)
            except (UnidentifiedImageError, OSError):
                old_path = os.path.join(self.folder_path, self.image_filenames[index])
                image_name = os.path.basename(old_path)
                folder = image_name.split('_')[0]
                new_path = os.path.join(
                    os.path.dirname(old_path).replace('unlabeled', 'labeled'),
                    folder,
                    image_name
                )
                image = Image.open(new_path).convert('RGB')
                return self.transform(image)


class DINODataset(Dataset):
    """Dataset for DINO - returns list of global and local crops"""

    def __init__(self, path, transform, v3=False):
        super().__init__()
        self.image_filenames = os.listdir(path)
        self.folder_path = path
        self.transform = transform
        self.v3 = v3

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        try:
            image = Image.open(
                os.path.join(self.folder_path, self.image_filenames[index])
            ).convert('RGB')
        except:
            old_path = os.path.join(self.folder_path, self.image_filenames[index])
            image_name = os.path.basename(old_path)
            folder = image_name.split('_')[0]
            new_path = os.path.join(
                os.path.dirname(old_path).replace('unlabeled', 'labeled'),
                folder,
                image_name
            )
            image = Image.open(new_path).convert('RGB')

        crops = self.transform(image)
        if self.v3:
            return crops[:2], crops[2:-2], crops[-2:]
        return crops[:2], crops[2:]


def get_train_dataloader(method="simclr", batch_size=None, config=None):
    """Get training dataloader for unlabeled data"""
    if config is None:
        from core.config import config as cfg
        config = cfg

    if batch_size is None:
        batch_size = getattr(getattr(config, method.capitalize()), 'batch_size', 128)

    train_folder = os.path.join(config.dataset_path, "train_unlabeled")

    if method.lower() == "simclr":
        from core.transforms import get_simclr_transforms
        transform = get_simclr_transforms()
        dataset = SimCLRDataset(train_folder, transform)
    elif method.lower() == "mae":
        from core.transforms import get_mae_transforms
        transform = get_mae_transforms()
        dataset = MAEDataset(train_folder, transform)
    elif method.lower() == "dino":
        from core.transforms import DINOTransform
        transform = DINOTransform()
        dataset = DINODataset(train_folder, transform)
    else:
        raise ValueError(f"Unknown method: {method}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


def get_linear_eval_dataloader(batch_size=256, config=None):
    """Get training dataloader for labeled data (linear evaluation)"""
    if config is None:
        from core.config import config as cfg
        config = cfg

    train_folder = os.path.join(config.dataset_path, "train_labeled")

    from core.transforms import get_eval_transforms
    transform = get_eval_transforms()
    dataset = ImageFolder(train_folder, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


def get_val_dataloader(batch_size=256, config=None):
    """Get validation dataloader"""
    if config is None:
        from core.config import config as cfg
        config = cfg

    val_folder = os.path.join(config.dataset_path, "val")

    from core.transforms import get_eval_transforms
    transform = get_eval_transforms()
    dataset = ImageFolder(val_folder, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
