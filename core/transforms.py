import torchvision.transforms as T
import math


def get_simclr_transforms(image_size=(224, 224)):
    """SimCLR augmentation pipeline"""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.32, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_mae_transforms(image_size=(224, 224)):
    """MAE augmentation pipeline"""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.32, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_eval_transforms(image_size=(224, 224)):
    """Evaluation transforms (no augmentation, only normalization)"""
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class DINOTransform:
    """DINO augmentation pipeline with global and local crops"""

    def __init__(self, global_crop_size=(224, 224), local_crop_size=96,
                 num_local_crops=4, global_crop_scale=(0.32, 1.0),
                 local_crop_scale=(0.05, 0.32), v3=False):
        self.global_transform = T.Compose([
            T.RandomResizedCrop(
                global_crop_size,
                scale=global_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.num_local_crops = num_local_crops
        self.local_transform = T.Compose([
            T.RandomResizedCrop(
                local_crop_size,
                scale=local_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.v3 = v3

    def __call__(self, image):
        crops = []
        # Two global crops
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        # Multiple local crops
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        if self.v3:
            # Add high-resolution crops for DINOv3 with GRAM
            high_res_size = (448, 448)  # Default high-res size
            gram_transform = T.Compose([
                T.Resize(high_res_size, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            crops.append(gram_transform(image))
            crops.append(gram_transform(image))

        return crops


class CosineScheduler:
    """Cosine annealing scheduler"""

    def __init__(self, base_value, final_value, total_steps):
        self.start_value = base_value
        self.end_value = final_value
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_value(self, current_step=None):
        if current_step is None:
            t = min(self.current_step, self.total_steps)
        else:
            t = current_step
        cosine = 0.5 * (1 + math.cos(math.pi * t / self.total_steps))
        return self.end_value - (self.end_value - self.start_value) * cosine


class LinearScheduler:
    """Linear scheduler"""

    def __init__(self, base_value, final_value, total_steps):
        self.start_value = base_value
        self.end_value = final_value
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_value(self, current_step=None):
        if current_step is None:
            t = min(self.current_step, self.total_steps)
        else:
            t = current_step
        progress = t / self.total_steps
        return self.start_value + (self.end_value - self.start_value) * progress
