import torch
import yaml
import os
from pathlib import Path


def load_config():
    """Load configuration from config.yaml file"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        return yaml_config
    else:
        # Return defaults if config.yaml doesn't exist
        return get_default_config()


def get_default_config():
    """Default configuration if config.yaml is not found"""
    return {
        "data_path": "./ssl_dataset",
        "checkpoint_path": "./checkpoints",
        "results_path": "./results",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_classes": 100,
    }


class Config:
    """Unified configuration class for all SSL methods"""

    # Load from config.yaml or use defaults
    _yaml_config = load_config()

    # Device and Dataset settings
    device = torch.device(_yaml_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    image_size = (224, 224)
    patch_size = 16
    embed_dim = 768
    num_classes = _yaml_config.get("num_classes", 100)

    dataset_path = _yaml_config.get("data_path", "./ssl_dataset")
    checkpoint_path = _yaml_config.get("checkpoint_path", "./checkpoints")
    results_path = _yaml_config.get("results_path", "./results")

    epochs_pretrain = 1000
    epochs_linear_eval = 20

    # SimCLR Configuration
    class SimCLR:
        epochs = 1000
        batch_size = 128
        lr = 3e-4
        temperature = 0.5
        projection_dim = 256
        image_size = (224, 224)
        weight_decay = 1e-6

    # MAE Configuration
    class MAE:
        epochs = 600
        batch_size = 256
        lr_base = 1.5e-4
        weight_decay = 0.05
        mask_ratio = 0.75
        image_size = (224, 224)

    # DINO Configuration
    class DINO:
        epochs = 1000
        batch_size = 96
        lr = 5e-4
        weight_decay_start = 0.04
        weight_decay_end = 0.4
        momentum_start = 0.996
        momentum_end = 1.0
        student_temp = 0.1
        teacher_temp_start = 0.04
        teacher_temp_end = 0.07
        teacher_temp_warmup_epochs = 30
        prototypes = 65536
        num_local_crops = 4
        image_size = (224, 224)

    # DINOv2 Configuration
    class DINOv2:
        epochs = 1000
        batch_size = 96
        lr = 1e-4
        weight_decay = 0.05
        llrd_factor = 0.98
        student_temp = 0.1
        teacher_temp = 0.07
        momentum = 0.999
        prototypes = 4096
        bottleneck_dim = 128
        hidden_dim = 768
        num_local_crops = 2
        mask_ratio = 0.5

    # DINOv3 Configuration
    class DINOv3:
        is_enabled = False
        gram_teacher_checkpoint = None

    # Evaluation Configuration
    class Eval:
        k_knn = 20
        epochs_linear = 20
        batch_size = 256


# Alias for backward compatibility
config = Config()
