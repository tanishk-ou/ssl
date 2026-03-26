# Architecture Overview

## Design Philosophy

This project uses a **hybrid modular architecture** that balances simplicity for end-users with flexibility for developers.

### Three-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Level 1: User Entry Points                                  │
│  - main.py: Unified training script                        │
│  - eval.py: Unified evaluation script                      │
│  - config.yaml: User-editable configuration                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Level 2: Method-Specific Modules (methods/)                 │
│  - simclr/: SimCLR implementation                           │
│    ├─ model.py: Vision Transformer + projection head       │
│    ├─ loss.py: Contrastive loss                            │
│    └─ trainer.py: Training loop                            │
│                                                              │
│  - mae/: Masked Autoencoder implementation                  │
│    ├─ model.py: Encoder + decoder with masking             │
│    ├─ loss.py: Reconstruction loss                         │
│    └─ trainer.py: Training loop                            │
│                                                              │
│  - dino/: Vision Transformer self-distillation             │
│    ├─ model.py: Student + teacher networks                 │
│    ├─ loss.py: Knowledge distillation loss                 │
│    └─ trainer.py: Training loop                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Level 3: Shared Infrastructure (core/ + utils/)             │
│                                                              │
│ Core Components:                                             │
│  - config.py: Unified configuration system                  │
│  - datasets.py: Dataset loaders for all methods             │
│  - transforms.py: Data augmentations                        │
│  - base_trainer.py: Abstract trainer class                  │
│  - schedulers.py: Learning rate and weight decay schedulers │
│                                                              │
│ Utilities:                                                   │
│  - eval_utils.py: k-NN and linear probe evaluation         │
│  - visualization.py: t-SNE, UMAP, attention visualizations │
│  - checkpointing.py: Save/load functionality                │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System (`core/config.py`)

**Purpose**: Centralized configuration management with YAML flexibility

```python
from core.config import config

# Access configuration
config.device          # 'cuda' or 'cpu'
config.dataset_path    # Path to dataset
config.SimCLR.batch_size
config.DINO.epochs
```

**Configuration File** (`config.yaml`):
```yaml
data_path: "./ssl_dataset"
checkpoint_path: "./checkpoints"
results_path: "./results"
device: "cuda"
num_classes: 100
```

### 2. Datasets (`core/datasets.py`)

**Three Dataset Classes**:
- `SimCLRDataset`: Returns two augmented views of same image
- `MAEDataset`: Returns single image for reconstruction
- `DINODataset`: Returns global and local crops

**DataLoader Factories**:
```python
from core.datasets import get_train_dataloader, get_val_dataloader

# Get training loader
train_loader = get_train_dataloader('simclr', batch_size=128)

# Get validation loader
val_loader = get_val_dataloader(batch_size=256)
```

### 3. Transforms/Augmentations (`core/transforms.py`)

**Key Classes**:
- `get_simclr_transforms()`: SimCLR augmentation pipeline
- `get_mae_transforms()`: MAE augmentation pipeline
- `get_eval_transforms()`: Evaluation transforms (no augmentation)
- `DINOTransform`: Complex transform with global + local crops
- `CosineScheduler`: Cosine annealing scheduler
- `LinearScheduler`: Linear interpolation scheduler

### 4. Base Trainer (`core/base_trainer.py`)

**Abstract Base Class** for all trainers:

```python
class BaseTrainer(ABC):
    """Abstract base for all SSL trainers"""

    @abstractmethod
    def train_step(self, batch):
        """Single training step - implemented by subclasses"""
        pass

    @abstractmethod
    def get_optimizer(self):
        """Get optimizer - implemented by subclasses"""
        pass

    def train(self, epochs):
        """Main training loop"""
        pass

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        pass
```

**Benefits**:
- Consistent training interface across methods
- Automatic logging and checkpointing
- Easy to extend with new methods

### 5. Method-Specific Trainers

Each method inherits from `BaseTrainer`:

```python
class SimCLRTrainer(BaseTrainer):
    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=...)

    def train_step(self, batch, optimizer):
        # SimCLR-specific training logic
        pass

class MAETrainer(BaseTrainer):
    def get_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=...)

    def train_step(self, batch, optimizer):
        # MAE-specific training logic
        pass

class DINOTrainer(BaseTrainer):
    def get_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=...)

    def train_step(self, batch, optimizer, epoch, total_epochs):
        # DINO-specific training logic with EMA update
        pass
```

## Data Flow Diagram

### Training Pipeline

```
config.yaml
    ↓
config.py (load configuration)
    ↓
main.py (parse arguments, select method)
    ├→ methods/{method}/model.py (create model)
    ├→ core/datasets.py (create dataloaders)
    ├→ methods/{method}/trainer.py (create trainer)
    ↓
Trainer.train() (training loop)
    ├→ data augmentation (core/transforms.py)
    ├→ model.forward() → loss computation
    ├→ loss.backward() → optimizer.step()
    ├→ logging + checkpointing
    ↓
checkpoints/{method}/model.pth (save weights)
```

### Evaluation Pipeline

```
eval.py (parse arguments)
    ↓
get_model() (load checkpoint)
    ↓
extract_features() (encode dataset)
    │
    ├→ knn_evaluation()
    │   └→ sklearn.neighbors.KNeighborsClassifier
    │       └→ Accuracy metric
    │
    └→ linear_probe_evaluation()
        ├→ nn.Linear (frozen backbone)
        ├→ training loop with CrossEntropyLoss
        └→ Accuracy metric

    ↓
results/{method}/evaluation_results.json (save results)
```

## Adding a New SSL Method

To add a new SSL method (e.g., BYOL):

1. **Create Method Directory**:
```bash
mkdir methods/byol
touch methods/byol/{__init__,model,loss,trainer}.py
```

2. **Create Model** (`methods/byol/model.py`):
```python
import torch.nn as nn

class BYOLModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Model implementation

    def forward(self, x):
        # Return embeddings
        pass
```

3. **Create Loss** (`methods/byol/loss.py`):
```python
def byol_loss(online_proj, target_proj):
    # Loss implementation
    pass
```

4. **Create Trainer** (`methods/byol/trainer.py`):
```python
from core.base_trainer import BaseTrainer

class BYOLTrainer(BaseTrainer):
    def get_optimizer(self):
        pass

    def train_step(self, batch, optimizer):
        pass
```

5. **Register in config** (`core/config.py`):
```python
class Config:
    class BYOL:
        batch_size = 256
        lr = 0.001
        # ... other params
```

6. **Add to main.py**:
```python
# In train_byol() function
from methods.byol.model import BYOLModel
from methods.byol.trainer import BYOLTrainer

# Add to argument parser
parser.add_argument('--method', choices=[..., 'byol'])

# Add to training function
if args.method == 'byol':
    train_byol(args)
```

## File Organization

### Core Infrastructure Files

| File | Purpose |
|------|---------|
| `config.yaml` | User-editable configuration |
| `core/config.py` | Python configuration loading |
| `core/datasets.py` | All dataset classes and loaders |
| `core/transforms.py` | All augmentation pipelines |
| `core/base_trainer.py` | Abstract trainer base class |
| `core/schedulers.py` | Learning rate schedulers |

### Method Files

Each method (simclr/, mae/, dino/) contains:
- `model.py`: Model architecture
- `loss.py`: Loss function
- `trainer.py`: Training implementation

### User Entry Points

| File | Purpose |
|------|---------|
| `main.py` | Training orchestrator |
| `eval.py` | Evaluation and visualization |
| `config.yaml` | Configuration |

### Results

| Directory | Contents |
|-----------|----------|
| `checkpoints/{method}/` | Trained models + logs |
| `results/{method}/` | Visualizations + JSON results |

## Design Patterns Used

### 1. **Factory Pattern**
```python
# config.py loads configuration
from core.config import config

# Dataloader factories create appropriate loaders
train_loader = get_train_dataloader('simclr')
```

### 2. **Abstract Base Class Pattern**
```python
# BaseTrainer defines interface
class SimpleClRTrainer(BaseTrainer):
    # Specific implementations
```

### 3. **Strategy Pattern**
```python
# Different methods use different losses but same training interface
trainer.train_step(batch, optimizer)
```

### 4. **Composition Over Inheritance**
```python
# Models compose components (encoder + projection head)
# Trainers compose models + optimizers
```

## Performance Considerations

### Memory Optimization
- Gradient checkpointing in Vision Transformers
- Batch accumulation for large batches
- Mixed precision training (automatic in trainers)

### Speed Optimization
- DataLoader with multiple workers
- Pin memory for GPU data transfer
- CUDA synchronization minimized

### Reproducibility
- Fixed random seeds in config
- Detailed logging of hyperparameters
- Checkpoint saving at regular intervals

## Testing Strategy

See `tests/` directory for:
- `test_models.py`: Model instantiation and forward pass
- `test_datasets.py`: Dataset loading and augmentation
- `test_training.py`: Training loop functionality
- `test_evaluation.py`: Evaluation metrics

## Future Extensions

1. **More Methods**: Add contrastive methods (BYOL, MoCo v3)
2. **Distributed Training**: Multi-GPU support
3. **Hyperparameter Optimization**: Automated tuning
4. **Federated Learning**: Privacy-preserving training
5. **Model Compression**: Knowledge distillation, quantization

---

For detailed usage, see [README.md](README.md)
