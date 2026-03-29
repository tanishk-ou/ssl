# Self-Supervised Visual Representation Learning

A comprehensive implementation of modern self-supervised learning methods for visual representation learning. This repository includes three state-of-the-art SSL methods with pre-trained models.

## Featured Methods

### 1. **DINO** - Vision Transformer Self-Distillation
- **Best performance**: 64.06% linear evaluation accuracy
- Vision Transformer backbone with self-distillation
- Momentum encoder with EMA updates

### 2. **SimCLR** - Contrastive Learning
- **High performance**: 60.22% linear evaluation accuracy
- Simplified contrastive learning framework
- Temperature-scaled contrastive loss

### 3. **MAE** - Masked Autoencoders
- **Decent performance**: 58.12% linear evaluation accuracy
- Vision Transformer with masked patch reconstruction and 75% masking ratio
- Asymmetric encoder-decoder architecture

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/tanishk-ou/ssl_project.git
cd ssl_project

# Install dependencies
pip install -r requirements.txt

# Edit config.yaml for your environment
nano config.yaml
```

<details>
<summary>GPU Memory Requirements</summary>

- DINO: ~24GB (batch size 96)
- SimCLR: ~16GB (batch size 128)
- MAE: ~16GB (batch size 256)

</details>

## Dataset Preparation

The code requires the SSL dataset with the following structure:

```
ssl_dataset/
├── train_unlabeled/     (130,000 images for pretraining)
├── train_labeled/       (labeled data for linear evaluation)
└── val/                 (5,000 validation images)
```

Download the dataset:

Manually download from [Google Drive](https://drive.google.com/file/d/1--11Usx6nRxI1v3EJ6Qd4LQ2WJJwbqP8):

```bash
unzip ssl_dataset_resized.zip -d ./ssl_dataset/
```

## Quick Start

### Using Pre-trained Models

Load and use pre-trained encoders:

```python
import torch
from methods.dino.model import DINOModel

# Load pre-trained DINO
model = DINOModel()
model.load_state_dict(torch.load('checkpoints/dino/model.pth'))
model.eval()

# Extract features
with torch.no_grad():
    features = model.student_encoder(images)  # Get embeddings
```

### Training from Scratch

Train a new model:

```bash
# Train DINO (recommended - best results)
python main.py --method dino --epochs 1000 --batch-size 96

# Train SimCLR
python main.py --method simclr --epochs 1000 --batch-size 128

# Train MAE
python main.py --method mae --epochs 600 --batch-size 256
```

### Evaluation

Evaluate pre-trained models:

```bash
# k-NN evaluation
python eval.py --method dino --k 20

# Linear probe evaluation
python eval.py --method dino --linear-epochs 20

# Full evaluation with results JSON
python eval.py --method dino --generate-results

# Generate visualizations (t-SNE, UMAP)
python eval.py --method dino --visualize
```

## Results

### Evaluation Metrics

#### DINO
- **k-NN Accuracy (k=20)**: 56.16%
- **Linear Probe Accuracy**: 64.06%
- **Training Loss**: 0.6237
- **Training Duration**: ~6 days (Nvidia TITAN RTX GPU)
- **Model Size**: 843 MB

#### SimCLR
- **k-NN Accuracy (k=20)**: 54.92%
- **Linear Probe Accuracy**: 60.22%
- **Training Loss**: 0.0236
- **Training Duration**: ~9 days (Nvidia TITAN RTX GPU)
- **Model Size**: 331 MB

#### MAE
- **k-NN Accuracy (k=20)**: 36.34%
- **Linear Probe Accuracy**: 58.12%
- **Training Loss**: 0.0118
- **Training Duration**: ~4 days (Nvidia TITAN RTX GPU)
- **Model Size**: 405 MB

### Visual Results

#### DINO — Self-Attention Maps

The DINO encoder learns to attend to semantically meaningful regions without any labels. The attention maps below show the model focusing on object boundaries and salient parts (heads, bodies, distinguishing features):

<p align="center">
  <img src="results/dino/attention_map_masks_high_contrast.png" width="700" alt="DINO high-contrast attention maps showing object-aware attention"/>
</p>

Multi-layer attention aggregation reveals how early, mid, and late transformer layers contribute complementary spatial information (R=Early, G=Mid, B=Late), alongside overall attention intensity:

<p align="center">
  <img src="results/dino/attention_map_masks_no_overlap.png" width="700" alt="DINO multi-layer attention aggregation (RGB per-layer, Plasma intensity)"/>
</p>

#### MAE — Masked Reconstruction

The MAE model learns rich visual representations by reconstructing images from only 25% of visible patches. Below are original images alongside their reconstructions from the heavily masked input:

<p align="center">
  <img src="results/mae/mae_reconstruction.png" width="700" alt="MAE reconstruction results — original vs reconstructed from 25% visible patches"/>
</p>

#### Feature Space Embeddings

t-SNE and UMAP projections of the learned feature spaces show how well each method clusters semantically similar images. Distinct color clusters indicate the model has learned meaningful class-separable representations without supervision.

**DINO** (best separation — clear cluster formation):

| t-SNE | UMAP |
|:---:|:---:|
| ![DINO t-SNE](results/dino/tsne.png) | ![DINO UMAP](results/dino/umap.png) |

**SimCLR** (strong contrastive clustering):

| t-SNE | UMAP |
|:---:|:---:|
| ![SimCLR t-SNE](results/simclr/tsne.png) | ![SimCLR UMAP](results/simclr/umap.png) |

**MAE** (generative — more diffuse embeddings, still structured):

| t-SNE | UMAP |
|:---:|:---:|
| ![MAE t-SNE](results/mae/tsne.png) | ![MAE UMAP](results/mae/umap.png) |

### Training Logs

Complete training logs for all three methods are in the [`logs/`](logs/) directory, providing epoch-by-epoch loss values and timestamps from the actual training runs:

| Method | Log File | Epochs | Training Period | GPU Time |
|--------|----------|--------|-----------------|----------|
| **DINO** | [`logger_dino.log`](logs/logger_dino.log) | 432 (early stopped) | Sep 30 – Oct 5, 2025 | ~6 days |
| **MAE** | [`logger_mae.log`](logs/logger_mae.log) | 600 | Sep 15 – Sep 19, 2025 | ~4 days |
| **SimCLR** | [`logger_simclr.log`](logs/logger_simclr.log) | 976 (early stopped) | Sep 19 – Sep 28, 2025 | ~9 days |

## Architecture

The repository uses a clean, modular architecture:

```
ssl-project/
├── main.py                  # Training orchestrator
├── eval.py                  # Evaluation script
├── config.yaml              # User configuration
│
├── methods/                 # Individual SSL methods
│   ├── simclr/
│   │   ├── model.py
│   │   ├── loss.py
│   │   └── trainer.py
│   ├── mae/
│   │   ├── model.py
│   │   ├── loss.py
│   │   └── trainer.py
│   └── dino/
│       ├── model.py
│       ├── loss.py
│       └── trainer.py
│
├── core/                    # Shared infrastructure
│   ├── config.py
│   ├── datasets.py          # Dataset loaders
│   ├── transforms.py        # Data augmentations
│   ├── base_trainer.py      # Abstract trainer
│   └── schedulers.py        # Learning rate schedulers
│
├── utils/                   # Utilities
│   ├── eval_utils.py        # Evaluation functions
│   └── visualization.py     # Visualization tools
│
├── checkpoints/             # Pre-trained models
│   ├── dino/model.pth
│   ├── simclr/model.pth
│   └── mae/model.pth
│
└── results/                 # Evaluation results & visualizations
    ├── dino/
    ├── simclr/
    └── mae/
│
├── logs/                    # Raw training logs (proof of training)
│   ├── logger_dino.log
│   ├── logger_mae.log
│   └── logger_simclr.log
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architectural information.

## Key Features

✅ **Three State-of-the-Art Methods**: DINO, SimCLR, MAE with unified interface

✅ **Pre-trained Model Weights**: All three methods with trained models included

✅ **Comprehensive Evaluation**: k-NN and Linear Probe evaluation metrics

✅ **Visualizations**: t-SNE and UMAP visualizations

✅ **Clean Codebase**: Modular design with easy-to-follow implementation

✅ **Extensive Logging**: Detailed training logs for each method

✅ **Flexible Configuration**: YAML-based config for easy customization

## Configuration

Edit `config.yaml` to customize paths and hyperparameters:

```yaml
# Data paths
data_path: "./ssl_dataset"
checkpoint_path: "./checkpoints"
results_path: "./results"

# Device
device: "cuda"  # or "cpu"

# Dataset
num_classes: 100
```

Method-specific configurations are in `core/config.py`.

## Methods in Detail

### DINO: Emerging Properties in Self-Supervised Vision Transformers
- **Paper**: [ICCV 2021](https://arxiv.org/abs/2104.14294)
- **Key Innovation**: Student-teacher self-distillation without labels
- **Strengths**:
  - Excellent unsupervised segmentation properties
  - Clear cluster formation in feature space
  - Strong linear evaluation performance
  - Interpretable attention maps

### SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
- **Paper**: [ICML 2020](https://arxiv.org/abs/2002.05709)
- **Key Innovation**: Large batch contrastive learning with momentum contrast
- **Strengths**:
  - Simple yet effective approach
  - Fast convergence
  - Strong contrastive properties
  - Widely adopted and studied

### MAE: Masked Autoencoders Are Scalable Vision Learners
- **Paper**: [CVPR 2022](https://arxiv.org/abs/2111.06377)
- **Key Innovation**: Simple masked reconstruction task (no labels needed)
- **Strengths**:
  - Efficient training with high masking ratio
  - Scalable to large models
  - Excellent for vision understanding tasks
  - Stable training dynamics

## Training Tips

1. **Batch Size Matters**: Larger batches typically improve performance
2. **Early Stopping**: All methods benefit from early stopping based on validation metrics
3. **Warm-up**: Consider learning rate warm-up for stability
4. **Data Augmentation**: Strong augmentations are critical for SSL
5. **GPU Memory**: Use gradient accumulation if memory is limiting

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `--batch-size 64`
- Use gradient accumulation
- Reduce image size in config

### Training is Too Slow
- Increase batch size (if memory allows)
- Use mixed precision training
- Reduce number of workers in dataloader

### Poor Evaluation Results
- Ensure data is properly downloaded and formatted
- Check that checkpoint files are loading correctly
- Verify feature extraction dimensions match expected values

## References

- **DINO**: Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. ICCV
- **SimCLR**: Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML
- **MAE**: He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR
- **Vision Transformer**: Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR

## Author

**Tanishk Gopalani**
- Roll No: 23/EE/266
- Email: gopalanitanishk@gmail.com
- College: Delhi Technological University (DTU)
- Branch: Electrical Engineering

## Acknowledgements

Thanks to:
- AIMS-DTU for the opportunity to work on SSL
- The authors of DINO, SimCLR, and MAE for their groundbreaking research
- The PyTorch and Hugging Face communities for excellent libraries

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

Cite the original papers for each method you use.

---

**Last Updated**: March 26, 2026
**Status**: Implementation Complete
