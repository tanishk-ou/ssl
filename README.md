# Self-Supervised Visual Representation Learning

This repository contains a complete implementation of self-supervised learning using two paradigms:
- SimCLR (Contrastive Learning)
- MAE (Masked Autoencoders)

## 👨‍💻 Author

- **Name:** Tanishk Gopalani  
- **Roll No:** 23/EE/266  
- **Email:** gopalanitanishk@gmail.com  
- **College:** Delhi Technological University (DTU)  
- **Branch:** Electrical Engineering  

## 📂 Folder Structure

```
SSL_Project/
│
├── README.md
├── requirements.txt
│
├── config.py
├── utils/
│   ├── transforms.py
│   ├── dataset_loader.py
│   ├── eval_utils.py
│
├── models/
│   ├── simclr.py
│   ├── mae.py
│   ├── decoder.py
│
├── loss/
│   ├── simclr_loss.py
│   ├── mae_loss.py
│
├── train/
│   ├── pretrain_simclr.py
│   ├── pretrain_mae.py
│   ├── linear_eval_simclr.py
│   ├── linear_eval_mae.py
│
├── visualize/
│   └── mae_reconstruction.py
│
├── main_simclr.py
├── main_mae.py
├── linear_eval.py
├── evaluate_models.py
```

## 🔧 Setup & Installation

```bash
git clone https://github.com/your_username/ssl_project.git
cd ssl_project
pip install -r requirements.txt
```

## 🧠 Pretraining

### SimCLR Pretraining
```bash
python main_simclr.py
```

### MAE Pretraining
```bash
python main_mae.py
```

## 📈 Linear Evaluation

```bash
python linear_eval.py
```

## 🧪 Final Evaluation

```bash
python evaluate_models.py
```

## 🧾 Results

| Method | Accuracy | F1 Score |
|--------|----------|----------|
| SimCLR | 34.25%   | 0.3012   |
| MAE    | 25.08%   | 0.2337   |

## 📁 Dataset Folder Structure

```
ssl_dataset/
├── train_unlabeled/
├── train_labeled/
└── val/
```

Place your dataset inside `ssl_dataset/` directory following the above structure.

## 📚 References

- SimCLR - Chen et al., ICML 2020
- MAE - He et al., CVPR 2022
- ResNet - He et al., CVPR 2016
- Vision Transformers - Dosovitskiy et al., ICLR 2021
- PyTorch, HuggingFace Docs

## 🙏 Acknowledgements

Thanks to AIMS-DTU and the evaluation team for providing this opportunity for me. This project task helped me understand Self-supervised visual learning really well.
