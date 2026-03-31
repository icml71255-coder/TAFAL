# TAFAL

This repository contains the code for **TAFAL**.  
It is anonymized for double-blind ICML review.

## Environment Setup

### Requirements
- Python 3.10  
- CUDA 12.x (for GPU experiments)  
- NVIDIA driver compatible with the PyTorch CUDA build  

### Create Conda Environment
```bash
conda create -n tafal python=3.10 -y
conda activate tafal
conda install pip -y
```

### Install PyTorch (GPU)
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Addition Baselines

| Method | ViT-B/16 Abs | ViT-B/16 Norm | ViT-B/32 Abs | ViT-B/32 Norm | ViT-L/14 Abs | ViT-L/14 Norm | RoBERTa Abs | RoBERTa Norm |
|--------|--------------|---------------|--------------|---------------|--------------|---------------|-------------|--------------|
| Pretrained | 53.09 | 58.98 | 44.19 | 49.33 | 61.33 | 65.86 | 35.51 | 39.38 |
| Task Arithmetic | 65.62 | 72.91 | 62.32 | 69.57 | 80.03 | 85.94 | 58.47 | 70.82 |
| ATLAS | 62.22 | 69.13 | 62.32 | 69.57 | 69.99 | 75.16 | 59.6 | 72.2 |
| TALoS | 69.8 | 77.55 | 64.14 | 71.6 | 76.86 | 82.53 | 63.54 | 76.97 |
| TauJp |  |  | 75.903 | 85.029 |  |  | 62.1 | 75.2 |
| KFAC |  |  | 74.977 | 83.762 |  |  | 59.61 | 72.21 |
| Iso-C | 76.24 | 85.24 | 73.88 | 82.89 | 84.12 | 90.38 | 62.7 | 75.95 |
| Iso-CTS | 77.3 | 86.38 | 75.61 | 84.82 | 84.57 | 90.88 | 61.8 | 74.86 |
| Task Singular Vectors | 76.88 | 85.96 | 74.89 | 83.98 | 84.07 | 90.39 | 67.03 | 81.19 |
| TAFAL (Ours) | 79.73 | 88.58 | 77.55 | 86.58 | 86.89 | 93.31 | 72.52 | 87.85 | 

### Task Negation (Retaining 95 percent of the pretrained accuracy)

| Method | ViT-B/16 Tar (↓) | ViT-B/16 Cont (↑) | ViT-B/32 Tar (↓) | ViT-B/32 Cont (↑) | ViT-L/14 Tar (↓) | ViT-L/14 Cont (↑) | RoBERTa Tar (↓) | RoBERTa Cont (↑) |
|--------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|
| Pretrained | 53.09 | 68.37 | 44.19 | 63.26 | 61.33 | 75.53 | 57.42 | 65.83 |
| Task Arithmetic | 19.86 | 64.58 | 23.4 | 63.82 | 19.45 | 72.96 | 37.17 | 65.53 |
| ATLAS | 18.55 | 67.02 | 18.44 | 61.89 | 17.07 | 73.8 | 55.44 | 64.42 |
| TALoS | 15.3 | 75.6 | 10.96 | 66.11 | 14.55 | 78.71 | 35.79 | 65.16 |
| TAFAL | 11.7 | 71.95 | 9.62 | 75.675 | 10.91 | 75.9 | 33.3 | 75.17 |

## Benchmark: Time & Storage Usage Across Methods For RoBERTa

ATLAS and TaLoS for 5 epochs, Taujp and KFAC for 3 epochs and 10-value hyperparameter search, Task Arithmetic, TSV and Iso-CTS for 10-value hyperparameter search, and TAFAL.

| Method                              | Time (seconds) | Storage (MB) |
|-------------------------------------|---------------:|------------:|
| Task Arithmetic (10-value search)   | 823            | N/A         |
| TAFAL                               | 701            | 4536        |
| ATLAS (5 epochs)                    | 2671           | N/A         |
| TaLoS (5 epochs)                    | 4763           | 3786        |
| TSV                                 | 3998           | 6713        |
| Iso-CTS                             | 3982           | 7351        |
| Taujp                               | 4156           | N/A         |
| KFAC                                | 4262           | 8100        |

### BLIP Addition Results

| Method | OK-VQA Abs | OK-VQA Norm | VQAv2 Abs | VQAv2 Norm | GQA Abs | GQA Norm | COCO Abs | COCO Norm | Flickr30k Abs | Flickr30k Norm |
|--------|------------|-------------|-----------|------------|---------|----------|----------|-----------|----------------|----------------|
| Pretrained | 28.70 | 50.52 | 65.60 | 91.02 | 29.28 | 45.88 | 80.25 | 90.50 | 0.482 | 74.61 |
| ATLAS | 49.50 | 87.14 | 66.00 | 91.63 | 31.29 | 48.62 | 84.88 | 92.16 | 0.532 | 82.60 |
| Task Arithmetic | 47.00 | 82.74 | 62.70 | 88.66 | 27.61 | 43.35 | 75.33 | 84.63 | 0.609 | 94.27 |
| TAFAL (Ours) | 53.20 | 93.66 | 68.10 | 93.24 | 33.54 | 51.83 | 88.59 | 95.79 | 0.628 | 97.21 |
| Iso-C | 49.33 | 86.84 | 63.40 | 89.30 | 30.04 | 46.62 | 63.86 | 72.40 | 0.530 | 82.04 |
| Iso-CTS | 46.66 | 82.14 | 65.10 | 90.69 | 30.67 | 47.41 | 83.92 | 90.99 | 0.600 | 92.87 |
| Task Singular Vectors | 52.80 | 92.95 | 64.80 | 90.58 | 31.92 | 49.06 | 72.84 | 82.19 | 0.580 | 89.78 |

### Lambda Ablation

| **Sun397 & ImageNet** |  |  | **MNIST & ImageNet** |  |  |
|--------------------------------|--|--|----------------------|--|--|
| Lambda | Sun397 (Target) | Imagenet (Control) | Lambda | MNIST | Imagenet |
|--------|-----------------|--------------------|--------|-------|----------|
| 0.1 | 57.35 | 80.23 | 0.1 | 12.97 | 75.54 |
| 0.5 | 56.92 | 79.91 | 0.5 | 12.97 | 75.54 |
| 1 | 54.97 | 78.05 | 1 | 12.97 | 75.54 |
| 5 | 51.01 | 74.69 | 10 | 12.97 | 75.54 |
| 10 | 49.49 | 72.56 | 20 | 12.97 | 75.54 |
| 50 | 47.59 | 69.01 | 50 | 12.97 | 75.54 |
| 100 | 46.93 | 67.96 | 100 | 12.97 | 75.54 |
| 500 | 46.08 | 66.79 |  |  |  |
| 1000 | 45.58 | 65.63 |  |  |  |
| 10,000 | 45.74 | 66.19 |  |  |  |
| 50,000 | 45.8 | 66.3 |  |  |  |

### Layer-wise Subspace Overlap

![Layer-wise Subspace Overlap](assets/layerwise_subspace_overlap.png)

## Notes
- All dependencies are specified in `requirements.txt`.
- This repository will be de-anonymized upon acceptance.
