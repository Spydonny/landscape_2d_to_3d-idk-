# landscape_2d_to_3d-idk-

A Python neural network project for converting 2D landscape images into 3D depth maps using DeepLabV3 + ResNet50.

> Work in progress — experimental research project.

## What it does

Takes a 2D landscape image as input and outputs a depth map / 3D representation using semantic segmentation and depth estimation techniques powered by the DeepLabV3 architecture with a ResNet50 backbone.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| PyTorch | Deep learning framework |
| DeepLabV3 + ResNet50 | Segmentation & depth model |
| CUDA | GPU acceleration |

## Project Structure

```
data/                       # Input images and datasets
deep_lab_v3_resnet50.py     # Model definition and inference
train.py                    # Training script
adasa.py                    # Data preprocessing / augmentation
cuda_check.py               # GPU availability check
rand_check.py               # Random seed / sanity checks
```

## Getting Started

```bash
# Check GPU availability
python cuda_check.py

# Train the model
python train.py

# Run inference
python deep_lab_v3_resnet50.py
```

## Requirements

Install dependencies:

```bash
pip install torch torchvision
```
