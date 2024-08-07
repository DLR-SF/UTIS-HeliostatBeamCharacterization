# UTIS-HeliostatBeamCharacterization
UNet-based Target Image Segmentation

## Overview

This project implements a UNet3+ framework for background separation and flux determination for calibration target images in solar tower plants.

## Directory Structure

- `data/` - Contains image data and data loading scripts.
- `logs/` - Directory to store training logs.
- `models/` - Directory for model architecture definitions, based on the [UNet3+Model]([URL](https://github.com/Owais-Ansari/Unet3plus)).
- `predict.py` - Script for running inference with a trained model.
- `__pycache__/` - Directory for Python cache files.
- `results.png` - Sample result image from the inference script.
- `trained_models/` - Directory to store trained model checkpoints.
- `train.py` - Script for training the UNet3+ model.
- `unet_pl_module.py` - PyTorch Lightning module for the UNet3+ model.
- `utils.py` - Utility functions for training and data processing.

## Usage

### Training

Run the training script:

```sh
python train.py
```

### Inference

Run the inference script:

```sh
python predict.py
```

### Results

Predicted images and a sample result grid are saved in the specified output directory and as results.png.

