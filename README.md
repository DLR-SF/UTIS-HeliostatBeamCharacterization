[![DOI](https://img.shields.io/badge/DOI-10.1016/j.solener.2024.112811-brightgreen)](https://doi.org/10.1016/j.solener.2024.112811)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![PyPI](https://img.shields.io/pypi/v/propulate)
![PyPI - Downloads](https://img.shields.io/pypi/dm/propulate)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)[![](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/7785/badge)](https://www.bestpractices.dev/projects/7785)
[![](https://img.shields.io/badge/Contact-propulate%40lists.kit.edu-orange)](mailto:propulate@lists.kit.edu)
[![Documentation Status](https://readthedocs.org/projects/propulate/badge/?version=latest)](https://propulate.readthedocs.io/en/latest/?badge=latest)
![](./coverage.svg)

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

