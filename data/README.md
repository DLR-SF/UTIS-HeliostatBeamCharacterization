# Data Directory

This directory contains all the datasets and images used for artificial data generation, training the UNet model, and testing the model's inference capabilities.

## Structure

The `data/` directory is organized into the following subdirectories:

### 1. `emptyTargetImages/`
   - **Description:** This folder contains images of empty target areas without any flux. These images serve as the background for generating artificial training data.
   - **Usage:** These images are combined with the flux images in the `fluxImages/` directory to create synthetic training samples. The UNet model uses these generated samples to learn how to separate the background from the flux.

### 2. `fluxImages/`
   - **Description:** This folder contains images representing the flux, or the concentrated solar power, typically seen on a target in solar tower plants.
   - **Usage:** These images are used to overlay the `emptyTargetImages` to simulate real scenarios in the artificial data generation process. The combined images are used for training the UNet model to predict the flux distribution.

### 3. `realSamples/`
   - **Description:** This folder contains real images of the target with actual flux distributions. These are captured from real-world scenarios in solar tower plants.
   - **Usage:** These real samples are used to test the inference capabilities of the trained UNet model. The model's ability to accurately predict flux from these real samples validates its effectiveness.

## How to Use
- The image data in this directory is loaded and processed via the `utis.dataset.py` script, which prepares the data for model training and inference. Make sure that the images are correctly organized and accessible for this script to function properly.

## Important Notes
- Ensure that all images have the same resolution before loading them for model training or inference.


## Example Usage

- **Training:** 
  - `python train-model` (uses generated data from `emptyTargetImages/` and `fluxImages/` for training)
  
- **Inference:**
  - `python run-inference` (runs inference on images in `realSamples/`)

This directory is crucial for the data-driven aspects of the UNet model in the UTIS-HeliostatBeamCharacterization project. Ensure that all data is correctly prepared and organized before running the model.
