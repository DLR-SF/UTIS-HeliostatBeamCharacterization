import torch
import data.dataset as dataset
from models.UNet_3Plus_ import UNet_3Plus
import os
from torchvision.utils import save_image, make_grid

import torch
import data.dataset as dataset
from models.UNet_3Plus_ import UNet_3Plus
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid

class UNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet_3Plus()


class UNetInference:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the UNetInference class.

        Args:
            model_path (str): Path to the saved model checkpoint.
            device (str): Device to run the inference on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        
        # Initialize the model with the same parameters used during training
        self.model = UNet().to(self.device)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load the model state dictionary from the checkpoint
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
    
    def predict(self, images):
        """
        Perform prediction on the given images.
        """
        with torch.no_grad():
            predictions = self.model.model(images)
        return predictions

def main():
    # Parameters
    model_path = 'trained_models/UNet3plus.ckpt'  # Path to the saved model checkpoint
    device = 'cpu'  # Device to run the inference on ('cpu' or 'cuda')

    # Load real samples
    real_samples = dataset.load_images('./data/realSamples').to(device)
    
    # Initialize the inference model
    unet_inference = UNetInference(model_path=model_path, device=device)
    
    # Perform prediction
    predictions = unet_inference.predict(real_samples)
    
    data = torch.cat([real_samples[:, 0], predictions[:, 0], predictions[:, 1]], dim=0)
    grid = make_grid(data.unsqueeze(1), nrow=real_samples.size(0)) 
    save_image(grid, 'results.png')
    
if __name__ == "__main__":
    main()
