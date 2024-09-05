import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

def load_images(path):
    """
    Load images from the specified directory and convert them to a tensor.

    Args:
        path (str): Path to the directory containing images.

    Returns:
        torch.Tensor: A tensor containing all the loaded images.
    """
    
    transform = transforms.ToTensor()
    image_data = [transform(Image.open(os.path.join(path, f))) for f in os.listdir(path) if f.endswith(".png")]

    return torch.stack(image_data)

class PNGDataset(Dataset):
    """
    A dataset class for loading and managing PNG images.
    """
    def __init__(self):
        """
        Initialize the dataset by loading empty target images and flux images.
        """
        self.emptyTargetImages = load_images('./data/emptyTargetImages')
        self.fluxImages = load_images('./data/fluxImages')

        print(f'Empty Target Images loaded with size: {self.emptyTargetImages.size()}')
        print(f'Flux Images loaded with size: {self.fluxImages.size()}')

    def getReflMap(self):
        """
        Compute the reflection map by averaging the empty target images.

        Returns:
            torch.Tensor: The normalized reflection map.
        """
        reflMap = torch.mean(self.emptyTargetImages, dim=0)
        reflMap = reflMap / torch.max(reflMap)
        return reflMap

    def __len__(self):
        return self.fluxImages.size(0)

    def __getitem__(self, idx):
        
        if isinstance(idx, (list, torch.Tensor)):
            random_indices = torch.randint(0, len(self.emptyTargetImages), (len(idx),))
        else:
            random_indices = torch.randint(0, len(self.emptyTargetImages), (1,)).item()

        imgs = self.emptyTargetImages[random_indices]
        fluxes = self.fluxImages[idx]
        return {'imgs': imgs, 'fluxes': fluxes}

def get_dataloader_and_reflmap(batch_size):
    """
    Get a DataLoader and reflection map for the dataset.
    """
    dataset = PNGDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=True)
    reflMap = dataset.getReflMap()
    return dataloader, reflMap

if __name__ == '__main__':
    dataset = PNGDataset()
