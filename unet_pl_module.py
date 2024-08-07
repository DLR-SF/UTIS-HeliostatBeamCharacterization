import torch
import pytorch_lightning as pl
import utils
import random
from models.UNet_3Plus_ import UNet_3Plus
from torchvision.utils import make_grid
import data.dataset as dataset
import torch.nn.functional as F

class UNet(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        
        self.model = UNet_3Plus()
        self.l2loss = torch.nn.MSELoss()
        self.reflMap = None
        self.realSamples = None
        
        # Thresholds for image scaling
        self.img_min = params['min_value_image']
        self.img_max = params['max_value_image']
        self.aug_min = params['augment_min_value']
        self.aug_max = params['augment_max_value']
        
        # Parameters for random flux shift
        self.shift_dev_x = params['augment_flux_center_dev_x']
        self.shift_dev_y = params['augment_flux_center_dev_y']
        self.shift_max_x = params['max_shift_x']
        self.shift_max_y = params['max_shift_y']
        
    def setup(self, stage: str) -> None:
        """
        Setup method to load reflection map and real samples.
        """
        self.reflMap = self.reflMap.to(self.device)
        self.realSamples = dataset.load_images('./data/realSamples').to(self.device)
        
    def training_step(self, batch, batch_idx):
        """
        Training step to process each batch of data.
        """
        emptyTargetImages_batch = batch["imgs"]
        fluxImages_batch = batch["fluxes"]
        
        # Apply augmentations to flux images
        fluxImages_batch = self.augmentFlux(fluxImages_batch)
        
        # Generate augmented images and compute the ground truth
        generated_images, emptyTargetImages_augmented = self.generateImage(emptyTargetImages_batch, fluxImages_batch.clone(), self.reflMap, augment=True)
        gt = torch.cat([fluxImages_batch, emptyTargetImages_augmented], dim=1)
        
        # Forward pass and compute loss
        out = self.model(generated_images)
        loss = self.l2loss(out, gt)
        
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        
        # Evaluation without augmentations every 10 batches
        if batch_idx % 10 == 0:
            generated_images, emptyTargetImages_augmented = self.generateImage(emptyTargetImages_batch, fluxImages_batch.clone(), self.reflMap, augment=False)
            gt = torch.cat([fluxImages_batch, emptyTargetImages_augmented], dim=1)
            
            out = self.model(generated_images)
            loss_val = utils.calc_flux_loss(out, gt)
                    
            self.log('val_loss', loss_val, on_epoch=True, sync_dist=True)
        
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        """
        Method to visualize results at the end of each training epoch.
        """
        self.model.eval()
        
        train_dataloader = self.trainer.train_dataloader
        batch = next(iter(train_dataloader))
        
        emptyTargetImages_batch = batch["imgs"][:9].to(self.device)
        fluxImages_batch = batch["fluxes"][:9].to(self.device)
        
        # Generate images and add to TensorBoard logger
        self.log_images(emptyTargetImages_batch, fluxImages_batch, "Results_Artificial", augment=False)
        self.log_images(emptyTargetImages_batch, fluxImages_batch, "Results_Artificial_Augmented", augment=True)
        
        # Show results on real data
        out = self.model(self.realSamples)
        data_real = torch.cat([self.realSamples[:, 0], out[:, 0], out[:, 1]], dim=0)
        grid = make_grid(data_real.unsqueeze(1), nrow=9) 
        self.logger.experiment.add_image("Results_Real", grid, self.current_epoch)
        
        self.model.train()
        
    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        optimizer = utils.AdamP(self.model.parameters(), lr=2e-4, weight_decay=1e-8)
        return optimizer
    
    def generateImage(self, aux, fluxes, reflMap, augment=True):
        """
        Generate augmented images by combining auxiliary images with flux images.
        """
        reflMap = reflMap.unsqueeze(1).repeat(aux.size(0), 1, 1, 1)

        # Apply augmentations during training to background
        if augment:
            augmented = self.augmentBackground(torch.cat([aux, reflMap], dim=1)) 
            augmented_aux = augmented[:, 0].unsqueeze(1)
            augmented_refl = augmented[:, 1].unsqueeze(1)
            max_ = self.aug_max
            min_ = self.aug_min
        else:
            augmented_aux = aux
            augmented_refl = reflMap
            max_ = self.img_max
            min_ = self.img_min

        refl_fluxes = torch.mul(fluxes, augmented_refl) 
        max_augmented_aux = augmented_aux.view(augmented_aux.size(0), -1).max(dim=1)[0]
        max_refl_fluxes = refl_fluxes.view(refl_fluxes.size(0), -1).max(dim=1)[0]

        # Calculate scaling factors to ensure the combined image max values are between [min_, max_]
        scale_upper = torch.clamp((max_ - max_augmented_aux) / max_refl_fluxes, min=0, max=1)
        scale_lower = torch.clamp((min_ - max_augmented_aux) / max_refl_fluxes, min=0, max=1)
        scaling_factors = scale_lower + (scale_upper - scale_lower) * torch.rand_like(scale_lower)

        # Apply scaling and add the images
        generated_images = augmented_aux + (refl_fluxes * scaling_factors.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        
        return generated_images, augmented_aux

    def augmentFlux(self, flux):
        """
        Apply augmentations such as random flips and shifts to the flux images.
        """
        # Random flips and transpose
        flux = utils.random_hflip(flux, prob=0.5)
        flux = utils.random_vflip(flux, prob=0.5)
        flux = utils.random_transpose(flux, prob=0.5)
        
        # Assign shift values based on target
        shift_x = self.shift_dev_x * torch.randn(flux.size(0))
        shift_y = self.shift_dev_y * torch.randn(flux.size(0))

        shift_x = torch.clamp(shift_x, -self.shift_max_x, self.shift_max_x)
        shift_y = torch.clamp(shift_y, -self.shift_max_y, self.shift_max_y)
        
        # Apply shifts to images
        shifted_images = [self.apply_shift(img, shift_x[i], shift_y[i]) for i, img in enumerate(flux)]

        # Convert list of tensors to a tensor
        flux = torch.stack(shifted_images)
        
        # Apply Gaussian filter
        flux = utils.gaussian_filter_batch(flux)
        
        return flux
    
    def apply_shift(self, img, shift_x, shift_y):
        """
        Apply shifting to a single image based on the given shift values.
        """
        # Calculate padding size. The max possible shift in any direction is considered for padding.
        padding_size_x = int(abs(shift_x))
        padding_size_y = int(abs(shift_y))
        padding_size = max(padding_size_x, padding_size_y)
        
        # Pad image
        padded_img = F.pad(img, (padding_size, padding_size, padding_size, padding_size), mode='constant')
        
        # Calculate the starting point for cropping
        start_x = padding_size + int(shift_x)
        start_y = padding_size + int(shift_y)
        
        # Crop the image to original size with the applied shift
        cropped_img = padded_img[:, start_y:start_y + img.size(1), start_x:start_x + img.size(2)]
        return cropped_img

    def augmentBackground(self, images):
        """
        Apply random augmentations to the background images.
        """
        # Random crop
        if random.random() > 0.25:    
            random_scale = utils.random_float(0.95, 0.99)
            images = utils.random_crop_and_resize(images, scale=random_scale)
            
        # Random scale
        scale_factor = utils.random_float(0.95, 1.05)
        images = scale_factor * images
        
        return images

    def log_images(self, emptyTargetImages_batch, fluxImages_batch, tag, augment):
        """
        Generate images and log them to TensorBoard.
        """
        generated_images, _ = self.generateImage(emptyTargetImages_batch, fluxImages_batch.clone(), self.reflMap, augment=augment)
        out = self.model(generated_images)
        data = torch.cat([generated_images[:, 0], out[:, 0], out[:, 1]], dim=0)
        grid = make_grid(data.unsqueeze(1), nrow=9) 
        self.logger.experiment.add_image(tag, grid, self.current_epoch)