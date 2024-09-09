import pytorch_lightning as pl
from dataset import get_dataloader_and_reflmap
from unet_pl_module import UNet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training script for UNet3+")
    
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device to use for training')
    parser.add_argument('--devices', type=int, nargs='+', default=[1], help='GPU device ids to use (if using GPU)')
    parser.add_argument('--model_name', type=str, default='UNet3plus', help='Name of the model')
    parser.add_argument('--state_dict', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs to train')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./trained_models', help='Directory to save model checkpoints')
    
    parser.add_argument('--min_value_image', type=float, default=0.8, help='Min pixel value for real images')
    parser.add_argument('--max_value_image', type=float, default=1.0, help='Max pixel value for real images')
    parser.add_argument('--augment_min_value', type=float, default=0.7, help='Min pixel value for augmented images')
    parser.add_argument('--augment_max_value', type=float, default=1.1, help='Max pixel value for augmented images')
    parser.add_argument('--augment_flux_center_dev_x', type=float, default=15, help='Std deviation for flux shift in x')
    parser.add_argument('--augment_flux_center_dev_y', type=float, default=15, help='Std deviation for flux shift in y')
    parser.add_argument('--max_shift_x', type=int, default=80, help='Max flux shift in x')
    parser.add_argument('--max_shift_y', type=int, default=80, help='Max flux shift in y')

    return parser.parse_args()

def train_model(args):
    # Parameters
    params = {
        'min_value_image': args.min_value_image,
        'max_value_image': args.max_value_image,
        'augment_min_value': args.augment_min_value,
        'augment_max_value': args.augment_max_value,
        'augment_flux_center_dev_x': args.augment_flux_center_dev_x,
        'augment_flux_center_dev_y': args.augment_flux_center_dev_y,
        'max_shift_x': args.max_shift_x,
        'max_shift_y': args.max_shift_y,
    }
    
    # Initialize model
    model = UNet(params)
    
    # Load state_dict if available
    if args.state_dict is not None:
        model = UNet.load_from_checkpoint(args.state_dict, params=params)
    
    # Get data loader of empty images and simulated fluxes
    train_dataloader, model.reflMap = get_dataloader_and_reflmap(batch_size=args.batch_size)
    
    # Add TensorBoard logger
    tensorboard_logger = TensorBoardLogger(save_dir=args.log_dir, name=args.model_name)
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=args.model_name + '-{epoch:02d}-{train_loss:.8f}',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )
    
    # Determine the accelerator and devices
    if args.device == 'gpu':
        accelerator = 'gpu'
        devices = args.devices
    else:
        accelerator = 'cpu'
        devices = None  # No specific device needs to be set for CPU
    
    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback]
    )
    
    # Train the model
    trainer.fit(model, train_dataloader)

if __name__ == "__main__":
    args = get_args()
    train_model(args)
