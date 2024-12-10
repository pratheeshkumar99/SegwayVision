import argparse
import yaml
import logging
import torch
import os
from datetime import datetime
from segmentation import (
    SegmentationConfig, SegmentationDataset, get_dataloader,
    FCN, LinkNet, UNet, DeepLabV3, Visualizer
)

def setup_logging(save_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(save_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_model(model_type: str, config: SegmentationConfig):
    """Get model based on type."""
    models = {
        'fcn': FCN,
        'linknet': LinkNet,
        'unet': UNet,
        'deeplabv3': DeepLabV3
    }
    return models[model_type](config)

def train(config: SegmentationConfig):
    """Main training function."""
    # Setup logging
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    setup_logging(config.CHECKPOINT_DIR)
    logger = logging.getLogger(__name__)
    
    # Create datasets and dataloaders
    logger.info("Creating datasets...")
    train_dataset = SegmentationDataset(config, mode='train')  # Create the training dataset using the SegmentationDataset class.
    train_size = len(train_dataset)

    print(f"Training dataset size: {train_size}")
    val_dataset = SegmentationDataset(config, mode='val')  # Create the validation dataset using the SegmentationDataset class.

    val_size = len(val_dataset)
    print(f"Validation dataset size: {val_size}")


    print("\n\n\n\n\n\n\n\n\n")
    
    train_loader = get_dataloader(train_dataset, config) # Create the training dataloader using the get_dataloader function
    val_loader = get_dataloader(val_dataset, config) # Create the validation dataloader using the get_dataloader function
    
    # Initialize model
    logger.info(f"Initializing {config.MODEL_TYPE} model...")
    model = get_model(config.MODEL_TYPE, config)
    model = model.to(config.DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Train model
    logger.info("Starting training...")
    history = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Save final model
    logger.info("Saving final model...")
    model.save_checkpoint('final_model.pth')
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='path to config file')
    args = parser.parse_args()

    # Load configuration
    config = SegmentationConfig()
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for k, v in config_dict.items():
                setattr(config, k, v) # Updating the config object with the values from the YAML file

    # Setup logging and create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True) # Create the checkpoint directory if it doesn't exist
    setup_logging(config.CHECKPOINT_DIR) # Setup logging configuration

    # Print device and availability
    print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Current device being used: {config.DEVICE}")

    # Train model
    model, history = train(config)
    
    # Visualize results
    visualizer = Visualizer(config)
    return model, history, visualizer

if __name__ == '__main__':
    main()



