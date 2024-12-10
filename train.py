### The code has been written using OOPs concepts. The code has been divided into multiple classes and methods to make it more modular and reusable. The code has been divided into the following classes:
# 1. SegmentationConfig : This class is used to store configuration parameters for the segmentation models. It uses dataclasses to define the configuration parameters and their default values. The __post_init__ method is used to create necessary directories and check if the data directory exists.
# 2. BaseModel : This is the base class for all segmentation models. It is a subclass of nn.Module and contains common functionality such as training, validation, and model saving/loading. It implements the forward method, which must be implemented by all subclasses. It also contains the train_model method, which is used to train the model and calculate metrics.
# 3. FCN : This class implements the Fully Convolutional Network (FCN) model for semantic segmentation. It is a subclass of BaseModel and implements the forward method to define the model architecture. It also contains the FCNDecoder class, which implements the decoder with skip connections.
# 4. SegmentationDataset : This class is used to create PyTorch datasets for segmentation tasks. It is a subclass of torch.utils.data.Dataset and implements the __len__ and __getitem__ methods to load images and masks. It uses the Albumentations library for data augmentation.
# 5. Augmentation : This class contains static methods to get training and validation transforms using the Albumentations library. The get_train_transforms method returns a list of augmentation transforms for training, while the get_val_transforms method returns a list of transforms for validation.
# 6. Visualizer : This class is used to visualize the model predictions and ground truth masks. It contains methods to plot images, masks, and predictions using Matplotlib.
# 7. train : This function is the main training function that creates datasets, dataloaders, and the model. It trains the model using the train_model method and saves the final model.
# 8. UNet, LinkNet, DeepLabV3 : These classes implement the UNet, LinkNet, and DeepLabV3 models for semantic segmentation. They are subclasses of BaseModel and implement the forward method to define the model architecture.


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
    model = get_model(config.MODEL_TYPE, config) # Get the model based on the model type
    model = model.to(config.DEVICE) # Move the model to the device
    
    criterion = torch.nn.CrossEntropyLoss() # Define the loss function: CrossEntropyLoss //The reason this loss function is used because this is multi-class segmentation problem where each pixel can belong to one of the 8 classes and the model is trained to predict the class of each pixel. The CrossEntropyLoss function is used to calculate the loss between the predicted and ground truth masks.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, # The optimizer object
        mode='min', # The mode to monitor the metric, ie specify whether to monitor the metric for increase or decrease
        factor=0.1, # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5, # Number of epochs with no improvement after which learning rate will be reduced
        verbose=True # If True, prints a message to stdout for each update
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



