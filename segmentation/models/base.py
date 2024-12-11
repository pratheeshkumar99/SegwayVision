import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from ..utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


"""
The base class BaseModel is defined in the models/base.py file. This class is the parent class for all segmentation models and contains common functionality such as training, validation, and model saving/loading. The BaseModel class is a subclass of nn.Module and implements the forward method, which must be implemented by all subclasses.
All segmentation models should inherit from the BaseModel class and implement the forward method to define the model architecture. The BaseModel class provides methods for training the model, calculating metrics, and saving/loading model checkpoints.
"""

class BaseModel(nn.Module):
    """Base class for all segmentation models."""
    
    def __init__(self, config: 'SegmentationConfig'):
        super().__init__()
        self.config = config # Initialize the config in the base class.
        self.metrics = SegmentationMetrics() # Initliaze the SegmentationMetrics  in the base class.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass to be implemented by subclasses.
        """Forward pass to be implemented by subclasses."""
        raise NotImplementedError
    
    """
    This method is used to train the model. It takes in the training loader, validation loader, criterion, optimizer and scheduler as input and returns the training history. This method is actually inherited by all the models and is used to train the model and return the training history.
    """
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Dict[str, list]:  # This is method common to all the segmentation models which is used to train the model and would be inherited by all the models.
        
        """Training loop with validation."""
        device = torch.device(self.config.DEVICE)
        self.to(device)
        
        # Initialize training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            # Training phase
            self.train()
            train_metrics = self._train_epoch(
                train_loader, criterion, optimizer, device
            )  # Train the model for one epoch.
            
            # Validation phase
            self.eval()
            val_metrics = self._validate_epoch(val_loader, criterion, device) # Validate the model for one epoch.
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])   # Reduces the learning rate when the validation loss stops improving. 
                else:
                    scheduler.step()
            
            # Update history
            for k in train_metrics:
                history[f'train_{k}'].append(train_metrics[k])
                history[f'val_{k}'].append(val_metrics[k])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Early stopping if the validation loss does not improve
            if patience_counter >= self.config.PATIENCE:
                logger.info("Early stopping triggered")
                break
        
        return history
    
    """
    This  method is used to train the model for one epoch. It takes in the training loader, criterion, optimizer and device as input and returns the training loss, iou and dice coefficients.
    """
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Dict[str, float]:
        """Run one training epoch."""
        epoch_loss = 0
        epoch_iou = 0
        epoch_dice = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            metrics = self.metrics.calculate_metrics(outputs, masks) # Calculate the dice and iou metrics for mdoel predictions and ground truth masks.
            
            # Update running metrics
            epoch_loss += loss.item()
            epoch_iou += metrics['iou']
            epoch_dice += metrics['dice']
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{metrics["iou"]:.4f}'
            })
        
        # Calculate epoch metrics
        num_batches = len(train_loader)
        return {
            'loss': epoch_loss / num_batches,
            'iou': epoch_iou / num_batches,
            'dice': epoch_dice / num_batches
        }
    

    """
    This metnod is used to validate the model for one epoch. It takes in the validation loader, criterion and device as input and returns the validation loss, iou and dice coefficients.
    """
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Dict[str, float]:
        """Run one validation epoch."""
        epoch_loss = 0
        epoch_iou = 0
        epoch_dice = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            for images, masks in progress_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = self(images)
                loss = criterion(outputs, masks)
                metrics = self.metrics.calculate_metrics(outputs, masks)
                
                epoch_loss += loss.item()
                epoch_iou += metrics['iou']
                epoch_dice += metrics['dice']
                
                progress_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_iou': f'{metrics["iou"]:.4f}'
                })
        
        num_batches = len(val_loader)
        return {
            'loss': epoch_loss / num_batches,
            'iou': epoch_iou / num_batches,
            'dice': epoch_dice / num_batches
        }
    

    """
    This method is used to log the epoch metrics. It takes in the epoch number, training metrics and validation metrics as input and logs the epoch number, training loss, training iou, training dice, validation loss, validation iou and validation dice.
    """
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch metrics."""
        logger.info(
            f"Epoch {epoch + 1}/{self.config.EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f}"
        )
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.CHECKPOINT_DIR, filename)
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")










        """
        Information about the Inheriting mdoels from the BaseModel class:

        Here is a table comparing the key differences between the FCN with VGG-19 backbone, U-Net, and LinkNet architectures, with comments:

        | Feature | FCN with VGG-19 | U-Net | LinkNet |
        | --- | --- | --- | --- |
        | Encoder Architecture | Uses pre-trained VGG-19 model | Uses custom convolutional layers | Uses pre-trained ResNet-34 model |
        | # FCN with VGG-19 leverages the pre-trained features of VGG-19, enabling transfer learning and potentially better performance with limited data. | # U-Net learns the features from scratch, providing more flexibility to adapt to the specific task and dataset. | # LinkNet combines the benefits of using a pre-trained encoder (ResNet-34) with the flexibility of a symmetric encoder-decoder architecture. |
        | Upsampling Method | Bilinear interpolation | Transposed convolution | Transposed convolution |
        | # FCN with VGG-19 uses a simpler upsampling method (bilinear interpolation), which may be less accurate but computationally more efficient. | # U-Net uses transposed convolution (deconvolution) layers for upsampling, which can learn the upsampling parameters during training. | # Like U-Net, LinkNet also uses transposed convolution layers for upsampling in the decoder. |
        | Skip Connections | Concatenation | Concatenation | Addition |
        | # FCN with VGG-19 uses concatenation-based skip connections, combining the encoder and decoder features. | # U-Net also uses concatenation-based skip connections, similar to FCN with VGG-19. | # LinkNet uses additive skip connections, where the encoder features are added to the decoder features. |
        | Architectural Symmetry | Asymmetric (heavier encoder, lighter decoder) | Symmetric (matching encoder-decoder structure) | Symmetric (matching encoder-decoder structure) |
        | # The asymmetric nature of FCN with VGG-19 may result in a more efficient inference, but the U-Net and LinkNet symmetric architectures can potentially learn better features. | # The symmetric structure of U-Net allows for a more balanced feature learning between the encoder and decoder. | # The symmetric encoder-decoder structure of LinkNet also allows for a more balanced feature learning. |
        | Use Cases | Better for transfer learning and limited data | Better for learning from scratch and full control over feature learning | Combines the benefits of pre-trained encoder and flexible architecture |
        | # FCN with VGG-19 is particularly suitable for transfer learning applications and scenarios with limited training data, as it can leverage the pre-trained VGG-19 features. | # U-Net is a better choice when you want to learn the features from scratch and have full control over the model architecture. | # LinkNet can be a good compromise, allowing you to use a pre-trained encoder (ResNet-34) while maintaining the flexibility of a symmetric encoder-decoder structure. |
                
                """