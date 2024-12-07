import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Optional
import cv2

class Visualizer:
    """Visualization utilities for segmentation results."""
    def __init__(self, config):
        self.config = config
        self.colors = list(config.COLORS.values())
    
    def visualize_batch(self, images: torch.Tensor, masks: torch.Tensor, 
                       predictions: torch.Tensor, num_samples: int = 4):
        """Visualize a batch of results."""
        batch_size = min(num_samples, images.shape[0])
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
        
        for idx in range(batch_size):
            # Original image
            img = self._denormalize_image(images[idx])
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title('Input Image')
            axes[idx, 0].axis('off')
            
            # Ground truth
            mask = self._mask_to_color(masks[idx])
            axes[idx, 1].imshow(mask)
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            # Prediction
            pred = self._mask_to_color(predictions[idx])
            axes[idx, 2].imshow(pred)
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _denormalize_image(self, img: torch.Tensor) -> np.ndarray:
        """Denormalize and convert to numpy array."""
        img = img.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img
    
    def _mask_to_color(self, mask: torch.Tensor) -> np.ndarray:
        """Convert mask to color image."""
        if mask.dim() == 4:  # Remove batch dimension if present
            mask = mask.squeeze(0)
        
        if mask.dim() == 3:  # One-hot encoded
            mask = torch.argmax(mask, dim=0)
        
        mask = mask.cpu().numpy()
        height, width = mask.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for idx, color in enumerate(self.colors):
            colored_mask[mask == idx] = color
        
        return colored_mask