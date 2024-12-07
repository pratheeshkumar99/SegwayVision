import torch
import numpy as np
from typing import Dict

class SegmentationMetrics:
    """Metrics for segmentation evaluation."""
    @staticmethod
    def dice_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, 
                        smooth: float = 1e-6) -> torch.Tensor:
        """Calculate Dice coefficient."""
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        
        intersection = (y_true == y_pred).float().sum()
        union = y_true.numel()
        
        return (2. * intersection + smooth) / (union + smooth)

    @staticmethod
    def iou_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, 
                       smooth: float = 1e-6) -> torch.Tensor:
        """Calculate IoU coefficient."""
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        
        intersection = ((y_pred == y_true) & (y_true > 0)).float().sum()
        union = ((y_pred > 0) | (y_true > 0)).float().sum()
        
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """Calculate all metrics."""
        dice = SegmentationMetrics.dice_coefficient(y_pred, y_true)
        iou = SegmentationMetrics.iou_coefficient(y_pred, y_true)
        
        return {
            'dice': dice.item(),
            'iou': iou.item()
        }
