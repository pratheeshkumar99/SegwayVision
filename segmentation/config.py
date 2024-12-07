# segmentation/config.py
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any
import torch
import os
from pathlib import Path

@dataclass
class SegmentationConfig:
    """Configuration for segmentation models."""
    # Model parameters
    MODEL_TYPE: str = 'linknet'  # ['fcn', 'linknet', 'unet', 'deeplabv3']
    BACKBONE: str = 'resnet34'
    NUM_CLASSES: int = 8
    INPUT_SHAPE: Tuple[int, int, int] = (256, 256, 3)
    
    # Training parameters
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    DEVICE: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    NUM_WORKERS: int = 4
    PATIENCE: int = 10 
    
    # Paths
    DATA_DIR: str = str(Path('IDD_data/IDD/idd20k_lite').absolute())
    CHECKPOINT_DIR: str = str(Path('checkpoints').absolute())
    
    # Class colors for visualization - using default_factory
    COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'drivable': (128, 64, 18),
        'non_drivable': (244, 35, 232),
        'living_things': (220, 20, 60),
        'vehicles': (0, 0, 230),
        'road_side': (220, 190, 40),
        'far_objects': (70, 70, 70),
        'sky': (70, 130, 180),
        'misc': (0, 0, 0)
    })
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        
        if not os.path.exists(self.DATA_DIR):
            raise RuntimeError(
                f"Data directory not found at {self.DATA_DIR}. "
                "Please run setup_data.py first."
            )