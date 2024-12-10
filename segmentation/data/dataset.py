import os
import cv2
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
from imutils import paths
from torchvision.transforms import ToTensor
from .augmentation import Augmentation
from torchvision.transforms.functional import to_tensor

logger = logging.getLogger(__name__)

class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation tasks."""
    def __init__(self, config: 'SegmentationConfig', mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.transforms = (Augmentation.get_train_transforms() if mode == 'train' 
                         else Augmentation.get_val_transforms())   # Gets the appropriate transformation based on the mode, if train then get train transforms else get val transforms which is defined in augmentation.py which used albumentations library
        
        # Get image and mask paths
        self.image_paths = self._get_image_paths()
        self.mask_paths = self._get_mask_paths() if mode != 'test' else None
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Load image
        image_path = self.image_paths[idx] # Get the image path
        image = cv2.imread(image_path) # Read the image using cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the image to RGB
        image = cv2.resize(image, self.config.INPUT_SHAPE[:2][::-1]) # Resize the image to the input shape

        # Load mask if not test mode


        """
        Note : The mask visually looks like a grayscale image but it is not a grayscale image. Infact the masks have unique pixel values each value mapping it different class.
        0 - Drivable
        1 - Non Drivable
        2 - Living Things
        3 - Vehicles
        4 - Road Side
        5 - Far Objects
        6 - Sky
        7 - Misc
        """

        mask = None
        if self.mode != 'test':
            mask_path = self.mask_paths[idx] # Get the mask path
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read the mask using cv2 in grayscale : shape (length, width)
            mask = cv2.resize(mask, self.config.INPUT_SHAPE[:2][::-1], interpolation=cv2.INTER_NEAREST) # Resize the mask to the input shape : shape (length, width) ---> (256, 256)
            mask = self._convert_mask_to_onehot(mask) # Convert the mask to one-hot encoding : shape (256, 256) ---> (256, 256, 8)

        # Apply transformations
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask) # Apply same transformation to image and mask
            image = transformed['image']
            mask = transformed['mask'] if mask is not None else None

        # Convert to tensors and ensure they are float32
        #Here we perform operation to convert the numpy array ---> torch tensor wihch is float32 and permute the dimension from (height, width, channels) ---> (channels, height, width). This is done because PyTorch expects the image in this format
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Ensure it's float32
        if mask is not None:
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # Ensure it's float32

        return image, mask if mask is not None else torch.zeros(1, *self.config.INPUT_SHAPE[:2])
    
    
    def _get_image_paths(self) -> List[str]: # Gets the image paths for the specified mode
        """Get image paths for the specified mode."""
        all_images = sorted(paths.list_images(os.path.join(self.config.DATA_DIR, 'leftImg8bit')))
        return [img for img in all_images if self.mode in img.split('/')[-3]]
    
    def _get_mask_paths(self) -> List[str]: # Gets the mask paths for the specified mode
        """Get mask paths for the specified mode."""
        all_masks = sorted(paths.list_images(os.path.join(self.config.DATA_DIR, 'gtFine')))
        return [mask for mask in all_masks if self.mode in mask.split('/')[-3]]
    
    def _convert_mask_to_onehot(self, mask: np.ndarray) -> torch.Tensor:  # Input shape : (256, 256) ---> Output shape : (256, 256, 8)
        """Convert mask to one-hot encoding."""
        onehot = np.zeros((*mask.shape, self.config.NUM_CLASSES), dtype=np.float32)
        for i in range(self.config.NUM_CLASSES):
            onehot[..., i] = (mask == i).astype(np.float32)
        return onehot

def get_dataloader(dataset: SegmentationDataset, config: 'SegmentationConfig') -> DataLoader:
    """Create DataLoader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=dataset.mode == 'train',
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )