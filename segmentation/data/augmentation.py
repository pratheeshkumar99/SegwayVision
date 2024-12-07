import albumentations as A
from albumentations.pytorch import ToTensorV2

class Augmentation:
    """Data augmentation utilities."""
    @staticmethod
    def get_train_transforms():
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ], p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @staticmethod
    def get_val_transforms():
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])