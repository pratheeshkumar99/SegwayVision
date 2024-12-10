import albumentations as A #Uses albumentation library for data augmentation over torchvision.transforms because it is faster and more efficient
from albumentations.pytorch import ToTensorV2



class Augmentation:
    """Data augmentation utilities."""
    @staticmethod
    def get_train_transforms():
        return A.Compose([
            A.RandomRotate90(p=0.5), #Randomly rotate the image by 90 degrees
            A.Flip(p=0.5), #Randomly flip the image
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5), #Randomly shift, scale and rotate the image
            A.OneOf([
                A.RandomBrightnessContrast(p=1), #Randomly change brightness and contrast
                A.RandomGamma(p=1), #Randomly change gamma
            ], p=0.5),  #Apply this augmentation with a probability of 0.5 if applied then either of the two will be applied
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #Normalize the image by subtracting the mean and dividing by the standard deviation
        ])
    
    @staticmethod
    def get_val_transforms():
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #Normalize the image by subtracting the mean and dividing by the standard deviation
        ])