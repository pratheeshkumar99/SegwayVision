# segmentation/models/__init__.py
from .fcn import FCN
from .linknet import LinkNet
from .unet import UNet
from .deeplabv3 import DeepLabV3

__all__ = ['FCN', 'LinkNet', 'UNet', 'DeepLabV3']