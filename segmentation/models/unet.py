# segmentation/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.base import BaseModel


# A custon ConvBlock class is defined which is used to create a double convolution block for the UNet architecture.

class ConvBlock(nn.Module):
    """Double convolution block for UNet."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # NO of filters = out_channels
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True), # ReLU activation
            nn.Conv2d(out_channels, out_channels, 3, padding=1), # NO of filters = out_channels
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True) # ReLU activation
        )

    def forward(self, x):
        return self.double_conv(x)
    
# This class inherits from the BaseModel class. The BaseModel class is a parent class for all the segmentation models. 
# The UNet class is a child class of the BaseModel class. The BaseModel class has a forward method that is to be implemented by the child classes. 
# The UNet class implements the forward method. The forward method of the UNet class takes an input tensor x and passes it through the UNet architecture to get
#  the output tensor. The UNet architecture consists of an encoder path and a decoder path with skip connections. The encoder path consists of four down-sampling 
# blocks, and the decoder path consists of four up-sampling blocks. The down-sampling blocks are implemented using the ConvBlock class, 
# which is a double convolution block. The ConvBlock class has two convolution layers with batch normalization and ReLU activation.
#  The up-sampling blocks are implemented using the nn.ConvTranspose2d layer, which performs up-sampling by a factor of 2.
#  The output of the UNet model is a tensor with the number of channels equal to the number of classes in the segmentation task. 
# The UNet class also has an _initialize_weights method that initializes the weights of the model using the Kaiming initialization method. 
# The UNet class is used to create a UNet model for semantic segmentation tasks.

class UNet(BaseModel):
    """UNet architecture for semantic segmentation."""
    def __init__(self, config):
        super().__init__(config)
        self.inc = ConvBlock(3, 64) # Input: (3, 256, 256) ----> Output: (64, 256, 256)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128)) # Input: (64, 256, 256) ----> Output: (128, 128, 128)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256)) # Input: (128, 128, 128) ----> Output: (256, 64, 64)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512)) # Input: (256, 64, 64) ----> Output: (512, 32, 32)
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 1024)) # Input: (512, 32, 32) ----> Output: (1024, 16, 16)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Input: (1024, 16, 16) ----> Output: (512, 32, 32)
        self.up_conv4 = ConvBlock(1024, 512) # Input: (1024, 32, 32) ----> Output: (512, 32, 32)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # Input: (512, 32, 32) ----> Output: (256, 64, 64)
        self.up_conv3 = ConvBlock(512, 256) # Input: (512, 64, 64) ----> Output: (256, 64, 64)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # Input: (256, 64, 64) ----> Output: (128, 128, 128)
        self.up_conv2 = ConvBlock(256, 128) # Input: (256, 128, 128) ----> Output: (128, 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # Input: (128, 128, 128) ----> Output: (64, 256, 256)
        self.up_conv1 = ConvBlock(128, 64) # Input: (128, 256, 256) ----> Output: (64, 256, 256)
        
        self.outc = nn.Conv2d(64, config.NUM_CLASSES, kernel_size=1) # Here the no of filters is equal to the number of classes in the segmentation task.
        
        self._initialize_weights()
    
    # The _initialize_weights method initializes the weights of the model using the Kaiming initialization method for the convolution layers and sets the weights of the batch normalization layers to ones and zeros.
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x) # Input: (3, 256, 256) ----> Output: (64, 256, 256)
        x2 = self.down1(x1) # Input: (64, 256, 256) ----> Output: (128, 128, 128)
        x3 = self.down2(x2) # Input: (128, 128, 128) ----> Output: (256, 64, 64)
        x4 = self.down3(x3) # Input: (256, 64, 64) ----> Output: (512, 32, 32)
        x5 = self.down4(x4) # Input: (512, 32, 32) ----> Output: (1024, 16, 16)
        
        # Decoder path with skip connections
        x = self.up4(x5) # Input: (1024, 16, 16) ----> Output: (512, 32, 32)
        x = torch.cat([x4, x], dim=1) # Concatenate along the channel dimension (512, 32, 32) + (512, 32, 32) ----> (1024, 32, 32)
        x = self.up_conv4(x) # Input: (1024, 32, 32) ----> Output: (512, 32, 32)
        
        x = self.up3(x) # Input: (512, 32, 32) ----> Output: (256, 64, 64)
        x = torch.cat([x3, x], dim=1) # Concatenate along the channel dimension (256, 64, 64) + (256, 64, 64) ----> (512, 64, 64)
        x = self.up_conv3(x) # Input: (512, 64, 64) ----> Output: (256, 64, 64)
        
        x = self.up2(x) # Input: (256, 64, 64) ----> Output: (128, 128, 128)
        x = torch.cat([x2, x], dim=1) # Concatenate along the channel dimension (128, 128, 128) + (128, 128, 128) ----> (256, 128, 128)
        x = self.up_conv2(x) # Input: (256, 128, 128) ----> Output: (128, 128, 128)
        
        x = self.up1(x) # Input: (128, 128, 128) ----> Output: (64, 256, 256)
        x = torch.cat([x1, x], dim=1) # Concatenate along the channel dimension (64, 256, 256) + (64, 256, 256) ----> (128, 256, 256)
        x = self.up_conv1(x) # Input: (128, 256, 256) ----> Output: (64, 256, 256)
        
        out = self.outc(x) # Input: (64, 256, 256) ----> Output: (NUM_CLASSES, 256, 256)
        return out