# segmentation/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.base import BaseModel

class ConvBlock(nn.Module):
    """Double convolution block for UNet."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(BaseModel):
    """UNet architecture for semantic segmentation."""
    def __init__(self, config):
        super().__init__(config)
        self.inc = ConvBlock(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 1024))
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(128, 64)
        
        self.outc = nn.Conv2d(64, config.NUM_CLASSES, kernel_size=1)
        
        self._initialize_weights()
    
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up4(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv4(x)
        
        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv1(x)
        
        out = self.outc(x)
        return out