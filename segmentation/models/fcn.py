import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict
from .base import BaseModel

class FCNEncoder(nn.Module):
    """FCN encoder based on VGG16."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())
        
        self.stage1 = nn.Sequential(*features[:5])   # 64 channels
        self.stage2 = nn.Sequential(*features[5:10])  # 128 channels
        self.stage3 = nn.Sequential(*features[10:17]) # 256 channels
        self.stage4 = nn.Sequential(*features[17:24]) # 512 channels
        self.stage5 = nn.Sequential(*features[24:])   # 512 channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning intermediate feature maps."""
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        
        return [x1, x2, x3, x4, x5]

class FCNDecoder(nn.Module):
    """FCN decoder with skip connections."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        # Decoder blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
        # Final classification
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass with skip connections."""
        [x1, x2, x3, x4, x5] = features
        
        # Decoder with skip connections
        x = self.block1(x5)
        x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x4], dim=1)
        
        x = self.block2(x)
        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        
        x = self.block3(x)
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        x = self.final(x)
        return x

class FCN(BaseModel):
    """Fully Convolutional Network for semantic segmentation."""
    
    def __init__(self, config: 'SegmentationConfig'):
        super().__init__(config)
        
        self.encoder = FCNEncoder(pretrained=True)
        self.decoder = FCNDecoder(num_classes=config.NUM_CLASSES)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[2:]
        
        # Encode
        features = self.encoder(x)
        
        # Decode
        x = self.decoder(features)
        
        # Ensure output size matches input size
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x
