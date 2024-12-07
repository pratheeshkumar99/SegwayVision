import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .base import BaseModel

class DecoderBlock(nn.Module):
    """LinkNet decoder block."""
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(middle_channels, middle_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class LinkNet(BaseModel):
    """LinkNet architecture for semantic segmentation."""
    def __init__(self, config: 'SegmentationConfig'):
        super().__init__(config)
        
        # Load pretrained ResNet34 as encoder
        resnet = models.resnet34(pretrained=True)
        
        # Encoder
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 32, 32)
        
        # Final convolutions
        self.final_conv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_conv2 = nn.Conv2d(32, config.NUM_CLASSES, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize decoder weights."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Decoder with skip connections
        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4 + e3)
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)
        
        # Final convolutions
        out = self.final_conv1(d1)
        out = self.final_conv2(out)
        
        # Ensure output size matches input size
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return out