import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .base import BaseModel

"""
This model is encoder decoder based model. Encoder is ResNet34 and decoder is Custom Conv2D Deconvolutional block.
This decoder has a bridge between encoder and decoder. This bridge is used to connect the encoder and decoder.
The bridge connection adds the output of the encoder to the output of the decoder. This helps in better learning of the model.
#There is a final convolution layer which is used to get the final output of the model channel size is equal to the number of classes.
"""

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
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # apply Cov2d, BatchNorm2d, ReLU : Input size 3x256x256 -----> 64x128x128
        self.encoder1 = resnet.layer1 # 64x128x128 -----> 64x128x128 Applied [3x3 Conv2d, BatchNorm2d, ReLU,3x3 Conv2d, BatchNorm2d] 3 times
        self.encoder2 = resnet.layer2 # 64 x 128 x 128 -----> 128 x 64 x 64 Applied 
        self.encoder3 = resnet.layer3 # 128 x 64 x 64 -----> 256 x 32 x 32
        self.encoder4 = resnet.layer4 # 256 x 32 x 32 -----> 512 x 16 x 16
        
        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256) # 512 x 16 x 16 -----> 256 x 32 x 32
        self.decoder3 = DecoderBlock(256, 128, 128)  # 256 x 32 x 32 -----> 128 x 64 x 64
        self.decoder2 = DecoderBlock(128, 64, 64) # 128 x 64 x 64 -----> 64 x 128 x 128
        self.decoder1 = DecoderBlock(64, 32, 32) # 64 x 128 x 128 -----> 32 x 256 x 256
         
        # Final convolutions
        self.final_conv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ) # 32 x 256 x 256 -----> 32 x 256 x 256
        self.final_conv2 = nn.Conv2d(32, config.NUM_CLASSES, 1) # 32 x 256 x 256 -----> 1 x 256 x 256
        
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
        e0 = self.encoder0(x) #input : 3x256x256 -----> 64x128x128
        e1 = self.encoder1(e0) # 64x128x128 -----> 64x128x128
        e2 = self.encoder2(e1) # 64x128x128 -----> 128x64x64
        e3 = self.encoder3(e2) # 128x64x64 -----> 256x32x32
        e4 = self.encoder4(e3) # 256x32x32 -----> 512x16x16
        
        # Decoder with skip connections
        d4 = self.decoder4(e4) # 512x16x16 -----> 256x32x32
        d3 = self.decoder3(d4 + e3) # 256x32x32 -----> 128x64x64
        d2 = self.decoder2(d3 + e2) # 128x64x64 -----> 64x128x128
        d1 = self.decoder1(d2 + e1) # 64x128x128 -----> 32x256x256
        
        # Final convolutions
        out = self.final_conv1(d1) # 32x256x256 -----> 32x256x256
        out = self.final_conv2(out) # 32x256x256 -----> number_classesx256x256
        
        # Ensure output size matches input size
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return out