import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict
from .base import BaseModel


"""
Example of how the FCN model works:
FCN Network Flow for input shape (224,224,3):

1. Input Stage:
- Input shape: (batch_size, 3, 224, 224)  # PyTorch expects (C,H,W) format

2. Encoder Path (Downsampling):
Stage 1:
- Input: (batch_size, 3, 224, 224)
- Output x1: (batch_size, 64, 112, 112)
- Halves spatial dimensions, increases channels to 64

Stage 2:
- Input: (batch_size, 64, 112, 112)
- Output x2: (batch_size, 128, 56, 56)
- Further halves spatial dimensions, increases channels to 128

Stage 3:
- Input: (batch_size, 128, 56, 56)
- Output x3: (batch_size, 256, 28, 28)
- Continues downsampling, increases channels to 256

Stage 4:
- Input: (batch_size, 256, 28, 28)
- Output x4: (batch_size, 512, 14, 14)
- Further downsampling, increases channels to 512

Stage 5:
- Input: (batch_size, 512, 14, 14)
- Output x5: (batch_size, 512, 7, 7)
- Final encoder output, maintains channels at 512

3. Decoder Path (Upsampling with Skip Connections):
Block 1:
- Input: x5 (batch_size, 512, 7, 7)
- After block1: (batch_size, 512, 7, 7)
- After upsample: (batch_size, 512, 14, 14)
- After concat with x4: (batch_size, 1024, 14, 14)  # Skip connection

Block 2:
- Input: (batch_size, 1024, 14, 14)
- After block2: (batch_size, 256, 14, 14)
- After upsample: (batch_size, 256, 28, 28)
- After concat with x3: (batch_size, 512, 28, 28)  # Skip connection

Block 3:
- Input: (batch_size, 512, 28, 28)
- After block3: (batch_size, 128, 28, 28)
- After upsample: (batch_size, 128, 56, 56)

Final Block:
- Input: (batch_size, 128, 56, 56)
- Output: (batch_size, num_classes, 56, 56)

4. Final Upsampling:
- Input: (batch_size, num_classes, 56, 56)
- Output: (batch_size, num_classes, 224, 224)
- Matches input spatial dimensions with class predictions

Final output provides a probability map for each class at each pixel position.
Each pixel has num_classes values (probabilities), and the class with highest
probability is the predicted segmentation class for that pixel.

Here the encoder part is pre-trained VGG16 model and the decoder part is implemented with skip connections and decoder uses a regular Conv2D layers with batch normalization and ReLU activation functions.

--------------- **************** ------------------ ****************  --------------- **************** ------------------ ****************

Differenciation between FCN and U-Net:

Key Architectural Differences between FCN and U-Net:

1. Encoder Architecture:
FCN Encoder:
- Uses pretrained VGG16 as feature extractor
- Extracts features using pretrained weights
- Architecture fixed by VGG structure
Example:
   vgg16 = models.vgg16(pretrained=True)
   features = list(vgg16.features.children())
   self.stage1 = nn.Sequential(*features[:5])  # Uses VGG blocks

U-Net Encoder:
- Custom Conv2D layers built from scratch
- Learns features during training
- Flexible architecture
Example:
   self.enc1 = nn.Sequential(
       nn.Conv2d(3, 64, 3, padding=1),
       nn.ReLU(),
       nn.Conv2d(64, 64, 3, padding=1),
       nn.ReLU()
   )

2. Decoder Architecture:
FCN Decoder:
- Uses regular Conv2D + Bilinear Interpolation for upsampling
- Simpler structure
Example:
   x = self.block1(x5)
   x = F.interpolate(x, size=x4.shape[2:], mode='bilinear')  # Bilinear upsampling
   x = torch.cat([x, x4], dim=1)

U-Net Decoder:
- Uses Transposed Convolutions (Deconvolution) for upsampling
- Learnable upsampling parameters
Example:
   self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # Deconvolution
   x = self.up1(x)
   x = torch.cat([x, skip], dim=1)

3. Key Architectural Differences:
Feature Extraction:
- FCN: Uses pretrained VGG features (transfer learning)
- U-Net: Learns features from scratch during training

Upsampling Method:
- FCN: Simple bilinear interpolation with F.interpolate
- U-Net: Learnable transposed convolutions

Architecture Symmetry:
- FCN: Asymmetric (heavy encoder, light decoder)
- U-Net: Symmetric (matching encoder-decoder structure)

Skip Connections:
- FCN: Uses VGG intermediate features
- U-Net: Uses custom encoder features

4. Use Case Advantages:
FCN Better For:
- Transfer learning applications
- Limited dataset scenarios
- When pretrained features are beneficial

U-Net Better For:
- Learning from scratch
- Custom architecture requirements
- When full control over feature learning is needed


"""

class FCNEncoder(nn.Module):
    """FCN encoder based on VGG16."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())
        
        self.stage1 = nn.Sequential(*features[:5])   # 64 channels Input shape : (batch_size, 3, 224, 224) ---> Output shape : (batch_size, 64, 112, 112)
        self.stage2 = nn.Sequential(*features[5:10])  # 128 channels # Input shape : (batch_size, 64, 112, 112) ---> Output shape : (batch_size, 128, 56, 56)
        self.stage3 = nn.Sequential(*features[10:17]) # 256 channels # Input shape : (batch_size, 128, 56, 56) ---> Output shape : (batch_size, 256, 28, 28) 
        self.stage4 = nn.Sequential(*features[17:24]) # 512 channels # Input shape : (batch_size, 256, 28, 28) ---> Output shape : (batch_size, 512, 14, 14)
        self.stage5 = nn.Sequential(*features[24:])   # 512 channels # Input shape : (batch_size, 512, 14, 14) ---> Output shape : (batch_size, 512, 7, 7)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning intermediate feature maps."""
        x1 = self.stage1(x) # Output shape : (batch_size, 64, 112, 112)
        x2 = self.stage2(x1) # Output shape : (batch_size, 128, 56, 56)
        x3 = self.stage3(x2) # Output shape : (batch_size, 256, 28, 28)
        x4 = self.stage4(x3) # Output shape : (batch_size, 512, 14, 14)
        x5 = self.stage5(x4) # Output shape : (batch_size, 512, 7, 7)
        
        return [x1, x2, x3, x4, x5] # Returns the feature maps at different stages of the encoder.

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
        x = self.block1(x5) # Input shape : (batch_size, 512, 7, 7) ---> Output shape : (batch_size, 512, 7, 7)
        x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True) # Upsample the feature map to the size of x4 : (batch_size, 512, 7, 7) ---> (batch_size, 512, 14, 14)
        x = torch.cat([x, x4], dim=1) # Concatenate the feature maps : (batch_size, 512, 14, 14) + (batch_size, 512, 14, 14) ---> (batch_size, 1024, 14, 14)
        
        x = self.block2(x) # Input shape : (batch_size, 1024, 14, 14) ---> Output shape : (batch_size, 256, 14, 14)
        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True) # Upsample the feature map to the size of x3 : (batch_size, 256, 14, 14) ---> (batch_size, 256, 28, 28)
        x = torch.cat([x, x3], dim=1) # Concatenate the feature maps : (batch_size, 256, 28, 28) + (batch_size, 256, 28, 28) ---> (batch_size, 512, 28, 28)
         
        x = self.block3(x) # Input shape : (batch_size, 512, 28, 28) ---> Output shape : (batch_size, 128, 28, 28)
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True) # Upsample the feature map to the size of x1 : (batch_size, 128, 28, 28) ---> (batch_size, 128, 56, 56)
        
        x = self.final(x) # Input shape : (batch_size, 128, 56, 56) ---> Output shape : (batch_size, num_classes, 56, 56)
        return x


"""
Inheriting the BaseModel class and implementing the FCN model. The FCN model is a Fully Convolutional Network for semantic segmentation. The FCN model consists of an encoder and a decoder. The encoder is based on VGG16 and the decoder consists of skip connections. The forward method is implemented to perform the forward pass. The FCN model is used to train the model and return the training history.
"""
class FCN(BaseModel):
    """Fully Convolutional Network for semantic segmentation."""
    
    def __init__(self, config: 'SegmentationConfig'):
        super().__init__(config)
        
        self.encoder = FCNEncoder(pretrained=True)
        self.decoder = FCNDecoder(num_classes=config.NUM_CLASSES)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[2:] # Get the input size
        
        # Encode
        features = self.encoder(x) # Get the features from the encoder : input shape : (batch_size, 3, 224, 224) ---> Output shape : [x1, x2, x3, x4, x5] where x1--> (batch_size, 64, 112, 112), x2--> (batch_size, 128, 56, 56), x3--> (batch_size, 256, 28, 28), x4--> (batch_size, 512, 14, 14), x5--> (batch_size, 512, 7, 7)
        
        # Decode
        x = self.decoder(features) # Get the output from the decoder : input shape : [x1, x2, x3, x4, x5] ---> Output shape : (batch_size, num_classes, 56, 56)
        
        # Ensure output size matches input size
        if x.shape[2:] != input_size: # If the output size does not match the input size
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True) # Upsample the output to the input size : (batch_size, num_classes, 56, 56) ---> (batch_size, num_classes, 224, 224)
        
        return x
