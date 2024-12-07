# Road Scene Semantic Segmentation

A PyTorch-based implementation of various deep learning architectures for semantic segmentation of unstructured road scenes using the Indian Driving Dataset (IDD).

![Model Results](IMAGES/results.png)

## Project Overview

This project implements and compares five popular deep learning architectures for semantic segmentation:
- FCN (Fully Convolutional Network)
- U-Net
- PSPNet
- LinkNet
- DeepLabV3+

The models are trained on the IDD-Lite dataset, which contains road scene images from Indian cities, annotated with 8 classes:
- Drivable area
- Non-drivable area
- Living things
- Vehicles
- Roadside objects
- Far objects
- Sky
- Miscellaneous



## Model Architectures
This project implements five deep learning architectures, each with its unique strengths for semantic segmentation:

### 1. Fully Convolutional Network (FCN)
```mermaid
[FCN diagram from previous response]
```

The FCN architecture transforms traditional classification networks into fully convolutional networks for semantic segmentation. Key features:
- Based on VGG16 backbone
- Replaces fully connected layers with 1x1 convolutions
- Uses skip connections from earlier layers for fine-grained prediction
- Multi-scale prediction fusion for better segmentation details

### 2. U-Net Architecture

```mermaid
[<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="800" height="400" fill="#ffffff"/>
    
    <!-- Encoder Path -->
    <g fill="#e6f3ff" stroke="#2980b9" stroke-width="2">
        <!-- Input -->
        <rect x="50" y="175" width="60" height="50" rx="5"/>
        <!-- Conv Blocks -->
        <rect x="130" y="150" width="60" height="100" rx="5"/>
        <rect x="210" y="125" width="60" height="150" rx="5"/>
        <rect x="290" y="100" width="60" height="200" rx="5"/>
    </g>

    <!-- Bottleneck -->
    <rect x="370" y="175" width="60" height="50" fill="#fff3e6" stroke="#e67e22" stroke-width="2" rx="5"/>

    <!-- Decoder Path -->
    <g fill="#e6ffe6" stroke="#27ae60" stroke-width="2">
        <!-- Upconv Blocks -->
        <rect x="450" y="100" width="60" height="200" rx="5"/>
        <rect x="530" y="125" width="60" height="150" rx="5"/>
        <rect x="610" y="150" width="60" height="100" rx="5"/>
        <!-- Output -->
        <rect x="690" y="175" width="60" height="50" rx="5"/>
    </g>

    <!-- Skip Connections -->
    <g stroke="#95a5a6" stroke-width="2" stroke-dasharray="5,5">
        <path d="M 240 150 L 500 150"/>
        <path d="M 320 125 L 480 125"/>
    </g>

    <!-- Labels -->
    <g font-family="Arial" font-size="12" fill="#333">
        <text x="55" y="205">Input</text>
        <text x="135" y="205">Conv1</text>
        <text x="215" y="205">Conv2</text>
        <text x="295" y="205">Conv3</text>
        <text x="375" y="205">Bridge</text>
        <text x="455" y="205">Up3</text>
        <text x="535" y="205">Up2</text>
        <text x="615" y="205">Up1</text>
        <text x="695" y="205">Output</text>
    </g>
</svg>]

U-Net features a symmetric encoder-decoder structure that's particularly effective for detailed segmentation:
- Contracting path (encoder) captures context
- Expanding path (decoder) enables precise localization
- Skip connections transfer detailed features from encoder to decoder
- Particularly effective at preserving fine structural details
- Our implementation uses a ResNet34 backbone for improved feature extraction

### 3. Pyramid Scene Parsing Network (PSPNet)
```mermaid
[PSPNet diagram from previous response]
```

PSPNet excels at capturing global context through its pyramid pooling module:
- Hierarchical global prior representation
- Multi-scale feature extraction through pyramid pooling
- Four levels of feature pooling (1×1, 2×2, 3×3, 6×6)
- Especially effective for complex scene understanding
- Better handling of objects at multiple scales

### 4. LinkNet
```mermaid
[LinkNet diagram from previous response]
```

LinkNet is designed for efficient semantic segmentation:
- Memory-efficient architecture with strong performance
- Direct connections between encoder and decoder blocks
- Residual connections for better gradient flow
- Lighter computational footprint compared to U-Net
- Ideal for real-time applications

### 5. DeepLabV3+
```mermaid
[DeepLabV3+ diagram from previous response]
```

DeepLabV3+ represents the state-of-the-art in semantic segmentation:
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale processing
- Multiple dilation rates (6, 12, 18) for broader receptive fields
- Encoder-decoder structure with ASPP module
- Fusion of low-level and high-level features
- Superior performance on boundary regions

## Architecture Comparison

| Architecture | Strengths | Best Use Cases | Memory Usage | Inference Speed |
|--------------|-----------|----------------|--------------|-----------------|
| FCN          | Simple, effective baseline | General segmentation | Medium | Fast |
| U-Net        | Fine detail preservation | Medical imaging, detailed segmentation | High | Medium |
| PSPNet       | Global context understanding | Complex scene parsing | High | Medium |
| LinkNet      | Efficiency, good performance | Real-time applications | Low | Fast |
| DeepLabV3+   | State-of-the-art accuracy | High-accuracy requirements | High | Slow |

## Results

Model performance comparison on IDD-Lite dataset:

| Architecture | Training Set | Testing Set | Mean F1 Score |
|--------------|-------------|-------------|---------------|
| FCN          | 0.9032      | 0.9034      | 0.687        |
| UNET         | 0.8784      | 0.7406      | 0.586        |
| PSPNET       | 0.9172      | 0.7385      | 0.733        |
| LINKNET      | 0.9231      | 0.7579      | 0.750        |
| DEEPLABV3+   | 0.8040      | 0.7712      | 0.787        |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/road-scene-segmentation.git
cd road-scene-segmentation

# Install dependencies
pip install -e .
```

### Requirements
- Python 3.7+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- albumentations >= 1.0.3
- OpenCV
- NumPy
- Matplotlib
- tqdm

## Dataset Setup

The project uses IDD-Lite dataset (~50MB). To set up the dataset:

```bash
python setup_data.py
```

This will download and organize the IDD-Lite dataset in the correct directory structure.

## Usage

### Training

To train a model:

```bash
python train.py --config config.yaml
```

Configure training parameters in `config.yaml`:
```yaml
MODEL_TYPE: 'unet'  # Options: 'fcn', 'unet', 'pspnet', 'linknet', 'deeplabv3'
BACKBONE: 'resnet34'
NUM_CLASSES: 8
BATCH_SIZE: 16
EPOCHS: 100
LEARNING_RATE: 0.001
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --config config.yaml --model-path checkpoints/final_model.pth
```

### Inference

For inference on a single image:

```python
from segmentation import SegmentationConfig, UNet, Visualizer
import cv2

# Initialize model and load weights
config = SegmentationConfig(MODEL_TYPE='unet')
model = UNet(config)
model.load_checkpoint('checkpoints/final_model.pth')

# Run inference
image = cv2.imread('path/to/image.jpg')
prediction = model.predict(image)
```

## Project Structure

```
├── segmentation/
│   ├── models/
│   │   ├── fcn.py
│   │   ├── unet.py
│   │   ├── pspnet.py
│   │   ├── linknet.py
│   │   └── deeplabv3.py
│   ├── config.py
│   ├── dataset.py
│   └── utils/
├── train.py
├── evaluate.py
├── setup_data.py
└── config.yaml
```

