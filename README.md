# Road Scene Semantic Segmentation

A PyTorch-based implementation of various deep learning architectures for semantic segmentation of unstructured road scenes using the Indian Driving Dataset (IDD).

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
graph LR
    subgraph VGG16_Backbone
        I[Input] --> C1[Conv Block 1]
        C1 --> C2[Conv Block 2]
        C2 --> C3[Conv Block 3]
        C3 --> C4[Conv Block 4]
        C4 --> C5[Conv Block 5]
    end
    subgraph FCN_Head
        C5 --> FC6[Conv 7x7]
        FC6 --> FC7[Conv 1x1]
        FC7 --> S1[Score]
    end
    subgraph Skip_Connections
        C4 --> S2[Score Pool4]
        C3 --> S3[Score Pool3]
        S1 --> U1[Upsample 2x]
        U1 --> F1[Fuse]
        S2 --> F1
        F1 --> U2[Upsample 2x]
        U2 --> F2[Fuse]
        S3 --> F2
        F2 --> U3[Upsample 8x]
        U3 --> O[Output]
    end
    style I fill:#f9f,stroke:#333
    style O fill:#9ff,stroke:#333
```

The FCN architecture transforms traditional classification networks into fully convolutional networks for semantic segmentation. Key features:
- Based on VGG16 backbone
- Replaces fully connected layers with 1x1 convolutions
- Uses skip connections from earlier layers for fine-grained prediction
- Multi-scale prediction fusion for better segmentation details

### 2. U-Net Architecture

```mermaid
graph TD
    subgraph Encoder
        I[Input Image] --> C1[Conv Block 1]
        C1 --> P1[MaxPool]
        P1 --> C2[Conv Block 2]
        C2 --> P2[MaxPool]
        P2 --> C3[Conv Block 3]
        C3 --> P3[MaxPool]
        P3 --> C4[Conv Block 4]
    end

    subgraph Bottleneck
        C4 --> B[Bottleneck]
    end

    subgraph Decoder
        B --> U1[UpConv 1]
        U1 --> D1[Conv Block 5]
        D1 --> U2[UpConv 2]
        U2 --> D2[Conv Block 6]
        D2 --> U3[UpConv 3]
        U3 --> D3[Conv Block 7]
        D3 --> O[Output]
    end

    %% Skip Connections
    C1 -.-> D3
    C2 -.-> D2
    C3 -.-> D1

    style I fill:#f9f,stroke:#333
    style O fill:#9ff,stroke:#333
    style B fill:#ff9,stroke:#333
```

### 3. Pyramid Scene Parsing Network (PSPNet)
```mermaid
graph TD
    subgraph Backbone
        I[Input] --> B[ResNet]
        B --> F[Feature Map]
    end
    subgraph Pyramid_Pooling
        F --> P1[Pool 1x1]
        F --> P2[Pool 2x2]
        F --> P3[Pool 3x3]
        F --> P4[Pool 6x6]
        P1 --> U1[Upsample]
        P2 --> U2[Upsample]
        P3 --> U3[Upsample]
        P4 --> U4[Upsample]
    end
    subgraph Final_Layers
        U1 --> C[Concat]
        U2 --> C
        U3 --> C
        U4 --> C
        F --> C
        C --> Conv[Conv 1x1]
        Conv --> Up[Upsample 8x]
        Up --> O[Output]
    end
    style I fill:#f9f,stroke:#333
    style O fill:#9ff,stroke:#333
```

PSPNet excels at capturing global context through its pyramid pooling module:
- Hierarchical global prior representation
- Multi-scale feature extraction through pyramid pooling
- Four levels of feature pooling (1×1, 2×2, 3×3, 6×6)
- Especially effective for complex scene understanding
- Better handling of objects at multiple scales

### 4. LinkNet
```mermaid
graph TD
    subgraph Encoder
        I[Input] --> E1[Encoder Block 1]
        E1 --> E2[Encoder Block 2]
        E2 --> E3[Encoder Block 3]
        E3 --> E4[Encoder Block 4]
    end
    subgraph Decoder
        E4 --> D4[Decoder Block 4]
        D4 --> D3[Decoder Block 3]
        D3 --> D2[Decoder Block 2]
        D2 --> D1[Decoder Block 1]
    end
    %% Skip Connections
    E1 -.-> D1
    E2 -.-> D2
    E3 -.-> D3
    E4 -.-> D4
    D1 --> F[Final Conv]
    F --> O[Output]
    style I fill:#f9f,stroke:#333
    style O fill:#9ff,stroke:#333
```

LinkNet is designed for efficient semantic segmentation:
- Memory-efficient architecture with strong performance
- Direct connections between encoder and decoder blocks
- Residual connections for better gradient flow
- Lighter computational footprint compared to U-Net
- Ideal for real-time applications

### 5. DeepLabV3+
```mermaid
graph TD
    subgraph Encoder
        I[Input] --> B[Backbone]
        B --> ASPP{ASPP Module}
    end
    subgraph ASPP_Module
        ASPP --> A1[1x1 Conv]
        ASPP --> A2[3x3 Rate 6]
        ASPP --> A3[3x3 Rate 12]
        ASPP --> A4[3x3 Rate 18]
        ASPP --> A5[Global Pool]
    end
    subgraph Decoder
        A1 & A2 & A3 & A4 & A5 --> C[Concat]
        C --> C1[Conv 1x1]
        B --> LF[Low-level Features]
        LF --> C2[Conv 1x1]
        C1 --> U1[Upsample 4x]
        U1 --> M[Merge]
        C2 --> M
        M --> U2[Upsample 4x]
        U2 --> O[Output]
    end
    style I fill:#f9f,stroke:#333
    style O fill:#9ff,stroke:#333
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


![Model Results](IMAGES/results.png)



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

