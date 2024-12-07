# example_usage.py
import torch
from segmentation import SegmentationConfig, UNet, Visualizer
import cv2
import matplotlib.pyplot as plt

def main():
    # Setup configuration
    config = SegmentationConfig(
        MODEL_TYPE='unet',
        BACKBONE='resnet34',
        NUM_CLASSES=8,
        INPUT_SHAPE=(256, 256, 3)
    )
    
    # Initialize model
    model = UNet(config)
    
    # Load pretrained weights (if available)
    try:
        model.load_checkpoint('/Users/pratheeshjp/Documents/SegwayVision/checkpoints/final_model.pth')
        print("Loaded pretrained weights")
    except:
        print("No pretrained weights found")
    
    # Example inference
    image_path = 'IDD_data/IDD/idd20k_lite/leftImg8bit/val/119/903127_image.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    image = cv2.resize(image, (256, 256))
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Visualize
    visualizer = Visualizer(config)
    visualizer.visualize_prediction(image, prediction.squeeze())
    plt.show()

if __name__ == '__main__':
    main()