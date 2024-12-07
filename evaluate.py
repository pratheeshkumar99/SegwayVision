import torch
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from segmentation import (
    SegmentationConfig, SegmentationDataset, get_dataloader,
    FCN, LinkNet, UNet, DeepLabV3, Visualizer, SegmentationMetrics
)

def evaluate_model(model, test_loader, config, save_dir=None):
    """Evaluate model on test set."""
    device = torch.device(config.DEVICE)
    model.eval()
    metrics = SegmentationMetrics()
    visualizer = Visualizer(config)
    
    total_metrics = {'dice': 0, 'iou': 0}
    examples = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            batch_metrics = metrics.calculate_metrics(outputs, masks)
            
            # Update metrics
            for k in total_metrics:
                total_metrics[k] += batch_metrics[k]
            
            # Save some examples
            if i < 5:  # Save first 5 batches for visualization
                examples.append((images, masks, outputs))
    
    # Calculate average metrics
    num_batches = len(test_loader)
    avg_metrics = {k: v/num_batches for k, v in total_metrics.items()}
    
    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(save_dir / 'metrics.txt', 'w') as f:
            for k, v in avg_metrics.items():
                f.write(f'{k}: {v:.4f}\n')
        
        # Save visualizations
        for i, (images, masks, outputs) in enumerate(examples):
            visualizer.visualize_batch(
                images, masks, outputs,
                save_path=save_dir / f'example_{i}.png'
            )
    
    return avg_metrics

# inference.py
def load_and_predict(image_path, model_path, config):
    """Load model and run inference on single image."""
    # Load model
    model = get_model(config.MODEL_TYPE, config)
    model.load_checkpoint(model_path)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, config.INPUT_SHAPE[:2])
    
    # Convert to tensor
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    image = transform(image=image)['image']
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(image)
        prediction = torch.softmax(output, dim=1)
        prediction = torch.argmax(prediction, dim=1)
    
    return prediction.squeeze().cpu().numpy()
