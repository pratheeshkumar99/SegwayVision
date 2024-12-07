# evaluate.py

import argparse
import torch
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from segmentation import (
    SegmentationConfig, 
    SegmentationDataset,
    get_dataloader,
    UNet, 
    FCN,
    LinkNet,
    DeepLabV3,
    Visualizer,
    SegmentationMetrics
)

def setup_logging(save_dir: str):
    """Setup logging configuration."""
    log_file = Path(save_dir) / f'evaluation_{datetime.now():%Y%m%d_%H%M%S}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_model(model_type: str, config: SegmentationConfig):
    """Get model based on type."""
    models = {
        'fcn': FCN,
        'linknet': LinkNet,
        'unet': UNet,
        'deeplabv3': DeepLabV3
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(models.keys())}")
    
    return models[model_type](config)

def evaluate(config: SegmentationConfig, model_path: str, save_dir: str):
    """Evaluate model on test set."""
    logger = setup_logging(save_dir)
    device = torch.device(config.DEVICE)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = get_model(config.MODEL_TYPE, config)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    logger.info("Creating test dataset")
    test_dataset = SegmentationDataset(config, mode='val')  # Using val set for evaluation
    test_loader = get_dataloader(test_dataset, config)
    
    # Initialize metrics and visualizer
    metrics = SegmentationMetrics()
    visualizer = Visualizer(config)
    
    # Evaluation loop
    logger.info("Starting evaluation")
    total_metrics = {'dice': 0, 'iou': 0}
    examples = []
    
    try:
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
                # Move to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate metrics
                batch_metrics = metrics.calculate_metrics(outputs, masks)
                
                # Update metrics
                for k in total_metrics:
                    total_metrics[k] += batch_metrics[k]
                
                # Save some examples for visualization
                if i < 5:  # Save first 5 batches
                    examples.append((images.cpu(), masks.cpu(), outputs.cpu()))
                
                # Clear some memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Calculate average metrics
        num_batches = len(test_loader)
        avg_metrics = {k: v/num_batches for k, v in total_metrics.items()}
        
        # Save results
        logger.info("Saving results")
        
        # Save metrics
        metrics_file = save_dir / 'metrics.txt'
        with open(metrics_file, 'w') as f:
            for k, v in avg_metrics.items():
                f.write(f'{k}: {v:.4f}\n')
            logger.info(f"Metrics saved to {metrics_file}")
        
        # Save visualizations
        for i, (images, masks, outputs) in enumerate(examples):
            save_path = save_dir / f'example_{i}.png'
            visualizer.visualize_batch(
                images, masks, outputs,
                save_path=save_path
            )
            logger.info(f"Visualization {i+1} saved to {save_path}")
        
        logger.info("Evaluation completed successfully")
        return avg_metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--config', type=str, required=True,
                      help='path to config file')
    parser.add_argument('--model-path', type=str, required=True,
                      help='path to model checkpoint')
    parser.add_argument('--save-dir', type=str, default='evaluation_results',
                      help='directory to save results')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = SegmentationConfig()
    for k, v in config_dict.items():
        setattr(config, k, v)
    
    # Run evaluation
    try:
        metrics = evaluate(config, args.model_path, args.save_dir)
        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()