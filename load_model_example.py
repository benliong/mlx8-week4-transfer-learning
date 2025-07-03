#!/usr/bin/env python3
"""
Example script showing how to load a saved model and use it for inference.

Usage:
    python load_model_example.py [model_path]
    
If no model_path is provided, it will use the most recent saved model.
"""

import argparse
import os
from model import Model
from utils import load_saved_model, list_saved_models, print_model_summary, get_device
from dataset import load_flickr30k_dataset, create_flickr30k_dataloaders
import torch
import logging
from utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def load_latest_model():
    """Load the most recent saved model."""
    models = list_saved_models()
    if not models:
        raise FileNotFoundError("No saved models found in 'saved_models' directory")
    
    latest_model = models[0]  # models are sorted by timestamp, newest first
    logger.info(f"Loading latest model: {latest_model['filename']}")
    return latest_model['path']


def run_inference_example(model_path=None):
    """Run inference example with a saved model."""
    
    # Load model
    if model_path is None:
        model_path = load_latest_model()
    
    logger.info(f"Loading model from: {model_path}")
    loaded_data = load_saved_model(model_path, Model)
    
    model = loaded_data['model']
    hyperparameters = loaded_data['hyperparameters']
    training_history = loaded_data['training_history']
    
    logger.info("✅ Model loaded successfully!")
    logger.info(f"Model was trained for {loaded_data['epoch']} epochs")
    
    if training_history:
        final_train_loss = training_history['training_losses'][-1]
        final_val_loss = training_history['validation_losses'][-1]
        logger.info(f"Final training loss: {final_train_loss:.4f}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Load test data (using same hyperparameters as training)
    logger.info("Loading test dataset...")
    datasets = load_flickr30k_dataset()
    dataloaders = create_flickr30k_dataloaders(
        datasets=datasets,
        image_size=hyperparameters["image_size"],
        batch_size=1,  # Use batch size 1 for inference
        tokenizer_name=hyperparameters["tokenizer_name"],
        max_caption_length=hyperparameters["max_caption_length"],
        use_all_captions=False
    )
    
    test_dataloader = dataloaders['test']
    
    # Run inference on a few test samples
    logger.info("Running inference on test samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= 3:  # Only process first 3 samples
                break
                
            batch_images = batch["image"]
            batch_input_ids = batch["input_ids"].to(get_device())
            batch_attention_mask = batch["attention_mask"].to(get_device())
            
            # Run inference
            outputs = model(
                images=batch_images,
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Input shape: {batch_input_ids.shape}")
            logger.info(f"  Output logits shape: {outputs.logits.shape}")
            logger.info(f"  Loss: {outputs.loss.item():.4f}")
            logger.info("-" * 40)
    
    logger.info("✅ Inference example completed!")


def main():
    parser = argparse.ArgumentParser(description="Load and test a saved model")
    parser.add_argument(
        "model_path", 
        nargs="?", 
        default=None,
        help="Path to saved model file (.pth). If not provided, uses the most recent model."
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all saved models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print_model_summary()
        return
    
    try:
        run_inference_example(args.model_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Available models:")
        print_model_summary()
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()