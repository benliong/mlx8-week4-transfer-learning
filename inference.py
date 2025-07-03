from model import Model
from utils import get_device, load_saved_model, list_saved_models, print_model_summary, get_device
from PIL import Image
import torch
import argparse
from utils import setup_logging
import logging
from transformers import AutoTokenizer
import os

setup_logging()
logger = logging.getLogger(__name__)

def inference(image, model, hyperparameters, training_history):
    model.eval()
    model = model.to(get_device())
    device = get_device()

    input_ids = torch.tensor([model.tokenizer.bos_token_id], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(images=[image], input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        output_string = model.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)[0]
        logger.info(f"Output string: {output_string}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image using a trained model")
    parser.add_argument("--model", "-m", required=True, help="Path to the model file (.pth)")
    parser.add_argument("--image", "-i", default="images/bes.jpg", help="Path to the input image (default: test.jpg)")
    
    try:
        args = parser.parse_args()
        
        # Check if the model file exists
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found.")
            print("\nAvailable models (sorted by date descending):")
            models = list_saved_models()
            if models:
                for i, model_info in enumerate(models, 1):
                    print(f"{i}. {model_info['filename']} (Size: {model_info['size_mb']:.1f} MB)")
            else:
                print("No saved models found.")
            exit(1)
            
    except SystemExit as e:
        # This catches the SystemExit raised by argparse when --model is missing
        if e.code == 2:  # argparse error code
            print("\nAvailable models (sorted by date descending):")
            models = list_saved_models()
            if models:
                for i, model_info in enumerate(models, 1):
                    print(f"{i}. {model_info['filename']} (Size: {model_info['size_mb']:.1f} MB)")
            else:
                print("No saved models found.")
        raise  # Re-raise the SystemExit to maintain normal argparse behavior

    model_data = load_saved_model(args.model, Model)
    model = model_data['model']
    hyperparameters = model_data['hyperparameters']
    training_history = model_data['training_history']
    num_epochs = model_data['epoch']
    logger.info("✅ Model loaded successfully!")
    logger.info(f"Model was trained for {num_epochs} epochs")

    # Process the image
    image = Image.open(args.image)
    image = image.resize((224, 224))
    image = image.convert("RGB")
    logger.info("✅ Image loaded successfully!")
    logger.info("✅ Image converted to RGB (224x224) successfully!")

    inference(image, model, hyperparameters, training_history)