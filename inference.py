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

    # Verify BOS token exists
    if model.tokenizer.bos_token_id is None:
        logger.error("BOS token not found in tokenizer!")
        return
    
    logger.info(f"Using BOS token: '{model.tokenizer.bos_token}' (ID: {model.tokenizer.bos_token_id})")
    input_ids = torch.tensor([model.tokenizer.bos_token_id], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # Generate text autoregressively
    max_length = 50  # Adjust as needed
    generated_tokens = [model.tokenizer.bos_token_id]
    
    with torch.no_grad():
        for _ in range(max_length):
            current_input_ids = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            current_attention_mask = torch.ones_like(current_input_ids, dtype=torch.long, device=device)
            
            outputs = model(images=[image], input_ids=current_input_ids, attention_mask=current_attention_mask)
            logits = outputs.logits
            
            # Get the next token (last token in the sequence)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Stop if we hit EOS token
            if hasattr(model.tokenizer, 'eos_token_id') and next_token_id == model.tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id)
        
        # Decode the full generated sequence (excluding BOS for cleaner output)
        output_string = model.tokenizer.decode(generated_tokens[1:], skip_special_tokens=True)
        logger.info(f"Generated text: {output_string}")
        return output_string

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
    checkpoint = model_data['checkpoint']
    logger.info("✅ Model loaded successfully!")
    logger.info(f"Model was trained for {num_epochs} epochs")
    
    # Use the exact BOS token ID that was saved during training
    saved_bos_token_id = checkpoint.get('bos_token_id')
    saved_vocab_size = checkpoint.get('tokenizer_vocab_size')
    
    if saved_bos_token_id is not None:
        # Ensure tokenizer setup matches training
        model.tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
        model.qwen_decoder.qwen_model.model.resize_token_embeddings(len(model.tokenizer))
        
        # Verify we got the same BOS token ID
        current_bos_id = model.tokenizer.bos_token_id
        current_vocab_size = len(model.tokenizer)
        
        if current_bos_id != saved_bos_token_id:
            logger.error(f"❌ BOS token ID mismatch! Training: {saved_bos_token_id}, Current: {current_bos_id}")
            logger.error("This will cause incorrect inference results!")
            exit(1)
        
        if saved_vocab_size and current_vocab_size != saved_vocab_size:
            logger.warning(f"⚠️ Vocab size mismatch! Training: {saved_vocab_size}, Current: {current_vocab_size}")
        
        logger.info(f"✅ BOS token ID verified: {current_bos_id} (matches training)")
    else:
        # Fallback for older saved models without BOS token ID
        logger.warning("⚠️ No saved BOS token ID found. Using current tokenizer setup.")
        model.tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
        model.qwen_decoder.qwen_model.model.resize_token_embeddings(len(model.tokenizer))
        logger.info(f"⚠️ Using BOS token ID: {model.tokenizer.bos_token_id} (not verified)")

    # Process the image
    image = Image.open(args.image)
    image = image.resize((224, 224))
    image = image.convert("RGB")
    logger.info("✅ Image loaded successfully!")
    logger.info("✅ Image converted to RGB (224x224) successfully!")

    inference(image, model, hyperparameters, training_history)