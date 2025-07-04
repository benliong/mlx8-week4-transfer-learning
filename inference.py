from utils import get_device, list_saved_models, get_device
from PIL import Image
import torch
import argparse
from utils import setup_logging
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from model import MultimodalClipQwenConfig, MultimodalClipQwenModel
import os

setup_logging()
logger = logging.getLogger(__name__)

def inference(image, model, max_length=None):
    model.eval()
    model = model.to(get_device())
    device = get_device()

    logger.info(f"Using BOS token: '{model.tokenizer.bos_token}' (ID: {model.tokenizer.bos_token_id})")
    input_ids = torch.tensor([model.tokenizer.bos_token_id], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # Generate text autoregressively
    # Use training max length by default, but allow override
    training_max_length = model.config.hyperparameters.get("max_caption_length", 128)    
    if max_length is None:
        max_length = training_max_length  # Use training length by default
        logger.info(f"Using training max length: {max_length} tokens")
    else:
        logger.info(f"Using custom max length: {max_length} tokens (training used: {training_max_length})")
        if max_length > training_max_length:
            logger.warning(f"⚠️ Generating longer than training length! May produce lower quality text.")
    
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
    parser.add_argument("--max-length", type=int, default=None, 
                       help="Maximum tokens to generate (default: use training max_caption_length)")
    
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
    
    device = get_device()

    AutoConfig.register("multimodal_clip_qwen", MultimodalClipQwenConfig)
    AutoModel.register(MultimodalClipQwenConfig, MultimodalClipQwenModel)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info(f"✅ Tokenizer ({args.model}) loaded successfully!")
    model = AutoModel.from_pretrained(args.model).to(device)
    model.tokenizer = tokenizer
    logger.info(f"✅ Model ({args.model}) loaded successfully!")
    
    # Process the image
    image = Image.open(args.image)
    image = image.resize((224, 224))
    image = image.convert("RGB")
    logger.info("✅ Image loaded successfully!")
    logger.info("✅ Image converted to RGB (224x224) successfully!")

    # Run inference with specified max length
    if args.max_length is not None:
        logger.info(f"Using command-line max length: {args.max_length}")
    
    inference(image, model, max_length=args.max_length)