from model import Model
from utils import get_device, load_model_for_inference, print_model_summary, get_device
from PIL import Image
import torch
import argparse
from utils import setup_logging
import logging
from transformers import AutoTokenizer

setup_logging()
logger = logging.getLogger(__name__)

def inference(image, model, hyperparameters, training_history):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    model.eval()
    model = model.to(get_device())

    prompt = "This photo shows "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(get_device())
    attention_mask = torch.ones_like(input_ids).to(get_device())

    with torch.no_grad():
        outputs = model(images=[image], input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        output_string = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)[0]
        logger.info(f"Output string: {output_string}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on an image using a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Selection Options:
  --model (no value)     : Use latest model
  --model 1              : Use latest model (by index)
  --model 2              : Use second latest model
  --model 20241205_143022: Use model matching timestamp
  --model epoch_1        : Use model matching partial string
  --model full/path      : Use specific path
  
  --list                 : Show all available models
        """
    )
    parser.add_argument(
        "--model", "-m", 
        nargs="?", 
        const=None,
        default=None,
        help="Model identifier (latest if not specified). See examples below."
    )
    parser.add_argument(
        "--image", "-i", 
        default="images/bes.jpg", 
        help="Path to the input image (default: images/bes.jpg)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available models and exit"
    )

    args = parser.parse_args()

    if args.list:
        print_model_summary()
        exit(0)

    try:
        # Use the new smart model loading system
        model_data = load_model_for_inference(args.model)
        model = model_data['model']
        hyperparameters = model_data['hyperparameters']
        training_history = model_data['training_history']
        
        # Process the image
        image = Image.open(args.image)
        image = image.resize((224, 224))
        image = image.convert("RGB")
        logger.info("✅ Image loaded successfully!")
        logger.info("✅ Image converted to RGB (224x224) successfully!")

        inference(image, model, hyperparameters, training_history)
        
    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        logger.info("\nAvailable models:")
        print_model_summary()
        exit(1)
    except ValueError as e:
        logger.error(f"❌ Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        raise