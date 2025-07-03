from model import Model
from utils import get_device, load_saved_model, list_saved_models, print_model_summary, get_device
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
    parser = argparse.ArgumentParser(description="Run inference on an image using a trained model")
    parser.add_argument("--model", "-m", required=True, help="Path to the model file (.pth)")
    parser.add_argument("--image", "-i", default="images/bes.jpg", help="Path to the input image (default: test.jpg)")

    args = parser.parse_args()

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