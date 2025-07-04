from sklearn.model_selection import train_test_split
import pandas as pd
from utils import setup_logging
import logging
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import MultimodalClipQwenModel
import torch.optim as optim
from utils import get_device
from dataset import load_flickr30k_dataset, create_flickr30k_dataloaders
from eval import evaluate
import torch
import os
import json
from datetime import datetime
import argparse
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model import MultimodalClipQwenConfig, MultimodalClipQwenModel

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

hyperparameters = {
    "batch_size": 4, 
    "num_epochs": 10,
    "learning_rate": 0.0001,
    "learning_rate_decay": 0.8,
    "learning_rate_decay_step": 1,
    "learning_rate_scheduler_enabled": True,
    "image_size": 224,
    "tokenizer_name": "Qwen/Qwen3-0.6B-Base",
    "max_caption_length": 128,
    "training_size_limit": None, # None for max
    "evaluation_enabled": False,
    "clip_model_name": "openai/clip-vit-base-patch32",
    "qwen_model_name": "Qwen/Qwen3-0.6B-Base",
    "mlp_hidden_size": 1024,
    "clip_hidden_size": 512,
}

config = MultimodalClipQwenConfig(
    clip_model_name=hyperparameters["clip_model_name"],
    qwen_model_name=hyperparameters["qwen_model_name"],
    mlp_hidden_size=hyperparameters["mlp_hidden_size"],
    clip_hidden_size=hyperparameters["clip_hidden_size"], 
    tokenizer_name=hyperparameters["tokenizer_name"]
)

def save_model(model, optimizer, epoch_num, loss, score, training_size, save_dir = "saved_models", training_history=None, tokenizer=None, timestamp=None):
    logger.info("Saving model...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model directory (AutoModel format)
    model_dir = os.path.join(save_dir, f"model_{timestamp}-epoch{epoch_num}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Save tokenizer in the model directory
    model.tokenizer.save_pretrained(model_dir)
    logger.info(f"✅ Tokenizer saved to: {model_dir}")
    
    # 2. Save model weights in AutoModel format
    model_weights_path = os.path.join(model_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_weights_path)
    logger.info(f"✅ Model weights saved to: {model_weights_path}")
    
    # 3. Create model configuration file (AutoModel compatible)
    config_data = {
        "model_type": "multimodal_clip_qwen",
        "architectures": ["MultimodalClipQwenModel"],
        "clip_model_name": "openai/clip-vit-base-patch32",
        "qwen_model_name": "Qwen/Qwen3-0.6B-Base",
        "mlp_hidden_size": 1024,
        "clip_hidden_size": 512,
        "freeze_clip": True,
        "freeze_qwen": True,
        "tokenizer_name": hyperparameters["tokenizer_name"],
        "torch_dtype": "bfloat16",
        # Additional metadata for backwards compatibility
        "hyperparameters": hyperparameters,
        "training_info": {
            "epoch": epoch_num,
            "num_epochs": hyperparameters["num_epochs"],
            "loss": loss,
            "score": score,
            "training_size": training_size,
            "timestamp": timestamp
        },
        "tokenizer_info": {
            "bos_token_id": model.tokenizer.bos_token_id,
            "vocab_size": len(model.tokenizer),
            "tokenizer_name": hyperparameters["tokenizer_name"]
        }
    }
    
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"✅ Model config saved to: {config_path}")
    
    # 4. Save training state (for resuming training)
    training_state_path = os.path.join(model_dir, "training_state.bin")
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history if training_history is not None else {},
        'epoch': epoch_num,
        'hyperparameters': hyperparameters,
    }, training_state_path)
    logger.info(f"✅ Training state saved to: {training_state_path}")
    
    # 5. Save training metadata (human-readable)
    metadata_path = os.path.join(model_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            'hyperparameters': hyperparameters,
            'training_history': training_history if training_history is not None else {},
            'model_dir': model_dir,
            'timestamp': timestamp,
            'epoch': epoch_num,
            'num_epochs': hyperparameters["num_epochs"],
            'loss': loss,
            'score': score,
            'training_size': training_size,
            'bos_token_id': model.tokenizer.bos_token_id,
            'tokenizer_vocab_size': len(model.tokenizer),
        }, f, indent=2)
    logger.info(f"✅ Training metadata saved to: {metadata_path}")
    
    logger.info(f"🎉 Model saved successfully in AutoModel format!")
    logger.info(f"📁 Model directory: {model_dir}")
    logger.info(f"📄 Files saved:")
    logger.info(f"   - config.json (model configuration)")
    logger.info(f"   - pytorch_model.bin (model weights)")
    logger.info(f"   - tokenizer files (tokenizer_config.json, vocab.json, etc.)")
    logger.info(f"   - training_state.bin (optimizer state for resuming)")
    logger.info(f"   - training_metadata.json (human-readable info)")
    logger.info(f"💡 You can now load this model using AutoModel.from_pretrained('{model_dir}')")
    
    return model_dir

def load_model(model_dir, device=None):
    """
    Load a model saved in AutoModel format
    
    Args:
        model_dir: Path to the directory containing the saved model
        device: Device to load the model on (defaults to get_device())
        
    Returns:
        tuple: (model, config_data, training_state) or None if loading fails
    """
    if device is None:
        device = get_device()
    
    logger.info(f"Loading model from: {model_dir}")
    
    try:
        # 1. Load configuration
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return None
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        logger.info(f"✅ Config loaded from: {config_path}")
        
        # 2. Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info(f"✅ Tokenizer loaded from: {model_dir}")
        
        # 3. Create model instance
        model = Model(saved_tokenizer=tokenizer)
        model = model.to(device)
        logger.info(f"✅ Model instance created")
        
        # 4. Load model weights
        model_weights_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_weights_path):
            logger.error(f"Model weights not found: {model_weights_path}")
            return None
            
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"✅ Model weights loaded from: {model_weights_path}")
        
        # 5. Load training state (optional)
        training_state = None
        training_state_path = os.path.join(model_dir, "training_state.bin")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            logger.info(f"✅ Training state loaded from: {training_state_path}")
        else:
            logger.info("ℹ️ Training state not found (not required for inference)")
        
        logger.info(f"🎉 Model loaded successfully!")
        logger.info(f"📄 Model type: {config_data.get('model_type', 'unknown')}")
        logger.info(f"📄 Epoch: {config_data.get('training_info', {}).get('epoch', 'unknown')}")
        logger.info(f"📄 Vocab size: {config_data.get('tokenizer_info', {}).get('vocab_size', 'unknown')}")
        
        return model, config_data, training_state
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return None

def train(model, training_dataloader, validation_dataloader, optimizer, device, epoch_num, num_epochs, timestamp, training_history=None, tokenizer=None):
    model.train()
    running_loss = 0
    log_interval = 10  # Log every 10 batches
    
    # Create progress bar
    pbar = tqdm(enumerate(training_dataloader), total=len(training_dataloader), desc=f"Training Epoch {epoch_num}/{num_epochs}")
    
    for batch_idx, batch in pbar:
        batch_images = batch["image"]
        batch_input_ids = batch["input_ids"].to(device)
        bos_token_id = model.tokenizer.bos_token_id
        batch_bos_ids = torch.full((batch_input_ids.shape[0], 1), bos_token_id, dtype=torch.long, device=device)
        batch_input_ids = torch.cat([batch_bos_ids, batch_input_ids], dim=1).to(device)
        batch_attention_mask = batch["attention_mask"].to(device)   
        batch_attention_mask = torch.cat([torch.ones((batch_input_ids.shape[0], 1), dtype=torch.long, device=device), batch_attention_mask], dim=1).to(device)


        # batch_attention_mask = batch["attention_mask"].to(device)  # ✅ Also get attention mask
        outputs = model(
            images=batch_images, 
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        # loss = loss_function(outputs, batch_captions)
        loss = outputs.loss
        loss.backward() # backpropagate the loss
        optimizer.step() # update the model parameters
        optimizer.zero_grad() # clear the gradients (reset)
        running_loss += loss.item()
        
        # Update progress bar with current loss
        current_avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{current_avg_loss:.4f}'
        })
        # Log loss periodically
        if (batch_idx + 1) % log_interval == 10:
            logger.info(f"Epoch {epoch_num}/{num_epochs}, Batch {batch_idx + 1}/{len(training_dataloader)}, "
                       f"Current Loss: {loss.item():.4f}, Running Avg Loss: {current_avg_loss:.4f}")

    if hyperparameters["evaluation_enabled"]:
        validation_loss, validation_score = evaluate(
            dataloader=validation_dataloader,
            model=model,
            epoch_num=epoch_num,
            num_epochs=num_epochs
        )
    else:
        validation_loss = None
        validation_score = None

    training_loss = running_loss / len(training_dataloader)
    training_size = len(training_dataloader)
    save_model(model, optimizer, epoch_num, training_loss, validation_score, training_size, training_history=training_history, tokenizer=tokenizer, timestamp=timestamp)
    logger.info(f"Training Epoch {epoch_num}/{num_epochs} completed!")
    logger.info(f"Training size: {training_size}")
    logger.info(f"Training loss: {training_loss}")
    if hyperparameters["evaluation_enabled"]:
        logger.info(f"Validation loss: {validation_loss}")
        logger.info(f"Validation score: {validation_score}")
    else:
        logger.info(f"Validation loss: N/A")
        logger.info(f"Validation score: N/A")
    
    if hyperparameters["learning_rate_scheduler_enabled"]:
        scheduler.step()
    
    return training_loss, validation_loss, validation_score, training_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--training-size-limit", 
        type=int, 
        help="Limit the number of training samples (None or 0 for unlimited)"
    )
    args = parser.parse_args()

    # Override training size limit if provided
    if args.training_size_limit is not None:
        # Treat 0 as unlimited (None)
        if args.training_size_limit == 0:
            hyperparameters["training_size_limit"] = None
            logger.info("Training size limit set to unlimited (0 provided)")
        else:
            hyperparameters["training_size_limit"] = args.training_size_limit
            logger.info(f"Training size limit overridden to: {args.training_size_limit}")
    else:
        logger.info(f"Using default training size limit: {hyperparameters['training_size_limit']}")

    datasets = load_flickr30k_dataset()
    dataloaders = create_flickr30k_dataloaders(
        datasets=datasets,
        image_size=hyperparameters["image_size"],
        batch_size=hyperparameters["batch_size"],
        tokenizer_name=hyperparameters["tokenizer_name"],
        max_caption_length=hyperparameters["max_caption_length"],
        use_all_captions=False,
        train_size_limit=hyperparameters["training_size_limit"]
    )
    
    # Extract individual dataloaders
    training_dataloader = dataloaders['train']
    validation_dataloader = dataloaders['validation']
    test_dataloader = dataloaders['test']

    logger.info("Flickr30K Dataset loading completed!")


    model = MultimodalClipQwenModel(config).to(get_device())
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    if hyperparameters["learning_rate_scheduler_enabled"]:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters["learning_rate_decay_step"], gamma=hyperparameters["learning_rate_decay"])

    # Track training metrics
    training_history = {
        "training_losses": [],
        "validation_losses": [],
        "validation_scores": [],
        "training_sizes": []
    }

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
    model.tokenizer = tokenizer
    logger.info(f"✅ Created fresh tokenizer with BOS token ({model.tokenizer.bos_token_id})")
    

    for epoch_num in range(hyperparameters["num_epochs"]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        training_loss, validation_loss, validation_score, training_size = train(
            model=model,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            optimizer=optimizer,
            device=get_device(),
            epoch_num=epoch_num+1,
            num_epochs=hyperparameters["num_epochs"],
            timestamp=timestamp,
            training_history=training_history,
            tokenizer=tokenizer
        )
        
        # Store training metrics
        training_history["training_losses"].append(training_loss)
        training_history["validation_losses"].append(validation_loss)
        training_history["validation_scores"].append(validation_score)
        training_history["training_sizes"].append(training_size)
    
    logger.info(f"Final training loss: {training_history['training_losses'][-1]:.4f}")
    if hyperparameters["evaluation_enabled"]:
        logger.info(f"Final validation loss: {training_history['validation_losses'][-1]:.4f}")
        logger.info(f"Final validation score: {training_history['validation_scores'][-1]:.4f}")
    else:
        logger.info(f"Final validation loss: N/A")
        logger.info(f"Final validation score: N/A")
    
