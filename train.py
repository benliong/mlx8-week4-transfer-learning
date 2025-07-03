from sklearn.model_selection import train_test_split
import pandas as pd
from utils import setup_logging
import logging
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Model
import torch.optim as optim
from utils import get_device
from dataset import load_flickr30k_dataset, create_flickr30k_dataloaders
from eval import evaluate
import torch
import os
import json
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

hyperparameters = {
    "batch_size": 4, 
    "num_epochs": 1,
    "learning_rate": 0.0001,
    "learning_rate_decay": 0.8,
    "learning_rate_decay_step": 1,
    "learning_rate_scheduler_enabled": True,
    "image_size": 224,
    "tokenizer_name": "Qwen/Qwen3-0.6B-Base",
    "max_caption_length": 128,
    "training_size_limit": 500, # None for max
    "evaluation_enabled": False,
}

def save_model(model, optimizer, epoch_num, loss, score, training_size, save_dir = "saved_models"):
    logger.info("Saving model...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save tokenizer first (most important for consistency!)
    tokenizer_dir = os.path.join(save_dir, f"tokenizer_{timestamp}-{epoch_num}")
    model.tokenizer.save_pretrained(tokenizer_dir)
    logger.info(f"✅ Tokenizer saved to: {tokenizer_dir}")
    
    # Save model state dict (recommended approach)
    model_path = os.path.join(save_dir, f"model_state_dict_{timestamp}-{epoch_num}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': hyperparameters,
        'training_history': training_history,
        'epoch': epoch_num,
        'num_epochs': hyperparameters["num_epochs"],
        'loss': loss,
        'score': score,
        'training_size': training_size,
        'tokenizer_dir': tokenizer_dir,  # Path to the saved tokenizer
        'bos_token_id': model.tokenizer.bos_token_id,  # Keep for verification
        'tokenizer_vocab_size': len(model.tokenizer),  # Keep for verification
    }, model_path)
    
    # Save hyperparameters and training history as JSON for easy access
    metadata_path = os.path.join(save_dir, f"training_metadata_{timestamp}-{epoch_num}.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            'hyperparameters': hyperparameters,
            'training_history': training_history,
            'model_path': model_path,
            'tokenizer_dir': tokenizer_dir,  # Include tokenizer path
            'timestamp': timestamp,
            'epoch': epoch_num,
            'num_epochs': hyperparameters["num_epochs"],
            'loss': loss,
            'score': score,
            'bos_token_id': model.tokenizer.bos_token_id,
            'tokenizer_vocab_size': len(model.tokenizer),
        }, f, indent=2)
    
    logger.info(f"Model saved successfully!")
    logger.info(f"Model state dict: {model_path}")
    logger.info(f"Training metadata: {metadata_path}")

def train(model, training_dataloader, validation_dataloader, optimizer, device, epoch_num, num_epochs, timestamp):
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
    save_model(model, optimizer, epoch_num, training_loss, validation_score, training_size)
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

    model = Model().to(get_device())
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    if hyperparameters["learning_rate_scheduler_enabled"]:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters["learning_rate_decay_step"], gamma=hyperparameters["learning_rate_decay"])

    # Track training metrics
    training_history = {
        "training_losses": [],
        "validation_losses": [],
        "validation_scores": []
    }

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
            timestamp=timestamp
        )
        
        # Store training metrics
        training_history["training_losses"].append(training_loss)
        training_history["validation_losses"].append(validation_loss)
        training_history["validation_scores"].append(validation_score)
    
    logger.info(f"Final training loss: {training_history['training_losses'][-1]:.4f}")
    if hyperparameters["evaluation_enabled"]:
        logger.info(f"Final validation loss: {training_history['validation_losses'][-1]:.4f}")
        logger.info(f"Final validation score: {training_history['validation_scores'][-1]:.4f}")
    else:
        logger.info(f"Final validation loss: N/A")
        logger.info(f"Final validation score: N/A")
    
