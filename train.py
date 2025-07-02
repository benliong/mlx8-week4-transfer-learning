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

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

hyperparameters = {
    "batch_size": 1, 
    "num_epochs": 1,
    "learning_rate": 0.001,
    "image_size": 224,
    "tokenizer_name": "Qwen/Qwen3-0.6B-Base",
    "max_caption_length": 128,
}

def train(model, dataloader, loss_function, optimizer, device, epoch_num, num_epochs):
    model.train()
    running_loss = 0
    log_interval = 10  # Log every 10 batches
    
    # Create progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_num}/{num_epochs}")
    
    for batch_idx, batch in pbar:
        batch_images = batch["image"]
        batch_input_ids = batch["input_ids"].to(device)
        batch_attention_mask = batch["attention_mask"].to(device)   

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
        if (batch_idx + 1) % log_interval == 0:
            logger.info(f"Epoch {epoch_num}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                       f"Current Loss: {loss.item():.4f}, Running Avg Loss: {current_avg_loss:.4f}")

    average_loss = running_loss / len(dataloader)
    return average_loss

if __name__ == "__main__":
    datasets = load_flickr30k_dataset()
    dataloaders = create_flickr30k_dataloaders(
        datasets=datasets,
        image_size=hyperparameters["image_size"],
        batch_size=hyperparameters["batch_size"],
        tokenizer_name=hyperparameters["tokenizer_name"],
        max_caption_length=hyperparameters["max_caption_length"],
        use_all_captions=False
    )
    
    # Extract individual dataloaders
    training_dataloader = dataloaders['train']
    validation_dataloader = dataloaders['validation']
    test_dataloader = dataloaders['test']

    logger.info("Flickr30K Dataset loading completed!")

    model = Model().to(get_device())
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    loss_function = nn.CrossEntropyLoss()
    # for epoch_num in range(hyperparameters["num_epochs"]):

    for epoch_num in range(hyperparameters["num_epochs"]):
        train_loss = train(
            model=model,
            dataloader=training_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=get_device(),
            epoch_num=0,
            num_epochs=1)
    
