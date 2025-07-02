import torch
from tqdm import tqdm
from dataset import load_flickr30k_dataset, create_flickr30k_dataloaders
from train import hyperparameters
from model import Model
from utils import get_device
import torch.nn as nn

def evaluate(dataloader, model, epoch_num, num_epochs):
    model.eval()
    validation_datasize = len(dataloader.dataset)
    batch_size = len(dataloader)
    validation_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_num}/{num_epochs}")
        for batch_idx, batch in pbar:
            images = batch["image"]
            labels = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            output = model(images=images, input_ids=labels, attention_mask=attention_mask) # (batch_size, 10)
            validation_loss += output.loss.item()
    
    validation_loss /= batch_size
    print(f"Epoch {epoch_num} Validation Loss: {validation_loss:.4f}")
    return validation_loss

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

    validation_dataloader = dataloaders['validation']
    test_dataloader = dataloaders['test']

    loss_function = nn.CrossEntropyLoss()
    model = Model()
    model = model.to(get_device())

    evaluate(
        dataloader=validation_dataloader,
        model=model,
        epoch_num=0,
        num_epochs=1
    )