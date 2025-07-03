# API Documentation

## Model Module (`model.py`)

### ClipEncoder

```python
class ClipEncoder(nn.Module):
    def __init__(self, freeze_clip=True)
```

**Purpose**: Encodes images using OpenAI's CLIP model to extract visual features.

**Parameters**:
- `freeze_clip` (bool): Whether to freeze CLIP parameters during training (default: True)

**Input**: List of PIL Images
**Output**: Tensor of shape `(batch_size, 512)` containing image embeddings

**Methods**:
- `forward(images)`: Processes images and returns feature embeddings

### QwenDecoder

```python
class QwenDecoder(nn.Module):
    def __init__(self, freeze_qwen=True)
```

**Purpose**: Generates text captions using Qwen language model.

**Parameters**:
- `freeze_qwen` (bool): Whether to freeze Qwen parameters during training (default: True)

**Input**: 
- `input_embeddings`: Combined image and text embeddings
- `labels`: Target token labels for loss calculation

**Output**: `CausalLMOutputWithPast` containing:
- `loss`: Scalar cross-entropy loss
- `logits`: Tensor of shape `(batch_size, seq_len, vocab_size)`

### Model

```python
class Model(nn.Module):
    def __init__(self, tokenizer_name="Qwen/Qwen3-0.6B-Base")
```

**Purpose**: Main model combining CLIP encoder, MLP adapter, and Qwen decoder.

**Parameters**:
- `tokenizer_name` (str): Hugging Face model name for tokenizer

**Architecture**:
- CLIP encoder (frozen)
- Linear layer: 512 → 1024 dimensions
- Qwen decoder (frozen)

**Forward Pass**:
```python
def forward(self, images, input_ids, attention_mask)
```

**Input**:
- `images`: List of PIL Images
- `input_ids`: Tokenized caption IDs
- `attention_mask`: Attention mask for input tokens

**Output**: Qwen model output with loss and logits

## Training Module (`train.py`)

### Hyperparameters

```python
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
    "training_size_limit": 500,
    "evaluation_enabled": False,
}
```

### Functions

#### save_model

```python
def save_model(model, optimizer, epoch_num, loss, score, training_size, save_dir="saved_models")
```

**Purpose**: Saves model checkpoint with metadata.

**Parameters**:
- `model`: PyTorch model to save
- `optimizer`: Optimizer state
- `epoch_num`: Current epoch number
- `loss`: Training loss
- `score`: Validation score
- `training_size`: Size of training dataset
- `save_dir`: Directory to save files

**Outputs**:
- Model state dictionary (`.pth`)
- Training metadata (`.json`)

#### train

```python
def train(model, training_dataloader, validation_dataloader, optimizer, device, epoch_num, num_epochs, timestamp)
```

**Purpose**: Executes one training epoch.

**Returns**: 
- `training_loss`: Average loss for the epoch
- `validation_loss`: Validation loss (if enabled)
- `validation_score`: ROUGE-L score (if enabled)
- `training_size`: Number of training batches

## Dataset Module (`dataset.py`)

### Functions

#### load_flickr30k_dataset

```python
def load_flickr30k_dataset()
```

**Purpose**: Downloads and loads Flickr30K dataset from Hugging Face.

**Returns**: Dictionary with train/validation/test splits

#### create_flickr30k_dataloaders

```python
def create_flickr30k_dataloaders(datasets, image_size, batch_size, tokenizer_name, max_caption_length, use_all_captions=False, train_size_limit=None)
```

**Purpose**: Creates PyTorch DataLoaders with preprocessing.

**Parameters**:
- `datasets`: Dataset splits from load_flickr30k_dataset()
- `image_size`: Target image size for resizing
- `batch_size`: Batch size for training
- `tokenizer_name`: Tokenizer for caption processing
- `max_caption_length`: Maximum tokens per caption
- `use_all_captions`: Whether to use all 5 captions per image
- `train_size_limit`: Limit training set size for experimentation

**Returns**: Dictionary with DataLoader objects for train/validation/test

## Evaluation Module (`eval.py`)

### evaluate

```python
def evaluate(dataloader, model, epoch_num, num_epochs)
```

**Purpose**: Evaluates model performance using ROUGE metrics.

**Parameters**:
- `dataloader`: Validation/test DataLoader
- `model`: Trained model to evaluate
- `epoch_num`: Current epoch for logging
- `num_epochs`: Total epochs for logging

**Returns**:
- `validation_loss`: Average validation loss
- `rouge_l_score`: ROUGE-L F1 score

**Process**:
1. Generates captions for validation images
2. Compares with ground truth using ROUGE-L
3. Calculates average loss and scores

## Inference Module (`inference.py`)

### generate_caption

```python
def generate_caption(model, image_path, max_length=50)
```

**Purpose**: Generates caption for a single image.

**Parameters**:
- `model`: Trained captioning model
- `image_path`: Path to input image
- `max_length`: Maximum caption length

**Returns**: Generated caption string

**Process**:
1. Loads and preprocesses image
2. Encodes image with CLIP
3. Generates caption with Qwen decoder
4. Decodes tokens to text

## Utilities Module (`utils.py`)

### Device Management

```python
def get_device()
```

**Purpose**: Automatically selects best available device (CUDA/CPU).

**Returns**: torch.device object

### Logging

```python
def setup_logging()
```

**Purpose**: Configures logging format and level.

**Features**:
- Timestamp formatting
- Multiple log levels
- Console and file output

### Data Processing

```python
def collate_fn(batch)
```

**Purpose**: Custom collation function for DataLoader.

**Handles**:
- Variable-length captions
- Image preprocessing
- Batch padding

## Model Loading (`load_model_example.py`)

### load_model_from_checkpoint

```python
def load_model_from_checkpoint(checkpoint_path, device)
```

**Purpose**: Loads saved model checkpoint.

**Parameters**:
- `checkpoint_path`: Path to saved model file
- `device`: Target device for model

**Returns**: 
- Loaded model
- Training metadata
- Hyperparameters

**Usage Example**:
```python
model, metadata, hyperparams = load_model_from_checkpoint("saved_models/model_state_dict_20241201_120000-1.pth", device)
```