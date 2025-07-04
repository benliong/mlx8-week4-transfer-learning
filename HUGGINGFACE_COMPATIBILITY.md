# HuggingFace Model Compatibility Guide

## Overview

Your custom Vision-Language model has been updated to be fully compatible with HuggingFace Transformers library. This means you can now:

- ✅ Save and load models using `save_pretrained()` and `from_pretrained()`
- ✅ Use `AutoModel` and `AutoTokenizer` for loading
- ✅ Share models on HuggingFace Hub
- ✅ Use standard HuggingFace tools and workflows

## What Changed

### 1. New Model Architecture

**Before:**
```python
from model import Model

# Old way
model = Model(tokenizer_name="Qwen/Qwen3-0.6B-Base")
```

**After:**
```python
from model import VisionLanguageModel
from configuration import VisionLanguageConfig

# New way with configuration
config = VisionLanguageConfig(
    clip_model_name="openai/clip-vit-base-patch32",
    qwen_model_name="Qwen/Qwen3-0.6B-Base",
    tokenizer_name="Qwen/Qwen3-0.6B-Base",
    max_caption_length=128,
    image_size=224,
)

model = VisionLanguageModel(config)
```

### 2. New Saving Format

**Before:**
```python
# Old way - saved as state dict
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # ...
}, 'model.pth')
```

**After:**
```python
# New way - HuggingFace format
model.save_pretrained("saved_models/my_model")
```

### 3. New Loading Format

**Before:**
```python
# Old way - manual loading
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

**After:**
```python
# New way - HuggingFace format
model = VisionLanguageModel.from_pretrained("saved_models/my_model")

# Or using AutoModel (after registration)
from transformers import AutoModel
model = AutoModel.from_pretrained("saved_models/my_model", trust_remote_code=True)
```

## File Structure

When you save a model using the new format, it creates a directory with these files:

```
saved_models/my_model/
├── config.json                 # Model configuration
├── pytorch_model.bin          # Model weights
├── tokenizer.json            # Tokenizer vocabulary and rules
├── tokenizer_config.json     # Tokenizer configuration
├── special_tokens_map.json   # Special tokens (BOS, EOS, etc.)
├── training_metadata.json    # Training history and hyperparameters
└── optimizer.pth            # Optimizer state (optional)
```

## Usage Examples

### 1. Training and Saving

```python
from model import VisionLanguageModel
from configuration import VisionLanguageConfig

# Create model with configuration
config = VisionLanguageConfig(
    clip_model_name="openai/clip-vit-base-patch32",
    qwen_model_name="Qwen/Qwen3-0.6B-Base",
    tokenizer_name="Qwen/Qwen3-0.6B-Base",
    max_caption_length=128,
    image_size=224,
)

model = VisionLanguageModel(config)

# Train your model...
# ...

# Save in HuggingFace format
model.save_pretrained("saved_models/my_vision_model")
```

### 2. Loading for Inference

```python
from model import VisionLanguageModel

# Load the model
model = VisionLanguageModel.from_pretrained("saved_models/my_vision_model")

# Use for inference
model.eval()
with torch.no_grad():
    outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
```

### 3. Using with AutoModel and AutoTokenizer

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig
from model import VisionLanguageModel
from configuration import VisionLanguageConfig

# Register your custom model (do this once)
AutoConfig.register("vision_language_model", VisionLanguageConfig)
AutoModel.register(VisionLanguageConfig, VisionLanguageModel)

# Load using AutoModel
model = AutoModel.from_pretrained("saved_models/my_vision_model", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("saved_models/my_vision_model")

# Use normally
inputs = tokenizer("A beautiful sunset", return_tensors="pt")
outputs = model(**inputs)
```

### 4. Sharing on HuggingFace Hub

```python
from huggingface_hub import login

# Login to HuggingFace
login()

# Push to Hub
model.push_to_hub("your-username/vision-language-model")

# Others can then load it
model = VisionLanguageModel.from_pretrained("your-username/vision-language-model")
```

## Backward Compatibility

The old `Model` class is still available for backward compatibility:

```python
from model import Model

# Legacy usage still works
model = Model(tokenizer_name="Qwen/Qwen3-0.6B-Base")

# But now it also supports HuggingFace saving
model.save_pretrained("saved_models/legacy_model")
```

## Migration Guide

### From Old Format to New Format

If you have models saved in the old format, you can convert them:

```python
from utils import load_saved_model
from model import Model

# Load old format
old_model_data = load_saved_model("old_model.pth", Model)
model = old_model_data['model']

# Save in new format
model.save_pretrained("saved_models/converted_model")
```

### Update Your Training Code

Replace your old saving code:

```python
# Old saving code
def save_model(model, optimizer, epoch, loss, save_dir):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # ...
    }, f"{save_dir}/model_{epoch}.pth")
```

With the new HuggingFace format:

```python
# New saving code
def save_model(model, optimizer, epoch, loss, save_dir):
    model_path = f"{save_dir}/model_epoch_{epoch}"
    model.save_pretrained(model_path)
    
    # Save optimizer separately if needed
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, f"{model_path}/optimizer.pth")
```

## Configuration Options

The `VisionLanguageConfig` class supports these parameters:

```python
config = VisionLanguageConfig(
    # Model architecture
    clip_model_name="openai/clip-vit-base-patch32",
    qwen_model_name="Qwen/Qwen3-0.6B-Base",
    image_embedding_dim=512,
    text_embedding_dim=1024,
    freeze_clip=True,
    freeze_qwen=True,
    
    # Training parameters
    max_caption_length=128,
    image_size=224,
    
    # Tokenizer parameters
    tokenizer_name="Qwen/Qwen3-0.6B-Base",
    bos_token="<|im_start|>",
    eos_token="<|im_end|>",
    pad_token="<|endoftext|>",
    unk_token="<|endoftext|>",
)
```

## Testing Your Setup

Run the example script to test everything:

```bash
python load_model_example.py
```

This will:
1. Create a model in HuggingFace format
2. Save it properly
3. Load it back using various methods
4. Verify compatibility with AutoModel and AutoTokenizer

## Benefits of HuggingFace Compatibility

1. **Standardization**: Your model follows industry-standard format
2. **Interoperability**: Works with all HuggingFace tools and libraries
3. **Sharing**: Easy to share models on HuggingFace Hub
4. **Documentation**: Automatic model cards and documentation
5. **Deployment**: Compatible with HuggingFace Inference API
6. **Community**: Access to the broader HuggingFace ecosystem

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you have the right imports:
   ```python
   from model import VisionLanguageModel
   from configuration import VisionLanguageConfig
   ```

2. **Loading Errors**: When using AutoModel, you might need:
   ```python
   model = AutoModel.from_pretrained("path", trust_remote_code=True)
   ```

3. **Tokenizer Issues**: Ensure the tokenizer is properly saved:
   ```python
   # This should create all necessary tokenizer files
   model.save_pretrained("save_path")
   ```

### Getting Help

If you encounter issues:
1. Check the file structure of your saved model
2. Verify all required files are present
3. Use the example script to test functionality
4. Check the logs for detailed error messages

## Summary

Your model is now fully compatible with HuggingFace Transformers! You can:

- ✅ Save with `model.save_pretrained()`
- ✅ Load with `VisionLanguageModel.from_pretrained()`
- ✅ Use with `AutoModel` and `AutoTokenizer`
- ✅ Share on HuggingFace Hub
- ✅ Maintain backward compatibility with old code

The new format provides better standardization, easier sharing, and full compatibility with the HuggingFace ecosystem.