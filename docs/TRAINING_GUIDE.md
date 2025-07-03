# Training Guide

This guide provides detailed instructions for training and evaluating the image captioning model.

## Quick Start

### 1. Basic Training
```bash
python train.py
```

This will:
- Download Flickr30K dataset automatically
- Train for 1 epoch with 500 samples
- Save model checkpoint in `saved_models/`

### 2. Full Dataset Training
Edit `train.py` and set:
```python
"training_size_limit": None,  # Use full dataset
"num_epochs": 5,              # Train for more epochs
```

Then run:
```bash
python train.py
```

### 3. Enable Evaluation
Edit `train.py` and set:
```python
"evaluation_enabled": True,
```

## Training Configuration

### Hyperparameter Tuning

Edit the `hyperparameters` dictionary in `train.py`:

```python
hyperparameters = {
    "batch_size": 8,                    # Increase if you have more GPU memory
    "num_epochs": 10,                   # More epochs for better convergence
    "learning_rate": 0.0001,            # Adjust based on training stability
    "learning_rate_decay": 0.9,         # Learning rate decay factor
    "learning_rate_decay_step": 2,      # Decay every N epochs
    "learning_rate_scheduler_enabled": True,
    "image_size": 224,                  # CLIP input size (don't change)
    "tokenizer_name": "Qwen/Qwen3-0.6B-Base",
    "max_caption_length": 128,          # Maximum tokens per caption
    "training_size_limit": None,        # None for full dataset
    "evaluation_enabled": True,         # Enable ROUGE evaluation
}
```

### Memory Optimization

For limited GPU memory:
```python
"batch_size": 2,                    # Reduce batch size
"training_size_limit": 1000,        # Limit training samples
```

For more GPU memory:
```python
"batch_size": 16,                   # Increase batch size
"training_size_limit": None,        # Use full dataset
```

## Training Process Details

### Data Flow

1. **Image Loading**: PIL images loaded from Flickr30K
2. **Image Encoding**: CLIP processes images → 512-dim features
3. **Feature Adaptation**: MLP maps 512-dim → 1024-dim
4. **Text Processing**: Captions tokenized with Qwen tokenizer
5. **Embedding Combination**: Image features + text embeddings
6. **Caption Generation**: Qwen generates next tokens
7. **Loss Calculation**: Cross-entropy loss on predictions

### Training Loop

Each epoch:
```python
for batch in training_dataloader:
    # 1. Process images through CLIP
    image_features = clip_encoder(batch_images)
    
    # 2. Adapt features with MLP
    adapted_features = mlp(image_features)
    
    # 3. Combine with text embeddings
    combined_embeddings = torch.cat([adapted_features, text_embeddings], dim=1)
    
    # 4. Generate captions with Qwen
    outputs = qwen_decoder(combined_embeddings, labels=labels)
    
    # 5. Compute loss and update
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Model Architecture Training

### Frozen Components
- **CLIP Parameters**: Vision encoder frozen to preserve pretrained features
- **Qwen Parameters**: Language model frozen to prevent catastrophic forgetting

### Trainable Components
- **MLP Adapter**: Only component that learns during training
- Maps CLIP's 512-dimensional space to Qwen's 1024-dimensional space

### Why This Works
- Leverages strong pretrained representations
- Minimal parameters to train (fast convergence)
- Prevents overfitting on small datasets
- Maintains model stability

## Evaluation

### Automatic Evaluation
Set `evaluation_enabled=True` in hyperparameters:
```python
"evaluation_enabled": True,
```

### Manual Evaluation
```bash
python eval.py
```

### Metrics Explained

#### ROUGE-L Score
- Measures longest common subsequence between generated and reference captions
- Range: 0.0 to 1.0 (higher is better)
- Typical scores: 0.3-0.6 for good models

#### Training Loss
- Cross-entropy loss during training
- Should decrease over epochs
- Typical range: 2.0-5.0 initially, converging to 1.0-2.0

## Model Checkpointing

### Automatic Saving
Models are automatically saved after each epoch:
```
saved_models/
├── model_state_dict_20241201_120000-1.pth
├── training_metadata_20241201_120000-1.json
├── model_state_dict_20241201_130000-2.pth
└── training_metadata_20241201_130000-2.json
```

### Loading Saved Models
```python
# Load model for inference
python load_model_example.py

# Or in code:
from load_model_example import load_model_from_checkpoint
model, metadata, hyperparams = load_model_from_checkpoint(
    "saved_models/model_state_dict_20241201_120000-1.pth", 
    device
)
```

## Monitoring Training

### Progress Tracking
- **Progress Bar**: Shows current batch, loss, and average loss
- **Logging**: Detailed logs in console
- **Checkpoints**: Model state saved after each epoch

### Key Metrics to Watch
1. **Training Loss**: Should decrease steadily
2. **Average Loss**: Running average should stabilize
3. **GPU Memory**: Monitor for out-of-memory errors
4. **ROUGE Score**: Should improve with training (if enabled)

### Sample Training Output
```
Training Epoch 1/5: 100%|██████████| 125/125 [02:30<00:00,  1.20s/it, Loss=2.3456, Avg Loss=2.4123]
✅ Model saved successfully!
📁 Model state dict: saved_models/model_state_dict_20241201_120000-1.pth
📊 Training loss: 2.4123
📈 Validation loss: 2.3456
🎯 ROUGE-L score: 0.3245
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce `batch_size`: `4 → 2 → 1`
- Reduce `training_size_limit`: `500 → 100`
- Close other GPU processes

#### 2. Slow Training
**Solutions**:
- Ensure GPU is being used (check logs)
- Increase `batch_size` if memory allows
- Use `training_size_limit` for faster experiments

#### 3. Poor ROUGE Scores
**Potential Causes**:
- Insufficient training epochs
- Learning rate too high/low
- Dataset size too small

**Solutions**:
- Train for more epochs
- Adjust learning rate: `0.0001 → 0.00005`
- Use full dataset (`training_size_limit=None`)

#### 4. Loss Not Decreasing
**Solutions**:
- Check learning rate (try 0.00005)
- Verify data loading (check batch contents)
- Ensure MLP is not frozen

### Performance Tips

#### For Experimentation
```python
hyperparameters = {
    "batch_size": 4,
    "num_epochs": 1,
    "training_size_limit": 100,    # Fast iteration
    "evaluation_enabled": False,   # Skip evaluation
}
```

#### For Best Results
```python
hyperparameters = {
    "batch_size": 8,
    "num_epochs": 10,
    "training_size_limit": None,   # Full dataset
    "evaluation_enabled": True,    # Monitor progress
    "learning_rate": 0.00005,      # Lower for stability
}
```

## Advanced Training

### Custom Dataset
To use your own dataset, modify `dataset.py`:
1. Implement custom dataset class
2. Follow Flickr30K format (image paths + captions)
3. Update DataLoader creation

### Model Variations
- **Larger Models**: Try `Qwen3-1.5B` or `Qwen3-7B`
- **Different Vision Encoders**: Experiment with other CLIP variants
- **Unfreezing**: Gradually unfreeze layers for fine-tuning

### Hyperparameter Sweeps
Use tools like Weights & Biases or MLflow:
```python
learning_rates = [0.0001, 0.00005, 0.00001]
batch_sizes = [4, 8, 16]

for lr in learning_rates:
    for bs in batch_sizes:
        hyperparameters["learning_rate"] = lr
        hyperparameters["batch_size"] = bs
        # Run training
```

## Next Steps

After successful training:
1. **Test Inference**: Use `inference.py` with new images
2. **Evaluate on Test Set**: Run full evaluation
3. **Deploy Model**: Create serving endpoint
4. **Fine-tune Further**: Experiment with unfreezing components