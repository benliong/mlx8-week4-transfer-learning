# MLX Week 4: Image Captioning with Transfer Learning

This project implements an image captioning model using transfer learning, combining CLIP (Contrastive Language-Image Pre-training) for image encoding and Qwen for text generation.

## Overview

The model architecture consists of:
- **CLIP Encoder**: Uses OpenAI's CLIP ViT-Base-Patch32 to encode images into 512-dimensional feature vectors
- **MLP Adapter**: Maps CLIP's 512-dimensional features to Qwen's 1024-dimensional embedding space
- **Qwen Decoder**: Uses Qwen3-0.6B-Base model for caption generation

## Features

- Image captioning using state-of-the-art vision-language models
- Transfer learning approach with frozen pretrained models
- Support for Flickr30K dataset
- Model evaluation with ROUGE scores
- Flexible training configuration
- Model saving and loading capabilities

## Installation

### Prerequisites
- Python 3.13+
- CUDA-compatible GPU (recommended)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd mlx8-w4-transferlearning
```

2. Install dependencies:
```bash
pip install -e .
```

## Dataset

The project uses the Flickr30K dataset, which contains:
- 31,783 images
- 158,915 captions (5 per image)
- Training/validation/test splits

The dataset is automatically downloaded and processed when running training scripts.

## Usage

### Training

Train the model with default hyperparameters:
```bash
python train.py
```

### Key Hyperparameters
- `batch_size`: 4 (default)
- `num_epochs`: 1 (default)
- `learning_rate`: 0.0001
- `max_caption_length`: 128
- `training_size_limit`: 500 (set to None for full dataset)

### Evaluation

Evaluate a trained model:
```bash
python eval.py
```

### Inference

Generate captions for new images:
```bash
python inference.py
```

### Loading Pre-trained Models

Example of loading a saved model:
```bash
python load_model_example.py
```

## Model Architecture

### CLIP Encoder (`ClipEncoder`)
- Input: PIL Images
- Output: 512-dimensional feature vectors
- Pretrained model: `openai/clip-vit-base-patch32`
- Parameters are frozen during training

### MLP Adapter
- Maps CLIP features (512-dim) to Qwen embedding space (1024-dim)
- Single linear layer: `nn.Linear(512, 1024)`
- Only trainable component in the pipeline

### Qwen Decoder (`QwenDecoder`)
- Input: Combined image and text embeddings
- Output: Generated captions
- Pretrained model: `Qwen/Qwen3-0.6B-Base`
- Parameters are frozen during training

## File Structure

```
├── model.py           # Model architecture definitions
├── train.py           # Training script and hyperparameters
├── eval.py            # Evaluation with ROUGE metrics
├── inference.py       # Single image caption generation
├── dataset.py         # Flickr30K dataset loading and preprocessing
├── utils.py           # Utility functions (logging, device management)
├── load_model_example.py  # Example of loading saved models
├── pyproject.toml     # Project dependencies
└── saved_models/      # Directory for saved model checkpoints
```

## Training Process

1. **Image Processing**: Images are processed through CLIP's image processor
2. **Feature Extraction**: CLIP generates 512-dimensional image features
3. **Feature Adaptation**: MLP maps features to Qwen's embedding space
4. **Caption Generation**: Features are concatenated with text embeddings
5. **Loss Calculation**: Cross-entropy loss on caption prediction

## Model Outputs

The model saves:
- Model state dictionaries (`.pth` files)
- Training metadata (JSON files)
- Hyperparameters and training history
- Loss and evaluation scores

## Evaluation Metrics

- **ROUGE-L**: Measures longest common subsequence overlap
- **Training Loss**: Cross-entropy loss during training
- **Validation Loss**: Loss on held-out validation set

## GPU Support

The project automatically detects and uses CUDA GPUs when available. Set device preferences in `utils.py`.

## Logging

Comprehensive logging is implemented throughout:
- Training progress and metrics
- Model loading/saving operations
- Device information
- Error handling

## Notes

- The model uses mixed precision (bfloat16) for efficiency
- CLIP and Qwen parameters are frozen to prevent catastrophic forgetting
- Only the MLP adapter layer is trained
- Supports both single caption and multiple caption training modes

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size in hyperparameters
2. **Tokenizer warnings**: Set `TOKENIZERS_PARALLELISM=false` (handled automatically)
3. **Missing dataset**: Ensure internet connection for automatic download

### Performance Tips
- Use GPU for faster training
- Adjust `training_size_limit` for faster experimentation
- Enable evaluation only when needed (set `evaluation_enabled=True`)

## License

This project is for educational purposes as part of MLX Week 4 assignment.