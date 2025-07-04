import logging
from contextlib import nullcontext

import torch
from torch.cuda.amp import autocast, GradScaler


START_TOKEN = 10  # After digits 0-9
END_TOKEN = 11
BLANK_TOKEN = 12


SIMPLE_MODEL_FILE = "data/simple.pth"
COMPLEX_MODEL_FILE = "data/complex.pth"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def amp_components(device, train=False):
    if device.type == "cuda" and train:
        return autocast(), GradScaler()
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext(), GradScaler(enabled=False)


def get_flickr30k_datasets():
    """
    Load and return the Flickr30k dataset splits.
    
    Returns:
        dict: Dictionary with 'train', 'validation', and 'test' dataset splits
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    
    # Load the dataset
    dataset = load_dataset("nlphuji/flickr30k")
    full_data = dataset['test']  # The dataset only has a 'test' split
    
    # Create train/validation/test splits (70%/15%/15%)
    total_indices = list(range(len(full_data)))
    
    # First split: separate out test set (15%)
    train_val_idx, test_idx = train_test_split(
        total_indices,
        test_size=0.15,
        random_state=42,
        shuffle=True
    )
    
    # Second split: separate train and validation
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.176,  # 15/(70+15) ≈ 0.176 to get 15% of total
        random_state=42,
        shuffle=True
    )
    
    return {
        'train': full_data.select(train_idx),
        'validation': full_data.select(val_idx),
        'test': full_data.select(test_idx)
    }


def flatten_captions(dataset):
    """
    Flatten a Flickr30k dataset to have one row per caption instead of per image.
    
    Args:
        dataset: A Flickr30k dataset split
        
    Returns:
        list: List of (image, caption) tuples
    """
    flattened = []
    for example in dataset:
        image = example['image']
        for caption in example['caption']:
            flattened.append((image, caption))
    return flattened


def sample_batch(dataset, batch_size=8):
    """
    Sample a batch of examples from the dataset.
    
    Args:
        dataset: A Flickr30k dataset split
        batch_size (int): Number of examples to sample
        
    Returns:
        list: List of sampled examples
    """
    import random
    indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))
    return [dataset[i] for i in indices]


def get_cache_info():
    """
    Get information about the Hugging Face datasets cache.
    
    Returns:
        dict: Cache information including size and location
    """
    import os
    from datasets import config
    
    cache_dir = config.HF_DATASETS_CACHE
    cache_info = {
        'cache_directory': cache_dir,
        'exists': os.path.exists(cache_dir)
    }
    
    if cache_info['exists']:
        # Get cache size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        cache_info['total_size_gb'] = total_size / (1024**3)
        cache_info['flickr30k_cached'] = os.path.exists(
            os.path.join(cache_dir, 'nlphuji___flickr30k')
        )
    
    return cache_info


def clear_dataset_cache(dataset_name=None):
    """
    Clear the datasets cache (use with caution).
    
    Args:
        dataset_name (str, optional): Specific dataset to clear, or None for all
    """
    from datasets import config
    import shutil
    import os
    
    if dataset_name:
        # Clear specific dataset
        cache_dir = config.HF_DATASETS_CACHE
        dataset_cache = os.path.join(cache_dir, dataset_name)
        if os.path.exists(dataset_cache):
            shutil.rmtree(dataset_cache)
            print(f"Cleared cache for {dataset_name}")
        else:
            print(f"No cache found for {dataset_name}")
    else:
        # Clear all datasets cache
        print("Warning: This will clear ALL cached datasets!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == 'yes':
            cache_dir = config.HF_DATASETS_CACHE
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print("All dataset cache cleared")
            else:
                print("No cache directory found")


def verify_flickr30k_cache():
    """
    Verify that Flickr30k is properly cached and show cache statistics.
    """
    cache_info = get_cache_info()
    
    print("📁 Cache Information:")
    print(f"  Location: {cache_info['cache_directory']}")
    print(f"  Exists: {cache_info['exists']}")
    
    if cache_info['exists']:
        print(f"  Total Size: {cache_info['total_size_gb']:.2f} GB")
        print(f"  Flickr30k Cached: {'✅ Yes' if cache_info['flickr30k_cached'] else '❌ No'}")
    
    return cache_info


def load_saved_model(model_path, model_class=None):
    """
    Load a saved model in HuggingFace format.
    
    Args:
        model_path (str): Path to the saved model directory (HuggingFace format)
        model_class (class, optional): Model class to instantiate. If not provided,
                                     will return only the checkpoint data.
    
    Returns:
        dict: Dictionary containing:
            - 'model': Instantiated model with loaded weights (if model_class provided)
            - 'optimizer_state_dict': Saved optimizer state (if available)
            - 'training_metadata': Training metadata
            - 'model_path': Path to the model directory
    """
    import json
    import os
    from model import VisionLanguageModel
    from transformers import AutoTokenizer
    
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # Check if it's a HuggingFace format directory
    config_file = os.path.join(model_path, "config.json")
    model_file = os.path.join(model_path, "pytorch_model.bin")
    tokenizer_config = os.path.join(model_path, "tokenizer_config.json")
    
    if os.path.exists(config_file) and os.path.exists(tokenizer_config):
        # New HuggingFace format
        logger.info(f"🔄 Loading HuggingFace format model from: {model_path}")
        
        # Load the model using the new format
        try:
            model = VisionLanguageModel.from_pretrained(model_path)
            logger.info("✅ Model loaded successfully using HuggingFace format")
            
            # Load additional metadata if available
            metadata_file = os.path.join(model_path, "training_metadata.json")
            optimizer_file = os.path.join(model_path, "optimizer.pth")
            
            training_metadata = None
            optimizer_state_dict = None
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    training_metadata = json.load(f)
                logger.info("✅ Training metadata loaded")
            
            if os.path.exists(optimizer_file):
                optimizer_data = torch.load(optimizer_file, map_location=get_device())
                optimizer_state_dict = optimizer_data.get('optimizer_state_dict')
                logger.info("✅ Optimizer state loaded")
            
            return {
                'model': model,
                'optimizer_state_dict': optimizer_state_dict,
                'training_metadata': training_metadata,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load HuggingFace format model: {e}")
            raise
    
    else:
        # Legacy format (old state dict format)
        logger.info(f"🔄 Loading legacy format model from: {model_path}")
        
        # Check if it's a .pth file
        if model_path.endswith('.pth'):
            return load_legacy_model(model_path, model_class)
        else:
            raise ValueError(f"Unsupported model format in directory: {model_path}")


def load_legacy_model(model_path, model_class=None):
    """
    Load a model saved in the old state dict format.
    
    Args:
        model_path (str): Path to the saved model file (.pth)
        model_class (class, optional): Model class to instantiate
    
    Returns:
        dict: Dictionary containing model and metadata
    """
    import json
    import os
    
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    
    result = {
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        'hyperparameters': checkpoint.get('hyperparameters'),
        'training_history': checkpoint.get('training_history'),
        'epoch': checkpoint.get('epoch'),
        'checkpoint': checkpoint
    }
    
    # If model class is provided, instantiate and load the model
    if model_class is not None:
        # Check if we have a saved tokenizer directory
        tokenizer_dir = checkpoint.get('tokenizer_dir')
        
        if tokenizer_dir and os.path.exists(tokenizer_dir):
            logger.info(f"🔄 Loading saved tokenizer from: {tokenizer_dir}")
            
            # Create model with saved tokenizer (BEST APPROACH!)
            try:
                from transformers import AutoTokenizer
                saved_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
                
                # Create model with the saved tokenizer
                model = model_class(tokenizer_name=None, saved_tokenizer=saved_tokenizer)
                
                logger.info(f"✅ Using saved tokenizer:")
                logger.info(f"   - Vocab size: {len(saved_tokenizer)}")
                logger.info(f"   - BOS token: '{saved_tokenizer.bos_token}' (ID: {saved_tokenizer.bos_token_id})")
                
            except Exception as e:
                logger.error(f"❌ Failed to load saved tokenizer: {e}")
                logger.warning("🔄 Falling back to fresh tokenizer creation...")
                model = model_class()  # Fallback to default
            
        else:
            # Fallback: Create new model (for backward compatibility)
            logger.warning("⚠️ No saved tokenizer found. Creating fresh tokenizer.")
            logger.warning("   This may cause tokenizer inconsistencies!")
            
            model = model_class()
            
            # Check for tokenizer size mismatch
            saved_vocab_size = checkpoint.get('tokenizer_vocab_size')
            current_vocab_size = len(model.tokenizer)
            
            if saved_vocab_size and saved_vocab_size != current_vocab_size:
                logger.warning(f"⚠️ Tokenizer size mismatch detected!")
                logger.warning(f"   Saved model vocab size: {saved_vocab_size}")
                logger.warning(f"   Current tokenizer size: {current_vocab_size}")
                logger.warning(f"   Attempting to resize embeddings...")
                
                # Resize the model's embeddings to match the saved size
                model.qwen_decoder.qwen_model.model.resize_token_embeddings(saved_vocab_size)
                logger.info(f"✅ Resized embeddings to {saved_vocab_size}")
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Model state dict loaded successfully")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.error(f"❌ Critical embedding size mismatch: {e}")
                logger.error("This indicates a tokenizer compatibility issue!")
                raise RuntimeError(f"Tokenizer size mismatch prevents model loading: {e}")
            else:
                raise e
        
        model.to(device)
        result['model'] = model
    
    return result


def list_saved_models(save_dir="saved_models"):
    """
    List all saved models in the specified directory (both HuggingFace and legacy formats).
    
    Args:
        save_dir (str): Directory containing saved models
        
    Returns:
        list: List of dictionaries with model information
    """
    import os
    import json
    from datetime import datetime
    
    if not os.path.exists(save_dir):
        print(f"Save directory '{save_dir}' does not exist.")
        return []
    
    models = []
    
    # Look for both HuggingFace format directories and legacy .pth files
    for filename in os.listdir(save_dir):
        full_path = os.path.join(save_dir, filename)
        
        if os.path.isdir(full_path):
            # Check if it's a HuggingFace format directory
            config_file = os.path.join(full_path, "config.json")
            tokenizer_config = os.path.join(full_path, "tokenizer_config.json")
            
            if os.path.exists(config_file) and os.path.exists(tokenizer_config):
                # HuggingFace format
                metadata_file = os.path.join(full_path, "training_metadata.json")
                metadata = None
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"Error reading metadata for {filename}: {e}")
                
                # Get directory size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(full_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                
                models.append({
                    'filename': filename,
                    'path': full_path,
                    'type': 'huggingface',
                    'timestamp': filename.split('_')[-1] if '_' in filename else filename,
                    'size_mb': total_size / (1024 * 1024),
                    'metadata': metadata
                })
        
        elif filename.startswith("model_state_dict_") and filename.endswith(".pth"):
            # Legacy format
            model_path = os.path.join(save_dir, filename)
            
            # Extract timestamp from filename
            timestamp_str = filename.replace("model_state_dict_", "").replace(".pth", "")
            
            # Look for corresponding metadata file
            metadata_file = os.path.join(save_dir, f"training_metadata_{timestamp_str}.json")
            metadata = None
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"Error reading metadata for {filename}: {e}")
            
            # Get file size
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            models.append({
                'filename': filename,
                'path': model_path,
                'type': 'legacy',
                'timestamp': timestamp_str,
                'size_mb': file_size_mb,
                'metadata': metadata
            })
    
    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return models


def print_model_summary(save_dir="saved_models"):
    """
    Print a summary of all saved models (both HuggingFace and legacy formats).
    
    Args:
        save_dir (str): Directory containing saved models
    """
    models = list_saved_models(save_dir)
    
    if not models:
        print(f"No saved models found in '{save_dir}'")
        return
    
    print(f"\n📊 Found {len(models)} saved model(s) in '{save_dir}':")
    print("-" * 80)
    
    for i, model_info in enumerate(models, 1):
        print(f"{i}. {model_info['filename']}")
        print(f"   📁 Size: {model_info['size_mb']:.1f} MB")
        print(f"   📅 Timestamp: {model_info['timestamp']}")
        print(f"   🔧 Format: {model_info['type'].upper()}")
        
        if model_info['metadata']:
            metadata = model_info['metadata']
            hyperparams = metadata.get('hyperparameters', {})
            history = metadata.get('training_history', {})
            
            print(f"   ⚙️  Epochs: {hyperparams.get('num_epochs', 'N/A')}")
            print(f"   📈 Batch Size: {hyperparams.get('batch_size', 'N/A')}")
            print(f"   🎯 Learning Rate: {hyperparams.get('learning_rate', 'N/A')}")
            
            if history.get('training_losses'):
                final_train_loss = history['training_losses'][-1]
                print(f"   📉 Final Training Loss: {final_train_loss:.4f}")
            
            if history.get('validation_losses'):
                final_val_loss = history['validation_losses'][-1]
                print(f"   📊 Final Validation Loss: {final_val_loss:.4f}")
                
        print("-" * 40)
