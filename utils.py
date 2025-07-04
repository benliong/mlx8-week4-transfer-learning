import logging
from contextlib import nullcontext

import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer


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
    Load a saved model state dict and associated metadata.
    
    Args:
        model_path (str): Path to the saved model file (.pth) or folder containing model files
        model_class (class, optional): Model class to instantiate. If not provided,
                                     will return only the checkpoint data.
    
    Returns:
        dict: Dictionary containing:
            - 'model': Instantiated model with loaded weights (if model_class provided)
            - 'optimizer_state_dict': Saved optimizer state
            - 'hyperparameters': Training hyperparameters
            - 'training_history': Training metrics history
            - 'epoch': Number of epochs trained
            - 'checkpoint': Raw checkpoint data
            - 'tokenizer': Loaded tokenizer (if available)
            - 'model_folder': Path to the model folder
    """
    import json
    import os
    
    # Handle both folder and file paths
    if os.path.isdir(model_path):
        # New folder structure
        model_folder = model_path
        actual_model_path = os.path.join(model_folder, "model_state_dict.pth")
        tokenizer_path = os.path.join(model_folder, "tokenizer")
        metadata_path = os.path.join(model_folder, "training_metadata.json")
    else:
        # Old file structure - try to find corresponding files
        model_folder = os.path.dirname(model_path)
        actual_model_path = model_path
        tokenizer_path = None
        metadata_path = None
        
        # Try to find metadata file with similar naming
        base_name = os.path.basename(model_path)
        if base_name.startswith("model_state_dict_"):
            timestamp_part = base_name.replace("model_state_dict_", "").replace(".pth", "")
            metadata_path = os.path.join(model_folder, f"training_metadata_{timestamp_part}.json")
    
    if not os.path.exists(actual_model_path):
        raise FileNotFoundError(f"Model file not found: {actual_model_path}")
    
    # Load the checkpoint
    device = get_device()
    checkpoint = torch.load(actual_model_path, map_location=device)
    
    result = {
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        'hyperparameters': checkpoint.get('hyperparameters'),
        'training_history': checkpoint.get('training_history'),
        'epoch': checkpoint.get('epoch'),
        'checkpoint': checkpoint,
        'model_folder': model_folder
    }
    
    # Load tokenizer if available
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            result['tokenizer'] = tokenizer
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
            result['tokenizer'] = None
    else:
        result['tokenizer'] = None
    
    # Load additional metadata from JSON if available
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # Add any additional metadata that might not be in the checkpoint
                for key, value in metadata.items():
                    if key not in result:
                        result[key] = value
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_path}: {e}")
    
    # If model class is provided, instantiate and load the model
    if model_class is not None:
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        result['model'] = model
        
        # If we loaded a tokenizer, assign it to the model
        if result['tokenizer'] is not None:
            model.tokenizer = result['tokenizer']
    
    return result


def list_saved_models(save_dir="saved_models"):
    """
    List all saved models in the specified directory.
    Handles both old file-based structure and new folder-based structure.
    
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
    
    # Look for both old file structure and new folder structure
    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        
        # Check if it's a folder (new structure)
        if os.path.isdir(item_path):
            model_file = os.path.join(item_path, "model_state_dict.pth")
            metadata_file = os.path.join(item_path, "training_metadata.json")
            
            if os.path.exists(model_file):
                # Extract timestamp from folder name
                timestamp_str = item
                
                # Load metadata
                metadata = None
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"Error reading metadata for {item}: {e}")
                
                # Get total folder size
                total_size_bytes = 0
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size_bytes += os.path.getsize(file_path)
                
                models.append({
                    'filename': item,
                    'path': item_path,
                    'is_folder': True,
                    'timestamp': timestamp_str,
                    'size_mb': total_size_bytes / (1024 * 1024),
                    'metadata': metadata
                })
        
        # Check if it's a model file (old structure)
        elif item.startswith("model_state_dict_") and item.endswith(".pth"):
            model_path = os.path.join(save_dir, item)
            
            # Extract timestamp from filename
            timestamp_str = item.replace("model_state_dict_", "").replace(".pth", "")
            
            # Look for corresponding metadata file
            metadata_file = os.path.join(save_dir, f"training_metadata_{timestamp_str}.json")
            metadata = None
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"Error reading metadata for {item}: {e}")
            
            # Get file size
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            models.append({
                'filename': item,
                'path': model_path,
                'is_folder': False,
                'timestamp': timestamp_str,
                'size_mb': file_size_mb,
                'metadata': metadata
            })
    
    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return models


def print_model_summary(save_dir="saved_models"):
    """
    Print a summary of all saved models.
    
    Args:
        save_dir (str): Directory containing saved models
    """
    import os
    
    models = list_saved_models(save_dir)
    
    if not models:
        print(f"No saved models found in '{save_dir}'")
        return
    
    print(f"\n📊 Found {len(models)} saved model(s) in '{save_dir}':")
    print("-" * 80)
    
    for i, model_info in enumerate(models, 1):
        print(f"{i}. {model_info['filename']}")
        
        # Show different info based on structure type
        if model_info.get('is_folder', False):
            print(f"   📁 Type: Folder (New Structure)")
            print(f"   📦 Total Size: {model_info['size_mb']:.1f} MB")
        else:
            print(f"   � Type: File (Legacy Structure)")
            print(f"   📁 Size: {model_info['size_mb']:.1f} MB")
        
        print(f"   �� Timestamp: {model_info['timestamp']}")
        
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
            
            if history.get('validation_losses') and history['validation_losses'][-1] is not None:
                final_val_loss = history['validation_losses'][-1]
                print(f"   📊 Final Validation Loss: {final_val_loss:.4f}")
        
        # Show additional info for folder structure
        if model_info.get('is_folder', False):
            folder_path = model_info['path']
            files_in_folder = []
            try:
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        files_in_folder.append(item)
                    elif os.path.isdir(item_path):
                        files_in_folder.append(f"{item}/")
                
                print(f"   📋 Contents: {', '.join(files_in_folder)}")
            except:
                pass
                
        print("-" * 40)
