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
