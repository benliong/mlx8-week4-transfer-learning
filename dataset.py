import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
import random
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from utils import setup_logging
from torch.utils.data import DataLoader
import logging
from torch.utils.data.dataloader import default_collate

logger = logging.getLogger("[Dataset]")

class Flickr30kDataset(Dataset):
    """
    Custom PyTorch Dataset for Flickr30k that handles image-caption pairs.
    
    This dataset:
    - Converts PIL images to tensors with proper preprocessing
    - Tokenizes captions for language model input
    - Handles multiple captions per image
    """
    
    def __init__(self, 
                 huggingface_dataset, 
                 tokenizer_name="Qwen/Qwen3-0.6B-Base", 
                 image_size=224, 
                 max_caption_length=128, 
                 use_random_caption=True):
        """
        Args:
            huggingface_dataset: The Hugging Face dataset split (train/val/test)
            tokenizer_name: Name of the tokenizer to use
            image_size: Size to resize images to
            max_caption_length: Maximum length for tokenized captions
            use_random_caption: If True, randomly select one caption per image
        """
        self.dataset = huggingface_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.use_random_caption = use_random_caption
        self.max_caption_length = max_caption_length
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Image preprocessing pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns a single (image, caption) pair.
        
        Returns:
            dict: {
                'image': torch.Tensor [3, H, W] - preprocessed image
                'caption': str - raw caption text
                'input_ids': torch.Tensor - tokenized caption
                'attention_mask': torch.Tensor - attention mask for caption
                'metadata': dict - original metadata (img_id, filename, etc.)
            }
        """
        item = self.dataset[idx]
        
        # Process image
        image = item['image']
        if isinstance(image, str):
            # If image is a path, load it
            image = Image.open(image).convert('RGB')
        
        image_tensor = self.image_transform(image)
        
        # Process caption
        captions = item['caption']
        if self.use_random_caption:
            # Randomly select one caption
            caption = random.choice(captions)
        else:
            # Use the first caption
            caption = captions[0]
        
        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_caption_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Metadata
        metadata = {
            'img_id': item['img_id'],
            'filename': item['filename'],
            'all_captions': captions,
            'selected_caption_idx': captions.index(caption) if self.use_random_caption else 0
        }
        
        return {
            'image': image,
            'caption': caption,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'metadata': metadata
        }

class Flickr30kCaptionDataset(Dataset):
    """
    Alternative dataset that creates separate samples for each caption.
    This gives you 5x more training samples (5 captions per image).
    """
    
    def __init__(self, 
                 huggingface_dataset, 
                 tokenizer_name="Qwen/Qwen3-0.6B-Base", 
                 image_size=224, 
                 max_caption_length=128):
        """
        Args:
            huggingface_dataset: The Hugging Face dataset split
            tokenizer_name: Name of the tokenizer to use
            image_size: Size to resize images to
            max_caption_length: Maximum length for tokenized captions
        """
        self.dataset = huggingface_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_caption_length = max_caption_length
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create flattened index mapping
        self.caption_indices = []
        for img_idx in range(len(self.dataset)):
            for cap_idx in range(len(self.dataset[img_idx]['caption'])):
                self.caption_indices.append((img_idx, cap_idx))
    
    def __len__(self):
        return len(self.caption_indices)
    
    def __getitem__(self, idx):
        """
        Returns a single (image, caption) pair from the flattened dataset.
        """
        img_idx, cap_idx = self.caption_indices[idx]
        item = self.dataset[img_idx]
        
        # Process image
        image = item['image']
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        image_tensor = self.image_transform(image)
        
        # Get specific caption
        caption = item['caption'][cap_idx]
        
        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_caption_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Metadata
        metadata = {
            'img_id': item['img_id'],
            'filename': item['filename'],
            'caption_idx': cap_idx,
            'total_captions': len(item['caption'])
        }
        
        return {
            'image': image_tensor,
            'caption': caption,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'metadata': metadata
        }

def load_flickr30k_dataset():
    """
    Load the Flickr30k dataset and split into train/validation/test sets.
    
    Returns:
        dict: Dictionary containing train, validation, and test datasets
    """
    logger.info("Loading Flickr30k dataset...")
    
    # Load the full Flickr30k dataset
    dataset = load_dataset("nlphuji/flickr30k")
    
    # Check what splits are available
    logger.info(f"Available splits: {list(dataset.keys())}")
    
    # This dataset only has 'test' split, so we'll use it as our full dataset
    # and create our own train/val/test splits
    if 'test' in dataset:
        full_data = dataset['test']
    else:
        # Fallback - use the first available split
        split_name = list(dataset.keys())[0]
        full_data = dataset[split_name]
        logger.info(f"Using split '{split_name}' as full dataset")
    
    logger.info(f"Total dataset size: {len(full_data)}")
    
    # Create train/validation/test splits (70%/15%/15%)
    total_indices = list(range(len(full_data)))
    
    # First split: separate out test set (15%)
    train_val_idx, test_idx = train_test_split(
        total_indices,
        test_size=0.15,
        random_state=42,
        shuffle=True
    )
    
    # Second split: separate train and validation from remaining data
    # This gives us 70% train, 15% val from the remaining 85%
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.176,  # 15/(70+15) ≈ 0.176 to get 15% of total
        random_state=42,
        shuffle=True
    )
    
    # Create the splits
    train_dataset = full_data.select(train_idx)
    val_dataset = full_data.select(val_idx)
    test_dataset = full_data.select(test_idx)
    
    logger.info(f"Final train size: {len(train_dataset)} ({len(train_dataset)/len(full_data)*100:.1f}%)")
    logger.info(f"Final validation size: {len(val_dataset)} ({len(val_dataset)/len(full_data)*100:.1f}%)")
    logger.info(f"Final test size: {len(test_dataset)} ({len(test_dataset)/len(full_data)*100:.1f}%)")
    
    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }

def custom_collate_fn(batch):
    # Bypass default_collate for 'image' field
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    batch_dict["input_ids"] = default_collate(batch_dict["input_ids"])
    batch_dict["attention_mask"] = default_collate(batch_dict["attention_mask"])
    # Leave "image" as list of PIL.Image for processor to handle later
    return batch_dict

def create_flickr30k_dataloaders(datasets, 
                                 image_size = 224,
                                 batch_size = 16, 
                                 num_workers = 4, 
                                 tokenizer_name = "Qwen/Qwen3-0.6B-Base",
                                 max_caption_length = 128,
                                 use_all_captions = False,
                                 train_size_limit = None):
    """
    Create DataLoaders for training, validation, and test sets.
    
    Args:
        datasets: Dict with 'train', 'validation', 'test' splits
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
        tokenizer_name: Tokenizer to use
        use_all_captions: If True, use all captions (5x more samples)
        train_size_limit: If provided, limit training set to this many samples
    
    Returns:
        dict: Dictionary with DataLoaders for each split
    """
    
    dataset_class = Flickr30kCaptionDataset if use_all_captions else Flickr30kDataset
    
    # Apply training size limit if specified
    pin_memory = torch.cuda.is_available()  # True only on NVIDIA GPU

    train_split = datasets['train']
    val_split = datasets['validation']  # Initialize val_split to avoid UnboundLocalError
    
    if train_size_limit is not None and train_size_limit > 0:
        original_train_size = len(train_split)
        train_size_limit = min(train_size_limit, original_train_size)
        train_split = train_split.select(range(train_size_limit))
        logger.info(f"Training set limited from {original_train_size} to {train_size_limit} samples")

        original_val_size = len(datasets['validation'])
        val_size_limit = min(train_size_limit, original_val_size)
        val_split = datasets['validation'].select(range(val_size_limit))
        logger.info(f"Validation set limited from {original_val_size} to {val_size_limit} samples")
    
    # Create datasets
    train_dataset = dataset_class(huggingface_dataset=train_split, tokenizer_name=tokenizer_name, image_size=image_size, max_caption_length=max_caption_length)
    val_dataset = dataset_class(huggingface_dataset=val_split, tokenizer_name=tokenizer_name, image_size=image_size, max_caption_length=max_caption_length)
    test_dataset = dataset_class(huggingface_dataset=datasets['test'], tokenizer_name=tokenizer_name, image_size=image_size, max_caption_length=max_caption_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader
    }

# Example usage and testing
if __name__ == "__main__":
    from utils import get_flickr30k_datasets
    
    print("Loading datasets...")
    datasets = get_flickr30k_datasets()
    
    print("Creating custom dataset...")
    custom_dataset = Flickr30kDataset(datasets['train'])
    
    print(f"Dataset length: {len(custom_dataset)}")
    
    # Test one sample
    sample = custom_dataset[0]
    print("\nSample structure:")
    for key, value in sample.items():
        if key == 'image':
            print(f"  {key}: {value.shape} (tensor)")
        elif key in ['input_ids', 'attention_mask']:
            print(f"  {key}: {value.shape} (tensor)")
        else:
            print(f"  {key}: {value}")
    
    print("\nTesting DataLoader...")
    dataloaders = create_flickr30k_dataloaders(datasets, batch_size=2)
    
    # Test one batch
    for batch in dataloaders['train']:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"First caption: {batch['caption'][0]}")
        break 