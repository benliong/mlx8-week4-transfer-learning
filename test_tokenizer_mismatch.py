#!/usr/bin/env python3
"""
Test script to demonstrate tokenizer size mismatch issues
"""

import torch
from transformers import AutoTokenizer
from utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

def simulate_training_tokenizer():
    """Simulate what happens during training"""
    print("=== TRAINING SCENARIO ===")
    
    # Load tokenizer (simulating training time)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    original_size = len(tokenizer)
    print(f"Original Qwen tokenizer size: {original_size}")
    
    # Add BOS token
    tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
    training_size = len(tokenizer)
    bos_id = tokenizer.bos_token_id
    
    print(f"After adding BOS token:")
    print(f"  - Tokenizer size: {training_size}")
    print(f"  - BOS token ID: {bos_id}")
    print(f"  - BOS token: '{tokenizer.bos_token}'")
    
    # Simulate creating embedding matrix
    embedding_matrix = torch.randn(training_size, 1024)  # vocab_size x hidden_dim
    print(f"Created embedding matrix shape: {embedding_matrix.shape}")
    
    return {
        'tokenizer_size': training_size,
        'bos_token_id': bos_id,
        'embedding_shape': embedding_matrix.shape,
        'tokenizer': tokenizer
    }

def simulate_inference_tokenizer(training_info):
    """Simulate what happens during inference with potentially updated tokenizer"""
    print("\n=== INFERENCE SCENARIO ===")
    
    # Load tokenizer again (simulating inference time - could be different!)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    original_size = len(tokenizer)
    print(f"Fresh Qwen tokenizer size: {original_size}")
    
    # Add BOS token again
    tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
    inference_size = len(tokenizer)
    bos_id = tokenizer.bos_token_id
    
    print(f"After adding BOS token:")
    print(f"  - Tokenizer size: {inference_size}")
    print(f"  - BOS token ID: {bos_id}")
    print(f"  - BOS token: '{tokenizer.bos_token}'")
    
    # Check for mismatches
    training_size = training_info['tokenizer_size']
    training_bos_id = training_info['bos_token_id']
    training_shape = training_info['embedding_shape']
    
    print(f"\n=== COMPATIBILITY CHECK ===")
    
    if inference_size != training_size:
        print(f"❌ VOCAB SIZE MISMATCH!")
        print(f"   Training: {training_size}, Inference: {inference_size}")
        print(f"   Difference: {inference_size - training_size} tokens")
        
        print(f"\n❌ EMBEDDING MATRIX INCOMPATIBLE!")
        print(f"   Saved embedding shape: {training_shape}")
        print(f"   Expected embedding shape: ({inference_size}, {training_shape[1]})")
        print(f"   This will cause PyTorch loading errors!")
    else:
        print(f"✅ Vocab sizes match: {training_size}")
    
    if bos_id != training_bos_id:
        print(f"❌ BOS TOKEN ID MISMATCH!")
        print(f"   Training: {training_bos_id}, Inference: {bos_id}")
        print(f"   This will cause incorrect model behavior!")
    else:
        print(f"✅ BOS token IDs match: {bos_id}")
    
    return {
        'size_mismatch': inference_size != training_size,
        'bos_mismatch': bos_id != training_bos_id,
        'size_diff': inference_size - training_size
    }

def demonstrate_loading_error(training_info):
    """Demonstrate what happens when trying to load mismatched embeddings"""
    print(f"\n=== DEMONSTRATING LOADING ERROR ===")
    
    # Create a mock state dict with training-time embedding size
    training_size = training_info['tokenizer_size']
    mock_state_dict = {
        'qwen_decoder.qwen_model.model.embed_tokens.weight': torch.randn(training_size, 1024)
    }
    
    print(f"Mock saved embedding matrix shape: {mock_state_dict['qwen_decoder.qwen_model.model.embed_tokens.weight'].shape}")
    
    # Create a fresh tokenizer (inference time)
    fresh_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    fresh_tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
    current_size = len(fresh_tokenizer)
    
    print(f"Current expected embedding size: ({current_size}, 1024)")
    
    if current_size != training_size:
        print(f"❌ Size mismatch detected!")
        print(f"   PyTorch would throw: RuntimeError: size mismatch for qwen_decoder.qwen_model.model.embed_tokens.weight")
        print(f"   Expected: [{current_size}, 1024], Got: [{training_size}, 1024]")
    else:
        print(f"✅ Sizes match - loading would succeed")

if __name__ == "__main__":
    print("🔍 Testing Tokenizer Size Mismatch Scenarios\n")
    
    # Simulate training
    training_info = simulate_training_tokenizer()
    
    # Simulate inference
    inference_info = simulate_inference_tokenizer(training_info)
    
    # Demonstrate loading error
    demonstrate_loading_error(training_info)
    
    print(f"\n📋 SUMMARY:")
    if inference_info['size_mismatch']:
        print(f"❌ This scenario would cause CRITICAL ERRORS in production!")
        print(f"   - Model loading would fail")
        print(f"   - Embedding matrix dimensions incompatible")
        print(f"   - Vocab size difference: {inference_info['size_diff']} tokens")
    else:
        print(f"✅ This scenario would work correctly")
    
    if inference_info['bos_mismatch']:
        print(f"❌ BOS token mismatch would cause incorrect model behavior!")
    
    print(f"\n💡 SOLUTION: Save and restore exact tokenizer state during training!")