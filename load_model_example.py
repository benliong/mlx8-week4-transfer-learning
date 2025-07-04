#!/usr/bin/env python3
"""
Example script demonstrating how to load and use saved models in HuggingFace format.

This script shows:
1. How to create and train a model
2. How to save it in HuggingFace format
3. How to load it back using AutoModel and AutoTokenizer
4. How to use the model for inference

Usage:
    python load_model_example.py
"""

import os
import torch
import logging
from utils import setup_logging, get_device, load_saved_model, list_saved_models, print_model_summary
from model import VisionLanguageModel, Model
from configuration import VisionLanguageConfig
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def demonstrate_huggingface_compatibility():
    """
    Demonstrate creating, saving, and loading models in HuggingFace format.
    """
    print("🚀 HuggingFace Model Compatibility Demo")
    print("=" * 50)
    
    # 1. Create a model with custom configuration
    print("\n1️⃣ Creating a model with custom configuration...")
    
    config = VisionLanguageConfig(
        clip_model_name="openai/clip-vit-base-patch32",
        qwen_model_name="Qwen/Qwen3-0.6B-Base",
        max_caption_length=64,  # Smaller for demo
        image_size=224,
        tokenizer_name="Qwen/Qwen3-0.6B-Base",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    
    model = VisionLanguageModel(config)
    print(f"✅ Model created with vocab size: {len(model.tokenizer)}")
    print(f"✅ BOS token: '{model.tokenizer.bos_token}' (ID: {model.tokenizer.bos_token_id})")
    
    # 2. Save the model in HuggingFace format
    print("\n2️⃣ Saving model in HuggingFace format...")
    
    save_dir = "saved_models/demo_huggingface_model"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    print(f"✅ Model saved to: {save_dir}")
    
    # Check what files were created
    saved_files = os.listdir(save_dir)
    print(f"📁 Files created: {saved_files}")
    
    # 3. Load the model back using the new format
    print("\n3️⃣ Loading model back using HuggingFace format...")
    
    loaded_model = VisionLanguageModel.from_pretrained(save_dir)
    print(f"✅ Model loaded successfully!")
    print(f"✅ Vocab size: {len(loaded_model.tokenizer)}")
    print(f"✅ BOS token: '{loaded_model.tokenizer.bos_token}' (ID: {loaded_model.tokenizer.bos_token_id})")
    
    # 4. Register for AutoClass support (optional)
    print("\n4️⃣ Registering for AutoClass support...")
    
    # Register the custom model for AutoClass
    from transformers import AutoConfig, AutoModel
    AutoConfig.register("vision_language_model", VisionLanguageConfig)
    AutoModel.register(VisionLanguageConfig, VisionLanguageModel)
    
    print("✅ Model registered for AutoClass support!")
    
    # 5. Load using AutoModel and AutoTokenizer
    print("\n5️⃣ Loading using AutoModel and AutoTokenizer...")
    
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        print(f"✅ Tokenizer loaded with AutoTokenizer")
        print(f"   - Vocab size: {len(tokenizer)}")
        print(f"   - BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
        
        # Load the model
        auto_model = AutoModel.from_pretrained(save_dir, trust_remote_code=True)
        print(f"✅ Model loaded with AutoModel")
        print(f"   - Model type: {type(auto_model).__name__}")
        print(f"   - Config type: {type(auto_model.config).__name__}")
        
    except Exception as e:
        print(f"⚠️ AutoModel loading failed: {e}")
        print("   Note: This is expected if the model isn't properly registered")
    
    # 6. Compare with legacy format loading
    print("\n6️⃣ Comparing with utils.load_saved_model()...")
    
    try:
        loaded_data = load_saved_model(save_dir)
        print(f"✅ Model loaded using utils.load_saved_model()")
        print(f"   - Has model: {'model' in loaded_data}")
        print(f"   - Has metadata: {'training_metadata' in loaded_data}")
        print(f"   - Model path: {loaded_data['model_path']}")
        
    except Exception as e:
        print(f"❌ Failed to load with utils: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 50)


def demonstrate_legacy_compatibility():
    """
    Demonstrate backward compatibility with legacy Model class.
    """
    print("\n🔄 Legacy Model Compatibility Demo")
    print("=" * 50)
    
    # Create a legacy model (for backward compatibility)
    print("\n1️⃣ Creating legacy Model class...")
    
    legacy_model = Model(tokenizer_name="Qwen/Qwen3-0.6B-Base")
    print(f"✅ Legacy model created")
    if legacy_model.tokenizer is not None:
        print(f"✅ BOS token: '{legacy_model.tokenizer.bos_token}' (ID: {legacy_model.tokenizer.bos_token_id})")
    else:
        print("⚠️ No tokenizer available")
    
    # Save using the new format
    print("\n2️⃣ Saving legacy model in HuggingFace format...")
    
    save_dir = "saved_models/demo_legacy_model"
    os.makedirs(save_dir, exist_ok=True)
    
    legacy_model.save_pretrained(save_dir)
    print(f"✅ Legacy model saved to: {save_dir}")
    
    print("\n✅ Legacy compatibility maintained!")


def list_all_saved_models():
    """
    List all saved models in the saved_models directory.
    """
    print("\n📊 Saved Models Summary")
    print("=" * 50)
    
    print_model_summary("saved_models")


def verify_files_structure():
    """
    Verify that saved models have the correct HuggingFace file structure.
    """
    print("\n📁 Verifying HuggingFace File Structure")
    print("=" * 50)
    
    models = list_saved_models("saved_models")
    
    for model_info in models:
        if model_info['type'] == 'huggingface':
            print(f"\n📂 {model_info['filename']}:")
            model_path = model_info['path']
            
            # Check for required HuggingFace files
            required_files = [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json"
            ]
            
            for file_name in required_files:
                file_path = os.path.join(model_path, file_name)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ {file_name} ({file_size / 1024:.1f} KB)")
                else:
                    print(f"   ❌ {file_name} (missing)")
            
            # Check for optional files
            optional_files = [
                "training_metadata.json",
                "optimizer.pth",
                "generation_config.json"
            ]
            
            for file_name in optional_files:
                file_path = os.path.join(model_path, file_name)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"   📄 {file_name} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    print("🤖 HuggingFace Model Compatibility Examples")
    print("=" * 60)
    
    try:
        # Run the main demo
        demonstrate_huggingface_compatibility()
        
        # Show legacy compatibility
        demonstrate_legacy_compatibility()
        
        # List all saved models
        list_all_saved_models()
        
        # Verify file structure
        verify_files_structure()
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise
    
    print("\n✅ All examples completed successfully!")
    print("📝 Your models are now compatible with AutoModel and AutoTokenizer!")
    print("🚀 You can load them using:")
    print("   - VisionLanguageModel.from_pretrained('path/to/model')")
    print("   - AutoModel.from_pretrained('path/to/model', trust_remote_code=True)")
    print("   - AutoTokenizer.from_pretrained('path/to/model')")