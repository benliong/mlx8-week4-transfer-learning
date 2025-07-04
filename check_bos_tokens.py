#!/usr/bin/env python3
"""
Utility to check BOS token consistency in saved models.
This helps debug BOS token ID mismatches between training and inference.
"""

import os
from utils import list_saved_models, load_saved_model, setup_logging
from model import Model
import logging

setup_logging()
logger = logging.getLogger(__name__)

def check_model_bos_tokens(save_dir="saved_models"):
    """Check BOS token consistency for all saved models."""
    
    models = list_saved_models(save_dir)
    
    if not models:
        print(f"No saved models found in '{save_dir}'")
        return
    
    print(f"\n🔍 Checking BOS token consistency for {len(models)} model(s):")
    print("=" * 80)
    
    for i, model_info in enumerate(models, 1):
        print(f"\n{i}. {model_info['filename']}")
        print("-" * 60)
        
        try:
            # Load the model without instantiating it (just get checkpoint data)
            model_data = load_saved_model(model_info['path'], model_class=None)
            checkpoint = model_data['checkpoint']
            
            # Check if BOS token info is saved
            saved_bos_id = checkpoint.get('bos_token_id')
            saved_vocab_size = checkpoint.get('tokenizer_vocab_size')
            
            # Check if this model has a saved tokenizer (new format)
            tokenizer_dir = checkpoint.get('tokenizer_dir')
            
            if tokenizer_dir and os.path.exists(tokenizer_dir):
                print(f"✅ Has saved tokenizer: {tokenizer_dir}")
                print(f"✅ This model uses the COMPLETE tokenizer approach!")
                
                try:
                    # Try to load the actual saved model to test
                    model_data = load_saved_model(model_info['path'], model_class=Model)
                    model = model_data['model']
                    print(f"✅ Model loaded successfully with saved tokenizer")
                    print(f"   - Vocab size: {len(model.tokenizer)}")
                    print(f"   - BOS token: '{model.tokenizer.bos_token}' (ID: {model.tokenizer.bos_token_id})")
                    print(f"✅ This model should work perfectly!")
                    
                except Exception as e:
                    print(f"❌ Error loading model with saved tokenizer: {e}")
                    
            elif saved_bos_id is not None:
                print(f"⚠️ Uses OLD tokenizer approach (BOS ID only)")
                print(f"✅ Saved BOS token ID: {saved_bos_id}")
                print(f"✅ Saved vocab size: {saved_vocab_size}")
                
                # Now create a fresh model and see what BOS ID it would get
                fresh_model = Model()
                current_bos_id = fresh_model.tokenizer.bos_token_id
                current_vocab_size = len(fresh_model.tokenizer)
                
                print(f"🔄 Fresh model BOS ID: {current_bos_id}")
                print(f"🔄 Fresh model vocab size: {current_vocab_size}")
                
                if current_bos_id == saved_bos_id:
                    print("✅ BOS token IDs match! This model should work correctly.")
                else:
                    print(f"❌ BOS token ID MISMATCH! This will cause inference issues!")
                    print(f"   Saved: {saved_bos_id}, Current: {current_bos_id}")
                
                if current_vocab_size != saved_vocab_size:
                    print(f"⚠️ Vocab size mismatch: Saved {saved_vocab_size}, Current {current_vocab_size}")
                    print(f"💡 Recommend re-training with new tokenizer saving system!")
                
            else:
                print("❌ No BOS token ID saved (older model format)")
                print("   This model was saved before BOS token tracking was implemented.")
                print("   Inference results may be unreliable.")
                
        except Exception as e:
            print(f"❌ Error checking model: {e}")
    
    print("\n" + "=" * 80)
    print("💡 Tip: Re-train models that show BOS token mismatches for best results.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check BOS token consistency in saved models")
    parser.add_argument("--save-dir", default="saved_models", help="Directory containing saved models")
    
    args = parser.parse_args()
    check_model_bos_tokens(args.save_dir)