#!/usr/bin/env python3
"""
Show how your precious BOS token is preserved across tokenizer files
"""

def show_bos_preservation():
    print("🎯 How Your Precious BOS Token is Preserved")
    print("=" * 60)
    
    print("When you save your tokenizer, your BOS token '<|im_start|>' gets saved in:")
    print()
    
    print("1. 📄 tokenizer.json")
    print("   Contains:")
    print("   • BOS token in vocabulary: '<|im_start|>': 151000")
    print("   • Token-to-ID mapping with your exact BOS ID")
    print("   • All the rules for how to tokenize text")
    print("   Example snippet:")
    print('   "vocab": {')
    print('     "<|im_start|>": 151000,')
    print('     "<|im_end|>": 151001,')
    print('     "hello": 1234,')
    print('     ...')
    print('   }')
    print()
    
    print("2. 🎯 special_tokens_map.json")
    print("   Contains:")
    print("   • Explicit BOS token definition")
    print("   • Maps the name 'bos_token' to your chosen token")
    print("   Example content:")
    print('   {')
    print('     "bos_token": "<|im_start|>",')
    print('     "eos_token": "<|im_end|>",')
    print('     "pad_token": "<|endoftext|>",')
    print('     "unk_token": "<|endoftext|>"')
    print('   }')
    print()
    
    print("3. ⚙️ tokenizer_config.json")
    print("   Contains:")
    print("   • BOS token configuration and behavior")
    print("   • Processing settings that affect BOS token usage")
    print("   Example snippet:")
    print('   {')
    print('     "bos_token": "<|im_start|>",')
    print('     "tokenizer_class": "QwenTokenizer",')
    print('     "add_bos_token": true,')
    print('     ...')
    print('   }')
    print()

def show_loading_process():
    print("🔄 What Happens When Loading Your BOS Token")
    print("=" * 60)
    
    print("Step 1: Load special_tokens_map.json")
    print("   Python reads: bos_token = '<|im_start|>'")
    print()
    
    print("Step 2: Load tokenizer_config.json")
    print("   Python reads: bos_token = '<|im_start|>', add_bos_token = true")
    print()
    
    print("Step 3: Load tokenizer.json")
    print("   Python finds: '<|im_start|>' → ID 151000 in vocabulary")
    print()
    
    print("Step 4: Create tokenizer object")
    print("   tokenizer.bos_token = '<|im_start|>'")
    print("   tokenizer.bos_token_id = 151000")
    print("   ✅ PERFECT CONSISTENCY!")
    print()

def show_what_if_missing():
    print("💥 What If BOS Token Was Missing From Files")
    print("=" * 60)
    
    scenarios = [
        {
            "missing_from": "special_tokens_map.json",
            "error": "KeyError: 'bos_token' not found",
            "result": "Can't access tokenizer.bos_token"
        },
        {
            "missing_from": "tokenizer.json vocabulary",
            "error": "Token '<|im_start|>' not in vocab",
            "result": "Can't convert token to ID"
        },
        {
            "missing_from": "tokenizer_config.json",
            "error": "No BOS token configuration",
            "result": "Wrong tokenizer behavior"
        }
    ]
    
    for scenario in scenarios:
        print(f"If missing from {scenario['missing_from']}:")
        print(f"   Error: {scenario['error']}")
        print(f"   Result: {scenario['result']}")
        print()
    
    print("🛡️ With ALL files saved: Your BOS token is TRIPLE-PROTECTED!")

def show_comparison():
    print("\n" + "=" * 60)
    print("📊 Old vs New Approach")
    print("=" * 60)
    
    print("OLD APPROACH (just save BOS token ID):")
    print("   Saved: bos_token_id = 151000")
    print("   Problem: What if new tokenizer assigns different ID?")
    print("   Risk: 💥 Silent failure with wrong BOS token")
    print()
    
    print("NEW APPROACH (save entire tokenizer):")
    print("   Saved: Complete tokenizer with '<|im_start|>' = 151000")
    print("   Loading: Uses EXACT same tokenizer that created the mapping")
    print("   Result: ✅ Guaranteed consistency!")
    print()
    
    print("🎯 Your BOS token journey:")
    print("   Training:  '<|im_start|>' → ID 151000")
    print("   Saving:    Entire tokenizer with this mapping")
    print("   Loading:   Same tokenizer, same mapping")
    print("   Inference: '<|im_start|>' → ID 151000 ✅")

if __name__ == "__main__":
    show_bos_preservation()
    show_loading_process()
    show_what_if_missing()
    show_comparison()