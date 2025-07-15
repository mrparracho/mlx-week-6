#!/usr/bin/env python3
"""
Debug script to check QWEN tokenizer properties
"""

from transformers import AutoTokenizer

def debug_tokenizer():
    """Debug the QWEN tokenizer to understand its properties."""
    
    print("Loading QWEN tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
    
    print(f"\nTokenizer class: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"BOS token: {tokenizer.bos_token}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    
    print(f"\nSpecial tokens:")
    print(f"  All special tokens: {tokenizer.all_special_tokens}")
    print(f"  Special tokens map: {tokenizer.special_tokens_map}")
    
    print(f"\nFirst 50 tokens in vocabulary:")
    for i in range(50):
        token = tokenizer.convert_ids_to_tokens(i)
        print(f"  {i}: {repr(token)}")
    
    print(f"\nLooking for string tokens...")
    string_tokens = []
    for i in range(min(1000, tokenizer.vocab_size)):
        token = tokenizer.convert_ids_to_tokens(i)
        if isinstance(token, str) and len(token) > 0 and not token.startswith('b"'):
            string_tokens.append((i, token))
            if len(string_tokens) >= 10:
                break
    
    print(f"Found {len(string_tokens)} string tokens:")
    for i, token in string_tokens:
        print(f"  {i}: {repr(token)}")
    
    print(f"\nTesting our fix...")
    
    # Test our fix approach
    if tokenizer.pad_token is None:
        # QWEN tokenizer has no EOS token, so we need to find a suitable pad token
        # Look for a string token in the vocabulary
        pad_token_found = False
        for i in range(min(1000, tokenizer.vocab_size)):
            token = tokenizer.convert_ids_to_tokens(i)
            if isinstance(token, str) and len(token) > 0 and not token.startswith('b"'):
                print(f"Setting pad_token to {repr(token)} (ID: {i})")
                tokenizer.pad_token = token
                tokenizer.pad_token_id = i
                pad_token_found = True
                break
        
        if not pad_token_found:
            print("No suitable string token found, using '<pad>'")
            # Fallback to a simple string
            tokenizer.pad_token = "<pad>"
            # Try to find this token in vocabulary or use 0
            try:
                pad_id = tokenizer.convert_tokens_to_ids("<pad>")
                print(f"Found '<pad>' at ID: {pad_id}")
                tokenizer.pad_token_id = pad_id
            except:
                print("'<pad>' not found in vocabulary, using ID 0")
                tokenizer.pad_token_id = 0
    
    print(f"\nFinal state after fix:")
    print(f"  Pad token: {tokenizer.pad_token}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")
    
    # Test tokenization
    print(f"\nTesting tokenization...")
    try:
        test_text = "Hello world"
        result = tokenizer(test_text, padding=True, truncation=True, max_length=10, return_tensors="pt")
        print(f"Tokenization successful!")
        print(f"Input IDs shape: {result['input_ids'].shape}")
        print(f"Attention mask shape: {result['attention_mask'].shape}")
    except Exception as e:
        print(f"Tokenization failed: {e}")

if __name__ == "__main__":
    debug_tokenizer() 