#!/usr/bin/env python3
"""
Debug script to examine dataset and identify why loss is constant
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from data import load_and_preprocess_data
from config import DataConfig

def debug_dataset(debug_mode: bool = True, sample_size: int = 50):
    """Debug the dataset to understand the constant loss issue."""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B", trust_remote_code=True)
    
    print("Loading and preprocessing data...")
    config = DataConfig()
    
    # Use sampling for fast debugging
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        config, tokenizer, debug_mode=debug_mode, sample_size=sample_size
    )
    
    print("\n" + "="*60)
    print("DEBUGGING DATASET")
    print("="*60)
    
    # Examine first few examples
    for i in range(min(3, len(train_dataset))):
        example = train_dataset[i]
        print(f"\nExample {i}:")
        print(f"  Input IDs length: {len(example['input_ids'])}")
        print(f"  Labels length: {len(example['labels'])}")
        print(f"  Labels range: {min(example['labels'])} to {max(example['labels'])}")
        print(f"  Non-100 labels: {sum(1 for l in example['labels'] if l != -100)}")
        print(f"  First 10 input_ids: {example['input_ids'][:10]}")
        print(f"  First 10 labels: {example['labels'][:10]}")
        
        # Check if labels match input_ids length
        if len(example['input_ids']) != len(example['labels']):
            print(f"  ⚠️  LENGTH MISMATCH: input_ids={len(example['input_ids'])}, labels={len(example['labels'])}")
        else:
            print(f"  ✅ Lengths match: {len(example['input_ids'])}")
        
        # Check if all labels are -100
        if all(l == -100 for l in example['labels']):
            print(f"  ⚠️  ALL LABELS ARE -100!")
        
        # Decode some tokens to see what we're working with
        input_text = tokenizer.decode(example['input_ids'][:50])
        print(f"  Sample input text: {input_text[:100]}...")
        
        # Find where labels change from -100 to actual tokens
        label_changes = []
        for j, (input_id, label) in enumerate(zip(example['input_ids'], example['labels'])):
            if label != -100:
                label_changes.append((j, input_id, label))
                if len(label_changes) >= 5:  # Show first 5 changes
                    break
        
        print(f"  First 5 label changes: {label_changes}")
    
    # Test a few batches to see if the issue persists
    print(f"\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60)
    
    from torch.utils.data import DataLoader
    from data import collate_fn
    
    # Create a small dataloader
    dataloader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Only test first 3 batches
            break
            
        print(f"\nBatch {i}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        
        # Check if shapes match
        if batch['input_ids'].shape == batch['labels'].shape:
            print(f"  ✅ Shapes match: {batch['input_ids'].shape}")
        else:
            print(f"  ⚠️  SHAPE MISMATCH: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
        
        # Check label values
        labels = batch['labels']
        non_neg_100 = (labels != -100).sum().item()
        total_labels = labels.numel()
        print(f"  Non-100 labels: {non_neg_100}/{total_labels} ({non_neg_100/total_labels*100:.1f}%)")
        
        # Check for constant values
        unique_labels = torch.unique(labels).tolist()
        print(f"  Unique label values: {len(unique_labels)} (range: {min(unique_labels)} to {max(unique_labels)})")

if __name__ == "__main__":
    # Use debug mode with small sample for fast iteration
    debug_dataset(debug_mode=True, sample_size=50) 