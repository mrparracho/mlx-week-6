#!/usr/bin/env python3
"""
Example script demonstrating how to resume training from a checkpoint
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import create_config_from_args
from model import setup_model_and_tokenizer
from data import load_and_preprocess_data
from train import train_model
from transformers import AutoTokenizer


def main():
    """Example of resuming training from a checkpoint."""
    
    # Example 1: Resume from a specific checkpoint
    print("="*80)
    print("EXAMPLE 1: Resume from specific checkpoint")
    print("="*80)
    
    # Check if we have a checkpoint to resume from
    output_dir = "./trained_model"
    checkpoint_dir = os.path.join(output_dir, "checkpoint-1000")  # Example checkpoint
    
    if os.path.exists(checkpoint_dir):
        print(f"Found checkpoint: {checkpoint_dir}")
        print("To resume from this checkpoint, run:")
        print(f"python main.py --resume_from_checkpoint {checkpoint_dir}")
    else:
        print(f"No checkpoint found at {checkpoint_dir}")
        print("First run training to create checkpoints:")
        print("python main.py --epochs 3 --save_steps 500")
    
    # Example 2: Resume from best model
    print("\n" + "="*80)
    print("EXAMPLE 2: Resume from best model")
    print("="*80)
    
    best_info_path = os.path.join(output_dir, "best_model_info.json")
    if os.path.exists(best_info_path):
        print("Found best model info. To resume from best model, run:")
        print("python main.py --resume_from_checkpoint best")
    else:
        print("No best model found. Train first to create best model.")
    
    # Example 3: List available checkpoints
    print("\n" + "="*80)
    print("EXAMPLE 3: List available checkpoints")
    print("="*80)
    
    if os.path.exists(output_dir):
        import glob
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoints:
            print("Available checkpoints:")
            for checkpoint in sorted(checkpoints):
                checkpoint_name = os.path.basename(checkpoint)
                print(f"  - {checkpoint_name}")
            print("\nTo resume from any checkpoint, use:")
            print("python main.py --resume_from_checkpoint ./trained_model/checkpoint-XXXX")
        else:
            print("No checkpoints found. Run training first.")
    else:
        print("No output directory found. Run training first.")
    
    # Example 4: Complete resume example
    print("\n" + "="*80)
    print("EXAMPLE 4: Complete resume workflow")
    print("="*80)
    
    print("1. Start initial training:")
    print("   python main.py --epochs 2 --save_steps 500")
    print()
    print("2. Interrupt training (Ctrl+C) or let it complete")
    print()
    print("3. Resume from last checkpoint:")
    print("   python main.py --resume_from_checkpoint ./trained_model/checkpoint-XXXX")
    print()
    print("4. Or resume from best model:")
    print("   python main.py --resume_from_checkpoint best")
    print()
    print("5. Continue with more epochs:")
    print("   python main.py --epochs 5 --resume_from_checkpoint best")
    
    # Example 5: Resume with different parameters
    print("\n" + "="*80)
    print("EXAMPLE 5: Resume with different training parameters")
    print("="*80)
    
    print("You can resume training with different parameters:")
    print("python main.py --resume_from_checkpoint best --learning_rate 5e-5 --epochs 10")
    print()
    print("Note: Some parameters like model architecture cannot be changed when resuming.")
    print("The following parameters can be modified:")
    print("  - learning_rate")
    print("  - batch_size")
    print("  - epochs")
    print("  - warmup_steps")
    print("  - weight_decay")
    print("  - save_steps")
    print("  - eval_steps")


if __name__ == "__main__":
    main() 