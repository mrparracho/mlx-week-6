#!/usr/bin/env python3
"""
Script to continue training the LoRA model for more epochs to improve performance.
"""

import os
import sys
from main import main

if __name__ == "__main__":
    # Set up arguments for continued training
    sys.argv = [
        "main.py",
        "--epochs", "5",  # Train for 5 more epochs
        "--resume_from_checkpoint", "best",  # Resume from the best checkpoint
        "--learning_rate", "5e-4",  # Use a lower learning rate for continued training
        "--eval_steps", "1000",  # Evaluate more frequently
        "--save_steps", "2000",  # Save checkpoints more frequently
    ]
    
    print("🚀 Continuing training for 5 more epochs...")
    print("📊 This should significantly improve the model's performance.")
    print("⏱️  Expected training time: 2-3 hours")
    
    # Run the main training function
    main() 