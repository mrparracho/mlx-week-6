#!/usr/bin/env python3
"""
Example usage of QWEN 7B LoRA fine-tuning with Chinchilla scaling laws
"""

import subprocess
import sys
import os


def run_training_example():
    """Run a basic training example."""
    print("Running basic training example...")
    
    cmd = [
        sys.executable, "main.py",
        "--model_name", "Qwen/Qwen-7B",
        "--dataset_name", "cnn_dailymail",
        "--lora_rank", "16",
        "--learning_rate", "1e-4",
        "--batch_size", "4",
        "--epochs", "1",  # Reduced for example
        "--output_dir", "./example_output",
        "--generate_samples"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Training example completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training example failed: {e}")


def run_evaluation_example():
    """Run evaluation example."""
    print("Running evaluation example...")
    
    cmd = [
        sys.executable, "main.py",
        "--eval_only",
        "--eval_samples", "5",
        "--output_dir", "./example_output"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Evaluation example completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation example failed: {e}")


def run_custom_config_example():
    """Run training with custom configuration."""
    print("Running custom configuration example...")
    
    cmd = [
        sys.executable, "main.py",
        "--model_name", "Qwen/Qwen-7B",
        "--dataset_name", "cnn_dailymail",
        "--lora_rank", "32",  # Higher rank
        "--lora_dropout", "0.2",  # Higher dropout
        "--learning_rate", "5e-5",  # Lower learning rate
        "--batch_size", "2",  # Smaller batch size
        "--epochs", "1",
        "--max_input_length", "512",  # Shorter sequences
        "--max_target_length", "64",  # Shorter summaries
        "--output_dir", "./custom_output",
        "--save_steps", "100",
        "--eval_steps", "100",
        "--logging_steps", "5"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Custom configuration example completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Custom configuration example failed: {e}")


def main():
    """Main function to run examples."""
    print("="*80)
    print("QWEN 7B LoRA FINE-TUNING EXAMPLES")
    print("="*80)
    
    print("\nAvailable examples:")
    print("1. Basic training example")
    print("2. Evaluation example")
    print("3. Custom configuration example")
    print("4. Run all examples")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_training_example()
    elif choice == "2":
        run_evaluation_example()
    elif choice == "3":
        run_custom_config_example()
    elif choice == "4":
        print("\nRunning all examples...")
        run_training_example()
        run_evaluation_example()
        run_custom_config_example()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main() 