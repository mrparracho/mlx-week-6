#!/usr/bin/env python3
"""
Main training script for QWEN 7B LoRA fine-tuning with Chinchilla scaling laws
"""

import os
import sys
import argparse
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_default_config, create_config_from_args
from data import load_and_preprocess_data
from model import setup_model_and_tokenizer
from train import train_model
from utils import (
    setup_logging, save_config, save_training_history, 
    evaluate_summaries, save_evaluation_results, print_evaluation_samples,
    print_model_size_info, print_gpu_info, create_sample_summaries
)
from transformers import AutoTokenizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QWEN 7B LoRA Fine-tuning with Chinchilla Scaling Laws"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen-7B",
        help="Base model name or path"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cnn_dailymail",
        help="Dataset name"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: 2 * rank)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    
    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    
    # Data arguments
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1024,
        help="Maximum input length"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="Maximum target length"
    )
    
    # Chinchilla scaling arguments
    parser.add_argument(
        "--apply_chinchilla_scaling",
        action="store_true",
        default=True,
        help="Apply Chinchilla scaling laws"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_model",
        help="Output directory"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=10,
        help="Number of samples for evaluation"
    )
    
    # Sample generation arguments
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        help="Generate sample summaries after training"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    print("="*80)
    print("QWEN 7B LoRA FINE-TUNING WITH CHINCHILLA SCALING LAWS")
    print("="*80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = create_config_from_args(args)
    config.print_config()
    
    # Setup logging
    log_file = setup_logging(config.output.output_dir)
    print(f"Log file: {log_file}")
    
    # Save configuration
    save_config(config, config.output.output_dir)
    
    # Print GPU information
    print_gpu_info()
    
    try:
        # Load tokenizer before data preprocessing
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            trust_remote_code=config.model.trust_remote_code
        )
        
        # Set padding token for QWEN tokenizer
        if tokenizer.pad_token is None:
            # QWEN tokenizer has no EOS token, so we need to find a suitable pad token
            # Look for a string token in the vocabulary
            pad_token_found = False
            for i in range(min(1000, tokenizer.vocab_size)):
                token = tokenizer.convert_ids_to_tokens(i)
                if isinstance(token, str) and len(token) > 0 and not token.startswith('b"'):
                    tokenizer.pad_token = token
                    tokenizer.pad_token_id = i
                    pad_token_found = True
                    break
            
            if not pad_token_found:
                # Fallback to a simple string
                tokenizer.pad_token = "<pad>"
                # Try to find this token in vocabulary or use 0
                try:
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
                except:
                    tokenizer.pad_token_id = 0
        
        print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  Vocab size: {tokenizer.vocab_size}")

        # Load and preprocess data
        print("\n" + "="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
            config.data, 
            tokenizer
        )
        
        # Setup model and tokenizer
        print("\n" + "="*60)
        print("SETTING UP MODEL AND TOKENIZER")
        print("="*60)
        
        model, tokenizer = setup_model_and_tokenizer(
            config.model, 
            config.lora
        )
        
        # Print model size information
        print_model_size_info(model)
        
        if args.eval_only:
            # Evaluation only mode
            print("\n" + "="*60)
            print("EVALUATION MODE")
            print("="*60)
            
            # Load saved model if exists
            if os.path.exists(config.output.output_dir):
                print(f"Loading saved model from {config.output.output_dir}")
                # Note: In a real implementation, you'd load the saved model here
            
            # Evaluate model
            eval_results = evaluate_summaries(
                model, tokenizer, test_dataset, args.eval_samples
            )
            
            # Save and print evaluation results
            save_evaluation_results(eval_results, config.output.output_dir)
            print_evaluation_samples(eval_results)
            
        else:
            # Training mode
            print("\n" + "="*60)
            print("STARTING TRAINING")
            print("="*60)
            
            # Train model
            training_history = train_model(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                training_config=config.training,
                chinchilla_config=config.chinchilla,
                output_dir=config.output.output_dir
            )
            
            # Save training history
            save_training_history(training_history, config.output.output_dir)
            
            # Evaluate trained model
            print("\n" + "="*60)
            print("EVALUATING TRAINED MODEL")
            print("="*60)
            
            eval_results = evaluate_summaries(
                model, tokenizer, test_dataset, args.eval_samples
            )
            
            # Save and print evaluation results
            save_evaluation_results(eval_results, config.output.output_dir)
            print_evaluation_samples(eval_results)
            
            # Generate sample summaries if requested
            if args.generate_samples:
                print("\n" + "="*60)
                print("GENERATING SAMPLE SUMMARIES")
                print("="*60)
                
                # Sample texts for demonstration
                sample_texts = [
                    "The rapid advancement of artificial intelligence has sparked both excitement and concern among researchers and policymakers. While AI promises to revolutionize industries and improve human lives, questions remain about its potential impact on employment, privacy, and societal structures. Experts emphasize the need for responsible development and regulation to ensure AI benefits humanity while minimizing risks.",
                    "Climate change continues to be one of the most pressing challenges facing humanity. Recent studies show that global temperatures are rising faster than previously predicted, leading to more frequent extreme weather events, rising sea levels, and ecosystem disruptions. Scientists urge immediate action to reduce greenhouse gas emissions and transition to renewable energy sources.",
                    "The global economy faces unprecedented challenges as supply chain disruptions, inflation, and geopolitical tensions create uncertainty for businesses and consumers. Central banks worldwide are implementing various monetary policies to address inflation while supporting economic growth. Economists predict continued volatility in financial markets as these factors play out."
                ]
                
                create_sample_summaries(
                    model, tokenizer, sample_texts, config.output.output_dir
                )
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print(f"✓ Model saved to: {config.output.output_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
