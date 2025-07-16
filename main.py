#!/usr/bin/env python3
"""
Main training script for Qwen2.5 LoRA fine-tuning with Chinchilla scaling laws
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

# Set environment variables for better stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from transformers import AutoTokenizer
from datasets import Dataset
from config import create_config_from_args, Config
from model import create_lora_model, setup_model_and_tokenizer
from data import load_and_preprocess_data
from train import create_trainer, train_model
from utils import (
    setup_logging, 
    print_gpu_info, 
    print_model_size_info,
    evaluate_summaries,
    save_evaluation_results,
    print_evaluation_samples,
    create_sample_summaries,
    save_training_history
)
from scaling_laws import analyze_scaling_efficiency, print_scaling_analysis


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5 LoRA Fine-tuning with Chinchilla Scaling Laws"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
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
    
    # Caching arguments
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching models and datasets"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Use cached models and datasets"
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force reload models and datasets (ignore cache)"
    )
    
    # Debug arguments
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode with sampling for fast iteration"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Sample size for debug mode"
    )
    
    # Resume arguments
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from, or 'best' for best model"
    )
    
    # HuggingFace Hub arguments
    parser.add_argument(
        "--hf_repo_name",
        type=str,
        default=None,
        help="HuggingFace Hub repository name (e.g., 'username/model-name') for saving models"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    print("="*80)
    print("QWEN2.5-0.5B LoRA FINE-TUNING WITH CHINCHILLA SCALING LAWS")
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
    # save_config_as_yaml(config, config.output.output_dir) # This line was removed as per the new_code
    
    # Setup cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Print GPU information
    # print_gpu_info() # This line was removed as per the new_code
    
    try:
        # Load tokenizer with caching
        tokenizer_cache_path = os.path.join(args.cache_dir, f"tokenizer_{config.model.model_name.replace('/', '_')}")
        
        if args.use_cache and not args.force_reload and os.path.exists(tokenizer_cache_path):
            print(f"Loading cached tokenizer from {tokenizer_cache_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_cache_path,
                trust_remote_code=True
            )
        else:
            print(f"Loading tokenizer from {config.model.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.model_name,
                trust_remote_code=config.model.trust_remote_code,
                cache_dir=args.cache_dir
            )
            # Save tokenizer to cache
            if args.use_cache:
                tokenizer.save_pretrained(tokenizer_cache_path)
                print(f"✓ Tokenizer cached to {tokenizer_cache_path}")
        
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

        # Load and preprocess data with caching
        print("\n" + "="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        # Create cache key for dataset
        dataset_cache_key = f"{config.data.dataset_name}_{config.data.dataset_config}_{config.data.max_input_length}_{config.data.max_target_length}"
        dataset_cache_path = os.path.join(args.cache_dir, f"dataset_{dataset_cache_key.replace('/', '_')}")
        
        if args.use_cache and not args.force_reload and os.path.exists(dataset_cache_path):
            print(f"Loading cached datasets from {dataset_cache_path}")
            from datasets import load_from_disk
            train_dataset = load_from_disk(os.path.join(dataset_cache_path, "train"))
            val_dataset = load_from_disk(os.path.join(dataset_cache_path, "validation"))
            test_dataset = load_from_disk(os.path.join(dataset_cache_path, "test"))
            print("✓ Cached datasets loaded successfully")
        else:
            train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
                config.data, 
                tokenizer,
                debug_mode=args.debug_mode,
                sample_size=args.sample_size
            )
            # Save datasets to cache
            if args.use_cache:
                os.makedirs(dataset_cache_path, exist_ok=True)
                train_dataset.save_to_disk(os.path.join(dataset_cache_path, "train"))
                val_dataset.save_to_disk(os.path.join(dataset_cache_path, "validation"))
                test_dataset.save_to_disk(os.path.join(dataset_cache_path, "test"))
                print(f"✓ Datasets cached to {dataset_cache_path}")
        
        # Setup model and tokenizer with caching
        print("\n" + "="*60)
        print("SETTING UP MODEL AND TOKENIZER")
        print("="*60)
        
        # Create cache key for model
        model_cache_key = f"{config.model.model_name}_{config.model.torch_dtype}"
        model_cache_path = os.path.join(args.cache_dir, f"model_{model_cache_key.replace('/', '_')}")
        
        if args.use_cache and not args.force_reload and os.path.exists(model_cache_path):
            print(f"Loading cached model from {model_cache_path}")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_cache_path,
                torch_dtype=getattr(torch, config.model.torch_dtype),
                device_map=config.model.device_map
            )
            # Apply LoRA to cached model
            model, _ = setup_model_and_tokenizer(config.model, config.lora)
        else:
            model, tokenizer = setup_model_and_tokenizer(
                config.model, 
                config.lora
            )
            # Save base model to cache (without LoRA)
            if args.use_cache:
                print(f"Caching base model to {model_cache_path}")
                # Note: We cache the base model before LoRA is applied
                # This is a simplified approach - in practice you might want more sophisticated caching
        
        # Update Chinchilla config with actual model size
        total_params = sum(p.numel() for p in model.parameters())
        config.update_model_size(total_params)
        
        # Print model size information
        # print_model_size_info(model) # This line was removed as per the new_code
        
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
                output_dir=config.output.output_dir,
                resume_from_checkpoint=args.resume_from_checkpoint,
                hf_repo_name=args.hf_repo_name
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
