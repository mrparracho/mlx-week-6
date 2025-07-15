"""
Utility functions for QWEN 7B LoRA fine-tuning
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np


def setup_logging(output_dir: str) -> str:
    """
    Set up logging directory and files.
    
    Args:
        output_dir: Output directory
        
    Returns:
        Path to log file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_log_{timestamp}.txt")
    
    return log_file


def save_config(config, output_dir: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
    """
    config_file = os.path.join(output_dir, "config.json")
    
    # Convert dataclass to dict if needed
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"✓ Configuration saved to {config_file}")


def load_config(config_file: str) -> Dict:
    """
    Load configuration from file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def save_training_history(history: Dict[str, Any], output_dir: str):
    """
    Save training history to file.
    
    Args:
        history: Training history dictionary
        output_dir: Output directory
    """
    history_file = os.path.join(output_dir, "training_history.json")
    
    # Convert numpy arrays to lists for JSON serialization
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            history_serializable[key] = value.tolist()
        elif isinstance(value, dict):
            history_serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            history_serializable[key] = value
    
    with open(history_file, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    print(f"✓ Training history saved to {history_file}")


def generate_summary(model, tokenizer, text: str, 
                    max_length: int = 128, temperature: float = 0.7) -> str:
    """
    Generate a summary for the given text.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Input text to summarize
        max_length: Maximum length of generated summary
        temperature: Sampling temperature
        
    Returns:
        Generated summary
    """
    try:
        # Tokenize input - NO TRUNCATION, NO PADDING
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False  # Don't pad during tokenization
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with enhanced numerical stability
        with torch.no_grad():
            try:
                # Add numerical stability measures
                torch.set_grad_enabled(False)
                
                # Use more conservative generation parameters for stability
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=max(0.1, temperature),  # Ensure temperature > 0
                    do_sample=True,
                    top_p=0.9,  # Add nucleus sampling for stability
                    top_k=50,   # Add top-k sampling for stability
                    repetition_penalty=1.1,  # Prevent repetition
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # Add numerical stability
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False  # Don't return scores to avoid numerical issues
                )
                
                # Handle different output formats
                if hasattr(outputs, 'sequences'):
                    generated_ids = outputs.sequences[0]
                else:
                    generated_ids = outputs[0]
                
                # Decode the generated text with error handling
                try:
                    generated_text = tokenizer.decode(
                        generated_ids[inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    return generated_text.strip()
                except Exception as decode_error:
                    print(f"⚠️  Error decoding generated text: {decode_error}")
                    return "[ERROR: Failed to decode generated text]"
                
            except RuntimeError as e:
                error_msg = str(e)
                if any(keyword in error_msg.lower() for keyword in [
                    "probability tensor", "device-side assert", "nan", "inf", 
                    "invalid", "assertion failed"
                ]):
                    print(f"⚠️  NUMERICAL INSTABILITY DETECTED during generation")
                    print(f"   Error: {error_msg}")
                    print(f"   This indicates the model needs more training or better hyperparameters")
                    return "[ERROR: Numerical instability - model needs more training]"
                else:
                    print(f"⚠️  Runtime error during generation: {error_msg}")
                    return f"[ERROR: {error_msg}]"
            except Exception as e:
                print(f"⚠️  Unexpected error during generation: {str(e)}")
                return f"[ERROR: {str(e)}]"
                    
    except Exception as e:
        print(f"⚠️  Error during generation: {str(e)}")
        return f"[ERROR: {str(e)}]"


def check_model_health(model, tokenizer: AutoTokenizer) -> bool:
    """
    Check if the model is healthy and ready for evaluation.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        
    Returns:
        True if model is healthy, False otherwise
    """
    print("🔍 Checking model health...")
    
    try:
        # Test with a simple input
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=False, padding=False)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Test forward pass
            outputs = model(**inputs)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                
                # Check for numerical issues
                if torch.isnan(logits).any():
                    print("⚠️  Model logits contain NaN values")
                    return False
                
                if torch.isinf(logits).any():
                    print("⚠️  Model logits contain Inf values")
                    return False
                
                # Check for reasonable logit values
                logit_range = logits.max().item() - logits.min().item()
                if logit_range > 1000:
                    print(f"⚠️  Model logits have extreme range: {logit_range:.2f}")
                    return False
                
                print("✓ Model health check passed")
                return True
            else:
                print("⚠️  Model output doesn't have logits attribute")
                return False
                
    except Exception as e:
        print(f"⚠️  Model health check failed: {str(e)}")
        return False


def evaluate_summaries(model, tokenizer: AutoTokenizer, 
                      test_dataset, num_samples: int = 10) -> Dict[str, Any]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Check model health first
    if not check_model_health(model, tokenizer):
        print("⚠️  Model failed health check. Skipping evaluation.")
        return {
            'samples': [],
            'avg_summary_length': 0,
            'avg_generation_time': 0,
            'error': 'Model failed health check - numerical instability detected'
        }
    
    print(f"Evaluating model on {num_samples} samples...")
    
    # Ensure we don't try to sample more than available
    actual_samples = min(num_samples, len(test_dataset))
    if actual_samples < num_samples:
        print(f"Warning: Requested {num_samples} samples but only {len(test_dataset)} available. Using {actual_samples} samples.")
    
    results = {
        'samples': [],
        'avg_summary_length': 0,
        'avg_generation_time': 0
    }
    
    generation_times = []
    summary_lengths = []
    
    # Sample from test dataset - convert numpy integers to Python integers
    indices = np.random.choice(len(test_dataset), actual_samples, replace=False).tolist()
    
    for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        example = test_dataset[idx]
        
        # Check if this is a preprocessed dataset (tokenized) or raw dataset
        if 'input_ids' in example:
            # This is a preprocessed dataset - we need to decode the input
            input_ids = example['input_ids']
            # Decode the input tokens to get the original text
            original_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            # Check if highlights field is available in preprocessed dataset
            if 'highlights' in example:
                target_summary = example['highlights']
            else:
                target_summary = "[Target not available in preprocessed dataset]"
        else:
            # This is a raw dataset - use the original fields
            original_text = example.get('article', '')
            target_summary = example.get('highlights', '')
        
        # Debug: Print dataset structure for first example
        if i == 0:
            print(f"Debug - Dataset keys: {list(example.keys())}")
            print(f"Debug - Original text length: {len(original_text)}")
            print(f"Debug - Original text preview: {original_text[:100]}...")
        
        # Generate summary
        start_time = time.time()
        generated_summary = generate_summary(model, tokenizer, original_text)
        generation_time = time.time() - start_time
        
        # Store results
        sample_result = {
            'original_text': original_text[:200] + "..." if len(original_text) > 200 else original_text,
            'target_summary': target_summary,
            'generated_summary': generated_summary,
            'generation_time': generation_time
        }
        
        results['samples'].append(sample_result)
        generation_times.append(generation_time)
        summary_lengths.append(len(generated_summary.split()))
    
    # Calculate averages
    results['avg_summary_length'] = np.mean(summary_lengths)
    results['avg_generation_time'] = np.mean(generation_times)
    
    print(f"✓ Evaluation completed")
    print(f"  Average summary length: {results['avg_summary_length']:.1f} words")
    print(f"  Average generation time: {results['avg_generation_time']:.2f} seconds")
    
    return results


def save_evaluation_results(results: Dict[str, Any], output_dir: str):
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Evaluation results saved to {results_file}")


def print_evaluation_samples(results: Dict[str, Any], num_samples: int = 3):
    """
    Print sample evaluation results.
    
    Args:
        results: Evaluation results
        num_samples: Number of samples to print
    """
    print("\n" + "="*80)
    print("EVALUATION SAMPLES")
    print("="*80)
    
    for i, sample in enumerate(results['samples'][:num_samples]):
        print(f"\nSample {i + 1}:")
        print(f"Original Text: {sample['original_text']}")
        print(f"Target Summary: {sample['target_summary']}")
        print(f"Generated Summary: {sample['generated_summary']}")
        print(f"Generation Time: {sample['generation_time']:.2f}s")
        print("-" * 80)


def calculate_model_size_mb(model) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_size_info(model):
    """
    Print model size information.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = calculate_model_size_mb(model)
    
    print(f"\nModel Size Information:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {model_size_mb:.2f} MB")
    print(f"  Trainable Ratio: {trainable_params/total_params:.2%}")


def create_sample_summaries(model, tokenizer: AutoTokenizer, 
                           sample_texts: List[str], output_dir: str):
    """
    Create sample summaries for given texts.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        sample_texts: List of sample texts
        output_dir: Output directory
    """
    print("Creating sample summaries...")
    
    summaries = []
    
    for i, text in enumerate(tqdm(sample_texts, desc="Generating summaries")):
        summary = generate_summary(model, tokenizer, text)
        
        summaries.append({
            'text_id': i + 1,
            'original_text': text,
            'generated_summary': summary
        })
    
    # Save summaries
    summaries_file = os.path.join(output_dir, "sample_summaries.json")
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    # Print summaries
    print("\n" + "="*80)
    print("SAMPLE SUMMARIES")
    print("="*80)
    
    for summary in summaries:
        print(f"\nText {summary['text_id']}:")
        print(f"Original: {summary['original_text'][:200]}...")
        print(f"Summary: {summary['generated_summary']}")
        print("-" * 80)
    
    print(f"\n✓ Sample summaries saved to {summaries_file}")


def check_gpu_memory():
    """
    Check GPU memory usage.
    
    Returns:
        Dictionary with GPU memory info
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_info = {
        "gpu_available": True,
        "gpu_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
        "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
        "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    }
    
    return gpu_info


def print_gpu_info():
    """Print GPU information."""
    gpu_info = check_gpu_memory()
    
    if not gpu_info["gpu_available"]:
        print("GPU not available")
        return
    
    print(f"\nGPU Information:")
    print(f"  Device: {gpu_info['device_name']}")
    print(f"  Memory Allocated: {gpu_info['memory_allocated']:.2f} GB")
    print(f"  Memory Reserved: {gpu_info['memory_reserved']:.2f} GB")
    print(f"  Total Memory: {gpu_info['memory_total']:.2f} GB")
    print(f"  Memory Usage: {gpu_info['memory_allocated']/gpu_info['memory_total']:.1%}") 


def detect_device(prefer_mps: bool = True) -> str:
    """
    Detect the best available device: CUDA (GPU), MPS (Apple Silicon), or CPU.
    Args:
        prefer_mps: If True, prefer MPS over CPU if CUDA is not available.
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if prefer_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def print_device_info():
    """
    Print information about the detected device.
    """
    device = detect_device()
    print(f"\nDevice detected: {device.upper()}")
    if device == 'cuda':
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif device == 'mps':
        print("  Using Apple Silicon MPS backend (Metal Performance Shaders)")
    else:
        print("  Using CPU (no GPU/MPS available)") 