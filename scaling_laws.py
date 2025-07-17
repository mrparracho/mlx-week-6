"""
Chinchilla Scaling Laws Implementation
Based on "Training Compute-Optimal Large Language Models" by Hoffmann et al.
"""

import math
from typing import Tuple, Optional


def calculate_optimal_model_size(compute_budget: float) -> float:
    """
    Calculate optimal model size in parameters using Chinchilla scaling laws.
    
    Args:
        compute_budget: Compute budget in FLOPs
        
    Returns:
        Optimal model size in parameters
    """
    # Chinchilla formula: N_opt = 0.6 * C^0.5
    return 0.6 * (compute_budget ** 0.5)


def calculate_optimal_training_tokens(compute_budget: float) -> float:
    """
    Calculate optimal number of training tokens using Chinchilla scaling laws.
    
    Args:
        compute_budget: Compute budget in FLOPs
        
    Returns:
        Optimal number of training tokens
    """
    # Chinchilla formula: D_opt = 20 * C^0.5
    return 20 * (compute_budget ** 0.5)


def estimate_compute_budget(model_size: float, training_tokens: float) -> float:
    """
    Estimate compute budget in FLOPs for given model size and training tokens.
    
    Args:
        model_size: Model size in parameters
        training_tokens: Number of training tokens
        
    Returns:
        Compute budget in FLOPs
    """
    # C ≈ 6 * N * D (approximate formula)
    return 6 * model_size * training_tokens


def calculate_lora_parameters(base_model_size: float, lora_rank: int, 
                            target_modules_ratio: float = 0.1) -> float:
    """
    Calculate LoRA parameter count for given base model and rank.
    
    Args:
        base_model_size: Base model size in parameters
        lora_rank: LoRA rank
        target_modules_ratio: Ratio of parameters in target modules
        
    Returns:
        Number of LoRA parameters
    """
    # LoRA adds 2 * rank * (input_dim + output_dim) per layer
    # Approximate: 2 * rank * (base_model_size * target_modules_ratio)
    return 2 * lora_rank * (base_model_size * target_modules_ratio)


def get_chinchilla_optimal_params(available_tokens: float, 
                                 base_model_size: float) -> Tuple[float, float, float]:
    """
    Get optimal parameters based on Chinchilla scaling laws.
    
    Args:
        available_tokens: Available training tokens
        base_model_size: Base model size in parameters
        
    Returns:
        Tuple of (optimal_model_size, optimal_tokens, compute_budget)
    """
    # Calculate current compute budget
    current_compute = estimate_compute_budget(base_model_size, available_tokens)
    
    # Calculate optimal parameters
    optimal_model_size = calculate_optimal_model_size(current_compute)
    optimal_tokens = calculate_optimal_training_tokens(current_compute)
    
    return optimal_model_size, optimal_tokens, current_compute


def analyze_scaling_efficiency(available_tokens: float, trainable_params: float, base_model_size: Optional[float] = None) -> dict:
    """
    Analyze scaling efficiency and provide recommendations for LoRA training.
    
    Args:
        available_tokens: Available training tokens
        trainable_params: Number of trainable LoRA parameters
        base_model_size: Base model size in parameters (for reference)
        
    Returns:
        Dictionary with analysis results
    """
    optimal_model_size, optimal_tokens, compute_budget = get_chinchilla_optimal_params(
        available_tokens, trainable_params
    )
    
    # Calculate efficiency metrics
    model_efficiency = trainable_params / optimal_model_size if optimal_model_size > 0 else float('inf')
    token_efficiency = available_tokens / optimal_tokens if optimal_tokens > 0 else float('inf')
    
    analysis = {
        "trainable_params": trainable_params,
        "base_model_size": base_model_size,
        "optimal_model_size": optimal_model_size,
        "available_tokens": available_tokens,
        "optimal_tokens": optimal_tokens,
        "compute_budget": compute_budget,
        "model_efficiency": model_efficiency,
        "token_efficiency": token_efficiency,
        "recommendations": []
    }
    
    # Generate recommendations
    if model_efficiency > 1.2:
        analysis["recommendations"].append(
            f"Consider reducing LoRA rank. Current trainable: {trainable_params:.1e}, Optimal: {optimal_model_size:.1e}"
        )
    elif model_efficiency < 0.8:
        analysis["recommendations"].append(
            f"Consider increasing LoRA rank. Current trainable: {trainable_params:.1e}, Optimal: {optimal_model_size:.1e}"
        )
    
    if token_efficiency < 0.8:
        analysis["recommendations"].append(
            f"Consider using more training data. Available: {available_tokens:.1e}, Optimal: {optimal_tokens:.1e}"
        )
    elif token_efficiency > 1.2:
        analysis["recommendations"].append(
            f"Consider using less training data. Available: {available_tokens:.1e}, Optimal: {optimal_tokens:.1e}"
        )
    
    return analysis


def format_scientific_notation(value: float) -> str:
    """Format large numbers in scientific notation."""
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.2f}"


def print_scaling_analysis(analysis: dict):
    """Print scaling analysis in a readable format."""
    print("\n" + "="*60)
    print("CHINCHILLA SCALING LAWS ANALYSIS (LoRA)")
    print("="*60)
    print(f"Trainable LoRA Parameters: {format_scientific_notation(analysis['trainable_params'])}")
    if analysis.get('base_model_size'):
        print(f"Base Model Size: {format_scientific_notation(analysis['base_model_size'])} parameters")
    print(f"Optimal Model Size: {format_scientific_notation(analysis['optimal_model_size'])} parameters")
    print(f"Available Tokens: {format_scientific_notation(analysis['available_tokens'])}")
    print(f"Optimal Tokens: {format_scientific_notation(analysis['optimal_tokens'])}")
    print(f"Compute Budget: {format_scientific_notation(analysis['compute_budget'])} FLOPs")
    print(f"Model Efficiency: {analysis['model_efficiency']:.2f}x")
    print(f"Token Efficiency: {analysis['token_efficiency']:.2f}x")
    
    if analysis['recommendations']:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"{i}. {rec}")
    else:
        print("\n✓ Configuration is close to optimal!")
    
    print("="*60) 