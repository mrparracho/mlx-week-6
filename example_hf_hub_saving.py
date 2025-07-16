#!/usr/bin/env python3
"""
Example script demonstrating how to save models to HuggingFace Hub
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """Example of saving models to HuggingFace Hub."""
    
    print("="*80)
    print("HUGGINGFACE HUB SAVING EXAMPLE")
    print("="*80)
    
    # Example 1: Basic usage with environment variable
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic usage with HF_TOKEN environment variable")
    print("="*60)
    
    print("1. Set your HuggingFace token as an environment variable:")
    print("   export HF_TOKEN=your_token_here")
    print()
    print("2. Run training with HuggingFace Hub saving:")
    print("   python main.py --hf_repo_name your-username/your-model-name")
    print()
    print("3. The model will be automatically saved to Hub when a new best model is found")
    
    # Example 2: Resume training with Hub saving
    print("\n" + "="*60)
    print("EXAMPLE 2: Resume training with HuggingFace Hub saving")
    print("="*60)
    
    print("1. Resume training and save to Hub:")
    print("   python main.py --resume_from_checkpoint best --hf_repo_name your-username/your-model-name")
    print()
    print("2. This will continue training and save new best models to Hub")
    
    # Example 3: Different repository names
    print("\n" + "="*60)
    print("EXAMPLE 3: Different repository naming conventions")
    print("="*60)
    
    print("Public repositories:")
    print("  python main.py --hf_repo_name your-username/qwen-lora-summarization")
    print("  python main.py --hf_repo_name your-username/qwen-0.5b-cnn-dailymail")
    print()
    print("Private repositories (set private=True in code):")
    print("  python main.py --hf_repo_name your-username/private-model")
    print()
    print("Organization repositories:")
    print("  python main.py --hf_repo_name your-org/qwen-summarization")
    
    # Example 4: Complete workflow
    print("\n" + "="*60)
    print("EXAMPLE 4: Complete workflow")
    print("="*60)
    
    print("1. Set up your environment:")
    print("   export HF_TOKEN=your_token_here")
    print()
    print("2. Start training with Hub saving:")
    print("   python main.py \\")
    print("     --epochs 3 \\")
    print("     --learning_rate 1e-4 \\")
    print("     --hf_repo_name your-username/qwen-summarization-v1")
    print()
    print("3. Monitor training - best models will be saved automatically")
    print()
    print("4. Resume training if needed:")
    print("   python main.py \\")
    print("     --resume_from_checkpoint best \\")
    print("     --epochs 5 \\")
    print("     --hf_repo_name your-username/qwen-summarization-v1")
    print()
    print("5. Your model will be available at:")
    print("   https://huggingface.co/your-username/qwen-summarization-v1")
    
    # Example 5: Using the saved model
    print("\n" + "="*60)
    print("EXAMPLE 5: Using the saved model")
    print("="*60)
    
    print("After training, you can use your model like this:")
    print()
    print("```python")
    print("from transformers import AutoTokenizer, AutoModelForCausalLM")
    print("from peft import PeftModel")
    print()
    print("# Load the base model")
    print("base_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')")
    print()
    print("# Load your LoRA adapter")
    print("model = PeftModel.from_pretrained(base_model, 'your-username/qwen-summarization-v1')")
    print()
    print("# Load the tokenizer")
    print("tokenizer = AutoTokenizer.from_pretrained('your-username/qwen-summarization-v1')")
    print()
    print("# Generate a summary")
    print("text = 'Your input text here...'")
    print("inputs = tokenizer(f'Summarize: {text}\\nTarget:', return_tensors='pt')")
    print("outputs = model.generate(**inputs, max_length=128)")
    print("summary = tokenizer.decode(outputs[0], skip_special_tokens=True)")
    print("print(summary)")
    print("```")
    
    # Check if HF_TOKEN is set
    print("\n" + "="*60)
    print("CURRENT SETUP CHECK")
    print("="*60)
    
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("✓ HF_TOKEN environment variable is set")
        print(f"  Token: {hf_token[:8]}...{hf_token[-4:] if len(hf_token) > 12 else '***'}")
    else:
        print("❌ HF_TOKEN environment variable is not set")
        print("   Set it with: export HF_TOKEN=your_token_here")
        print("   Get your token from: https://huggingface.co/settings/tokens")
    
    print("\n" + "="*80)
    print("✓ HuggingFace Hub saving is now integrated into your training pipeline!")
    print("="*80)


if __name__ == "__main__":
    main() 