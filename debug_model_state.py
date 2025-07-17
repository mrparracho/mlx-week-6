#!/usr/bin/env python3
"""
Debug script to analyze the current model state and explain the issues.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def analyze_model_state():
    """Analyze the current model state and explain issues."""
    
    print("🔍 ANALYZING MODEL STATE")
    print("=" * 50)
    
    # Load training history
    with open("trained_model/training_history.json", "r") as f:
        history = json.load(f)
    
    training_losses = history["training_losses"]
    validation_losses = history["validation_losses"]
    
    print(f"📊 Training Statistics:")
    print(f"  Total training steps: {len(training_losses)}")
    print(f"  Initial training loss: {training_losses[0]:.2f}")
    print(f"  Final training loss: {training_losses[-1]:.2f}")
    print(f"  Best validation loss: {min(validation_losses):.2f}")
    print(f"  Final validation loss: {validation_losses[-1]:.2f}")
    
    print(f"\n📈 Loss Analysis:")
    print(f"  Training loss improvement: {training_losses[0] - training_losses[-1]:.2f}")
    print(f"  Validation loss improvement: {validation_losses[0] - validation_losses[-1]:.2f}")
    
    # Check if model is under-trained
    print(f"\n⚠️  ISSUES IDENTIFIED:")
    
    if training_losses[-1] > 5.0:
        print(f"  ❌ Training loss is still too high ({training_losses[-1]:.2f})")
        print(f"     - Expected: < 3.0 for coherent generation")
        print(f"     - Current: {training_losses[-1]:.2f}")
    
    if validation_losses[-1] > 5.0:
        print(f"  ❌ Validation loss is still too high ({validation_losses[-1]:.2f})")
        print(f"     - Expected: < 3.0 for coherent generation")
        print(f"     - Current: {validation_losses[-1]:.2f}")
    
    if len(training_losses) < 10000:
        print(f"  ❌ Model is under-trained")
        print(f"     - Current steps: {len(training_losses)}")
        print(f"     - Recommended: > 50,000 steps for summarization")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. Continue training for at least 5 more epochs")
    print(f"  2. Use a lower learning rate (5e-4 instead of 1e-2)")
    print(f"  3. Monitor validation loss - stop when it plateaus")
    print(f"  4. Expected final loss should be < 3.0 for good generation")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"  Run: python continue_training.py")
    print(f"  This will train for 5 more epochs with better parameters")

def test_model_generation():
    """Test the current model's generation capabilities."""
    
    print(f"\n🧪 TESTING MODEL GENERATION")
    print("=" * 50)
    
    try:
        # Load model and tokenizer
        print("Loading model...")
        base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        model = PeftModel.from_pretrained(base_model, "trained_model/checkpoint-50000")
        tokenizer = AutoTokenizer.from_pretrained("trained_model/checkpoint-50000")
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Test generation
        test_text = "Summarize: The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"Test input: {test_text}")
        print(f"Generated output: {generated_text}")
        
        # Check for nonsensical output
        if any(char in generated_text for char in ['@', '(', ')', '界', '砀', '阗']):
            print(f"❌ Model is generating nonsensical characters")
            print(f"   This confirms the model needs more training")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    analyze_model_state()
    test_model_generation() 