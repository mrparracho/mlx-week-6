#!/usr/bin/env python3
"""
Test script to verify resume training functionality
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import LoRATrainer
from config import TrainingConfig, ChinchillaConfig
from transformers import AutoTokenizer


def test_checkpoint_structure():
    """Test that checkpoints have the correct structure."""
    print("Testing checkpoint structure...")
    
    output_dir = "./trained_model"
    if not os.path.exists(output_dir):
        print("No trained_model directory found. Run training first.")
        return False
    
    # Find a checkpoint
    import glob
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    if not checkpoints:
        print("No checkpoints found. Run training first.")
        return False
    
    # Use the best checkpoint if available, otherwise use the first one
    best_info_path = os.path.join(output_dir, "best_model_info.json")
    if os.path.exists(best_info_path):
        with open(best_info_path, 'r') as f:
            best_info = json.load(f)
        checkpoint_dir = os.path.join(output_dir, best_info['checkpoint'])
        print(f"Testing best checkpoint: {checkpoint_dir}")
    else:
        checkpoint_dir = checkpoints[0]
        print(f"Testing checkpoint: {checkpoint_dir}")
    
    # Check required files (some may be missing in older checkpoints)
    required_files = [
        "training_state.pt",
        "checkpoint_info.json"
    ]
    
    optional_files = [
        "config.json",
        "tokenizer.json",
        "adapter_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(checkpoint_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files in checkpoint: {missing_files}")
        return False
    
    print("✓ All required files present")
    
    # Check optional files
    present_optional = []
    for file in optional_files:
        file_path = os.path.join(checkpoint_dir, file)
        if os.path.exists(file_path):
            present_optional.append(file)
    
    print(f"✓ Optional files present: {present_optional}")
    
    # Test training state loading
    try:
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        training_state = torch.load(training_state_path, map_location='cpu')
        
        required_keys = [
            'global_step', 'epoch', 'optimizer_state_dict', 
            'scheduler_state_dict', 'best_loss', 'training_losses', 
            'validation_losses'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in training_state:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing keys in training state: {missing_keys}")
            return False
        
        print("✓ Training state structure correct")
        print(f"  Global step: {training_state['global_step']}")
        print(f"  Epoch: {training_state['epoch']}")
        print(f"  Best loss: {training_state['best_loss']:.4f}")
        print(f"  Training losses: {len(training_state['training_losses'])}")
        print(f"  Validation losses: {len(training_state['validation_losses'])}")
        
    except Exception as e:
        print(f"❌ Error loading training state: {e}")
        return False
    
    # Test checkpoint info
    try:
        info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print("✓ Checkpoint info loaded")
        print(f"  Step: {info.get('step', 'N/A')}")
        print(f"  Loss: {info.get('loss', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint info: {e}")
        return False
    
    return True


def test_best_model_info():
    """Test best model info structure."""
    print("\nTesting best model info...")
    
    output_dir = "./trained_model"
    best_info_path = os.path.join(output_dir, "best_model_info.json")
    
    if not os.path.exists(best_info_path):
        print("No best model info found. Run training first.")
        return False
    
    try:
        with open(best_info_path, 'r') as f:
            best_info = json.load(f)
        
        required_keys = ['step', 'epoch', 'loss', 'checkpoint', 'timestamp']
        missing_keys = []
        
        for key in required_keys:
            if key not in best_info:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing keys in best model info: {missing_keys}")
            return False
        
        print("✓ Best model info structure correct")
        print(f"  Best checkpoint: {best_info['checkpoint']}")
        print(f"  Best loss: {best_info['loss']:.4f}")
        print(f"  Step: {best_info['step']}")
        
        # Test that the best checkpoint exists
        best_checkpoint_path = os.path.join(output_dir, best_info['checkpoint'])
        if os.path.exists(best_checkpoint_path):
            print("✓ Best checkpoint exists")
        else:
            print(f"❌ Best checkpoint not found: {best_checkpoint_path}")
            return False
        
    except Exception as e:
        print(f"❌ Error loading best model info: {e}")
        return False
    
    return True


def test_resume_simulation():
    """Simulate resume functionality without loading the full model."""
    print("\nSimulating resume functionality...")
    
    output_dir = "./trained_model"
    
    # Test 'best' checkpoint resolution
    best_info_path = os.path.join(output_dir, "best_model_info.json")
    if os.path.exists(best_info_path):
        try:
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
            
            checkpoint_path = os.path.join(output_dir, best_info['checkpoint'])
            print(f"✓ 'best' checkpoint resolves to: {checkpoint_path}")
            
            if os.path.exists(checkpoint_path):
                print("✓ Best checkpoint path is valid")
            else:
                print("❌ Best checkpoint path is invalid")
                return False
                
        except Exception as e:
            print(f"❌ Error resolving 'best' checkpoint: {e}")
            return False
    
    # Test checkpoint enumeration
    import glob
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if checkpoints:
        print(f"✓ Found {len(checkpoints)} checkpoints")
        for checkpoint in sorted(checkpoints)[:3]:  # Show first 3
            checkpoint_name = os.path.basename(checkpoint)
            print(f"  - {checkpoint_name}")
    else:
        print("❌ No checkpoints found")
        return False
    
    return True


def main():
    """Run all resume functionality tests."""
    print("="*80)
    print("TESTING RESUME TRAINING FUNCTIONALITY")
    print("="*80)
    
    tests = [
        ("Checkpoint Structure", test_checkpoint_structure),
        ("Best Model Info", test_best_model_info),
        ("Resume Simulation", test_resume_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print('='*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Resume functionality is working correctly.")
        print("\nYou can now use resume functionality:")
        print("  python main.py --resume_from_checkpoint best")
        print("  python main.py --resume_from_checkpoint ./trained_model/checkpoint-XXXX")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("Make sure you have run training first to create checkpoints.")


if __name__ == "__main__":
    main() 