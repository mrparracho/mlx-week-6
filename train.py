"""
Training module for Qwen2.5 LoRA fine-tuning with Chinchilla scaling laws
"""

import os
import time
import random
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import Dataset
from tqdm import tqdm
import numpy as np
from config import TrainingConfig, ChinchillaConfig
from scaling_laws import analyze_scaling_efficiency, print_scaling_analysis
from data import collate_fn


class HuggingFaceHubSaver:
    """Class to handle saving models to Hugging Face Hub."""
    
    def __init__(self, repo_name: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize the HuggingFace Hub saver.
        
        Args:
            repo_name: Repository name on Hugging Face Hub (e.g., "username/model-name")
            token: Hugging Face token (if None, will try to get from HF_TOKEN env var)
        """
        self.repo_name = repo_name
        self.token = token or os.getenv('HF_TOKEN')
        self.enabled = bool(self.token and self.repo_name)
        
        if self.enabled:
            print(f"✓ HuggingFace Hub saving enabled for: {self.repo_name}")
        else:
            if not self.token:
                print("⚠️  No HuggingFace token found. Set HF_TOKEN environment variable to enable Hub saving.")
            if not self.repo_name:
                print("⚠️  No repository name provided. Set repo_name to enable Hub saving.")
    
    def save_to_hub(self, model, tokenizer, training_config: TrainingConfig, 
                   best_loss: float, global_step: int, epoch: int) -> bool:
        """
        Save the model to Hugging Face Hub.
        
        Args:
            model: The trained model
            tokenizer: The tokenizer
            training_config: Training configuration
            best_loss: Best validation loss achieved
            global_step: Current global step
            epoch: Current epoch
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            print(f"\n🚀 Saving model to Hugging Face Hub: {self.repo_name}")
            
            # Create commit message with training info
            commit_message = f"LoRA fine-tuned model - Loss: {best_loss:.4f}, Step: {global_step}, Epoch: {epoch}"
            
            # Save model
            model.push_to_hub(
                self.repo_name,
                token=self.token,
                commit_message=commit_message,
                private=False  # Set to True if you want a private repository
            )
            
            # Save tokenizer
            tokenizer.push_to_hub(
                self.repo_name,
                token=self.token,
                commit_message=f"Tokenizer for {commit_message}"
            )
            
            # Create and save model card
            self._create_model_card(training_config, best_loss, global_step, epoch)
            
            print(f"✓ Model successfully saved to: https://huggingface.co/{self.repo_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save model to Hugging Face Hub: {e}")
            return False
    
    def _create_model_card(self, training_config: TrainingConfig, best_loss: float, 
                          global_step: int, epoch: int):
        """Create a model card with training information."""
        try:
            model_card_content = f"""---
language:
- en
license: mit
tags:
- text-generation
- summarization
- lora
- qwen
---

# LoRA Fine-tuned Qwen2.5 Model for Summarization

This model is a LoRA fine-tuned version of Qwen2.5-0.5B for text summarization tasks.

## Training Information

- **Base Model**: Qwen/Qwen2.5-0.5B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Best Validation Loss**: {best_loss:.4f}
- **Training Steps**: {global_step}
- **Epochs**: {epoch}
- **Learning Rate**: {training_config.learning_rate}
- **Batch Size**: {training_config.batch_size}
- **LoRA Rank**: 16
- **LoRA Alpha**: 32

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model = PeftModel.from_pretrained(model, "{self.repo_name}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{self.repo_name}")

# Generate summary
input_text = "Your input text here..."
inputs = tokenizer(f"Summarize: {{input_text}}\\nTarget:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Details

This model was fine-tuned using LoRA with the following configuration:
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- LoRA dropout: 0.1
- Weight decay: {training_config.weight_decay}
- Warmup steps: {training_config.warmup_steps}

The model was trained on the CNN/DailyMail dataset for text summarization.
"""
            
            # Save model card
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            if self.repo_name:  # Ensure repo_name is not None
                api.upload_file(
                    path_or_fileobj=model_card_content.encode(),
                    path_in_repo="README.md",
                    repo_id=self.repo_name,
                    commit_message="Add model card"
                )
            
        except Exception as e:
            print(f"Warning: Could not create model card: {e}")


class LoRATrainer:
    """Trainer for LoRA fine-tuning with Chinchilla scaling laws."""
    
    def __init__(self, model, tokenizer: AutoTokenizer, 
                 training_config: TrainingConfig, 
                 chinchilla_config: ChinchillaConfig,
                 resume_from_checkpoint: Optional[str] = None,
                 hf_repo_name: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.chinchilla_config = chinchilla_config
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Initialize HuggingFace Hub saver
        self.hf_saver = HuggingFaceHubSaver(repo_name=hf_repo_name)
        
        # Use CUDA if available, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Move model to device (use to_empty for meta tensors)
        try:
            self.model.to_empty(device=self.device)
            print(f"✓ Model moved to device: {self.device}")
        except Exception as e:
            print(f"Warning: Failed to move model to {self.device}: {e}")
            print("Falling back to CPU...")
            self.device = torch.device("cpu")
            self.model.to_empty(device=self.device)
            print("✓ Model moved to CPU")
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_losses = []
        self.validation_losses = []
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory or 'best' for best model
        """
        import json
        import glob
        import os
        
        print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        
        # Handle 'best' checkpoint
        if checkpoint_path == 'best':
            # Try to find best_model_info.json in common locations
            possible_dirs = [
                './trained_model',
                './output',
                '.',
            ]
            best_info_path = None
            output_dir = None
            for dir_path in possible_dirs:
                test_path = os.path.join(dir_path, 'best_model_info.json')
                if os.path.exists(test_path):
                    best_info_path = test_path
                    output_dir = dir_path
                    break
            if best_info_path is None:
                # Try to find any best_model_info.json in the workspace
                matches = glob.glob('**/best_model_info.json', recursive=True)
                if matches:
                    best_info_path = matches[0]
                    output_dir = os.path.dirname(best_info_path)
            if best_info_path is None:
                raise ValueError("No best model info found. Cannot resume from 'best' checkpoint.")
            assert output_dir is not None, "Output directory for best checkpoint could not be determined."
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
                checkpoint_path = os.path.join(output_dir, best_info['checkpoint'])
                print(f"Found best checkpoint: {checkpoint_path}")
        
        # Load model and tokenizer
        try:
            print(f"Loading model from {checkpoint_path}")
            # Try to detect if this is a PEFT/LoRA checkpoint
            import os
            from transformers import AutoModelForCausalLM
            model_loaded = False
            try:
                from peft import PeftModel, PeftConfig
                peft_config_path = os.path.join(checkpoint_path, 'adapter_config.json')
                if os.path.exists(peft_config_path):
                    print("Detected PEFT/LoRA checkpoint. Loading base model and applying LoRA adapter...")
                    peft_config = PeftConfig.from_pretrained(checkpoint_path)
                    if not peft_config.base_model_name_or_path:
                        raise ValueError("PEFT config is missing base_model_name_or_path. Cannot load base model.")
                    base_model = AutoModelForCausalLM.from_pretrained(str(peft_config.base_model_name_or_path))
                    # Load the PEFT adapter
                    self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
                    model_loaded = True
            except ImportError:
                print("peft not installed, skipping PEFT/LoRA loading.")
            if not model_loaded:
                # Fallback to standard HuggingFace model loading
                self.model = self.model.__class__.from_pretrained(checkpoint_path)
            self.tokenizer = self.tokenizer.__class__.from_pretrained(checkpoint_path)
            
            # Move model to device
            self.model.to_empty(device=self.device)
            print(f"✓ Model loaded and moved to {self.device}")
            
            # Ensure model is in training mode and parameters are trainable
            self.model.train()
            
            # For PEFT/LoRA models, ensure adapter parameters are trainable
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                print("⚠️  No trainable parameters found. Attempting to fix...")
                # Try to enable training for LoRA parameters
                for name, param in self.model.named_parameters():
                    if any(keyword in name.lower() for keyword in ['lora', 'adapter', 'peft']):
                        param.requires_grad = True
                        print(f"  Enabled training for: {name}")
                
                # Check again
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                if len(trainable_params) == 0:
                    raise ValueError("No trainable parameters found after loading checkpoint. Model may not be properly configured for training.")
                else:
                    print(f"✓ Found {len(trainable_params)} trainable parameters")
            else:
                print(f"✓ Found {len(trainable_params)} trainable parameters")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
        if os.path.exists(training_state_path):
            try:
                print(f"Loading training state from {training_state_path}")
                training_state = torch.load(training_state_path, map_location=self.device)
                
                # Restore training state
                self.global_step = training_state.get('global_step', 0)
                self.epoch = training_state.get('epoch', 0)
                self.best_loss = training_state.get('best_loss', float('inf'))
                self.training_losses = training_state.get('training_losses', [])
                self.validation_losses = training_state.get('validation_losses', [])
                
                print(f"✓ Training state restored:")
                print(f"  Global step: {self.global_step}")
                print(f"  Epoch: {self.epoch}")
                print(f"  Best loss: {self.best_loss:.4f}")
                print(f"  Training losses: {len(self.training_losses)}")
                print(f"  Validation losses: {len(self.validation_losses)}")
                
                # Note: Optimizer and scheduler will be recreated in setup_optimizer_and_scheduler
                # with the correct state loaded
                
            except Exception as e:
                print(f"Warning: Could not load training state: {e}")
                print("Starting from beginning...")
        else:
            print("No training state found. Starting from beginning...")
        
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Set up optimizer and learning rate scheduler with resume support.
        
        Args:
            num_training_steps: Total number of training steps
        """
        print("Setting up optimizer and scheduler...")
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Create scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Load optimizer and scheduler state if resuming
        if self.resume_from_checkpoint:
            training_state_path = os.path.join(self.resume_from_checkpoint, 'training_state.pt')
            if os.path.exists(training_state_path):
                try:
                    training_state = torch.load(training_state_path, map_location=self.device)
                    
                    # Load optimizer state
                    if 'optimizer_state_dict' in training_state:
                        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                        print("✓ Optimizer state restored")
                    
                    # Load scheduler state
                    if 'scheduler_state_dict' in training_state and self.scheduler:
                        self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                        print("✓ Scheduler state restored")
                        
                except Exception as e:
                    print(f"Warning: Could not restore optimizer/scheduler state: {e}")
        
        print(f"✓ Optimizer and scheduler created")
        print(f"  Learning rate: {self.training_config.learning_rate}")
        print(f"  Weight decay: {self.training_config.weight_decay}")
        print(f"  Warmup steps: {self.training_config.warmup_steps}")
        print(f"  Total steps: {num_training_steps}")
        if self.resume_from_checkpoint:
            print(f"  Resuming from step: {self.global_step}")
    
    def create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader for the dataset.
        
        Args:
            dataset: Dataset to create loader for
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader instance
        """
        # Disable multiprocessing on MPS to prevent resource tracker issues
        num_workers = 0 if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else self.training_config.dataloader_num_workers
        pin_memory = False if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else self.training_config.dataloader_pin_memory
        
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    def calculate_loss(self, batch: Dict) -> torch.Tensor:
        """
        Calculate loss for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with gradient scaling for numerical stability
        # Disable mixed precision for stability - use regular forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Get loss and check for numerical issues
        loss = outputs.loss
        
        # Additional numerical stability checks
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  LOSS IS NAN/INF: {loss.item()}")
            # Return a small positive loss instead of NaN/Inf
            return torch.tensor(1.0, device=self.device, requires_grad=True)
        
        # Check if loss is too large (indicates instability)
        if loss.item() > 100.0:
            print(f"⚠️  LOSS TOO LARGE: {loss.item()}")
            # Clip the loss to prevent gradient explosion
            loss = torch.clamp(loss, max=100.0)
        
        return loss
    

    
    def train_step(self, batch: Dict) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Calculate loss with numerical stability checks
        loss = self.calculate_loss(batch)
        
        # Numerical stability checks
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  NUMERICAL INSTABILITY DETECTED at step {self.global_step}")
            print(f"  Loss value: {loss.item()}")
            print(f"  Batch size: {batch['input_ids'].shape[0]}")
            print(f"  Input shape: {batch['input_ids'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            
            # Check for valid labels
            valid_labels = (batch['labels'] != -100).sum().item()
            print(f"  Valid label positions: {valid_labels}")
            
            # Skip this batch if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Skipping batch due to invalid loss")
                return float('inf')  # Return high loss to indicate problem
        
        # Backward pass with gradient stability checks
        loss.backward()
        
        # Check for gradient issues
        grad_norm = 0.0
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_grad = True
                param_grad_norm = param.grad.norm().item()
                # Check for NaN/Inf using Python math functions
                if math.isnan(param_grad_norm) or math.isinf(param_grad_norm):
                    print(f"⚠️  GRADIENT CONTAINS NAN/INF: {param_grad_norm}")
                    # Zero out problematic gradients
                    param.grad.zero_()
                else:
                    grad_norm += param_grad_norm ** 2
        
        if not has_grad:
            print(f"⚠️  NO GRADIENTS FOUND at step {self.global_step}")
            return float('inf')
        
        grad_norm = grad_norm ** 0.5
        
        # Gradient clipping with stability check
        if grad_norm > 0 and not math.isnan(grad_norm) and not math.isinf(grad_norm):
            # Use a more conservative gradient clipping
            clip_value = min(self.training_config.max_grad_norm, grad_norm * 0.5)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                clip_value
            )
        
        # Optimizer step with error handling
        try:
            self.optimizer.step()
            self.scheduler.step()
        except Exception as e:
            print(f"⚠️  OPTIMIZER STEP FAILED: {e}")
            return float('inf')
        
        return loss.item()
    
    def validate(self, val_dataloader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss = self.calculate_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def cleanup_old_checkpoints(self, output_dir: str):
        """
        Clean up old checkpoints to prevent symlink issues.
        
        Args:
            output_dir: Output directory
        """
        import glob
        import shutil
        
        # Find all checkpoint directories
        checkpoint_pattern = os.path.join(output_dir, 'checkpoint-*')
        checkpoints = glob.glob(checkpoint_pattern)
        
        # Keep only the most recent checkpoint and remove others
        if len(checkpoints) > 1:
            # Sort by modification time (newest first)
            checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove old checkpoints (keep only the newest)
            for old_checkpoint in checkpoints[1:]:
                try:
                    shutil.rmtree(old_checkpoint)
                    print(f"✓ Removed old checkpoint: {os.path.basename(old_checkpoint)}")
                except Exception as e:
                    print(f"Warning: Could not remove old checkpoint {old_checkpoint}: {e}")
    
    def save_checkpoint(self, output_dir: str, is_best: bool = False, current_loss: Optional[float] = None):
        """
        Save a checkpoint with smart best model management.
        
        Args:
            output_dir: Directory to save checkpoint
            is_best: Whether this is the best model so far
            current_loss: Current validation loss for comparison
        """
        import shutil
        import json
        
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        # Save checkpoint metadata
        checkpoint_info = {
            'step': self.global_step,
            'epoch': self.epoch,
            'loss': current_loss if current_loss is not None else float('inf'),
            'best_loss': self.best_loss,
            'timestamp': time.time()
        }
        
        with open(os.path.join(checkpoint_dir, 'checkpoint_info.json'), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        if is_best and current_loss is not None:
            self._update_best_model(output_dir, checkpoint_dir, current_loss)
        
        # Clean up old checkpoints (keep only recent ones)
        self._cleanup_old_checkpoints(output_dir)

    def _update_best_model(self, output_dir: str, checkpoint_dir: str, current_loss: float):
        """
        Smart best model update based on loss comparison.
        
        Args:
            output_dir: Output directory
            checkpoint_dir: Current checkpoint directory
            current_loss: Current validation loss
        """
        import shutil
        import json
        
        best_dir = os.path.join(output_dir, 'best')
        best_info_path = os.path.join(output_dir, 'best_model_info.json')
        
        # Check if we have a previous best model
        previous_best_loss = float('inf')
        if os.path.exists(best_info_path):
            try:
                with open(best_info_path, 'r') as f:
                    best_info = json.load(f)
                    previous_best_loss = best_info.get('loss', float('inf'))
            except (json.JSONDecodeError, FileNotFoundError):
                previous_best_loss = float('inf')
        
        # Only update if current loss is better
        if current_loss < previous_best_loss:
            print(f"🏆 New best model! Loss improved from {previous_best_loss:.4f} to {current_loss:.4f}")
            
            # Remove existing best model
            if os.path.exists(best_dir):
                try:
                    if os.path.islink(best_dir):
                        os.unlink(best_dir)
                    else:
                        shutil.rmtree(best_dir)
                except Exception as e:
                    print(f"Warning: Could not remove old best model: {e}")
            
            # Create new best model (try symlink first, fallback to copy)
            try:
                # Try symlink
                checkpoint_name = os.path.basename(checkpoint_dir)
                os.symlink(checkpoint_name, best_dir)
                print("✓ Best model symlinked")
            except OSError:
                # Fallback to copy
                try:
                    shutil.copytree(checkpoint_dir, best_dir)
                    print("✓ Best model copied")
                except Exception as e:
                    print(f"Error copying best model: {e}")
                    return
            
            # Update best model info
            best_info = {
                'step': self.global_step,
                'epoch': self.epoch,
                'loss': current_loss,
                'checkpoint': os.path.basename(checkpoint_dir),
                'timestamp': time.time()
            }
            
            with open(best_info_path, 'w') as f:
                json.dump(best_info, f, indent=2)
            
            print(f"✓ Best model info saved to {best_info_path}")
            
            # Save to HuggingFace Hub if enabled
            if self.hf_saver.enabled:
                print("🔄 Saving new best model to HuggingFace Hub...")
                success = self.hf_saver.save_to_hub(
                    self.model, 
                    self.tokenizer, 
                    self.training_config,
                    current_loss,
                    self.global_step,
                    self.epoch
                )
                if success:
                    print("✓ Best model saved to HuggingFace Hub")
                else:
                    print("⚠️  Failed to save to HuggingFace Hub")
        else:
            print(f"📊 Current loss {current_loss:.4f} not better than best {previous_best_loss:.4f}")

    def _cleanup_old_checkpoints(self, output_dir: str, keep_last: int = 3):
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            output_dir: Output directory
            keep_last: Number of recent checkpoints to keep
        """
        import glob
        import shutil
        
        # Find all checkpoint directories
        checkpoint_pattern = os.path.join(output_dir, 'checkpoint-*')
        checkpoints = glob.glob(checkpoint_pattern)
        
        if len(checkpoints) <= keep_last:
            return
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Remove old checkpoints (keep only the most recent ones)
        for old_checkpoint in checkpoints[keep_last:]:
            try:
                shutil.rmtree(old_checkpoint)
                print(f"🗑️  Removed old checkpoint: {os.path.basename(old_checkpoint)}")
            except Exception as e:
                print(f"Warning: Could not remove old checkpoint {old_checkpoint}: {e}")

    def get_best_model_info(self, output_dir: str) -> dict:
        """
        Get information about the best model.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary with best model information
        """
        import json
        
        best_info_path = os.path.join(output_dir, 'best_model_info.json')
        
        if os.path.exists(best_info_path):
            try:
                with open(best_info_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return {
            'step': 0,
            'epoch': 0,
            'loss': float('inf'),
            'checkpoint': None,
            'timestamp': None
        }
    
    def _cleanup_multiprocessing_resources(self):
        """
        Comprehensive cleanup of multiprocessing resources at the end of training.
        This handles DataLoader workers and other multiprocessing resources.
        """
        try:
            import os
            import signal
            import psutil
            import multiprocessing
            
            print("🧹 Cleaning up multiprocessing resources...")
            
            # Clean up any remaining child processes
            current_pid = os.getpid()
            process = psutil.Process(current_pid)
            children = process.children(recursive=True)
            
            if children:
                print(f"  Found {len(children)} child processes to clean up")
                for child in children:
                    try:
                        child.terminate()
                        child.wait(timeout=2)
                    except:
                        try:
                            child.kill()
                            child.wait(timeout=1)
                        except:
                            pass
            
            # Clean up multiprocessing resource tracker
            try:
                if hasattr(multiprocessing, 'resource_tracker'):
                    multiprocessing.resource_tracker._CLEANUP_CALLED = True
            except:
                pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("✓ Multiprocessing resources cleaned up")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up multiprocessing resources: {e}")
    
    def apply_chinchilla_scaling(self, train_dataset: Dataset) -> Dict:
        """
        Apply Chinchilla scaling laws analysis.
        
        Args:
            train_dataset: Training dataset
            
        Returns:
            Scaling analysis results
        """
        if not self.chinchilla_config.apply_scaling_laws:
            return {}
        
        print("\nApplying Chinchilla scaling laws...")
        
        # Calculate total tokens in training dataset - use efficient sampling for large datasets
        dataset_size = len(train_dataset)
        if dataset_size > 10000:
            # For large datasets, sample to estimate total tokens
            sample_size = min(1000, dataset_size // 100)  # Sample 1% or 1000 examples
            sample_indices = random.sample(range(dataset_size), sample_size)
            sample_dataset = train_dataset.select(sample_indices)
            
            # Fix the dataset access issue
            sample_tokens = 0
            for i in range(len(sample_dataset)):
                example = sample_dataset[i]
                if 'input_ids' in example:
                    sample_tokens += len(example['input_ids'])
            
            total_tokens = int(sample_tokens * (dataset_size / sample_size))
            print(f"Estimated total tokens from {sample_size} sample examples")
        else:
            # For small datasets, count all tokens
            total_tokens = 0
            for i in range(len(train_dataset)):
                example = train_dataset[i]
                if 'input_ids' in example:
                    total_tokens += len(example['input_ids'])
        
        # Check if base_model_size is available
        if self.chinchilla_config.base_model_size is None:
            print("Warning: base_model_size not set, skipping scaling analysis")
            return {}
        
        # Calculate trainable parameters for LoRA
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Analyze scaling efficiency using trainable LoRA parameters and base model size
        analysis = analyze_scaling_efficiency(
            available_tokens=total_tokens,
            trainable_params=trainable_params,
            base_model_size=total_params
        )
        
        # Print analysis
        print_scaling_analysis(analysis)
        
        return analysis
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              output_dir: str) -> Dict[str, Union[List[float], Dict]]:
        """
        Main training loop with resume support.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for saving models
            
        Returns:
            Dictionary with training history
        """
        print("\n" + "="*60)
        if self.resume_from_checkpoint:
            print("RESUMING TRAINING")
        else:
            print("STARTING TRAINING")
        print("="*60)
        
        # Apply Chinchilla scaling analysis
        scaling_analysis = self.apply_chinchilla_scaling(train_dataset)
        
        # Create data loaders
        train_dataloader = self.create_dataloader(train_dataset, shuffle=True)
        val_dataloader = self.create_dataloader(val_dataset, shuffle=False)
        
        # Calculate total training steps
        num_training_steps = len(train_dataloader) * self.training_config.max_epochs
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(num_training_steps)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean up old checkpoints to prevent symlink issues
        self.cleanup_old_checkpoints(output_dir)
        
        # Training loop
        start_time = time.time()
        
        # Calculate starting epoch and step for resume
        if self.resume_from_checkpoint:
            # Calculate how many epochs we've completed
            steps_per_epoch = len(train_dataloader)
            completed_epochs = self.global_step // steps_per_epoch
            starting_epoch = completed_epochs
            
            print(f"Resuming from epoch {starting_epoch + 1}, step {self.global_step}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Completed epochs: {completed_epochs}")
            print(f"Will train for {self.training_config.max_epochs - starting_epoch} more epochs")
        else:
            starting_epoch = 0
        
        for epoch in range(starting_epoch, self.training_config.max_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.training_config.max_epochs}")
            
            # Training
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            # Skip batches if resuming mid-epoch
            if self.resume_from_checkpoint and epoch == starting_epoch:
                steps_in_current_epoch = self.global_step % len(train_dataloader)
                if steps_in_current_epoch > 0:
                    print(f"Skipping {steps_in_current_epoch} steps in current epoch")
                    # Skip the progress bar iterator for completed steps
                    for _ in range(steps_in_current_epoch):
                        try:
                            next(iter(progress_bar))
                        except StopIteration:
                            break
            
            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.training_losses.append(loss)
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': self.global_step
                })
                
                # Validation
                if self.global_step % self.training_config.eval_steps == 0:
                    val_loss = self.validate(val_dataloader)
                    self.validation_losses.append(val_loss)
                    print(f"Validation Loss: {val_loss:.4f}")
                    
                    # Save checkpoint if best so far
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(output_dir, is_best=True, current_loss=val_loss)
                        print(f"✓ New best model saved (loss: {val_loss:.4f})")
                    else:
                        # Still save checkpoint but not as best
                        self.save_checkpoint(output_dir, is_best=False, current_loss=val_loss)
                
                # Save checkpoint
                if self.global_step % self.training_config.save_steps == 0:
                    self.save_checkpoint(output_dir)
            
            # End of epoch validation
            val_loss = self.validate(val_dataloader)
            self.validation_losses.append(val_loss)
            print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

            # Check if this is the best model at end of epoch
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(output_dir, is_best=True, current_loss=val_loss)
                print(f"✓ New best model saved at end of epoch (loss: {val_loss:.4f})")
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n✓ Training completed in {total_time/3600:.2f} hours")
        print(f"✓ Best validation loss: {self.best_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(output_dir)
        
        # Clean up multiprocessing resources
        self._cleanup_multiprocessing_resources()
        
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'scaling_analysis': scaling_analysis
        }


def create_trainer(model, tokenizer: AutoTokenizer, 
                  training_config: TrainingConfig,
                  chinchilla_config: ChinchillaConfig,
                  resume_from_checkpoint: Optional[str] = None,
                  hf_repo_name: Optional[str] = None) -> LoRATrainer:
    """
    Create a trainer instance with resume support.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        training_config: Training configuration
        chinchilla_config: Chinchilla configuration
        resume_from_checkpoint: Path to checkpoint to resume from
        hf_repo_name: HuggingFace Hub repository name for saving models
        
    Returns:
        LoRATrainer instance
    """
    return LoRATrainer(model, tokenizer, training_config, chinchilla_config, resume_from_checkpoint, hf_repo_name)


def train_model(model, tokenizer: AutoTokenizer, 
                train_dataset: Dataset, val_dataset: Dataset,
                training_config: TrainingConfig,
                chinchilla_config: ChinchillaConfig,
                output_dir: str,
                resume_from_checkpoint: Optional[str] = None,
                hf_repo_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Train the model with resume support.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_config: Training configuration
        chinchilla_config: Chinchilla configuration
        output_dir: Output directory
        resume_from_checkpoint: Path to checkpoint to resume from
        hf_repo_name: HuggingFace Hub repository name for saving models
        
    Returns:
        Training history
    """
    # Create trainer
    trainer = create_trainer(model, tokenizer, training_config, chinchilla_config, resume_from_checkpoint, hf_repo_name)
    
    # Start training
    history = trainer.train(train_dataset, val_dataset, output_dir)
    
    return history 