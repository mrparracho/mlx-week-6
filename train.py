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


class LoRATrainer:
    """Trainer for LoRA fine-tuning with Chinchilla scaling laws."""
    
    def __init__(self, model, tokenizer: AutoTokenizer, 
                 training_config: TrainingConfig, 
                 chinchilla_config: ChinchillaConfig,
                 resume_from_checkpoint: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.chinchilla_config = chinchilla_config
        self.resume_from_checkpoint = resume_from_checkpoint
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
        
        print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        
        # Handle 'best' checkpoint
        if checkpoint_path == 'best':
            # Find the best model checkpoint
            output_dir = os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.'
            best_info_path = os.path.join(output_dir, 'best_model_info.json')
            
            if os.path.exists(best_info_path):
                with open(best_info_path, 'r') as f:
                    best_info = json.load(f)
                    checkpoint_path = os.path.join(output_dir, best_info['checkpoint'])
                    print(f"Found best checkpoint: {checkpoint_path}")
            else:
                raise ValueError("No best model info found. Cannot resume from 'best' checkpoint.")
        
        # Load model and tokenizer
        try:
            print(f"Loading model from {checkpoint_path}")
            self.model = self.model.__class__.from_pretrained(checkpoint_path)
            self.tokenizer = self.tokenizer.__class__.from_pretrained(checkpoint_path)
            
            # Move model to device
            self.model.to_empty(device=self.device)
            print(f"✓ Model loaded and moved to {self.device}")
            
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
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=shuffle,
            pin_memory=self.training_config.dataloader_pin_memory,
            num_workers=self.training_config.dataloader_num_workers,
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
        
        # Analyze scaling efficiency
        analysis = analyze_scaling_efficiency(
            available_tokens=total_tokens,
            base_model_size=self.chinchilla_config.base_model_size
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
                for _ in range(steps_in_current_epoch):
                    try:
                        next(iter(progress_bar))
                    except StopIteration:
                        break
                print(f"Skipped {steps_in_current_epoch} steps in current epoch")
            
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
        
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'scaling_analysis': scaling_analysis
        }


def create_trainer(model, tokenizer: AutoTokenizer, 
                  training_config: TrainingConfig,
                  chinchilla_config: ChinchillaConfig,
                  resume_from_checkpoint: Optional[str] = None) -> LoRATrainer:
    """
    Create a trainer instance with resume support.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        training_config: Training configuration
        chinchilla_config: Chinchilla configuration
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        LoRATrainer instance
    """
    return LoRATrainer(model, tokenizer, training_config, chinchilla_config, resume_from_checkpoint)


def train_model(model, tokenizer: AutoTokenizer, 
                train_dataset: Dataset, val_dataset: Dataset,
                training_config: TrainingConfig,
                chinchilla_config: ChinchillaConfig,
                output_dir: str,
                resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
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
        
    Returns:
        Training history
    """
    # Create trainer
    trainer = create_trainer(model, tokenizer, training_config, chinchilla_config, resume_from_checkpoint)
    
    # Start training
    history = trainer.train(train_dataset, val_dataset, output_dir)
    
    return history 