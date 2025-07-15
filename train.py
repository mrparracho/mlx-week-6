"""
Training module for QWEN 7B LoRA fine-tuning with Chinchilla scaling laws
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
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
                 chinchilla_config: ChinchillaConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.chinchilla_config = chinchilla_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device (use to_empty for meta tensors)
        self.model.to_empty(device=self.device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_losses = []
        self.validation_losses = []
        
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Set up optimizer and learning rate scheduler.
        
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
        
        print(f"✓ Optimizer and scheduler created")
        print(f"  Learning rate: {self.training_config.learning_rate}")
        print(f"  Weight decay: {self.training_config.weight_decay}")
        print(f"  Warmup steps: {self.training_config.warmup_steps}")
        print(f"  Total steps: {num_training_steps}")
    
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
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    def train_step(self, batch: Dict) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Calculate loss
        loss = self.calculate_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.training_config.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
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
    
    def save_checkpoint(self, output_dir: str, is_best: bool = False):
        """
        Save a checkpoint.
        
        Args:
            output_dir: Directory to save checkpoint
            is_best: Whether this is the best model so far
        """
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        if is_best:
            # Create best model symlink
            best_dir = os.path.join(output_dir, 'best')
            if os.path.exists(best_dir):
                os.remove(best_dir)
            os.symlink(checkpoint_dir, best_dir)
    
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
        
        # Calculate total tokens in training dataset
        total_tokens = sum(len(example['input_ids']) + len(example['target_ids']) 
                          for example in train_dataset)
        
        # Analyze scaling efficiency
        analysis = analyze_scaling_efficiency(
            available_tokens=total_tokens,
            base_model_size=self.chinchilla_config.base_model_size
        )
        
        # Print analysis
        print_scaling_analysis(analysis)
        
        return analysis
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              output_dir: str) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for saving models
            
        Returns:
            Dictionary with training history
        """
        print("\n" + "="*60)
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
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.training_config.max_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.training_config.max_epochs}")
            
            # Training
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.training_losses.append(loss)
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Logging
                if self.global_step % self.training_config.logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-self.training_config.logging_steps:])
                    print(f"Step {self.global_step}: Loss = {avg_loss:.4f}")
                
                # Validation
                if self.global_step % self.training_config.eval_steps == 0:
                    val_loss = self.validate(val_dataloader)
                    self.validation_losses.append(val_loss)
                    print(f"Validation Loss: {val_loss:.4f}")
                    
                    # Save checkpoint if best so far
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(output_dir, is_best=True)
                        print(f"✓ New best model saved (loss: {val_loss:.4f})")
                
                # Save checkpoint
                if self.global_step % self.training_config.save_steps == 0:
                    self.save_checkpoint(output_dir)
            
            # End of epoch validation
            val_loss = self.validate(val_dataloader)
            self.validation_losses.append(val_loss)
            print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
        
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
                  chinchilla_config: ChinchillaConfig) -> LoRATrainer:
    """
    Create a trainer instance.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        training_config: Training configuration
        chinchilla_config: Chinchilla configuration
        
    Returns:
        LoRATrainer instance
    """
    return LoRATrainer(model, tokenizer, training_config, chinchilla_config)


def train_model(model, tokenizer: AutoTokenizer, 
                train_dataset: Dataset, val_dataset: Dataset,
                training_config: TrainingConfig,
                chinchilla_config: ChinchillaConfig,
                output_dir: str) -> Dict[str, Any]:
    """
    Train the model.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_config: Training configuration
        chinchilla_config: Chinchilla configuration
        output_dir: Output directory
        
    Returns:
        Training history
    """
    # Create trainer
    trainer = create_trainer(model, tokenizer, training_config, chinchilla_config)
    
    # Start training
    history = trainer.train(train_dataset, val_dataset, output_dir)
    
    return history 