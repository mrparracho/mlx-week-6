"""
Data loading and preprocessing for CNN/DailyMail summarization dataset
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from tqdm import tqdm
from config import DataConfig
import torch
from torch.nn.utils.rnn import pad_sequence
import random


class SummarizationDataLoader:
    """Data loader for summarization tasks."""
    
    def __init__(self, config: DataConfig, tokenizer: PreTrainedTokenizer, 
                 debug_mode: bool = False, sample_size: Optional[int] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.debug_mode = debug_mode
        self.sample_size = sample_size
        self._tokenizer_cache = {}  # For optional caching
        if debug_mode and sample_size is None:
            self.sample_size = 100  # Default sample size for debugging
    
    def load_cnn_dailymail_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load CNN/DailyMail dataset.
        
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        print("Loading CNN/DailyMail dataset...")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                trust_remote_code=True
            )
            
            train_dataset = dataset[self.config.train_split]
            validation_dataset = dataset[self.config.validation_split]
            test_dataset = dataset[self.config.test_split]
            
            # Apply sampling if in debug mode
            if self.debug_mode:
                print(f"🔧 DEBUG MODE: Sampling {self.sample_size} examples from each split")
                
                # Sample from each split
                train_indices = random.sample(range(len(train_dataset)), 
                                            min(self.sample_size, len(train_dataset)))
                val_indices = random.sample(range(len(validation_dataset)), 
                                          min(self.sample_size // 10, len(validation_dataset)))
                test_indices = random.sample(range(len(test_dataset)), 
                                           min(self.sample_size // 10, len(test_dataset)))
                
                train_dataset = train_dataset.select(train_indices)
                validation_dataset = validation_dataset.select(val_indices)
                test_dataset = test_dataset.select(test_indices)
                
                print(f"✓ Sampled {len(train_dataset)} training examples")
                print(f"✓ Sampled {len(validation_dataset)} validation examples")
                print(f"✓ Sampled {len(test_dataset)} test examples")
            else:
                print(f"✓ Loaded {len(train_dataset)} training examples")
                print(f"✓ Loaded {len(validation_dataset)} validation examples")
                print(f"✓ Loaded {len(test_dataset)} test examples")
            
            return train_dataset, validation_dataset, test_dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def format_for_summarization(self, example: Dict) -> Dict:
        """
        Format example for summarization task.
        
        Args:
            example: Raw dataset example
            
        Returns:
            Formatted example with input and target
        """
        # Extract article and highlights
        article = example.get('article', '')
        highlights = example.get('highlights', '')
        
        # Format input with prefix
        input_text = f"{self.config.input_prefix}{article}"
        
        # Format target with prefix
        target_text = f"{self.config.target_prefix}{highlights}"
        
        return {
            'input': input_text,
            'target': target_text,
            'article': article,
            'highlights': highlights
        }
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Optimized tokenization function for causal LM training (no truncation).
        """
        model_max_length = getattr(self.tokenizer, 'model_max_length', 2048)
        input_texts = examples['input']
        target_texts = examples['target']
        # Pre-tokenize input texts to find boundaries efficiently
        input_tokenized = self.tokenizer(
            input_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        # Combine texts efficiently
        combined_texts = [f"{input_text} {target_text}" for input_text, target_text in zip(input_texts, target_texts)]
        # Tokenize combined texts in one batch operation (no truncation)
        tokenized = self.tokenizer(
            combined_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors=None
        )
        # Create labels efficiently
        labels = []
        input_lengths = [len(ids) for ids in input_tokenized['input_ids']]
        for input_ids, input_len in zip(tokenized['input_ids'], input_lengths):
            label = [-100] * len(input_ids)
            boundary = min(input_len, len(input_ids))
            label[boundary:] = input_ids[boundary:]
            labels.append(label)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        print("Preprocessing dataset...")
        formatted_dataset = dataset.map(
            self.format_for_summarization,
            remove_columns=dataset.column_names,
            desc="Formatting for summarization"
        )
        tokenized_dataset = formatted_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=formatted_dataset.column_names,
            desc="Tokenizing"
        )
        print("✓ Dataset preprocessing completed")
        return tokenized_dataset
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with statistics
        """
        dataset_size = len(dataset)
        
        # Determine if we should use sampling
        use_sampling = dataset_size > 10000 or self.debug_mode
        
        if use_sampling:
            # Use sampling for very large datasets or debug mode
            if self.debug_mode and self.sample_size:
                sample_size = min(self.sample_size, dataset_size)
                desc = f"Calculating statistics (debug sample: {sample_size})"
            else:
                sample_size = min(5000, dataset_size // 10)  # Sample 10% or 5000, whichever is smaller
                desc = f"Calculating statistics (sampled: {sample_size})"
            
            sample_indices = random.sample(range(dataset_size), sample_size)
            sample_dataset = dataset.select(sample_indices)
            
            input_lengths = []
            label_lengths = []
            
            for example in tqdm(sample_dataset, desc=desc):
                input_lengths.append(len(example['input_ids']))
                # Count non-padding labels efficiently
                labels = example['labels']
                valid_count = sum(1 for l in labels if l != -100)
                label_lengths.append(valid_count)
            
            # Calculate sample statistics
            if not input_lengths or not label_lengths:
                # Handle empty dataset case
                stats = {
                    'num_examples': 0,
                    'avg_input_length': 0.0,
                    'avg_target_length': 0.0,
                    'max_input_length': 0,
                    'max_target_length': 0,
                    'min_input_length': 0,
                    'min_target_length': 0,
                    'total_input_tokens': 0,
                    'total_target_tokens': 0,
                    'total_tokens': 0,
                    'sampled': True,
                    'sample_size': sample_size,
                    'extrapolated': True
                }
            else:
                avg_input_length = np.mean(input_lengths)
                avg_target_length = np.mean(label_lengths)
                max_input_length = np.max(input_lengths)
                max_target_length = np.max(label_lengths)
                min_input_length = np.min(input_lengths)
                min_target_length = np.min(label_lengths)
                
                # Extrapolate to full dataset
                total_input_tokens = int(avg_input_length * dataset_size)
                total_target_tokens = int(avg_target_length * dataset_size)
                
                stats = {
                    'num_examples': dataset_size,
                    'avg_input_length': avg_input_length,
                    'avg_target_length': avg_target_length,
                    'max_input_length': max_input_length,
                    'max_target_length': max_target_length,
                    'min_input_length': min_input_length,
                    'min_target_length': min_target_length,
                    'total_input_tokens': total_input_tokens,
                    'total_target_tokens': total_target_tokens,
                    'total_tokens': total_input_tokens + total_target_tokens,
                    'sampled': True,
                    'sample_size': sample_size,
                    'extrapolated': True
                }
            
        else:
            # For smaller datasets, process all examples but in batches
            batch_size = 1000
            input_lengths = []
            label_lengths = []
            
            for i in tqdm(range(0, dataset_size, batch_size), desc="Calculating statistics"):
                end_idx = min(i + batch_size, dataset_size)
                batch = dataset.select(range(i, end_idx))
                
                for example in batch:
                    input_lengths.append(len(example['input_ids']))
                    # Count non-padding labels efficiently
                    labels = example['labels']
                    valid_count = sum(1 for l in labels if l != -100)
                    label_lengths.append(valid_count)
            
            if not input_lengths or not label_lengths:
                # Handle empty dataset case
                stats = {
                    'num_examples': 0,
                    'avg_input_length': 0.0,
                    'avg_target_length': 0.0,
                    'max_input_length': 0,
                    'max_target_length': 0,
                    'min_input_length': 0,
                    'min_target_length': 0,
                    'total_input_tokens': 0,
                    'total_target_tokens': 0,
                    'total_tokens': 0,
                    'sampled': False,
                    'extrapolated': False
                }
            else:
                stats = {
                    'num_examples': len(dataset),
                    'avg_input_length': np.mean(input_lengths),
                    'avg_target_length': np.mean(label_lengths),
                    'max_input_length': np.max(input_lengths),
                    'max_target_length': np.max(label_lengths),
                    'min_input_length': np.min(input_lengths),
                    'min_target_length': np.min(label_lengths),
                    'total_input_tokens': sum(input_lengths),
                    'total_target_tokens': sum(label_lengths),
                    'total_tokens': sum(input_lengths) + sum(label_lengths),
                    'sampled': False,
                    'extrapolated': False
                }
        
        return stats
    
    def print_dataset_info(self, train_dataset: Dataset, val_dataset: Dataset, 
                          test_dataset: Dataset):
        """Print information about the datasets."""
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)
        
        # Training dataset stats
        train_stats = self.get_dataset_statistics(train_dataset)
        print(f"\nTRAINING DATASET:")
        print(f"  Examples: {train_stats['num_examples']:,}")
        print(f"  Avg Input Length: {train_stats['avg_input_length']:.1f} tokens")
        print(f"  Avg Target Length: {train_stats['avg_target_length']:.1f} tokens")
        print(f"  Total Tokens: {train_stats['total_tokens']:,}")
        if train_stats.get('sampled'):
            print(f"  📊 Sampled: {train_stats['sample_size']} examples (extrapolated)")
        
        # Validation dataset stats
        val_stats = self.get_dataset_statistics(val_dataset)
        print(f"\nVALIDATION DATASET:")
        print(f"  Examples: {val_stats['num_examples']:,}")
        print(f"  Avg Input Length: {val_stats['avg_input_length']:.1f} tokens")
        print(f"  Avg Target Length: {val_stats['avg_target_length']:.1f} tokens")
        print(f"  Total Tokens: {val_stats['total_tokens']:,}")
        if val_stats.get('sampled'):
            print(f"  📊 Sampled: {val_stats['sample_size']} examples (extrapolated)")
        
        # Test dataset stats
        test_stats = self.get_dataset_statistics(test_dataset)
        print(f"\nTEST DATASET:")
        print(f"  Examples: {test_stats['num_examples']:,}")
        print(f"  Avg Input Length: {test_stats['avg_input_length']:.1f} tokens")
        print(f"  Avg Target Length: {test_stats['avg_target_length']:.1f} tokens")
        print(f"  Total Tokens: {test_stats['total_tokens']:,}")
        if test_stats.get('sampled'):
            print(f"  📊 Sampled: {test_stats['sample_size']} examples (extrapolated)")
        
        # Overall stats
        total_examples = train_stats['num_examples'] + val_stats['num_examples'] + test_stats['num_examples']
        total_tokens = train_stats['total_tokens'] + val_stats['total_tokens'] + test_stats['total_tokens']
        
        print(f"\nOVERALL:")
        print(f"  Total Examples: {total_examples:,}")
        print(f"  Total Tokens: {total_tokens:,}")
        print(f"  Tokens per Example: {total_tokens/total_examples:.1f}")
        
        # Show sampling info if any dataset was sampled
        if any(stats.get('sampled', False) for stats in [train_stats, val_stats, test_stats]):
            print(f"\n📊 SAMPLING INFO:")
            print(f"  Debug Mode: {'Yes' if self.debug_mode else 'No'}")
            if self.debug_mode and self.sample_size:
                print(f"  Sample Size: {self.sample_size}")
            print(f"  Statistics are extrapolated from samples")
        
        print("="*60)


def create_data_loader(config: DataConfig, tokenizer: PreTrainedTokenizer, 
                      debug_mode: bool = False, sample_size: Optional[int] = None) -> SummarizationDataLoader:
    """
    Create a data loader instance.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer for text processing
        debug_mode: Whether to use debug mode with sampling
        sample_size: Size of sample to use in debug mode
        
    Returns:
        SummarizationDataLoader instance
    """
    return SummarizationDataLoader(config, tokenizer, debug_mode, sample_size)


def load_and_preprocess_data(config: DataConfig, tokenizer: PreTrainedTokenizer,
                           debug_mode: bool = False, sample_size: Optional[int] = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and preprocess the CNN/DailyMail dataset.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer for text processing
        debug_mode: Whether to use debug mode with sampling
        sample_size: Size of sample to use in debug mode
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    # Create data loader
    data_loader = create_data_loader(config, tokenizer, debug_mode, sample_size)
    
    # Load raw datasets
    train_raw, val_raw, test_raw = data_loader.load_cnn_dailymail_dataset()
    
    # Preprocess datasets
    train_dataset = data_loader.preprocess_dataset(train_raw)
    val_dataset = data_loader.preprocess_dataset(val_raw)
    test_dataset = data_loader.preprocess_dataset(test_raw)
    
    # Print dataset information
    data_loader.print_dataset_info(train_dataset, val_dataset, test_dataset)
    
    return train_dataset, val_dataset, test_dataset 


def collate_fn(batch, pad_token_id=0):
    # Dynamic padding to the longest sequence in the batch
    batch_input_ids = [x['input_ids'] for x in batch]
    batch_attention_mask = [x['attention_mask'] for x in batch]
    batch_labels = [x['labels'] for x in batch]
    max_length = max(len(ids) for ids in batch_input_ids)
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
        pad_len = max_length - len(input_ids)
        padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
        padded_attention_mask.append(attention_mask + [0] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
        'labels': torch.tensor(padded_labels, dtype=torch.long)
    } 