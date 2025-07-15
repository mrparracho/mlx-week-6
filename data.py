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


class SummarizationDataLoader:
    """Data loader for summarization tasks."""
    
    def __init__(self, config: DataConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
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
        Tokenize input and target texts.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples
        """
        # QWEN tokenizer does not support padding tokens; must pad manually after tokenization
        # Tokenize inputs (no return_tensors)
        inputs = self.tokenizer(
            examples['input'],
            max_length=self.config.max_input_length,
            truncation=self.config.truncation,
            padding=False  # No padding
        )
        
        # Tokenize targets (no return_tensors)
        targets = self.tokenizer(
            examples['target'],
            max_length=self.config.max_target_length,
            truncation=self.config.truncation,
            padding=False  # No padding
        )
        
        # Create labels (same as target input_ids for causal LM)
        labels = [target_ids[:] for target_ids in targets['input_ids']]
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels,
            'target_ids': targets['input_ids'],
            'target_attention_mask': targets['attention_mask']
        }
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess dataset for training.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Preprocessed dataset
        """
        print("Preprocessing dataset...")
        
        # Format for summarization
        formatted_dataset = dataset.map(
            self.format_for_summarization,
            remove_columns=dataset.column_names,
            desc="Formatting for summarization"
        )
        
        # Tokenize
        tokenized_dataset = formatted_dataset.map(
            self.tokenize_function,
            batched=True,
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
        input_lengths = []
        target_lengths = []
        
        for example in tqdm(dataset, desc="Calculating statistics"):
            input_lengths.append(len(example['input_ids']))
            target_lengths.append(len(example['target_ids']))
        
        stats = {
            'num_examples': len(dataset),
            'avg_input_length': np.mean(input_lengths),
            'avg_target_length': np.mean(target_lengths),
            'max_input_length': np.max(input_lengths),
            'max_target_length': np.max(target_lengths),
            'min_input_length': np.min(input_lengths),
            'min_target_length': np.min(target_lengths),
            'total_input_tokens': sum(input_lengths),
            'total_target_tokens': sum(target_lengths),
            'total_tokens': sum(input_lengths) + sum(target_lengths)
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
        
        # Validation dataset stats
        val_stats = self.get_dataset_statistics(val_dataset)
        print(f"\nVALIDATION DATASET:")
        print(f"  Examples: {val_stats['num_examples']:,}")
        print(f"  Avg Input Length: {val_stats['avg_input_length']:.1f} tokens")
        print(f"  Avg Target Length: {val_stats['avg_target_length']:.1f} tokens")
        print(f"  Total Tokens: {val_stats['total_tokens']:,}")
        
        # Test dataset stats
        test_stats = self.get_dataset_statistics(test_dataset)
        print(f"\nTEST DATASET:")
        print(f"  Examples: {test_stats['num_examples']:,}")
        print(f"  Avg Input Length: {test_stats['avg_input_length']:.1f} tokens")
        print(f"  Avg Target Length: {test_stats['avg_target_length']:.1f} tokens")
        print(f"  Total Tokens: {test_stats['total_tokens']:,}")
        
        # Overall stats
        total_examples = train_stats['num_examples'] + val_stats['num_examples'] + test_stats['num_examples']
        total_tokens = train_stats['total_tokens'] + val_stats['total_tokens'] + test_stats['total_tokens']
        
        print(f"\nOVERALL:")
        print(f"  Total Examples: {total_examples:,}")
        print(f"  Total Tokens: {total_tokens:,}")
        print(f"  Tokens per Example: {total_tokens/total_examples:.1f}")
        
        print("="*60)


def create_data_loader(config: DataConfig, tokenizer: PreTrainedTokenizer) -> SummarizationDataLoader:
    """
    Create a data loader instance.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer for text processing
        
    Returns:
        SummarizationDataLoader instance
    """
    return SummarizationDataLoader(config, tokenizer)


def load_and_preprocess_data(config: DataConfig, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and preprocess the CNN/DailyMail dataset.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer for text processing
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    # Create data loader
    data_loader = create_data_loader(config, tokenizer)
    
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
    """
    Custom collate function to pad input_ids, attention_mask, labels, etc. for QWEN.
    Args:
        batch: List of dicts from the dataset
        pad_token_id: Token ID to use for padding (default 0)
    Returns:
        Dict of padded tensors
    """
    input_ids = [torch.tensor(x['input_ids'], dtype=torch.long) for x in batch]
    attention_mask = [torch.tensor(x['attention_mask'], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x['labels'], dtype=torch.long) for x in batch]
    target_ids = [torch.tensor(x['target_ids'], dtype=torch.long) for x in batch]
    target_attention_mask = [torch.tensor(x['target_attention_mask'], dtype=torch.long) for x in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
    target_attention_mask_padded = pad_sequence(target_attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'target_ids': target_ids_padded,
        'target_attention_mask': target_attention_mask_padded
    } 