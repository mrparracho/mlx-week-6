"""
Configuration for Qwen2.5-0.5B LoRA Fine-tuning with Chinchilla Scaling Laws
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import os
import yaml
from datasets import Dataset


@dataclass
class ChinchillaConfig:
    """Chinchilla scaling laws configuration."""
    base_model_size: Optional[float] = None  # Trainable LoRA parameters for scaling analysis
    total_model_size: Optional[float] = None  # Total model parameters for reference
    dataset_tokens: float = 230e6  # CNN/DailyMail total tokens (~287K examples * 800 tokens)
    apply_scaling_laws: bool = True
    compute_budget: Optional[float] = None  # Will be calculated
    optimal_model_size: Optional[float] = None  # Will be calculated
    optimal_training_tokens: Optional[float] = None  # Will be calculated


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""
    rank: int = 16 
    alpha: int = 32  # 2 * rank
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            # Qwen2.5 uses separate q_proj, k_proj, v_proj, o_proj for attention
            # and gate_proj, up_proj, down_proj for MLP
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] #, "gate_proj", "up_proj", "down_proj"]
        if self.alpha is None:
            self.alpha = 2 * self.rank


@dataclass
class TrainingConfig:
    """Training configuration optimized for 2-hour epochs."""
    learning_rate: float = 5e-5  # Much higher for faster convergence
    batch_size: int = 1  # Increased batch size
    gradient_accumulation_steps: int = 8  # Reduced for faster steps
    max_epochs: int = 3
    warmup_steps: int = 500  # Increased warmup for higher LR
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Increased for higher LR
    save_steps: int = 2000  # Save less frequently
    eval_steps: int = 2000  # Evaluate less frequently
    logging_steps: int = 100  # Log more frequently
    save_total_limit: int = 2  # Keep fewer checkpoints
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = True  # Enable for faster data loading
    dataloader_num_workers: int = 4  # Re-enable for faster loading
    
    # Memory and stability settings
    dataloader_drop_last: bool = True

@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "cnn_dailymail"
    dataset_config: str = "3.0.0"
    max_input_length: int = 1024
    max_target_length: int = 128
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    input_prefix: str = "Summarize: "
    target_prefix: str = "\nTarget:"

    def load_cnn_dailymail_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load the CNN/DailyMail dataset."""
        from datasets import load_dataset
        dataset = load_dataset(self.dataset_name, self.dataset_config)
        
        # Sample a subset for faster training
        max_train_samples = 50000  # Reduce from ~287K to 50K
        max_val_samples = 5000
        max_test_samples = 5000
        
        if len(dataset['train']) > max_train_samples:
            dataset['train'] = dataset['train'].select(range(max_train_samples))
        if len(dataset['validation']) > max_val_samples:
            dataset['validation'] = dataset['validation'].select(range(max_val_samples))
        if len(dataset['test']) > max_test_samples:
            dataset['test'] = dataset['test'].select(range(max_test_samples))
        
        return dataset['train'], dataset['validation'], dataset['test']


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    trust_remote_code: bool = True
    use_cache: bool = False
    torch_dtype: str = "float16"
    device_map: str = "auto"


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "./trained_model"
    save_model: bool = True
    save_tokenizer: bool = True
    save_config: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""
    chinchilla: ChinchillaConfig = None  # type: ignore
    lora: LoRAConfig = None  # type: ignore
    training: TrainingConfig = None  # type: ignore
    data: DataConfig = None  # type: ignore
    model: ModelConfig = None  # type: ignore
    output: OutputConfig = None  # type: ignore
    
    def __post_init__(self):
        if self.chinchilla is None:
            self.chinchilla = ChinchillaConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.output is None:
            self.output = OutputConfig()
    
    def update_from_args(self, args):
        """Update configuration from command line arguments."""
        if hasattr(args, 'model_name') and args.model_name:
            self.model.model_name = args.model_name
        if hasattr(args, 'dataset_name') and args.dataset_name:
            self.data.dataset_name = args.dataset_name
        if hasattr(args, 'lora_rank') and args.lora_rank:
            self.lora.rank = args.lora_rank
            self.lora.alpha = 2 * args.lora_rank
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'batch_size') and args.batch_size:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'epochs') and args.epochs:
            self.training.max_epochs = args.epochs
        if hasattr(args, 'output_dir') and args.output_dir:
            self.output.output_dir = args.output_dir
        if hasattr(args, 'apply_chinchilla_scaling'):
            self.chinchilla.apply_scaling_laws = args.apply_chinchilla_scaling
    
    def update_model_size(self, trainable_params: float, total_params: float = None):
        """Update Chinchilla config with trainable LoRA parameters for scaling analysis."""
        self.chinchilla.base_model_size = trainable_params  # Use trainable params for scaling
        if total_params:
            self.chinchilla.total_model_size = total_params  # Store total for reference
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        
        print(f"\nMODEL:")
        print(f"  Model Name: {self.model.model_name}")
        print(f"  Trust Remote Code: {self.model.trust_remote_code}")
        print(f"  Torch Dtype: {self.model.torch_dtype}")
        
        print(f"\nLoRA:")
        print(f"  Rank: {self.lora.rank}")
        print(f"  Alpha: {self.lora.alpha}")
        print(f"  Dropout: {self.lora.dropout}")
        print(f"  Target Modules: {self.lora.target_modules}")
        
        print(f"\nTRAINING:")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Gradient Accumulation Steps: {self.training.gradient_accumulation_steps}")
        print(f"  Max Epochs: {self.training.max_epochs}")
        print(f"  Warmup Steps: {self.training.warmup_steps}")
        print(f"  Weight Decay: {self.training.weight_decay}")
        
        print(f"\nDATA:")
        print(f"  Dataset: {self.data.dataset_name}")
        print(f"  Max Input Length: {self.data.max_input_length}")
        print(f"  Max Target Length: {self.data.max_target_length}")
        print(f"  Input Prefix: '{self.data.input_prefix}'")
        print(f"  Target Prefix: '{self.data.target_prefix}'")
        
        print(f"\nCHINCHILLA SCALING:")
        print(f"  Apply Scaling Laws: {self.chinchilla.apply_scaling_laws}")
        if self.chinchilla.base_model_size is not None:
            print(f"  Trainable LoRA Parameters: {self.chinchilla.base_model_size:.1e}")
        if self.chinchilla.total_model_size is not None:
            print(f"  Total Model Parameters: {self.chinchilla.total_model_size:.1e}")
        if self.chinchilla.base_model_size is None:
            print(f"  Model Size: Will be calculated dynamically")
        print(f"  Dataset Tokens: {self.chinchilla.dataset_tokens:.1e}")
        
        print(f"\nOUTPUT:")
        print(f"  Output Directory: {self.output.output_dir}")
        print(f"  Save Model: {self.output.save_model}")
        print(f"  Push to Hub: {self.output.push_to_hub}")
        
        print("="*60)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def create_config_from_args(args) -> Config:
    """Create configuration from command line arguments."""
    config = get_default_config()
    config.update_from_args(args)
    return config


def save_config_as_yaml(config: Config, output_dir: str, filename: str = "configs.yml"):
    """Save configuration as YAML file."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, filename)
    
    # Convert config to dictionary
    config_dict = asdict(config)
    
    # Save as YAML
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"✓ Configuration saved to {config_path}")
    return config_path 