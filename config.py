"""
Configuration for QWEN 7B LoRA Fine-tuning with Chinchilla Scaling Laws
"""

from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class ChinchillaConfig:
    """Chinchilla scaling laws configuration."""
    base_model_size: float = 7e9  # QWEN 7B parameters
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
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.alpha is None:
            self.alpha = 2 * self.rank


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 0


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
    target_prefix: str = ""


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen/Qwen-7B"
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
        
        print(f"\nCHINCHILLA SCALING:")
        print(f"  Apply Scaling Laws: {self.chinchilla.apply_scaling_laws}")
        print(f"  Base Model Size: {self.chinchilla.base_model_size:.1e}")
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