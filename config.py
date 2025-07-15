"""
Configuration for Qwen2.5-0.5B LoRA Fine-tuning with Chinchilla Scaling Laws
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
import os
import yaml


@dataclass
class ChinchillaConfig:
    """Chinchilla scaling laws configuration."""
    base_model_size: Optional[float] = None  # Will be calculated dynamically
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
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if self.alpha is None:
            self.alpha = 2 * self.rank


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 5e-5  # Reduced from 1e-4 for better stability
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
    dataloader_num_workers: int = 4
    
    # Numerical stability settings
    use_amp: bool = True  # Automatic Mixed Precision
    use_gradient_checkpointing: bool = True  # Memory efficiency
    use_fp16: bool = False  # Use float16 for training (set to False for stability)
    fp16_full_eval: bool = False  # Use float16 for evaluation
    dataloader_drop_last: bool = True  # Drop incomplete batches
    
    # Boundary detection configuration
    use_boundary_detection: bool = True
    boundary_loss_weight: float = 0.1
    boundary_position_weight: float = 2.0
    boundary_smooth_window: int = 4


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


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    trust_remote_code: bool = True
    use_cache: bool = False
    torch_dtype: str = "float32"
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
    
    def update_model_size(self, model_size: float):
        """Update Chinchilla config with actual model size."""
        self.chinchilla.base_model_size = model_size
    
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
        print(f"  Use Boundary Detection: {self.training.use_boundary_detection}")
        if self.training.use_boundary_detection:
            print(f"  Boundary Loss Weight: {self.training.boundary_loss_weight}")
            print(f"  Boundary Position Weight: {self.training.boundary_position_weight}")
            print(f"  Boundary Smooth Window: {self.training.boundary_smooth_window}")
        
        print(f"\nDATA:")
        print(f"  Dataset: {self.data.dataset_name}")
        print(f"  Max Input Length: {self.data.max_input_length}")
        print(f"  Max Target Length: {self.data.max_target_length}")
        print(f"  Input Prefix: '{self.data.input_prefix}'")
        print(f"  Target Prefix: '{self.data.target_prefix}'")
        
        print(f"\nCHINCHILLA SCALING:")
        print(f"  Apply Scaling Laws: {self.chinchilla.apply_scaling_laws}")
        if self.chinchilla.base_model_size is not None:
            print(f"  Base Model Size: {self.chinchilla.base_model_size:.1e}")
        else:
            print(f"  Base Model Size: Will be calculated dynamically")
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