"""
LoRA model implementation for Qwen2.5 fine-tuning
"""

import os
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from config import ModelConfig, LoRAConfig

# Try to import BitsAndBytesConfig, but handle case where it's not available
try:
    from transformers.utils.quantization_config import BitsAndBytesConfig
    import torch
    # Check if CUDA is available for bitsandbytes
    BITSANDBYTES_AVAILABLE = torch.cuda.is_available()
    if not BITSANDBYTES_AVAILABLE:
        print("Warning: CUDA not available. Using full precision training.")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Using full precision training.")


def setup_numerical_stability():
    """Set up numerical stability settings for different devices."""
    # Set torch settings for better numerical stability
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Set default tensor type based on device
    if torch.cuda.is_available():
        # Use float32 for better stability on CUDA
        torch.set_default_dtype(torch.float32)
        # Enable gradient scaler for mixed precision if needed
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("✓ CUDA numerical stability settings applied")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS settings
        torch.set_default_dtype(torch.float32)
        print("✓ MPS numerical stability settings applied")
    else:
        # CPU settings
        torch.set_default_dtype(torch.float32)
        print("✓ CPU numerical stability settings applied")


class LoRAModel:
    """LoRA model wrapper for Qwen2.5 fine-tuning."""
    
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None
        
        # Set up numerical stability
        setup_numerical_stability()
        
    def load_base_model(self) -> AutoModelForCausalLM:
        """
        Load the base Qwen2.5 model.
        
        Returns:
            Loaded base model
        """
        print(f"Loading base model: {self.model_config.model_name}")
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": self.model_config.trust_remote_code,
            "torch_dtype": getattr(torch, self.model_config.torch_dtype),
            "use_cache": self.model_config.use_cache
        }
        
        # Set device map based on availability
        if torch.cuda.is_available():
            model_kwargs["device_map"] = self.model_config.device_map
            # Use float32 for better stability on CUDA
            model_kwargs["torch_dtype"] = torch.float32
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Use MPS if available
            model_kwargs["device_map"] = "mps"
        else:
            # Fall back to CPU
            model_kwargs["device_map"] = "cpu"
        
        # Add quantization if bitsandbytes is available and not on CUDA (for stability)
        if BITSANDBYTES_AVAILABLE and not torch.cuda.is_available():
            print("Using 4-bit quantization for memory efficiency")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            print("Using full precision (no quantization available or disabled for CUDA stability)")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            **model_kwargs
        )
        
        print("✓ Base model loaded successfully")
        return model
    
    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load the tokenizer for the model.
        
        Returns:
            Loaded tokenizer
        """
        print(f"Loading tokenizer: {self.model_config.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("✓ Tokenizer loaded successfully")
        return tokenizer
    
    def create_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Returns:
            LoRA configuration
        """
        lora_config = LoraConfig(
            r=self.lora_config.rank,
            lora_alpha=self.lora_config.alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        print(f"✓ LoRA config created: rank={self.lora_config.rank}, alpha={self.lora_config.alpha}")
        return lora_config
    
    def apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Apply LoRA to the base model.
        
        Args:
            model: Base model
            
        Returns:
            Model with LoRA applied
        """
        print("Applying LoRA to model...")
        
        # Create LoRA config
        lora_config = self.create_lora_config()
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        print("✓ LoRA applied successfully")
        return model
    
    def setup_model(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Set up the complete model with LoRA.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load base model
        self.model = self.load_base_model()
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        
        # Apply LoRA
        self.model = self.apply_lora(self.model)
        
        return self.model, self.tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Get model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Get LoRA parameters
        lora_params = sum(p.numel() for name, p in self.model.named_parameters() 
                         if "lora" in name.lower())
        
        info = {
            "model_name": self.model_config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "lora_parameters": lora_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
            "lora_rank": self.lora_config.rank,
            "lora_alpha": self.lora_config.alpha,
            "target_modules": self.lora_config.target_modules
        }
        
        return info
    
    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Model Name: {info['model_name']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Non-trainable Parameters: {info['non_trainable_parameters']:,}")
        print(f"LoRA Parameters: {info['lora_parameters']:,}")
        print(f"Trainable Ratio: {info['trainable_ratio']:.2%}")
        print(f"LoRA Rank: {info['lora_rank']}")
        print(f"LoRA Alpha: {info['lora_alpha']}")
        print(f"Target Modules: {info['target_modules']}")
        print("="*60)
    
    def save_model(self, output_dir: str):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        print(f"Saving model to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Model saved to {output_dir}")
    
    def load_saved_model(self, model_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading saved model from {model_path}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.trust_remote_code,
            device_map=self.model_config.device_map,
            torch_dtype=getattr(torch, self.model_config.torch_dtype)
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        print(f"✓ Model loaded from {model_path}")
        return self.model, self.tokenizer


def create_lora_model(model_config: ModelConfig, lora_config: LoRAConfig) -> LoRAModel:
    """
    Create a LoRA model instance.
    
    Args:
        model_config: Model configuration
        lora_config: LoRA configuration
        
    Returns:
        LoRAModel instance
    """
    return LoRAModel(model_config, lora_config)


def setup_model_and_tokenizer(model_config: ModelConfig, 
                             lora_config: LoRAConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Set up model and tokenizer with LoRA.
    
    Args:
        model_config: Model configuration
        lora_config: LoRA configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Create LoRA model
    lora_model = create_lora_model(model_config, lora_config)
    
    # Setup model and tokenizer
    model, tokenizer = lora_model.setup_model()
    
    # Print model information
    lora_model.print_model_info()
    
    return model, tokenizer 