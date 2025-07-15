# Qwen2.5-0.5B LoRA Fine-tuning for Summarization

This project implements fine-tuning of the Qwen2.5-0.5B base model using Low-Rank Adaptation (LoRA) for supervised fine-tuning (SFT) to create high-quality text summaries.

## Overview

The project leverages the Qwen2.5-0.5B base model and applies LoRA techniques to efficiently fine-tune the model for summarization tasks. LoRA reduces the computational requirements and memory footprint while maintaining model performance.

## Key Features

- **Base Model**: Qwen2.5-0.5B (Qwen2.5-0.5B-base)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: Text Summarization
- **Framework**: MLX (Apple's machine learning framework)

## Project Structure

```
mlx-week-6/
├── README.md                 # Project documentation
├── main.py                   # Main training script
├── config.py                 # Configuration management
├── data.py                   # Data loading and preprocessing
├── model.py                  # LoRA model implementation
├── train.py                  # Training logic
├── utils.py                  # Utility functions
├── scaling_laws.py           # Chinchilla scaling laws
├── example_usage.py          # Example usage scripts
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── .gitignore               # Git ignore file
```

## Requirements

- Python 3.13+
- PyTorch 2.0+
- Transformers 4.35+
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets (HuggingFace)
- MLX (Apple's ML framework)
- Qwen2.5-0.5B model weights (automatically downloaded)
- CNN/DailyMail dataset (automatically downloaded)

**Note**: BitsAndBytes quantization is automatically used on Linux/Windows systems for memory efficiency. On macOS, the model will run in full precision.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mlx-week-6
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The Qwen2.5-0.5B model and CNN/DailyMail dataset will be automatically downloaded on first run.

## Usage

### Basic Training

```bash
python main.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --dataset_name "cnn_dailymail" \
  --lora_rank 16 \
  --learning_rate 1e-4 \
  --batch_size 4 \
  --epochs 3 \
  --output_dir "./trained_model" \
  --generate_samples
```

### Evaluation Only

```bash
python main.py \
  --eval_only \
  --eval_samples 10 \
  --output_dir "./trained_model"
```

### Custom Configuration

```bash
python main.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --lora_rank 32 \
  --lora_dropout 0.2 \
  --learning_rate 5e-5 \
  --batch_size 2 \
  --max_input_length 512 \
  --max_target_length 64 \
  --output_dir "./custom_model"
```

### Example Usage

Run the example script for guided usage:

```bash
python example_usage.py
```

### Testing Boundary Detection

Test the boundary detection functionality:

```bash
# Test boundary detection loss calculation
python test_boundary_detection.py

# Demonstrate boundary detection in training
python example_boundary_training.py
```

## Training

### LoRA Configuration

The model is fine-tuned using LoRA parameters:
- **LoRA Rank**: 16 (default, configurable)
- **LoRA Alpha**: 32 (2 × rank, configurable)
- **Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (attention and MLP layers)
- **Dropout**: 0.1 (configurable)
- **Task Type**: Causal Language Modeling

### Chinchilla Scaling Laws

The training incorporates Chinchilla scaling laws for optimal compute efficiency:
- **Optimal Model Size**: Calculated based on compute budget
- **Optimal Training Tokens**: Calculated based on compute budget
- **Efficiency Analysis**: Automatic recommendations for model/data scaling

### Training Process

1. **Data Preprocessing**: CNN/DailyMail articles are formatted with "Summarize: " prefix and "\nTarget:" for summaries
2. **Tokenization**: Input and target texts are tokenized with Qwen2.5 tokenizer
3. **Boundary Detection**: Automatic detection of input/target boundaries for enhanced loss calculation
4. **LoRA Application**: Low-rank adapters are applied to attention and MLP layers
5. **Training Loop**: Supervised fine-tuning with gradient accumulation and boundary-aware loss
6. **Validation**: Regular evaluation on validation set
7. **Checkpointing**: Automatic saving of best models

### Boundary Detection

The training incorporates intelligent boundary detection to improve the model's ability to learn the transition from input context to target generation:

- **Automatic Boundary Detection**: Identifies transitions from input tokens (marked with -100) to target tokens
- **Enhanced Loss Calculation**: Applies higher loss weights to boundary positions to improve transition learning
- **Smooth Transition Window**: Configurable window size for gradual transition learning
- **Configurable Weights**: Adjustable boundary loss weight and position weight for fine-tuning
- **Clear Delimiters**: Uses "Summarize: " prefix for input and "\nTarget:" prefix for summaries

**Configuration Options:**
```python
# Enable/disable boundary detection
use_boundary_detection: bool = True

# Weight for boundary loss component (default: 0.1)
boundary_loss_weight: float = 0.1

# Weight multiplier for boundary positions (default: 2.0)
boundary_position_weight: float = 2.0

# Number of positions after boundary for smooth transition (default: 4)
boundary_smooth_window: int = 4
```

## Model Architecture

- **Base Model**: Qwen2.5-0.5B with 502 million parameters
- **LoRA Configuration**: Low-rank adaptation for efficient fine-tuning
- **Quantization**: 4-bit quantization for memory efficiency
- **Target Layers**: Attention projection matrices (q_proj, k_proj, v_proj, o_proj) and MLP layers (gate_proj, up_proj, down_proj)
- **Output**: Text summaries with improved quality and coherence
- **Memory Usage**: ~2GB GPU memory (with quantization)

## Performance

### Expected Results

- **Training Time**: ~30-60 minutes on RTX 4090 (3 epochs)
- **Memory Usage**: ~2GB GPU memory with 4-bit quantization (Linux/Windows), ~4GB with full precision (macOS)
- **Model Size**: ~502M parameters base + ~8.8M LoRA parameters
- **Inference Speed**: ~0.5-1 seconds per summary generation

### Evaluation Metrics

- **Loss Reduction**: Significant improvement over base model
- **Summary Quality**: Coherent and relevant summaries
- **Generation Speed**: Fast inference with LoRA adapters
- **Memory Efficiency**: 98%+ parameter reduction with LoRA

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make your changes
5. Add tests if applicable
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Qwen team for the Qwen2.5 base model
- MLX team for the framework
- LoRA paper authors for the efficient fine-tuning method 