# Multi-Task CAD Generation Using Compact Decoder-Only Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Multi-Task CAD Generation Using Compact Decoder-Only Models"** accepted at IEEE ICAAIML 2026.

This repository provides a lightweight, unified framework for generating parametric CAD models from multiple input modalities: **B-Rep geometry**, **images**, and **natural language descriptions**. Our approach leverages compact decoder-only language models with modality-specific encoders and efficient parameter adaptation techniques.

---

## 🔬 Overview

Traditional CAD generation systems are limited to single modalities and often require large, specialized architectures. We introduce **CADCoder**, a family of compact multi-task models that:

- **Unified Architecture**: Single decoder-only backbone (DeepSeek-Coder 1.3B-7B) handles all modalities
- **Modality-Specific Encoders**: Lightweight prefix encoders for B-Rep, image, and text inputs
- **Efficient Adaptation**: LoRA fine-tuning enables rapid training with minimal parameters
- **High Quality Output**: Generates executable Python code (CadQuery/RapidCAD-Py) for downstream CAD systems
- **Comprehensive Evaluation**: Chamfer distance, F-score, normal consistency, CodeBLEU, and IoU metrics

### Key Features

✅ **Multi-modal support**: B-Rep, images, text descriptions, and sequence completion  
✅ **Memory efficient**: 4-bit quantization, gradient checkpointing, LoRA (rank 64)  
✅ **Production ready**: Full training/inference pipeline with checkpointing and logging  
✅ **Comprehensive metrics**: Geometric accuracy, code quality, and execution success rates  
✅ **Extensible**: Easy to add new encoders or CAD output formats  

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Evaluation & Inference](#-evaluation--inference)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Hardware Requirements](#-hardware-requirements)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Installation

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Key Dependencies

- **PyTorch** 2.4.0+ with CUDA support
- **Lightning** 2.0+ for training orchestration
- **Transformers** 4.36+ (HuggingFace)
- **DeepSpeed** 0.10+ for distributed training
- **PEFT** 0.5+ for LoRA implementation
- **DGL** 1.1+ for graph neural networks (B-Rep encoder)
- **CadQuery** for CAD code execution
- **Weights & Biases** for experiment tracking

See [requirements.txt](requirements.txt) or [setup.py](setup.py) for complete dependency list.

---

## 📦 Data Preparation

### 1. Download Dataset

We use the **Omni-CAD** dataset for training:

```bash
mkdir data
cd data
wget https://huggingface.co/datasets/jingwei-xu-00/Omni-CAD/resolve/main/Omni-CAD.zip
unzip Omni-CAD.zip
cd ..
```

### 2. Configure Data Path

Edit [config.yaml](config.yaml) to specify your data location:

```yaml
data:
  root_dir: "/path/to/your/data"
  train_ratio: 0.8
  val_ratio: 0.1
```

### 3. Generate CAD Code

Convert JSON representations to executable Python code:

```bash
python scripts/create_cad_code.py
```

**Note**: Requires RapidCAD-Py library installed in your environment.

### 4. Create Modality-Specific Data

Generate B-Rep graphs and rendered images:

```bash
# Generate B-Rep boundary representations
python scripts/create_brep_data.py

# Convert STEP files to graph structures
python scripts/step_to_brep_graph.py

# Render CAD models to images (if using image modality)
python scripts/create_image_data.py
```

**Note**: Text descriptions are directly sourced from Omni-CAD—no additional preprocessing needed.

---

## 🏗️ Model Architecture

### Architecture Details

```
Input (B-Rep/Image/Text)
    ↓
Modality Encoder (frozen or lightly trained)
    ↓
Adaptive Projection Layer
    ↓
Virtual Prefix Tokens (12 tokens, dim=4096)
    ↓
DeepSeek-Coder Decoder (LoRA-adapted)
    ↓
CAD Code Generation (max 512 tokens)
```

**Key Innovations**:
- **Prefix Tuning**: 12 virtual tokens encode modality-specific information
- **LoRA Adaptation**: Rank-64 low-rank updates to decoder attention layers
- **Alignment Loss**: Auxiliary loss aligns encoder/decoder representations (weight 0.1-1.0)
- **Quantization**: 4-bit NF4 quantization for memory efficiency (optional)

### Encoder Specifications

- **B-Rep Encoder**: 4-layer GAT with UV-Net curve/surface encoders (256-dim embeddings)
- **Text Encoder**: BERT-base-uncased (768-dim) with 2-layer MLP projection
- **Image Encoder**: ViT-base-patch16 (224×224) with patch token selection

---

## 🎯 Training

### Single Modality Training

Use the unified training script with modality-specific configurations:

```bash
# B-Rep to CAD code
python train.py --config cadcoderbrep

# Text to CAD code
python train.py --config cadcodertext

# Image to CAD code
python train.py --config cadcoderimage

# Sequence completion
python train.py --config cadcoderseqcompl
```

### Configuration Files

All hyperparameters are specified in YAML files under [config/](config/):

- [cadcoderbrep.yaml](config/cadcoderbrep.yaml): B-Rep modality settings
- [cadcodertext.yaml](config/cadcodertext.yaml): Text modality settings
- [cadcoderimage.yaml](config/cadcoderimage.yaml): Image modality settings
- [cadcoderseqcompl.yaml](config/cadcoderseqcompl.yaml): Sequence completion settings
- [config.yaml](config.yaml): Global defaults

**Key Hyperparameters**:

```yaml
training:
  max_epochs: 75
  batch_size: 1-2  # Adjust based on GPU memory
  learning_rate: 1e-4
  gradient_clip_val: 1.0

cadcoder:
  num_virtual_tokens: 12
  lora_r: 64
  quantization: false  # Enable for 4-bit training
  use_lora: true

inference:
  max_new_tokens: 512
  temperature: 0.2
  top_k: 10
```

### Multi-GPU Training

Enable DeepSpeed for distributed training:

```yaml
deepspeed:
  multi_gpu: true
  zero_stage: 1  # Stage 1, 2, or 3
```

```bash
# Launch with DeepSpeed
deepspeed --num_gpus=4 train.py --config cadcoderbrep
```

### Logging & Experiment Tracking

Supported loggers:
- **Weights & Biases** (recommended): Real-time metrics, model artifacts, code samples
- **MLflow**: Local tracking with artifact storage
- **TensorBoard**: Basic metric visualization

Configure in [config.yaml](config.yaml):

```yaml
logging:
  logger: "wandb"  # or "mlflow", "tensorboard"
  logger_mode: "online"  # or "offline", "disabled"
  logger_tag: "experiment_name"
```

---

## 📊 Evaluation & Inference

### Automatic Evaluation

All metrics reported in the paper are computed automatically during validation:

- **Geometric Metrics**:
  - Chamfer Distance (lower is better)
  - F-Score @ 0.05 threshold (higher is better)
  - Normal Consistency (higher is better)
  - Intersection over Union (higher is better)

- **Code Quality Metrics**:
  - CodeBLEU (BLEU + AST + dataflow similarity)
  - Syntax validity rate
  - Execution success rate

- **Generation Metrics**:
  - Tokens per second
  - Invalid code ratio
  - Average sequence length

### Running Evaluation

```bash
# Evaluate on test set
python train.py --config cadcoderbrep --mode inference

# Run HumanEval-style evaluation
python scripts/run_humaneval.py --checkpoint /path/to/model.ckpt

# Detailed metrics per model
python scripts/detailed_model_metrics.py
```

### Inference on Custom Inputs

```python
from models import BrepCadcoder
from omegaconf import OmegaConf

# Load config and checkpoint
config = OmegaConf.load("config/cadcoderbrep.yaml")
model = BrepCadcoder.load_from_checkpoint("model.ckpt", config=config)

# Generate CAD code from B-Rep
brep_input = load_brep_graph("example.step")
cad_code = model.generate(brep_input, max_length=512)
print(cad_code)
```

### Visualization

Visualize predictions vs. ground truth:

```bash
# Generate comparison plots
python scripts/visualize_cadcoder_brep_results.py

# Analyze invalid code reasons
python scripts/visualize_invalid_reasons.py

# Compare model performance
python scripts/visualize_base_model_performance.py
```

---

## 📁 Repository Structure

```
cadcoder/
├── config/                          # Configuration files
│   ├── config.yaml                  # Global defaults
│   ├── cadcoderbrep.yaml            # B-Rep modality config
│   ├── cadcodertext.yaml            # Text modality config
│   ├── cadcoderimage.yaml           # Image modality config
│   └── cadcoderseqcompl.yaml        # Sequence completion config
│
├── data_loader/                     # Dataset implementations
│   ├── brep_to_rapidcadpy_dataset.py    # B-Rep dataset
│   ├── text_to_rapidcadpy_dataset.py    # Text dataset
│   ├── image_to_rapidcadpy_dataset.py   # Image dataset
│   ├── shaft_dataset.py                 # Sequence completion dataset
│   └── rapidcadpy_dataset.py            # Base dataset class
│
├── models/                          # Model architectures
│   ├── cadcoder_base.py             # Shared base model (training loop)
│   ├── cadcoder_brep.py             # B-Rep variant
│   ├── cadcoder_text.py             # Text variant
│   ├── cadcoder_image.py            # Image variant
│   ├── cadcoder_seq_completion.py   # Sequence completion variant
│   ├── encoder_brep.py              # Graph neural network encoder
│   ├── encoder_text.py              # BERT-based text encoder
│   ├── encoder_image.py             # ViT-based image encoder
│   ├── alignment_loss.py            # Encoder-decoder alignment loss
│   ├── config.py                    # Configuration dataclasses
│   ├── configuration_brepcadcoder.py # HF model config
│   └── modeling_brepcadcoder.py     # HF model wrapper
│
├── modules/                         # Utilities
│   ├── cad_evaluator.py             # Evaluation metrics
│   ├── execute_cad_code.py          # Safe code execution
│   ├── brep_to_graph.py             # B-Rep graph conversion
│   └── custom_logging.py            # Logging callbacks
│
├── scripts/                         # Data processing & analysis
│   ├── create_cad_code.py           # JSON → Python conversion
│   ├── create_brep_data.py          # Generate B-Rep files
│   ├── step_to_brep_graph.py        # STEP → graph conversion
│   ├── run_humaneval.py             # Evaluation script
│   ├── visualize_*.py               # Visualization scripts
│   └── prepare_deepseek_data.py     # Fine-tuning data prep
│
├── train.py                         # Main training entry point
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
└── README.md                        # This file
```

---

## 💻 Hardware Requirements

### Recommended Setup

Our experiments were conducted on:
- **GPUs**: 4× NVIDIA L40S (48GB each)
- **Total GPU Memory**: 196GB
- **System RAM**: 256GB
- **Storage**: ~500GB for dataset and checkpoints

### Minimum Requirements

With reduced batch size and quantization:
- **GPUs**: 1× NVIDIA RTX 4090 (24GB) or equivalent
- **GPU Memory**: 24GB (with 4-bit quantization)
- **System RAM**: 64GB
- **Storage**: ~200GB

### Memory Optimization Tips

```yaml
# Enable in config YAML
cadcoder:
  quantization: true  # 4-bit quantization (~50% memory reduction)
  
optimizer:
  gradient_checkpointing: true  # Trade compute for memory
  accumulate_grad_batches: 4    # Effective batch size increase
  
training:
  batch_size: 1  # Reduce if OOM
```

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{Berger2026MultiTaskCAD,
  title     = {Multi-Task CAD Generation Using Compact Decoder-Only Models},
  author    = {Berger, Elias and Mehlst{\"a}ubl, Jan and Saske, Bernhard and Paetzold-Byhain, Kristin},
  booktitle = {Proceedings of the IEEE 2026 International Conference on Advances in Artificial Intelligence and Machine Learning},
  year      = {2026},
  address   = {Tokyo, Japan},
  month     = mar,
  publisher = {IEEE}
}
```

---

## Acknowledgments

- **Omni-CAD Dataset**: [jingwei-xu-00/Omni-CAD](https://huggingface.co/datasets/jingwei-xu-00/Omni-CAD)
- **DeepSeek-Coder**: [deepseek-ai](https://github.com/deepseek-ai/DeepSeek-Coder)
- **CadQuery**: [CadQuery Project](https://github.com/CadQuery/cadquery)
- **RapidCAD-Py**: [RapidCad-Py](https://github.com/rapidcad/rapidcadpy)
