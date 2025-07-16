# DeepONet-CFD: Deep Neural Operator Network for CFD Surrogate Modeling

## Project Overview

DeepONet-CFD is a deep learning-based surrogate model for computational fluid dynamics (CFD) simulations. It uses Deep Operator Networks (DeepONet) architecture to predict fluid flow properties, significantly reducing computational cost compared to traditional CFD methods.

### Key Features

- DeepONet architecture with trunk and branch networks
- Distributed training with PyTorch FSDP (Fully Sharded Data Parallel)
- Mixed precision training support
- Gradient accumulation for large batch training
- Comprehensive logging and checkpointing

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU
- PyTorch 2.0+

### Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
Final_Swin+deeponet/
│
├── config/                 # Configuration files
│   └── default.yaml       # Default model and training settings
│
├──data/
│   ├── train.csv 
│   └── val.csv 
|
├── dataloader/            # Data loading utilities
│   ├── Dataset.py         # CFD dataset implementation
│   ├── build_dataloader.py# DataLoader construction
│   └── load_data.py       # Data loading helpers
│
├── loss/                  # Loss function definitions
│   └── loss2.py          # Custom loss for CFD prediction
│
├── model/                 # Model architecture
│   ├── builder.py         # Model construction utilities
│   └── NeuralNetworks.py # Network definitions
│
├── utils/                 # Utility functions
│   ├── arg_parser.py     # Command line argument parsing
│   ├── checkpoint.py     # Model checkpointing
│   ├── distributed.py    # Distributed training setup
│   ├── logger.py         # Logging utilities
│   └── process_dict.py   # State dict processing
│
├── main_train.py               # Main training script
└── README.md             # Project documentation
```

## Usage

### Training

To train the model with distributed data parallel:

```bash
cd /Final_Swin+deeponet
torchrun --nproc_per_node=NUM_GPUS ./main_train_swin.py
```

Where `NUM_GPUS` is the number of GPUs you want to use.

### Configuration

Model and training parameters can be configured in `config/default.yaml`. Key configuration options include:

```yaml
model:
  branch_img:
    embed_size: 24
  trunk:
    in_dim: 4
    out_dim: 256
    hidden: 64
    layers: 4
  branch_bc:
    in_dim: 7
    out_dim: 64
    hidden: 64
    layers: 4
  branch_bp:
    in_dim: 1
    out_dim: 4
train:
  dtype: bf16         # fp32 | fp16 | bf16
  learning_rate : 1e-4
  epochs: 10
  batch_size: 8
  resume_epoch: 0
```
