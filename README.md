# Aneumo: A Large-Scale Multimodal Aneurysm Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2505.14717-b31b1b.svg)](https://arxiv.org/abs/2505.14717)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/SAIS-Life-Science/Aneumo)

## Project Overview

**Aneumo** is a large-scale, comprehensive, **multimodal** cerebral aneurysm hemodynamics dataset designed to advance machine learning and computational fluid dynamics (CFD) research.

This dataset is built on 427 real aneurysm geometries and includes:

-   **10,660** high-precision 3D models (synthetic aneurysm evolutions).
-   A massive **95,940** total simulation cases, uniquely including:
    -   **85,280 steady-state** simulations (under 8 flow conditions).
    -   **10,660 high-fidelity transient (pulsatile)** simulations.
-   **Pre-computed WSS**: **Wall Shear Stress (WSS)**, velocity, and pressure fields are pre-computed for **all 95,940 cases**.
-   **Multimodal Inputs**: Rich geometric representations including **Segmentation Masks**, **Point Clouds**, and **Meshes**.

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/fig1_3.png?raw=true" width="800px">
</div>
<p align="center"><b>Figure 1:</b> The multimodal data generation pipeline for the Aneumo dataset. (a) Patient-derived geometries are processed into "Aneurysm-free" shapes, then deformed into synthetic models and their corresponding 3D segmentation masks. (b) Key spatial modalities for each geometry include its CFD Mesh, a sampled Point Cloud, and the resulting hemodynamic fields (Velocity, Pressure, WSS) at a representative snapshot. (c) Example of the <b>transient (pulsatile)</b> data, showing the velocity field's evolution at five time points (t=0.2s to 1.0s) within one cardiac cycle.</p>


This project provides deep learning benchmark code for:

1.  **Steady-State Models** — "Syn-to-Real" task (training on synthetic Aneumo, testing on real AneuX data):
    - DeepONet-based model
    - Hybrid model combining Swin Transformer with DeepONet (for **multimodal inputs: masks + point clouds**)

2.  **Transient Models** — Temporal WSS prediction from pulsatile CFD data:
    - **Aneumo (Temporal DeepONet V2)**: History encoder (Transformer/MLP) + Swin Transformer geometry encoder + DeepONet with cross-attention
    - **Baselines**: FNO, 3D U-Net, MeshGraphNet
    - **Cross-geometry Generalization**: Multi-case training and evaluation with geometry-split and time-split protocols

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/network.png?raw=true"  width="800px">
  <p><b>Figure 2:</b> Schematic illustration of the DeepONet-SwinT model architecture for predicting aneurysm hemodynamic parameters.</p>
</div>

## Dataset Features and Contributions

### Key Contributions

1.  **First Large-Scale, Dual-Mode Hemodynamics Dataset**: Provides **95,940** total hemodynamic data samples. This is the first dataset to include **10,660 high-fidelity transient (pulsatile) simulations** at this scale, filling a critical gap for spatio-temporal AI models.
2.  **Pre-Computed WSS for All Cases**: To maximize usability and lower the barrier for AI research, the critical **Wall Shear Stress (WSS)** fields are pre-computed for **all 95,940** steady-state and transient cases.
3.  **Rich Multimodal Inputs**: Based on 427 real geometries, 10,660 models are provided, each represented as a **3D Mesh**, a **Segmentation Mask**, and a **Point Cloud**, supporting diverse CVPR tasks (GNNs, Transformers, CNNs).
4.  **Massive Scale & "Syn-to-Real" Benchmark**: The dataset generation required over **11 million CPU core-hours**. We also provide a benchmark for **Syn-to-Real** generalization (training on synthetic Aneumo data, testing on unseen real AneuX data).

### Data Format and Organization

The dataset provides multiple formats to support different research tasks:

| File Type                                   | Description                                                |
| ------------------------------------------- | ---------------------------------------------------------- |
| Mask/*case_id*.nii.gz                     | Segmented vessel ROI area for vessel region identification |
| Stl/*case_id*.stl                         | Surface mesh of vessel geometry                            |
| Mesh/*case_id*.msh                        | Computational mesh files for fluid simulation              |
| VTK/*flow*/inlet.vtp                      | Fluid characteristics at inlet boundary                    |
| VTK/*flow*/internal.vtu                   | Fluid characteristics in internal volume                   |
| VTK/*flow*/outlet.vtp                     | Fluid characteristics at outlet boundary                   |
| VTK/*flow*/wall.vtp                       | Fluid characteristics at wall boundary                     |
| NPY/*flow*/array_inlet_*case_id*.npy    | Numerical data for inlet boundary conditions               |
| NPY/*flow*/array_internal_*case_id*.npy | Numerical data for internal volume conditions              |
| NPY/*flow*/array_outlet_*case_id*.npy   | Numerical data for outlet boundary conditions              |
| NPY/*flow*/array_wall_*case_id*.npy     | Numerical data for wall boundary conditions                |

All data formats are compatible with mainstream machine learning frameworks (PyTorch, TensorFlow) and standard CFD analysis tools.

## Repository Structure

```
├── inference_deeponet.py         # [Steady] DeepONet inference script
├── inference_swint.py            # [Steady] Swin+DeepONet inference script
├── cfd_opt_deeponet/             # [Steady] DeepONet model implementation
│   ├── main_train.py             # Training script
│   ├── checkpoint/               # Pre-trained checkpoints
│   ├── config/                   # Configuration files
│   ├── model/                    # Model architecture
│   └── ...
├── cfd_opt_swin_deeponet/        # [Steady] Swin+DeepONet model implementation
│   ├── main_train_swin.py        # Training script
│   ├── checkpoint/               # Pre-trained checkpoints
│   ├── config/                   # Configuration files
│   ├── model/                    # Model architecture
│   └── ...
├── transient/                    # [Transient] Core model (Temporal DeepONet V2)
│   ├── models.py                 # Network architectures
│   ├── dataset.py                # Temporal CFD dataset loader
│   ├── losses.py                 # Loss functions
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── inference.py              # Full-mesh inference
├── baselines/                    # [Transient] Baseline models
│   ├── models/                   # FNO, U-Net, MeshGraphNet
│   ├── datasets/                 # Voxelized & cross-geometry datasets
│   ├── train.py                  # Single-case training
│   ├── train_cross.py            # Cross-geometry training (multi-GPU DDP)
│   ├── evaluate.py               # Single-case evaluation
│   └── evaluate_cross.py         # Cross-geometry evaluation
├── Data_preprocessing/           # Data preprocessing (steady-state + transient)
├── scripts/                      # [Transient] Shell scripts for training
├── visualization/                # [Transient] Result visualization tools
├── real_data/                    # [Steady] Sample data for inference testing
│   ├── cfd_data/                 # CFD simulation data samples
│   └── img_data/                 # 3D image data samples
├── result/                       # [Transient] WSS comparison visualization
├── fig/                          # Figures for documentation
├── MPs.csv                       # Morphometric parameters of 10,660 aneurysms (computed via VMTK)
├── Connection.csv                # Geometry-to-base-case mapping
└── datasheet_aneumo.md           # Dataset datasheet (Gebru et al. framework)
```

## Quick Start

### Environment Requirements

- Python 3.8+
- CUDA-enabled GPU
- PyTorch 2.0+

### Installing Dependencies

```bash
# For transient models and baselines (root-level)
pip install -r requirements.txt

# Some baselines require additional packages:
# MeshGraphNet: pip install torch_scatter (match your PyTorch + CUDA version)
# FNO: pip install neuralop
# Geometry encoder: pip install monai
```

For steady-state models, install from the subfolder:
```bash
# DeepONet model
cd cfd_opt_deeponet && pip install -r requirements.txt

# Swin+DeepONet model
cd cfd_opt_swin_deeponet && pip install -r requirements.txt
```

---

## Steady-State Models

### Training

```bash
# DeepONet model
cd cfd_opt_deeponet
torchrun --nproc_per_node=NUM_GPUS ./main_train.py

# Swin+DeepONet model
cd cfd_opt_swin_deeponet
torchrun --nproc_per_node=NUM_GPUS ./main_train_swin.py
```

Model parameters and configurations can be adjusted in the `config/default.yaml` file in each model directory. Each implementation includes:

- Distributed training with DDP (DistributedDataParallel)
- Mixed precision training for faster computation
- Checkpoint saving and resumption
- Comprehensive logging and monitoring
- Customizable loss functions and optimizers

### Inference

```bash
# Using DeepONet model for inference
python inference_deeponet.py

# Using Swin+DeepONet model for inference
python inference_swint.py
```

### Sample Data

The repository includes sample data in the `real_data` directory for quick testing:

```
real_data/
  ├── cfd_data/              # CFD simulation data
  │   ├── m=0.002/           # Flow rate = 0.002 kg/s
  │   │   └── 4.npz          # Case ID 4 
  │   └── m=0.003/           # Flow rate = 0.003 kg/s
  │       └── 4.npz          # Case ID 4
  └── img_data/              # 3D image data for Swin+DeepONet model
      └── 4.npy              # Image data for case ID 4
```

---

## Transient Models

### Data Format (HDF5)

Input data for transient models is stored in HDF5 format:

```
case_XXX.h5
├── mesh/
│   ├── coords         [N, 3]       # Node coordinates (x, y, z)
│   ├── node_type      [N]          # 0=internal, 1=inflow, 2=outflow, 3=wall
│   ├── wall_indices   [M]          # Indices of wall nodes
│   └── edges          [E, 2]       # Mesh connectivity
├── fields/
│   ├── velocity       [T, N, 3]    # Velocity (u, v, w)
│   ├── pressure       [T, N, 1]    # Pressure
│   └── wss            [T, M, 3]    # Wall shear stress (on wall nodes)
└── time_values        [T]          # Timestep values
```

### Data Preprocessing

Convert raw VTK simulation output to HDF5:

```bash
# Single case
python Data_preprocessing/convert_single.py --case_id 201

# Multi-case (100 cases for cross-geometry experiments)
python Data_preprocessing/convert_multi_cross.py --num-workers 16
```

### Training

#### Aneumo (Temporal DeepONet V2)

```bash
# With Swin geometry encoder
bash scripts/train_aneumo.sh 0 with_swin

# Without geometry encoder (ablation)
bash scripts/train_aneumo.sh 0 no_swin

# Or directly:
python -m transient.train \
    --h5_path data/case.h5 \
    --output_vars wss \
    --history_encoder transformer \
    --boundary_cut 0.1 \
    --epochs 10000
```

#### Baselines

```bash
# All baselines on single GPU
bash scripts/train_baselines.sh 0

# Individual baseline
python -m baselines.train --model fno --h5_path data/case.h5 --epochs 10000
python -m baselines.train --model unet --h5_path data/case.h5 --epochs 10000
python -m baselines.train --model mgn --h5_path data/case.h5 --epochs 10000
```

#### Cross-Geometry Generalization

```bash
# Geometry split (hold out deforms)
bash scripts/train_cross_geometry.sh fno
bash scripts/train_cross_geometry.sh deeponet

# Time split (front 80% train, back 20% test)
bash scripts/train_cross_time.sh mgn

# Multi-GPU
NGPU=4 bash scripts/train_cross_geometry.sh unet
```

### Evaluation

```bash
# Aneumo
python -m transient.evaluate --checkpoint checkpoint/v2_with_swin/best_model.pt

# Baselines
python -m baselines.evaluate --model fno --checkpoint checkpoint_baselines/fno/best_model.pt

# Cross-geometry
python -m baselines.evaluate_cross --model fno --split_mode geometry
```

### Model Architecture

#### Aneumo (Temporal DeepONet V2)

```
Input: History frames (t-N to t) + Query coordinates (t+1 to t+M)
                    │
    ┌───────────────┼───────────────┐
    │               │               │
History Encoder  Swin Geometry   Trunk Network
(Transformer/MLP) Encoder(3D)   (MLP + CrossAttn)
    │               │               │
    └───────┬───────┘               │
            │                       │
      Branch Fusion            Query Features
     (history + geo)                │
            │                       │
            └───── DeepONet ────────┘
                    │
              WSS Prediction
```

#### Baselines

| Model | Type | Input | Parameters |
|-------|------|-------|-----------|
| FNO | Spectral | Voxelized 3D grid | ~1M |
| U-Net | CNN | Voxelized 3D grid | ~1M |
| MeshGraphNet | GNN | Mesh graph | ~1M |

---

## Visualization

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/Vmax.png?raw=true" alt="Vmax" width="45%">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/dp.png?raw=true" alt="DP" width="45%">
</div>
<p align="center"><b>Figure 3:</b> Hemodynamic parameter visualization: maximum velocity (left) and normalized pressure difference (right)</p>

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/inference_cfd.png?raw=true"  width="800px">
  <p><b>Figure 4:</b> Comparison of predicted pressure and velocity fields by DeepONet and DeepONet-SwinT with CFD ground truth, including (a) Wall pressure contour plots, (b) Internal pressure contour plots, and (c) Internal velocity contour plots.</p>
</div>

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/wss.png?raw=true"  width="800px">
  <p><b>Figure 5:</b> The model accurately forecasts the dynamic propagation of high-stress regions and pulsatile peaks over time.</p>
</div>

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/result/comparison_wss.png?raw=true"  width="800px">
  <p><b>Figure 6:</b> Transient WSS comparison across models — predicted vs. ground truth Wall Shear Stress fields.</p>
</div>

## Morphometric Parameters

The file [`MPs.csv`](MPs.csv) provides 15 clinically relevant morphometric parameters for all 10,660 aneurysm geometries, computed using [VMTK](http://www.vmtk.org/). Parameters include Size, Neck Width (NW), Aspect Ratio (AR), Height, Max Diameter, Volume, Surface Area, Bottleneck Factor (BF), Non-Sphericity Index (NSI), Ellipticity Index (EI), Undulation Index (UI), and more.

These parameters have been validated against clinical ranges (see [`datasheet_aneumo.md`](datasheet_aneumo.md) Section 8 for details).

## Dataset Datasheet

A comprehensive dataset datasheet following the [Gebru et al. (2021)](https://doi.org/10.1145/3458723) framework is provided in [`datasheet_aneumo.md`](datasheet_aneumo.md), covering motivation, composition, collection process, preprocessing, intended uses, distribution, and maintenance.

## Data Access

-   **Massive Scale**: The dataset required over **11 million CPU core-hours** to generate.
-   **Total Storage**:
    -   **Steady-State (85,280 cases)**: ~23.1 TB (raw), processed to **4.0 TB**.
    -   **Transient (10,660 cases)**: **100+ TB** (raw).

-   **Download & Availability**:
    -   **Available Now**: The 4.0 TB processed **steady-state dataset** (containing Velocity & Pressure fields) is available at: [Aneumo Dataset on HuggingFace](https://huggingface.co/datasets/SAIS-Life-Science/Aneumo).
    -   **Coming Soon**: Due to the massive scale, the **steady-state WSS fields** and the entire **100+ TB transient dataset** are being sliced for easier access. We are gradually uploading the full transient dataset to Hugging Face, with **1,000 already uploaded**. **Please stay tuned!**

-   For detailed data descriptions, generation methods, and benchmark results, please refer to our [paper](https://arxiv.org/abs/2505.14717).

## Citation

If you use the Aneumo dataset or our code in your research, please cite:

```bibtex
@article{li2025aneumo,
  title={Aneumo: A Large-Scale Multimodal Aneurysm Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks},
  author={Xigui Li and Yuanye Zhou and Feiyang Xiao and Xin Guo and Chen Jiang and Tan Pan and Xingmeng Zhang and Cenyu Liu and Zeyun Miao and Jianchao Ge and Xiansheng Wang and Qimeng Wang and Yichi Zhang and Wenbo Zhang and Fengping Zhu and Limei Han and Yuan Qi and Chensen Lin and Yuan Cheng},
  journal={arXiv preprint arXiv:2505.14717},
  year={2025}
}
```

## Acknowledgements

This dataset incorporates real aneurysm data partly sourced from the [AneuX dataset](https://github.com/hirsch-lab/aneuxdb). We express our gratitude for their open-source contributions.
