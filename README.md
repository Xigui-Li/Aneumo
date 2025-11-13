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


This project also provides deep learning benchmark code for our **"Syn-to-Real"** task (training on synthetic Aneumo, testing on real AneuX data) for efficient hemodynamic prediction:

1.  DeepONet-based model
2.  Hybrid model combining Swin Transformer with DeepONet (for **multimodal inputs: masks + point clouds**)

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

## Code Structure and Functionality

The project contains implementations of two main deep learning models and their inference code:

### 1. DeepONet Model (`cfd_opt-deeponet`)

- CFD surrogate model based on DeepONet architecture
- Implementation files:
  - `main_train.py` - Training script
  - `inference_deeponet.py` - Inference script in root directory
- Features:
  - Efficient prediction of hemodynamic parameters without requiring image data
  - Distributed training support
  - Mixed precision computation
  - Checkpoint resumption functionality

### 2. Swin+DeepONet Hybrid Model (`cfd_opt-swin+deeponet`)

- Hybrid architecture combining Swin Transformer with DeepONet
- Implementation files:
  - `main_train_swin.py` - Training script
  - `inference_swint.py` - Inference script in root directory
- Features:
  - Better spatial feature capture using Swin Transformer for 3D medical images
  - Joint extraction of fluid dynamics features with DeepONet
  - Support for distributed training and mixed precision computation

### Repository Structure

```
├── inference_deeponet.py         # DeepONet model inference script
├── inference_swint.py            # Swin+DeepONet model inference script
├── cfd_opt_deeponet/             # DeepONet model implementation
│   ├── main_train.py             # Training script
│   ├── checkpoint/               # Model checkpoints
│   │   └── deeponet/             # Pre-trained DeepONet checkpoints
│   ├── config/                   # Configuration files
│   ├── model/                    # Model architecture definition
│   └── ...
├── cfd_opt_swin_deeponet/        # Swin+DeepONet model implementation
│   ├── main_train_swin.py        # Training script
│   ├── checkpoint/               # Model checkpoints
│   │   └── deeponet_swin/        # Pre-trained Swin+DeepONet checkpoints
│   ├── config/                   # Configuration files
│   ├── model/                    # Model architecture definition
│   └── ...
└── real_data/                    # Sample data for inference testing
│   ├── cfd_data/                 # CFD simulation data samples
│   └── img_data/                 # 3D image data samples
├── Data_preprocessing/           # Data preprocessing scripts and instructions    
```

## Quick Start

### Environment Requirements

- Python 3.8+
- CUDA-enabled GPU
- PyTorch 2.0+
- Additional dependencies listed in the `requirements.txt` files in each subfolder

### Installing Dependencies

```bash
# For DeepONet model
cd cfd_opt-deeponet
pip install -r requirements.txt

# Or for Swin+DeepONet model
cd cfd_opt-swin+deeponet
pip install -r requirements.txt
```

### Sample Data and Quick Test

The repository includes sample data in the `real_data` directory for quick testing:

```bash
# Sample CFD data structure
real_data/
  ├── cfd_data/              # CFD simulation data
  │   ├── m=0.002/           # Flow rate = 0.002 kg/s
  │   │   └── 4.npz          # Case ID 4 
  │   └── m=0.003/           # Flow rate = 0.003 kg/s
  │       └── 4.npz          # Case ID 4
  └── img_data/              # 3D image data for Swin+DeepONet model
      └── 4.npy              # Image data for case ID 4
```


### Training Models

```bash
# DeepONet model
cd cfd_opt-deeponet
torchrun --nproc_per_node=NUM_GPUS ./main_train.py

# Swin+DeepONet model
cd cfd_opt-swin+deeponet
torchrun --nproc_per_node=NUM_GPUS ./main_train_swin.py
```

Model parameters and configurations can be adjusted in the `config/default.yaml` file in each model directory. Each implementation includes:

- Distributed training with DDP (DistributedDataParallel)
- Mixed precision training for faster computation
- Checkpoint saving and resumption
- Comprehensive logging and monitoring
- Customizable loss functions and optimizers

### Inference

Two ready-to-use inference scripts are provided at the root level of the repository:

1. `inference_deeponet.py` - For running inference with DeepONet model
2. `inference_swint.py` - For running inference with the hybrid Swin Transformer + DeepONet model

The repository includes pre-trained model checkpoints and sample data for quick testing:

```bash
# Using DeepONet model for inference
python inference_deeponet.py

# Using Swin+DeepONet model for inference
python inference_swint.py
```


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


## Data Access

-   **Massive Scale**: The dataset required over **11 million CPU core-hours** to generate.
-   **Total Storage**:
    -   **Steady-State (85,280 cases)**: ~23.1 TB (raw), processed to **4.0 TB**.
    -   **Transient (10,660 cases)**: **100+ TB** (raw).

-   **Download & Availability**:
    -   **Available Now**: The 4.0 TB processed **steady-state dataset** (containing Velocity & Pressure fields) is available at: [Aneumo Dataset on HuggingFace 🤗](https://huggingface.co/datasets/SAIS-Life-Science/Aneumo).
    -   ⏳ **Coming Soon**: Due to the massive scale, the **steady-state WSS fields** and the entire **100+ TB transient dataset** are currently being sliced for easier access. We are uploading these to Hugging Face incrementally. **Please stay tuned!**

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
