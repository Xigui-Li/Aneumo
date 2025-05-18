# Aneumo: A Large-Scale Multimodal Aneurysm Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2501.09980-b31b1b.svg)](https://arxiv.org/abs/2501.09980)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/SAIS-Life-Science/Aneumo)

## Project Overview

**Aneumo** is a large-scale, comprehensive cerebral aneurysm hemodynamics dataset designed to advance machine learning and computational fluid dynamics (CFD) research related to cerebral aneurysms. The dataset includes:

- 10,660 high-precision 3D models generated through controlled deformation techniques based on 427 real aneurysm geometries
- 85,280 hemodynamic simulation data sets under 8 different flow conditions
- Medical imaging-style segmentation masks and multiple data representation formats


<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/workflow.png"  width="1200px">
  <p><b>Figure 1:</b> Workflow for deforming patient-specific aneurysm models and simulating vascular hemodynamics. (a) Patient-derived aneurysmal geometries are first processed to remove the aneurysm and recover a healthy vascular shape. Controlled geometric deformations are then applied to generate synthetic aneurysm models. (b) CFD meshes are created for the deformed geometries, followed by simulations of blood flow velocity and pressure fields for hemodynamic analysis.</p>
</div>



This project also provides two deep learning CFD surrogate modeling code implementations for efficient prediction of hemodynamic parameters:

1. DeepONet-based model
2. Hybrid model combining Swin Transformer with DeepONet

![Model Architecture](/Users/lixigui/Downloads/manuscript/1st/1/figures/network.pdf)

<p align="center"><b>Figure 2:</b> Schematic illustration of the DeepONet-SwinT model architecture for predicting aneurysm hemodynamic parameters</p>

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/network.pdf"  width="1200px">
  <p><b>Figure 2:</b> Schematic illustration of the DeepONet-SwinT model architecture for predicting aneurysm hemodynamic parameters.</p>
</div>



## Dataset Features and Contributions

### Key Contributions

1. **First Large-Scale High-Fidelity Hemodynamics Dataset**: Provides 85,280 hemodynamic data samples (velocity/pressure fields) generated through CFD simulations under various physiological flow conditions (0.001–0.004 kg/s), filling a critical gap in AI-driven hemodynamic modeling.
2. **Diverse Aneurysm Geometry Evolution Data**: Based on 427 real aneurysm geometries, 10,660 high-quality 3D models were generated through controlled deformation techniques, comprehensively capturing aneurysm geometric evolution across different stages and supporting quantitative modeling of rupture risk.
3. **Multimodal Data Fusion Framework**: The dataset includes high-resolution binary segmentation mask images precisely aligned with CFD parameters, supporting multimodal learning tasks and facilitating multi-scale feature mapping in complex flow environments.

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

The project contains implementations of two main deep learning models:

### 1. DeepONet Model (`cfd_opt-deeponet`)

- CFD surrogate model based on DeepONet architecture
- Implementation file: `v3_8.py`
- Features:
  - Efficient prediction of hemodynamic parameters
  - Distributed training support
  - Mixed precision computation
  - Checkpoint resumption functionality

### 2. Swin+DeepONet Hybrid Model (`cfd_opt-swin+deeponet`)

- Hybrid architecture combining Swin Transformer with DeepONet
- Implementation file: `v3_swin_bs8.py`
- Features:
  - Better spatial feature capture using Swin Transformer
  - Joint extraction of fluid dynamics features with DeepONet
  - Support for distributed training and mixed precision computation
  - Inference scripts provided (`inference3.py`, `inference4.py`)

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

### Training Models

```bash
# DeepONet model
cd cfd_opt-deeponet
torchrun --nproc_per_node=NUM_GPUS ./v3_8.py

# Swin+DeepONet model
cd cfd_opt-swin+deeponet
torchrun --nproc_per_node=NUM_GPUS ./v3_swin_bs8.py
```

Model parameters and configurations can be adjusted in the `config/default.yaml` file in each model directory.

### Inference

```bash
# Using Swin+DeepONet model for inference
cd cfd_opt-swin+deeponet
python inference_swint.py
```

## Visualization

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/Vmax.png" alt="Vmax" width="45%">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/dp.png" alt="DP" width="45%">
</div>
<p align="center"><b>Figure 3:</b> Hemodynamic parameter visualization: maximum velocity (left) and normalized pressure difference (right)</p>

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/inference_cfd.pdf"  width="1200px">
  <p><b>Figure 4:</b> Comparison of predicted pressure and velocity fields by DeepONet and DeepONet-SwinT with CFD ground truth, including (a) Wall pressure contour plots, (b) Internal pressure contour plots, and (c) Internal velocity contour plots.</p>
</div>



## Data Access

- The dataset is available for download, with a total size of approximately 4TB; each compressed package contains 40 case_ids (e.g., 1-40)
- Download link: [https://huggingface.co/datasets/SAIS-Life-Science/Aneumo](https://huggingface.co/datasets/SAIS-Life-Science/Aneumo)
- For detailed data descriptions, generation methods, and experimental results, please refer to our [paper](https://arxiv.org/abs/2501.09980).

## Citation

If you use the Aneumo dataset or our code in your research, please cite:

```bibtex
@article{li2025aneumo,
  title={Aneumo: A Large-Scale Comprehensive Synthetic Dataset of Aneurysm Hemodynamics},
  author={Li, Xigui and Zhou, Yuanye and Xiao, Feiyang and Guo, Xin and Zhang, Yichi and Jiang, Chen and Ge, Jianchao and Wang, Xiansheng and Wang, Qimeng and Zhang, Taiwei and others},
  journal={arXiv preprint arXiv:2501.09980},
  year={2025}
}
```

## Acknowledgements

This dataset incorporates real aneurysm data partly sourced from the [AneuX dataset](https://github.com/hirsch-lab/aneuxdb). We express our gratitude for their open-source contributions.
