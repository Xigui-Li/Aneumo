# Aneumo: A Large-Scale Comprehensive Synthetic Dataset for Aneurysm Hemodynamics

## Basic Information

* This repository hosts the **Aneumo** dataset, a large-scale and comprehensive synthetic dataset specifically designed for the study of aneurysm hemodynamics.

* The **Aneumo** dataset comprises **10,000+ 3D models (real and synthetic)**, categorized as follows:
   
  - **466 real aneurysm models**, sourced from the AneuX dataset ([https://github.com/hirsch-lab/aneuxdb](https://github.com/hirsch-lab/aneuxdb)).
  - **466 aneurysm-free models**, representing normal vascular structures.
  - **9,534 deformed aneurysm models**, encompassing diverse aneurysm shapes, sizes, and locations.

* These synthetic models were generated through **resection** and **deformation** operations, based on **466 real aneurysm cases**.

* The dataset further includes:
  - **10,000+ Medical image-like segmentation mask files**, enabling the development and testing of segmentation algorithms.
  - **80,000+ Hemodynamic data** obtained at **eight steady-state flow rates (0.001 to 0.004 kg/s)**, covering critical parameters such as velocity, pressure, and wall shear stress.

* **The dataset will be made available for download soon. Stay tuned for updates.    Please refer to [the paper](https://arxiv.org/abs/2501.09980) for more details.**

## Visual Overview

Below are key visualizations illustrating the dataset generation process and sample outputs:

<div align="center">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/fig1%20workflow.png" alt="Workflow" width="1200px">
  <p><b>Figure 1:</b> Workflow of the Aneumo dataset generation process</p>
</div>

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/Vmax.png" alt="Vmax" style="height: 300px;">
  <img src="https://github.com/Xigui-Li/Aneumo/blob/main/fig/dp.png" alt="DP" style="height: 300px;">
</div>
<p align="center"><b>Figure 2:</b> Visualization of hemodynamic parameters: Maximum velocity (left) and normalized pressure difference (right)</p>

## Acknowledgments

The development of the **Aneumo** dataset was made possible through the use of real-world aneurysm data, with partial contributions from the **AneuX dataset** ([https://github.com/hirsch-lab/aneuxdb](https://github.com/hirsch-lab/aneuxdb)).
