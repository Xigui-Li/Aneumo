# CFD Data Processing and Model Training Pipeline

This repository contains a complete workflow for handling CFD data, corresponding 3D medical images, deep learning model training (DeepONet, Swin Transformer), and post-processing for visualization.
If you want to run the imagePreprocess code, just run the 'DataPreprocess.py' and revise the configuration in 'Preprocess_config.yaml' file.

## Table of Contents

1. [Raw CFD Data](#1-raw-cfd-data)
2. [Raw Image Data](#2-raw-image-data)
3. [VTK → NPY Conversion](#3-vtk--npy-conversion)
4. [Image Preprocessing (ROI Extraction)](#4-image-preprocessing-roi-extraction)
5. [Signed Distance Function (Wall Distance)](#5-signed-distance-function-wall-distance)
6. [CFD Data Structuring (Before Input to Network)](#6-cfd-data-structuring-before-input-to-network)
7. [Model Parameters](#7-model-parameters)
8. [Model Training Logs](#8-model-training-logs)
9. [Post-Processing (NPY → VTK)](#9-post-processing-npy--vtk)
10. [NPY → Joblib Conversion](#10-npy--joblib-conversion)
11. [Summary](#11-summary)
12. [Directory Structure](#12-directory-structure)
13. [Usage](#13-usage)

---

## 1. Raw CFD Data

- Path: `raw_cfd/{case_id}/m=0.001/`
- Includes:
  - Boundary files: `inlet.vtp`, `outlet.vtp`, `wall.vtp`
  - Internal fluid region: `internal.vtu`
  - Fields: `[x, y, z, p, u, v, w]`
- Source: OpenFOAM results (VTK format)

---

## 2. Raw Image Data

- Path: `niidata/`
- Includes files: `{ID}.nii.gz`
- Shape: `[L, H, W, 1]`
- Source: 3D Slicer exports

---

## 3. VTK → NPY Conversion

- Path: `npydata/`
- Scripts:
  - [`vtk2npy.py`](./vtk2npy.py): Converts boundary data (`.vtp`) and internal `.vtu` data to NumPy arrays (`.npy`), each containing `[x, y, z, p, u, v, w]`.

---

## 4. Image Preprocessing (ROI Extraction)

- Path: `data/corped_image/`
- Scripts:
  - [`imagePreprocess_ori.py`](./imagePreprocess_ori.py): Extracts ROI without downsampling.
  - [`imagePreprocess2.py`](./imagePreprocess2.py): Adds downsampling.
  - [`imagePreprocess2-1.py`](./imagePreprocess2-1.py): Adds background padding.

---

## 5. Signed Distance Function (Wall Distance)

- Path: `npydata/`
- Script: [`calWallDistance.py`](./calWallDistance.py)
- This step adds the SDF to the original CFD data, yielding `[x, y, z, sdf, p, u, v, w]`.

---

## 6. CFD Data Structuring (Before Input to Network)

- Paths:
  - Training data: `data/cfd/`
  - Inference data: `data/cfd_pred/`
- Fields after preprocessing:
  - `X_sup`: `[x, y, z, sdf]`
  - `Y_sup`: `[p, u, v, w]`
  - `X_inlet`: `[x, y, z, p, u, v, w]`
  - `Simple_inlet`: `[x0, y0, z0, nx, ny, nz, flow_rate]`
- Scripts:
  - [`cfdPreprocess.py`](./cfdPreprocess.py)
  - [`cfdPreprocess_pred.py`](./cfdPreprocess_pred.py): For inference and preserves a 0.5 mm inlet region.

---

## 7. Model Parameters

- Checkpoints typically located in `checkpoint/.../`.
- Store networks (DeepONet, Swin-based, or others) with different epochs and logs.

---

## 8. Model Training Logs

- Path: `checkpoint/.../`
- Contains details such as:
  - L2, MNAE (p, u, v, w)
  - Training / Testing time

---

## 9. Post-Processing (NPY → VTK)

- Path: `checkpoint/.../` (where the `.npy` inference outputs are saved)
- Script:
  - [`npy2vtk.py`](./npy2vtk.py): Converts the `.npy` predictions to `.vtk` or `.vtu` for visualization. Also computes `diff.vtk` to compare predictions vs. ground truth.

---

## 10. NPY → Joblib Conversion

- Script: [`transfer_data2onejob.py`](./transfer_data2onejob.py)
- Merges `.npy` / `.npz` format data into a single `.job` file for parallel processing or streamlined workflows.

---

## 11. Summary

This project demonstrates a full workflow from raw data → preprocessing → model training & evaluation → results post-processing. Please follow the data flow in the exact sequence, ensuring consistent data formats for the model. For large-scale automation, it is recommended to merge these scripts into a unified pipeline, driven by a single configuration file.

---

## 12. Directory Structure

```
Dataprepprocessing
├── calWallDistance.py
├── cfdPreprocess.py
├── cfdPreprocess_pred.py
├── imagePreprocess_ori.py
├── imagePreprocess2.py
├── imagePreprocess2-1.py
├── inference_real_deeponet.py
├── inference_real_swin.py
├── npy2vtk.py
├── README.md
├── transfer_data2onejob.py
└── vtk2npy.py
```

---

## 13. Usage

Below is a typical sequence to execute the pipeline. Adjust paths and parameters as needed:

1. **Converting VTK to NPY**

   ```bash
   python vtk2npy.py
   ```

   Reads `.vtu` or `.vtp` to create `[x, y, z, p, u, v, w].npy`.
2. **Preprocessing Images**

   ```bash
   python imagePreprocess_ori.py
   ```

   Generates cropped or downsampled 3D medical images.
3. **Calculating Wall Distance**

   ```bash
   python calWallDistance.py
   ```

   Computes `[x, y, z, sdf, p, u, v, w]`.
4. **CFD Data Formatting**

   ```bash
   python cfdPreprocess.py
   ```

   Produces `data/cfd/...` containing `X_sup`, `Y_sup`, etc.
5. **Training Models**Use model scripts (DeepONet or Swin) and keep checkpoints in `checkpoint/.../`.
6. **Inference**

   ```bash
   python inference_real_swin.py
   ```

   Saves predictions `[x, y, z, p, u, v, w].npy`.
7. **NPY → VTK**

   ```bash
   python npy2vtk.py
   ```

   Creates `.vtk` files for visualization and computes error files (e.g., `diff.vtk`).
8. **Optional: Convert NPY → Joblib**

   ```bash
   python npy2joblib.py
   ```

   Merges data into `.job` for easy loading and parallel processing.

---
