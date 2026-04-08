# Datasheet for Aneumo Dataset

Following the framework proposed by Gebru et al. (2021), "Datasheets for Datasets," *Communications of the ACM*, 64(12), 86--92.

---

## 1. Motivation

**For what purpose was the dataset created?**
Aneumo was created to address the critical lack of large-scale, high-fidelity benchmarks that integrate 3D geometry with dynamic physical fields for Scientific Machine Learning (SciML) in biofluid dynamics. Existing datasets in cerebral aneurysm hemodynamics suffer from limited scale (< 1,000 cases), absence of transient (pulsatile) simulation data, or incomplete multimodal alignment. Aneumo is designed as a premier benchmark for AI-driven simulation, Digital Twins, and 4D spatio-temporal forecasting.

**Who created the dataset and on behalf of which entity?**
The dataset was created by Xigui Li, Yuanye Zhou, Feiyang Xiao, Xin Guo, Chen Jiang, Tan Pan, Xingmeng Zhang, Zeyun Miao, Cenyu Liu, Xiansheng Wang, Qimeng Wang, Yichi Zhang, Wenbo Zhang, Hongwei Zhang, Ruoxi Jiang, Fengping Zhu, Limei Han, Yuan Qi, Chensen Lin, and Yuan Cheng, affiliated with Fudan University (Shanghai, China), Shanghai Academy of Artificial Intelligence for Science (Shanghai, China), and The Hong Kong Polytechnic University (Hong Kong, China).

**Who funded the creation of the dataset?**
This work was supported by the National Natural Science Foundation of China (Grant No. 82394432 and 92249302) and the Shanghai Municipal Science and Technology Major Project (Grant No. 2023SHZDZX02). The computations were performed using the CFFF platform of Fudan University.

**Any other comments?**
The dataset is intended primarily as a research benchmark for the data mining and machine learning communities. It is not intended for direct clinical decision-making without proper validation against in-vivo measurements.

---

## 2. Composition

**What do the instances that comprise the dataset represent?**
Each instance represents a 3D cerebral aneurysm vascular geometry paired with its corresponding Computational Fluid Dynamics (CFD) simulation results. The geometries are derived from real patient anatomy through synthetic deformation of clinically sourced 3D models.

**How many instances are there in total?**
- **3D Geometries**: 10,660 unique vascular models
- **Steady-State Simulations**: 85,280 cases (10,660 geometries x 8 flow rates)
- **Transient Simulations**: 10,660 pulsatile flow sequences (one per geometry, each containing 100 time steps over one cardiac cycle)
- **Total Simulations**: 95,940

**Does the dataset contain all possible instances or is it a sample of instances from a larger set?**
The dataset is a sample. The 10,660 geometries were generated from 427 base cases in the publicly available AneuX dataset via at least 20 randomized non-rigid deformation operations per base case. These 427 base cases are themselves a subset of clinically acquired intracranial aneurysm models.

**What data does each instance consist of?**

For each geometry:
| Data Type | Format | Description |
|-----------|--------|-------------|
| Surface Mesh | `.stl` | Stereo-lithography surface mesh of the vascular geometry |
| Computational Mesh | `.msh` | Polyhedral volume mesh used for CFD simulation |
| Segmentation Mask | `.nii.gz` | Voxelized binary ROI in NIfTI format, compatible with medical imaging pipelines |
| Point Cloud | `.npz` | Surface points sampled from the geometry |

For each simulation:
| Data Type | Format | Description |
|-----------|--------|-------------|
| Inlet Fields | `.vtp` | Fluid characteristics at the inlet boundary |
| Internal Fields | `.vtu` | Volumetric velocity (u, v, w), pressure (p) within the lumen |
| Outlet Fields | `.vtp` | Fluid characteristics at the outlet boundary |
| Wall Fields | `.vtp` | Wall Shear Stress (WSS) vector field at the vessel wall |
| ML-Ready Arrays | `.npy` | NumPy arrays for inlet, internal, outlet, and wall boundary conditions |

**Is there a label or target associated with each instance?**
The CFD simulation fields (velocity, pressure, WSS) serve as ground-truth targets for supervised learning tasks. No manual annotations or clinical labels (e.g., rupture status, aneurysm subtype) are provided.

**Is any information missing from individual instances?**
- Patient-level clinical metadata (age, sex, medical history, rupture status) is not available, as the upstream AneuX dataset does not provide this information.
- Demographic composition of the original 427 patient cases is unknown.
- Original medical imaging data (CTA/MRI) is not included; only derived 3D surface models are used.

**Are relationships between individual instances made explicit?**
Yes. Each deformed geometry is traceable to one of the 427 base AneuX cases. The case numbering scheme preserves this relationship. Steady-state and transient simulations sharing the same geometry ID are explicitly linked.

**Are there recommended data splits?**
Yes. For the Generalization Benchmark (steady-state):
- **Training**: 1,280 cases (160 geometries x 8 flow conditions)
- **Validation**: 80 cases (40 geometries x 2 flow conditions), with no geometric overlap with training
- **Test**: 20 cases from 10 original (undeformed) AneuX geometries, held out from the generation process, serving as a "Syn-to-Real" generalization test

For the Dynamics Benchmark (transient):
- **Training**: First 80% of time steps (t_1 to t_80) from the selected geometry(ies)
- **Test**: Last 20% of time steps (t_81 to t_100)

**Does the dataset contain data that might be considered confidential?**
No. All geometries are derived from the publicly available, de-identified AneuX dataset. No raw clinical data, imaging data, or personally identifiable information (PII) is included.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**
No.

---

## 3. Collection Process

**How was the data associated with each instance acquired?**
The data was generated through a multi-stage computational pipeline:

1. **Geometry Generation**: 427 base aneurysm models from the AneuX dataset were processed using Geomagic Wrap 2024 for mesh repair and quality inspection. Aneurysm regions were removed and synthetically re-introduced via randomized non-rigid deformations (amplitude range [0.5, 1.0]) to create 10,660 diverse geometries. A transition band was automatically generated to ensure smooth blending with the parent vessel.

2. **Mesh Generation**: ANSYS Fluent Meshing 2025 R2 was used to generate unstructured polyhedral meshes. Minimum mesh size: 0.15 mm. Boundary layer: 10 inflation layers with growth rate 1.2.

3. **CFD Simulation**: OpenFOAM v2312 was used with the Finite Volume Method (FVM):
   - Steady-state: icoFoam solver with PISO algorithm, 8 distinct mass flow rates (0.001--0.004 kg/s)
   - Transient: pisoFoam solver with pulsatile inlet waveform, time step dt = 10^-6 s, CFL < 1, data saved from the 5th cardiac cycle

4. **Segmentation Mask Generation**: 3D Slicer was used to voxelize STL models into NIfTI-format binary segmentation masks.

**What mechanisms or procedures were used to collect the data?**
All procedures were computational. No human subjects were recruited. The geometry deformation, mesh generation, and CFD simulations were orchestrated using Slurm and Kubernetes on a high-performance computing cluster, peaking at 10,000 concurrent CPU cores.

**If the dataset is a sample from a larger set, what was the sampling strategy?**
The 427 base cases were taken from the AneuX dataset (750 total cases). The selection was based on geometric quality and suitability for CFD meshing. Each base case underwent at least 20 randomized deformations.

**Who was involved in the data collection process?**
The dataset generation was performed by the author team. Critical steps (aneurysm region assessment, deformation design, boundary condition definition) were reviewed and validated by clinical physicians and computational fluid dynamics experts.

**Over what timeframe was the data collected?**
The simulation campaign totaled over 11 million CPU core-hours. The exact calendar timeframe is not specified but spans approximately 2024--2025.

**Were any ethical review processes conducted?**
The base geometries come from the publicly available AneuX dataset (CC BY 4.0), which was de-identified and anonymized by its original creators. Our work involves only computational augmentation (geometry deformation and physics simulation) of these already-anonymized 3D surface models. No new human participants were recruited, no raw clinical data was collected, and no PII is involved.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes. The preprocessing pipeline includes:
- **Mesh Repair**: Geomagic Wrap Mesh Doctor tool to fix non-manifold edges, self-intersections, cusps, small tunnels, and holes
- **Surface Smoothing**: Remesh and Fit Surface (NURBS reconstruction) for geometric fidelity
- **Format Conversion**: STL to STEP (via Geomagic Wrap), STEP to SCDOC (via ANSYS SpaceClaim), mesh generation (.msh via ANSYS Fluent Meshing)
- **Quality Control**: Mesh sensitivity analysis confirming <0.01% relative difference between 0.15 mm and 0.10 mm mesh resolutions; residual convergence verification (velocity residuals < 10^-9, pressure residuals < 10^-5 for steady-state)
- **Field Computation**: WSS fields were pre-computed from velocity gradients and included for all 95,940 cases

**Is the software that was used to preprocess/clean/label the data available?**
- Geomagic Wrap 2024 (commercial, 3D Systems Inc.)
- ANSYS SpaceClaim 2025 R2 (commercial, ANSYS Inc.)
- ANSYS Fluent Meshing 2025 R2 (commercial, ANSYS Inc.)
- OpenFOAM v2312 (open-source, https://www.openfoam.com/)
- 3D Slicer (open-source, https://www.slicer.org/)

**Was the raw data saved in addition to the preprocessed/cleaned/labeled data?**
The raw CFD simulation output exceeds 100 TB. The curated, compressed public release is approximately 24 TB (4 TB for steady-state, 20 TB for transient).

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes. The accompanying paper establishes two benchmark tasks:
1. **Generalization Benchmark (Steady-State)**: Geometry-conditioned prediction of pressure and velocity fields using DeepONet and DeepONet-SwinT architectures, with ablation studies on batch size, point density, flow condition diversity, and scaling law analysis.
2. **Dynamics Benchmark (Transient)**: 4D Wall Shear Stress (WSS) forecasting using a Temporal DeepONet architecture with cross-attention mechanism.

Additional baselines have been evaluated: FNO, 3D UNet, and MeshGraphNet.

**Is there a repository that links to any or all papers or systems that use the dataset?**
Yes. The main repository is at https://github.com/Xigui-Li/Aneumo. The dataset is hosted on Hugging Face.

**What (other) tasks could the dataset be used for?**
- 3D generative modeling of complex vascular geometries
- Physics-informed 3D reconstruction from segmentation masks or point clouds
- Shape-conditioned spatio-temporal forecasting
- Multi-modal learning (geometry-to-field translation)
- Neural operator development and benchmarking
- Hemodynamic parameter prediction for clinical risk assessment (with appropriate validation)
- Foundation Model pre-training for biofluid dynamics

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
- All geometries are synthetically deformed from 427 base cases; the morphological diversity, while validated against clinical statistics, may not capture the full spectrum of rare aneurysm morphologies.
- Blood is modeled as Newtonian fluid (density 1050 kg/m^3, dynamic viscosity 0.00345 Pa.s). This is widely accepted for intracranial arteries at relevant shear rates but may introduce minor inaccuracies in low-shear-rate regions within aneurysm sacs.
- Vessel walls are assumed rigid (no fluid-structure interaction).
- The inlet boundary condition uses the largest cross-section as inlet, which is a standard practice but may differ from patient-specific flow conditions.

**Are there tasks for which the dataset should not be used?**
- **Direct clinical decision-making**: The dataset is intended for research benchmarking. Predictions from models trained on this data should not be used for clinical diagnosis or treatment planning without rigorous validation against in-vivo measurements.
- **Patient identification**: Although the data is fully anonymized, the dataset should not be used in attempts to re-identify patients.

---

## 6. Distribution

**How will the dataset be distributed?**
- **Dataset**: Hosted on Hugging Face with version control, accessible via standard Hugging Face APIs
- **Code**: Hosted on GitHub at https://github.com/Xigui-Li/Aneumo, including data preprocessing pipelines, deep learning model implementations (DeepONet, DeepONet-SwinT, Temporal DeepONet, FNO, 3D UNet, MeshGraphNet), and inference scripts

**When will the dataset be distributed?**
The dataset is publicly available as of the submission date.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**
The dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Users may freely share, adapt, and build upon the material, subject to appropriate credit.

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**
The upstream AneuX dataset is also distributed under CC BY 4.0. No additional restrictions apply.

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?**
No.

---

## 7. Maintenance

**Who is supporting/hosting/maintaining the dataset?**
The dataset is maintained by the first author (Xigui Li) and the corresponding authors (Xin Guo, Chensen Lin, Yuan Cheng) at Fudan University and Shanghai Academy of Artificial Intelligence for Science.

**How can the owner/curator/manager of the dataset be contacted?**
- Xigui Li: lixigui@fudan.edu.cn
- Xin Guo: guoxin@sais.org.cn
- Chensen Lin: linchensen@fudan.edu.cn
- Yuan Cheng: cheng_yuan@fudan.edu.cn

**Is there an erratum?**
Not at this time. Any corrections will be documented in the GitHub repository's CHANGELOG and on the Hugging Face dataset card.

**Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?**
Yes. The maintainers plan to:
- Release versioned updates with changelogs on Hugging Face
- Expand simulation conditions (e.g., additional boundary conditions, non-Newtonian rheology models) in future versions
- Incorporate additional base geometries as they become available from collaborating clinical institutions
- Add new baseline model results as the community contributes

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**
The dataset contains only anonymized 3D geometric models and computational simulation results. No direct personal data is retained. The upstream AneuX dataset was de-identified prior to public release by its original creators.

**Will older versions of the dataset continue to be supported/hosted/maintained?**
Yes. Hugging Face's version control system ensures that all previous versions remain accessible.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Yes. Contributions can be made via pull requests to the GitHub repository (https://github.com/Xigui-Li/Aneumo). The maintainers welcome community contributions including new baseline results, additional preprocessing tools, and downstream task implementations.

---

## 8. Morphometric Validation Summary

To validate the physiological plausibility of the synthetic geometries, 15 clinically relevant morphometric parameters were computed for all 10,000+ aneurysm geometries:

| Parameter | Mean +/- SD | Clinical Range | Reference |
|-----------|-------------|----------------|-----------|
| Size (mm) | 3.39 +/- 0.77 | 1--15 | ISUIA Study |
| Neck Width (mm) | 3.29 +/- 0.80 | 1--10 | Debrun et al. |
| Aspect Ratio | 1.17 +/- 0.56 | 0.5--4.0 | Ujiie 1999, Weir 2003 |
| Height (mm) | 3.70 +/- 1.06 | 1--15 | -- |
| Max Diameter (mm) | 5.02 +/- 1.16 | 2--20 | ISUIA |
| Volume (mm^3) | 45.09 +/- 31.61 | 1--500 | -- |
| Non-Sphericity Index | 0.236 +/- 0.021 | 0.05--0.35 | Raghavan 2005 |
| Undulation Index | 0.011 +/- 0.028 | 0--0.3 | Raghavan 2005 |
| Bottleneck Factor | 1.58 +/- 0.58 | 0.8--3.0 | Dhar 2008 |

All parameters fall within clinically documented ranges, confirming that the synthetic geometry pipeline produces physiologically plausible aneurysm morphologies. The full morphometric table is publicly available at https://github.com/Xigui-Li/Aneumo/blob/main/MPs.csv.
