# 3D Resection & Surgical Planning of Brain Tumors Using Deep Learning + Reinforcement Learning (DL + RL)

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch_2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![MONAI: Medical AI](https://img.shields.io/badge/MONAI-1.3+-00A67E?logo=monai)](https://monai.io/)
[![RL: Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-2A62BC)](https://stable-baselines3.readthedocs.io/)
[![Environment: Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-008080)](https://gymnasium.farama.org/)
[![UI: Plotly Dash](https://img.shields.io/badge/Dashboard-Plotly_Dash-3F4F75?logo=plotly)](https://dash.plotly.com/)

**An end-to-end autonomous neurosurgical planning framework** that combines **3D multi-modal Deep Learning** for perception with **Continuous-Space Reinforcement Learning** for trajectory optimization. The system bridges discrete tumor segmentation with kinematically constrained, safety-aware surgical tool paths, replacing sparse binary rewards with dense probabilistic safety gradients.

---
# 3D Resection & Surgical Planning of Brain Tumors Using Deep Learning + Reinforcement Learning (DL + RL)

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch_2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![MONAI: Medical AI](https://img.shields.io/badge/MONAI-1.3+-00A67E?logo=monai)](https://monai.io/)
[![RL: Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-2A62BC)](https://stable-baselines3.readthedocs.io/)
[![Environment: Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-008080)](https://gymnasium.farama.org/)
[![UI: Plotly Dash](https://img.shields.io/badge/Dashboard-Plotly_Dash-3F4F75?logo=plotly)](https://dash.plotly.com/)

**An end-to-end autonomous neurosurgical planning framework** that combines **3D multi-modal Deep Learning** for perception with **Continuous-Space Reinforcement Learning** for trajectory optimization. The system bridges discrete tumor segmentation with kinematically constrained, safety-aware surgical tool paths, replacing sparse binary rewards with dense probabilistic safety gradients.

---

## Architecture Overview

```text
[ BraTS 2021 + ReMIND2Reg Multi-Modal 3D MRI / iUS ]
                       │
                       ▼
       ┌───────────────────────────────┐
       │  PHASE 1: Data Harmonization  │  ➔ 2mm Isotropic Resampling
       │  & 2.5D Pseudo-RGB Extraction │  ➔ Z-Score / Percentile Norm
       │                               │  ➔ Central Orthogonal Slices
       │                               │    (Axial / Coronal / Sagittal)
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │  PHASE 2: Attention U-Net     │  ➔ Quadrupled Capacity Generalist
       │  & Boundary-Aware Edge Loss   │  ➔ DiceFocal + Morphological Edge L1
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │  PHASE 3: 3D SwinUNETR        │  ➔ Frozen Pretrained Backbone
       │  Transfer Learning Safety Head│  ➔ 5×5×5 Peritumoral Danger Margin
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │  PHASE 4: Preoperative Context│  ➔ Continuous Sigmoid Safety Field
       │  & Hybrid Cost Map Generation │  ➔ Burr Hole Entry Kinematics
       │                               │  ➔ 3D Euclidean Distance Transform
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │  PHASE 5: Kaggle RL Backend   │  ➔ Continuous MDP (Spherical Retractor)
       │  (Proximal Policy Opt. - PPO) │  ➔ Dense Reward: ΔEOR + Risk + Smoothness
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │  PHASE 6: Colab Frontend      │  ➔ Deterministic Rollouts (< 3s)
       │  & Clinical Validation        │  ➔ Interactive Dashboard + iUS Validation
       └───────────────────────────────┘
```

## Repository Contents

Execute the notebooks sequentially from Phase 1 to Phase 6:

| File                                      | Phase | Module Focus                          | Primary Deliverables |
|-------------------------------------------|-------|---------------------------------------|----------------------|
| `phase-1-dl-2.ipynb`                      | **1** | Data Harmonization & Ingestion        | 2mm isotropic volumes, `Extract25DSlicesd` |
| `phase-2-brats-pretraining-dl-3.ipynb`    | **2** | Generalist Segmentation               | Attention U-Net, Boundary-Aware Loss |
| `phase-3-transfer-learning-for-deep-learning (2).ipynb` | **3** | 3D Volumetric Safety Head             | Frozen SwinUNETR + Continuous Safety Head |
| `phase-4-preoperative-surgical-context.ipynb` | **4** | Surgical Context & Cost Mapping       | `healed_surgical_context.npz` (Safety + EDT + Burr Hole) |
| `Phase5_Kaggle_Training_Backend.ipynb`    | **5** | Reinforcement Learning Training       | PPO Agent (`ppo_best_model.zip`, `vecnorm.pkl`) |
| `Phase6_Colab_Frontend.ipynb`             | **6** | Inference & Dashboard                 | Deterministic trajectories + Plotly Dash UI |

---

## Core Algorithmic Details

### 1. Datasets & Preprocessing
- **BraTS 2020/2021**: Multi-modal MRI (T1, T1ce, T2, FLAIR) with labels for NCR/NET, Edema, and Enhancing Tumor.
- **ReMIND2Reg**: Pre-op MRI + intra-op Ultrasound for resection cavity validation.
- **Harmonization**: 2 mm isotropic resampling (trilinear for MRI, nearest-neighbor for masks), Z-score normalization (brain-only), percentile clipping for iUS, and RAS orientation.

### 2. Perception (Phases 1–3)
- **Phase 1**: Memory-efficient 2.5D orthogonal slice extraction around tumor centroid.
- **Phase 2**: Quad-capacity 2D Attention U-Net trained with hybrid **DiceFocal + Morphological Edge Loss** for sharp boundary awareness.
- **Phase 3**: Transfer learning with frozen 3D SwinUNETR backbone + lightweight continuous safety head. Produces probabilistic safety fields instead of discrete masks.

### 3. Planning & Reinforcement Learning (Phases 4–5)
- **Phase 4**: Generates continuous safety probability field $P_{\text{safety}} \in [0,1]$ via Sigmoid, burr-hole entry point, and 3D Euclidean Distance Transform to danger zones.
- **Phase 5**: Custom Gymnasium `SurgicalEnv` with spherical kinematic constraints (retractor tool pivoting from burr hole). Trained with **PPO** (Stable-Baselines3) using a dense multi-objective reward combining Extent of Resection (EOR), risk avoidance, movement smoothness, and distance buffering.

### 4. Validation & Visualization (Phase 6)
- Deterministic policy rollouts.
- Quantitative validation against real intraoperative ultrasound cavities (Dice Similarity Coefficient, pEOR).
- Interactive Plotly Dash dashboard with 3D brain/tumor meshes, safety heatmaps, and multi-planar slice viewers.

---

## Installation & Setup

```bash
git clone https://github.com/yourusername/DL-PBL-brain-tumor-resection-and-surgical-planning.git
cd DL-PBL-brain-tumor-resection-and-surgical-planning

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "monai[all]>=1.3.0" stable-baselines3[extra] gymnasium>=0.29.0
pip install dash dash-bootstrap-components plotly matplotlib scipy scikit-image nibabel jupyterlab

jupyter lab

Cloud (Phases 5–6)

Phase 5: Run on Kaggle (GPU T4/P100) with attached datasets.
Phase 6: Run on Google Colab. Upload models from Phase 5 and launch the Dash dashboard.
```
