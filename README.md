
# EFSNet

**EFSNet** is a lightweight and efficient stereo matching network designed for real-time applications on resource-constrained devices. It combines hybrid cost volume aggregation, a dual-path refinement strategy, and a novel **LinearLog loss** to achieve state-of-the-art accuracy-speed trade-offs.

## 🔍 Overview

Stereo matching is crucial for 3D perception in autonomous driving and robotics. EFSNet introduces the following core innovations:

- **Hybrid 2D-3D Aggregation**: Fuses 3D group-wise cost volume with 2D correlation-based features.
- **Dual-Path Refinement**: Refines disparity maps via two parallel branches—one for detail preservation and one for robustness in low-texture regions.
- **LinearLog Loss**: A novel loss function that balances sensitivity to both small and large errors during training.

## 🧠 Highlights

- ⏱ **Real-time** inference: 16–18 ms on RTX 3090 for lite and standard versions.
- 📏 **Accurate**: EPE of **0.64 px** (standard) on SceneFlow.
- 🪶 **Lightweight**: < 5M parameters and < 30 GFLOPs for the standard version.

## 📊 Performance

| Method              | Params (M) | FLOPs (G) | EPE (px) | Runtime (ms) |
|---------------------|------------|-----------|----------|--------------|
| EFSNet-lite (Ours)  | 3.35       | 26.57     | 0.72     | **16**       |
| EFSNet (Ours)       | 4.59       | 29.93     | **0.64** | **18**       |
| LightStereo-M       | 7.64       | 36.36     | 0.62     | 23           |
| HITNet              | 0.42       | 50.23     | 0.55     | 36           |

## 📁 Dataset Support

- **SceneFlow** (synthetic)
- **KITTI 2012 / 2015** (real-world benchmarks)
- **ITB Stereo Dataset** ([Download here](https://drive.google.com/drive/folders/1iPFSPcVk-VzIRdKufC5CuyP7orhPoQWY?usp=drive_link)) – captured using a ZED 1 camera in a university driving scenario

## 🧱 Architecture

```
Stereo Image Pair
      │
  [MobileNetV2]
      │
Feature Extraction
      │
  ┌───────────────┬─────────────────┐
  │ 3D GWC Volume │ 2D Corr Volume  │
  │ (Low-res)     │ (High-res)      │
  └──────┬────────┴────────┬────────┘
     Hybrid 2D-3D Aggregation (IRBs)
                  │
        Dual-Path Refinement
    (detail path & global path)
                  │
         Cost Volume Regression
                  │
           Disparity Estimation
```

## 🛠 Installation

Under Construction

## 🚀 Usage

### Training

Under Construction

### Inference

Under Construction

### Evaluation

Under Construction

## 🧪 Ablation & Experiments

| Configuration            | EPE   |
|--------------------------|-------|
| Full EFSNet              | 0.662 |
| w/o Coarse Volume        | 0.713 |
| w/o Dual-path Refinement | 0.698 |
| w/ LinearLog Loss        | **0.637** |

## 📜 Citation

Under construction

