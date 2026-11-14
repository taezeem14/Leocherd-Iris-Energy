# 🌌 Leocherd Iris Energy
### Python-First Universal AI Video-to-3D Rigging & Motion Reconstruction Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![glTF 2.0](https://img.shields.io/badge/glTF-2.0%20Valid-brightgreen.svg)](https://www.khronos.org/gltf/)
[![Blender Rigify](https://img.shields.io/badge/Blender-Rigify%20Ready-orange.svg)](https://www.blender.org/)
[![Kaggle GPU](https://img.shields.io/badge/Kaggle-GPU%20Acceleration-blueviolet.svg)](experiments/kaggle/)

**Leocherd Iris Energy** is an extensible, Python-first 3D computer vision and rigging engine that reconstructs articulated skeletal structures, biomechanically regularized forward kinematics, and production-ready 3D animation assets directly from standard monocular video footage (`.mp4`, `.mov`).

---

## 🌟 Key Highlights

- **🎥 Monocular Video to 3D Articulation**: Direct frame extraction, subject detection, 3D landmark estimation, and temporal trajectory imputation.
- **🦾 Anatomical Stylized Mannequin**: Procedural, watertight humanoid proxy mesh with sculpted ribcage, deltoids, tapered limbs, face visor, and smooth Linear Blend Skinning (LBS).
- **📐 Tri-Axial Orthonormal Kinematic Solvers**: Joint orientation frames computed with Aim and Pole reference vectors to guarantee **zero roll/twist distortion** and zero inside-out flipping.
- **⚡ Hybrid Cloud + Local Architecture**: Run heavy GPU inference (MediaPipe Heavy / ViTPose / HMR2.0) on free Kaggle T4/P100 GPUs and import pre-computed trajectories locally in milliseconds.
- **📦 Multi-Format Asset Export**:
  - **glTF 2.0**: Embedded skinned mesh, inverse bind matrices, and animation tracks (100% strict Khronos validator clean).
  - **BVH**: Industry-standard Biovision Hierarchy motion capture file compatible with Unreal Engine, Unity, and MotionBuilder.
  - **Blender Scene**: Native `.blend` compiling with armature generation and automated NLA action retargeting.
- **🌐 FastAPI REST Microservice**: Fully asynchronous REST endpoints with OpenAPI/Swagger docs for job submission and status polling.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/taezeem14/Leocherd-Iris-Energy.git
cd Leocherd-Iris-Energy
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Run Local Reconstruction (CPU / ONNX)

```bash
python main.py reconstruct data/samples/online_walking_sample.mp4 --morphology human --fps 15 --output-dir outputs/my_reconstruction
```

### 3. Kaggle Cloud GPU Acceleration

1. Open [`experiments/kaggle/leocherd_iris_gpu_inference.ipynb`](experiments/kaggle/leocherd_iris_gpu_inference.ipynb).
2. Run on Kaggle with GPU T4 x2 and download `iris_kaggle_pose_data.json`.
3. Import locally in seconds:
```bash
python main.py import-kaggle iris_kaggle_pose_data.json --output-dir outputs/kaggle_rig
```
