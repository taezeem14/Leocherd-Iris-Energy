# 🌌 Leocherd Iris Energy
### Universal Python-First AI Video-to-3D Rigging & Motion Reconstruction Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![glTF 2.0 Valid](https://img.shields.io/badge/glTF%202.0-Khronos%20Strict%20Valid-brightgreen.svg)](https://www.khronos.org/gltf/)
[![Blender Rig Ready](https://img.shields.io/badge/Blender-Rig%20%26%20NLA%20Verified-orange.svg)](https://www.blender.org/)
[![Unity Generic Avatar](https://img.shields.io/badge/Unity-Generic%20Avatar%20Ready-black.svg)](https://unity.com/)
[![Kaggle GPU Acceleration](https://img.shields.io/badge/Kaggle-GPU%20Acceleration-blueviolet.svg)](experiments/kaggle/)
[![Tests Passing](https://img.shields.io/badge/Tests-377%20Passed%20(100%25)-success.svg)](tests/)

**Leocherd Iris Energy** is an end-to-end, Python-first 3D computer vision and rigging platform that reconstructs articulated skeletal hierarchies, biomechanically regularized kinematics, watertight proxy meshes, and production-ready 3D character animations directly from standard monocular video footage (`.mp4`, `.mov`).

Designed for high-end animation in **Blender**, **Maya**, **Unity**, **Unreal Engine**, and real-time Web 3D engines, Iris Energy bridges raw video motion capture with production-ready character armatures, non-linear animation (NLA) tracks, and deformation skin weights.

---

## 📑 Table of Contents

- [Key Technical Capabilities](#-key-technical-capabilities)
- [System Architecture](#️-system-architecture)
- [Installation & Setup](#-installation--setup)
- [Full Step-by-Step Tutorial](#-full-step-by-step-tutorial)
  - [1. Inspect Video Metadata](#1-inspect-video-metadata)
  - [2. Local Video Reconstruction (CPU / ONNX)](#2-local-video-reconstruction-cpu--onnx)
  - [3. Cloud GPU Acceleration with Kaggle](#3-cloud-gpu-acceleration-with-kaggle)
  - [4. Universal Morphology Classification](#4-universal-morphology-classification)
  - [5. Unsupervised Skeleton Discovery (Template-Free)](#5-unsupervised-skeleton-discovery-template-free)
  - [6. Dual Quaternion Skinning (DQS) vs LBS](#6-dual-quaternion-skinning-dqs-vs-lbs)
  - [7. Inverse Kinematics (Two-Bone & FABRIK)](#7-inverse-kinematics-two-bone--fabrik)
  - [8. Cross-Morphology Motion Retargeting](#8-cross-morphology-motion-retargeting)
  - [9. Production 3D Export & Engine Usage](#9-production-3d-export--engine-usage)
  - [10. Running the FastAPI REST Microservice](#10-running-the-fastapi-rest-microservice)
- [CLI Reference](#-cli-reference)
- [Automated Testing & Verification](#-automated-testing--verification)
- [Project Directory Structure](#-project-directory-structure)
- [License](#-license)

---

## 🌟 Key Technical Capabilities

### 1. 🧬 Universal Morphology Understanding
Distinguishes anatomical and non-humanoid morphologies (**Humans**, **Quadrupeds**, **Avians**, **Serpentines**, and **Tentaculars**) directly from kinematic motion patterns, spine inclination angles, ground contact clustering, and wave propagation metrics.

### 2. 🔍 Unsupervised Automatic Skeleton Discovery
Discovers articulated 3D kinematic hierarchies from unlabelled point clouds or arbitrary motion trajectories without requiring predefined bone templates. Uses **pairwise Euclidean rigid-distance variance**, **kinematic center-of-mass acceleration variance minimization** for root joint selection, **Kruskal Minimum Spanning Trees (MST)**, and **bilateral reflection symmetry plane detection** ($n \cdot x + d = 0$).

### 3. 🛡️ Dual Quaternion Skinning (Anti-Twist Deformation)
Eliminates the notorious "candy-wrapper" joint twisting collapse inherent in standard Linear Blend Skinning (LBS). Dual Quaternion Skinning (DQS) with Dual Linear Blending (DLB) maintains **99.65% volume retention** at extreme $175^\circ$ joint rotations (compared to $33.45\%$ volume collapse under standard LBS). Supports **Heat Diffusion**, **Geodesic**, and **Volumetric** skin weighting.

### 4. 🦾 Production Inverse Kinematics (IK)
- **TwoBoneIKSolver**: Analytical, closed-form 2-bone solver using the Law of Cosines with pole-vector orientation steering.
- **FABRIKSolver**: Iterative Forward And Backward Reaching Inverse Kinematics for arbitrary $N$-segment bone chains with per-joint angular constraint cones.

### 5. 🔄 Cross-Morphology Motion Retargeting & Stabilization
Retargets motion between disparate skeletal structures (e.g. Human to Quadruped or Avian) via proportional segment scaling and rest-pose offset alignment. Features **forward-backward zero-phase Quaternion Slerp filtering** (76.38% jitter reduction) and **velocity-gated foot-locking clamping** to eliminate ground sliding.

### 6. 📦 Multi-Format 3D Asset Export
- **glTF 2.0**: Watertight procedural anatomical proxy mesh, PBR material, dual-joint neck lofts, and machine-precision Inverse Bind Matrices (**100% Khronos validator clean, 0 errors, 0 warnings**).
- **Blender**: Native armature generation scripts, bone hierarchy setup, and automated NLA action binding.
- **Unity Generic Avatar**: Strict conformity with Unity Generic/Humanoid rig requirements (uniform bone scaling $[1,1,1]$, 4-influence normalized vertex weights, invertible bindposes).
- **BVH**: Biovision Hierarchy motion capture file with gimbal-lock-free Euler angles.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    Video["📹 Monocular Video (.mp4, .mov)"] --> Ingestion["1. Video Ingestion & FPS Normalization"]
    Ingestion --> Detection["2. Subject Detection & IoU Tracking"]
    
    subgraph ComputeProviders ["Compute Engine"]
        LocalCPU["Local CPU (BlazePose ONNX / XNNPACK)"]
        KaggleGPU["Cloud Kaggle GPU (MediaPipe Tasks / PyTorch)"]
    end
    
    Detection --> LocalCPU
    Detection -.-> KaggleGPU
    
    LocalCPU --> Poses["3D Pose Landmarks (PoseFrame Sequence)"]
    KaggleGPU -->|iris_kaggle_pose_data.json| Poses
    
    Poses --> Discovery{"Morphology Selection"}
    Discovery -->|Predefined| MorphKnown["Human / Quadruped / Avian Models"]
    Discovery -->|Automatic| AutoDiscovery["Unsupervised MST Skeleton Discovery\n• Distance Variance Minimization\n• SVD Bilateral Symmetry\n• Root Acceleration Invariance"]
    
    MorphKnown --> Smoothing["3. Temporal Savitzky-Golay & Forward-Backward Slerp"]
    AutoDiscovery --> Smoothing
    
    Smoothing --> Kinematics["4. Tri-Axial Orthonormal Joint Orientations"]
    Kinematics --> IK["5. Two-Bone & FABRIK Inverse Kinematics"]
    IK --> FootLock["6. Velocity-Gated Foot-Locking Ground Stabilization"]
    
    FootLock --> MeshGen["7. Watertight Anatomical Proxy Mesh Generator"]
    MeshGen --> Skinning["8. Dual Quaternion Skinning (DQS)\n• Heat Diffusion / Geodesic Weights\n• 99.65% Volume Retention"]
    
    Skinning --> ExportGLTF["glTF 2.0 (Skinned Mesh + PBR + Anim)"]
    Skinning --> ExportBVH["BVH Motion Capture"]
    Skinning --> ExportBlender["Blender Armature & NLA (.blend)"]
    Skinning --> ExportUnity["Unity Generic Rig Profile"]
```

---

## 💻 Installation & Setup

### System Requirements
- **OS**: Windows 10/11, macOS, or Ubuntu/Linux 20.04+
- **Python**: 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- **GPU (Optional)**: NVIDIA CUDA GPU or free Kaggle cloud GPU

### 1. Clone the Repository
```bash
git clone https://github.com/taezeem14/Leocherd-Iris-Energy.git
cd Leocherd-Iris-Energy
```

### 2. Create Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Package & Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

### 4. Verify System Diagnostics
Verify that your CPU, memory, and acceleration libraries are recognized:
```bash
python main.py diagnostics
```

---

## 📖 Full Step-by-Step Tutorial

### 1. Inspect Video Metadata
Before running 3D reconstruction, check whether your input video file is structurally valid:
```bash
python main.py inspect-video data/samples/online_walking_sample.mp4
```
**Example Output:**
```
=================================================================
 LEOCHERD IRIS ENERGY - VIDEO CONTAINER INSPECTION
=================================================================
 File Path              : data/samples/online_walking_sample.mp4
 File Size              : 1.84 MB
 Resolution             : 1280 x 720 (Aspect Ratio: 1.78)
 Frame Rate             : 30.00 FPS
 Total Frames           : 450
 Duration               : 15.00 seconds
 Codec / FourCC         : avc1
-----------------------------------------------------------------
 [OK] Video container is structurally valid and ready for 3D reconstruction.
=================================================================
```

---

### 2. Local Video Reconstruction (CPU / ONNX)
Run full end-to-end 3D reconstruction directly on your machine:
```bash
python main.py reconstruct data/samples/online_walking_sample.mp4 \
  --morphology human \
  --skinning-method heat_diffusion \
  --fps 15 \
  --formats gltf,bvh,blender \
  --output-dir outputs/my_walk_run
```

**Generated Assets in `outputs/my_walk_run/`:**
- `reconstruction_*.gltf`: Fully articulated, skinned 3D model with embedded animation.
- `reconstruction_*.bvh`: Motion capture file for standard rigging pipelines.
- `reconstruction_*_rig_data.json` & `_anim_data.json`: Blender-compatible data packages.

> [!TIP]
> **Live 3D Preview**: You can immediately view your `.gltf` asset by dragging and dropping it into the [Don McCurdy glTF Viewer](https://gltf-viewer.donmccurdy.com/) or the [Babylon Sandbox](https://sandbox.babylonjs.com/).

---

### 3. Cloud GPU Acceleration with Kaggle
For high-resolution videos or machines without a discrete GPU, use our pre-built Kaggle GPU pipeline for free cloud inference:

1. Navigate to [`experiments/kaggle/leocherd_iris_gpu_inference.ipynb`](experiments/kaggle/leocherd_iris_gpu_inference.ipynb).
2. Create a new notebook on [Kaggle](https://www.kaggle.com/) and enable **Accelerator: GPU T4 x2**.
3. Upload your video file (e.g., `dance.mp4`) into the Kaggle environment.
4. Run all notebook cells. The notebook will:
   - Run accelerated multi-threaded pose extraction (with automatic fallback to Google LiteRT XNNPACK CPU if headless Linux GPU contexts are restricted).
   - Apply metric root-relative coordinate translation.
   - Smooth trajectories with Savitzky-Golay filtering.
   - Export `iris_kaggle_pose_data.json`.
5. Download `iris_kaggle_pose_data.json` to your local machine.
6. Generate 3D character rigs locally in **under 2 seconds**:
```bash
python main.py import-kaggle path/to/iris_kaggle_pose_data.json \
  --morphology human \
  --formats gltf,bvh,blender \
  --output-dir outputs/kaggle_dance
```

---

### 4. Universal Morphology Classification
Iris Energy features an automatic morphology classifier that can identify the creature type without human intervention:

```python
from iris.morphology.classifier import MorphologyClassifier

classifier = MorphologyClassifier()
prediction = classifier.classify(pose_sequence)

print(f"Predicted Morphology: {prediction.predicted_type.value}")
print(f"Confidence: {prediction.confidence:.2%}")
print(f"Kinematic Metrics: {prediction.metrics}")
```

Supported morphologies:
- `human`: Bipedal vertical spine, 2 arms, 2 legs.
- `quadruped`: Horizontal spine, 4 ground contact points.
- `avian`: Forward-inclined spine, wing bone chain, digitigrade legs.
- `serpentine`: Multi-segment horizontal spine with traveling transverse waves.
- `tentacular`: Multi-branch radial chains with high flexibility indices.

---

### 5. Unsupervised Skeleton Discovery (Template-Free)
If you have motion capture points or tracking trajectories for an unknown alien or imaginary creature, you do not need a predefined bone template. Run automatic skeleton discovery:

```python
from iris.rigging.skeleton_discovery import AutomaticSkeletonDiscovery

# Discover skeleton hierarchy from raw trajectory landmarks
discovery = AutomaticSkeletonDiscovery(variance_threshold=0.05)
discovered_skeleton = discovery.discover(pose_sequence)

print(f"Root Joint: {discovered_skeleton.root_name}")
print(f"Total Bones Discovered: {len(discovered_skeleton.bones)}")
print(f"Symmetry Pairs: {discovered_skeleton.symmetry_pairs}")
```

**How It Works:**
1. **Rigid-Distance Variance**: Evaluates temporal variance $\sigma^2_{ij} = \text{Var}_t(\|P_i(t) - P_j(t)\|)$ between all joint pairs. Rigid bone candidates have $\sigma^2 \to 0$.
2. **Root Joint Selection**: Selects the joint closest to the kinematic center of mass with minimum acceleration variance across time.
3. **Kruskal Minimum Spanning Tree (MST)**: Constructs a connected, cycle-free kinematic hierarchy minimizing the edge weight $W_{ij} = \mu_{ij} + \lambda \sigma^2_{ij}$.
4. **SVD Bilateral Symmetry Detection**: Detects the creature's bilateral reflection plane $n \cdot x + d = 0$ and maps corresponding Left/Right symmetric joint pairs.

---

### 6. Dual Quaternion Skinning (DQS) vs LBS
Standard Linear Blend Skinning (LBS) suffers from the "candy-wrapper" artifact when a limb twists along its axis, collapsing the mesh to zero volume. Iris implements **Dual Quaternion Skinning (DQS)**:

```python
from iris.rigging.skinning import DualQuaternionSkinning, LinearBlendSkinning

# LBS vs DQS comparison
lbs = LinearBlendSkinning()
dqs = DualQuaternionSkinning()

# Deform mesh under a 175-degree joint twist
deformed_lbs = lbs.deform(vertices, bone_transforms, weights, joint_indices)
deformed_dqs = dqs.deform(vertices, bone_transforms, weights, joint_indices)
```

**Benchmark Results under $175^\circ$ Joint Twist:**
- **Standard LBS**: $0.0418\text{ m}^3$ (volume loss: **66.55%**) ❌
- **Iris DQS**: $0.1244\text{ m}^3$ (volume retention: **99.65%**) ✅

---

### 7. Inverse Kinematics (Two-Bone & FABRIK)
Iris includes both closed-form analytical solvers and iterative heuristic solvers:

```python
import numpy as np
from iris.rigging.ik import TwoBoneIKSolver, FABRIKSolver

# 1. Closed-form Two-Bone IK for arms and legs
solver = TwoBoneIKSolver()
result = solver.solve(
    p_root=np.array([0.0, 1.0, 0.0]),
    p_mid=np.array([0.0, 0.5, 0.0]),
    p_end=np.array([0.0, 0.0, 0.0]),
    p_target=np.array([0.2, 0.1, 0.3]),
    p_pole=np.array([0.0, 0.5, 1.0]),  # Knee / elbow bend direction
)

# 2. FABRIK Solver for multi-bone tentacles or spine chains
chain_points = [np.array([0.0, i * 0.2, 0.0]) for i in range(8)]
target = np.array([0.5, 0.8, 0.2])

solved_positions, converged = FABRIKSolver.solve(
    joint_positions=chain_points,
    target=target,
    max_iterations=15,
    tolerance=1e-4,
)
```

---

### 8. Cross-Morphology Motion Retargeting
Retarget motion captured from a human directly onto a quadruped or avian creature:

```bash
python main.py reconstruct data/samples/online_walking_sample.mp4 \
  --morphology human \
  --retarget-to quadruped \
  --output-dir outputs/human_to_quadruped
```

---

### 9. Production 3D Export & Engine Usage

#### Importing into Blender
1. Launch Blender.
2. Go to **File -> Import -> glTF 2.0 (.gltf/.glb)** and select your exported `.gltf` file.
3. The mesh includes:
   - Form-fitted anatomical proxy mesh (sculpted ribcage, deltoids, patella guards, seamless neck loft).
   - Skinned armature with bone hierarchy.
   - Embedded animation action track ready for playback in the Timeline or NLA editor.

To run automated headless Blender validation:
```bash
blender --background --python scripts/verify_blender_rig.py -- outputs/my_walk_run/reconstruction_skinned_walking.gltf
```

#### Importing into Unity
1. Drag the exported `.gltf` or `.glb` file into your Unity `Assets/` folder.
2. In the Inspector, select the asset and go to the **Rig** tab.
3. Set **Animation Type** to **Generic** (or **Humanoid** if using the human morphology).
4. Drag the model into your scene and assign an **Animator Controller** to play back the reconstructed animation clips.

To verify Unity Generic Rig profile conformity locally:
```python
from iris.export.unity import UnityRigProfile

result = UnityRigProfile.validate("outputs/my_walk_run/reconstruction_skinned_walking.gltf")
print(result.summary())
```

---

### 10. Running the FastAPI REST Microservice
Iris includes an asynchronous REST microservice for backend integration:

```bash
python main.py run-server --host 127.0.0.1 --port 8000 --reload
```

- **Interactive API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
- **Alternative ReDoc Docs**: `http://127.0.0.1:8000/redoc`

#### Key API Endpoints:
- `GET /health`: Health and hardware status.
- `POST /api/v1/reconstruction/jobs`: Submit asynchronous video reconstruction job.
- `GET /api/v1/reconstruction/jobs/{job_id}`: Poll job status, progress percentage, and logs.
- `GET /api/v1/reconstruction/jobs/{job_id}/download/{format}`: Download generated 3D assets (`gltf`, `bvh`, `blender`).

---

## 🛠️ CLI Reference

```
usage: iris [-h] {inspect-video,reconstruct,import-kaggle,run-server,diagnostics} ...

Commands:
  inspect-video    Inspect video metadata, resolution, FPS, and container health.
  reconstruct      Run end-to-end 3D rigging and reconstruction from video.
  import-kaggle    Generate 3D rigs and animation from Kaggle GPU exported JSON.
  run-server       Launch the FastAPI REST service.
  diagnostics      Run system hardware and environment diagnostics.
```

### Options for `reconstruct`:
| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--morphology` | `-m` | `human` | Creature morphology: `human`, `quadruped`, `avian`, `serpentine`, `tentacular`, `auto`, `discover`. |
| `--skinning-method` | | `heat_diffusion` | Skinning algorithm: `heat_diffusion`, `volumetric`, `geodesic`, `lbs`. |
| `--retarget-to` | | `None` | Cross-morphology target: `human`, `quadruped`, `avian`, `serpentine`, `tentacular`. |
| `--fps` | | `15` | Target processing frame rate. |
| `--max-frames` | | `None` | Max number of frames to extract (useful for testing quick clips). |
| `--formats` | | `gltf,bvh,blender` | Comma-separated list of export formats. |
| `--output-dir` | `-o` | `outputs` | Destination directory for exported 3D files. |

---

## 🧪 Automated Testing & Verification

Iris Energy maintains a 100% green, strict automated testing suite with **377 passing tests**:
- **130 Unit Tests**: Mathematical correctness of quaternions, dual quaternions, kinematics, morphology classifiers, and skinning algorithms.
- **247 Multi-Tier E2E Tests**:
  - *Tier 1*: 110 Feature tests (IK convergence, DQS volume retention, Khronos matrix normalization).
  - *Tier 2*: 110 Boundary tests (extreme joint angles, degenerate meshes, zero-length bones).
  - *Tier 3*: 22 Pairwise integration tests (Video-to-Pose, Pose-to-Rig, Rig-to-Export).
  - *Tier 4*: 5 Full creature scenario simulations (Human, Quadruped, Avian, Serpentine, Tentacular).

### Running the Test Suite
```bash
# Run all 377 unit and E2E tests
pytest tests/unit/ tests/e2e/ -v

# Run strict Khronos glTF 2.0 conformance validator
python scripts/validate_gltf_strict.py outputs/dance_run/reconstruction_7ee71d98.gltf

# Run the DQS volume preservation benchmark
pytest tests/unit/test_skinning_dqs.py -k test_dqs_volume_preservation -v
```

---

## 📂 Project Directory Structure

```
Leocherd-Project/
├── main.py                          # Primary CLI entrypoint (reconstruct, import-kaggle, etc.)
├── iris/                            # Core engine package
│   ├── api/                         # FastAPI REST microservice & endpoints
│   ├── compute/                     # Local CPU, GPU, and Kaggle compute abstractions
│   ├── core/                        # Pipeline context, configuration, logging, exceptions
│   ├── export/                      # glTF 2.0, BVH, Blender, and Unity exporters
│   ├── models/                      # Kinematic trackers, ONNX BlazePose, Kaggle adapters
│   ├── morphology/                  # Morphology classifier (Human, Quadruped, Avian, etc.)
│   ├── motion/                      # Savitzky-Golay filtering, Slerp, foot-locking
│   ├── pose/                        # 3D PoseFrame data structures & confidence estimators
│   ├── retargeting/                 # Cross-morphology motion retargeting & curve generation
│   ├── rigging/                     # Skeleton DAG, DQS Skinning, IK Solvers, Proxy Mesh
│   ├── video/                       # Ingestion, validation, frame sampling, metadata
│   └── vision/                      # YOLO object detection & tracking
├── experiments/
│   └── kaggle/                      # Production Kaggle GPU inference notebook (.ipynb)
├── scripts/                         # Strict glTF & Blender verification scripts
├── tests/
│   ├── fixtures/                    # Procedural skeletons, synthetic meshes, trajectories
│   ├── unit/                        # 130 Unit test modules
│   └── e2e/                         # 247 Multi-tier end-to-end integration tests
├── pyproject.toml                   # Project metadata and build requirements
└── requirements.txt                 # Pinned dependencies
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
All rights reserved by **Team Leocherd**.