# Leocherd Iris Energy - Project Status

## Executive Summary
Leocherd Iris Energy is a Python-first, hardware-aware computer vision and 3D rigging platform.
As of Milestone 1, the foundational architecture, video processing, neural pose estimation, temporal motion processing, morphology plugins, graph skeleton reconstruction, and kinematic regularization subsystems have been successfully engineered, verified, and committed to version control (`ee0754e`).

---

## Component Audit Matrix

| Subsystem | Module Path | Status | Validation Summary |
| :--- | :--- | :--- | :--- |
| **System Diagnostics** | `scripts/system_diagnostics.py` | Complete | Tested on host hardware (Intel Pentium B950, 8GB RAM, no CUDA, Windows 10). Auto-selects `CPU_LIGHTWEIGHT` profile. |
| **Installer & Cache** | `scripts/install.py` | Complete | Implements Drive D: cache redirection (`--cache-dir D:\...`) preventing disk exhaustion on partition C:. |
| **Configuration & Pipeline** | `iris.core.config`, `pipeline` | Complete | Pydantic v2 settings, EventBus, error hierarchy, stage lifecycle manager. |
| **Video Ingestion** | `iris.video` | Complete | Format validation, magic byte checks, OpenCV metadata probe, memory-bounded streaming extractor with resolution scaling. |
| **Detection & Tracking** | `iris.vision` | Complete | Bounding box regression, contour motion detector, IoU multi-subject tracker, 2D overlay visualizer. |
| **Pose Estimation** | `iris.pose` | Complete | Strongly-typed 3D landmarks, confidence filtering, humanoid rig synthesizer, and ONNX BlazePose adapter. |
| **Model Weights** | `iris.models` | Complete | Downloaded and validated official OpenCV Zoo BlazePose ONNX model (`5.3 MB`) in `data/models/`. Built-in kinematic fallback available. |
| **Motion Engine** | `iris.motion` | Complete | Time-series trajectory representation, central-difference velocity/acceleration derivatives, One Euro adaptive filtering, spline imputation. |
| **Morphology Engine** | `iris.morphology` | Complete | Plugin system with Human (19 joints), Biped, Quadruped (mammal/reptile), Serpentine (N-joint), Tentacular, and Custom JSON graphs. |
| **Rigging & Kinematics** | `iris.rigging` | Complete | Full quaternion rotation algebra (`QuaternionMath`), tree skeleton with recursive bone length enforcement, Linear Blend Skinning (LBS). |
| **Animation Engine** | `iris.retargeting.animation` | Complete | Animation keyframe generation, root motion tracking, local bone quaternion curves. |
| **Compute Providers** | `iris.compute` | In Progress | Hybrid Local CPU/GPU + Kaggle Notebook compute abstraction. |
| **Asset Exporters** | `iris.export` | In Progress | Native glTF 2.0 with skeleton/skin/animation, BVH exporter, Blender automation. |
| **API & CLI** | `iris.api`, `main.py` | In Progress | FastAPI REST service and CLI tool (`iris inspect-video`, `iris reconstruct`). |

---

## Active Hardware & Resource Profile
- **Host CPU:** Intel Pentium B950 @ 2.10 GHz (2 physical cores / 2 threads, Sandy Bridge).
- **Instruction Support:** SSE4.1, SSE4.2 (No AVX/AVX2).
- **RAM:** 7.9 GB visible (~1.8 GB free).
- **Host GPU:** Intel HD Graphics (No CUDA).
- **Primary Execution Engine:** ONNX Runtime 1.31 CPU Execution Provider with multi-threading set to 2 intra-op threads.
- **Drive Partitioning:** All virtual environments, pip wheel caches, and models isolated to Drive D: (27.4 GB free).
