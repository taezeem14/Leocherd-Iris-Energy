# Leocherd Iris Energy

> **Python-First Universal AI Video-to-3D Rigging and Motion Reconstruction Engine**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/hardware-CPU%20Lightweight%20%2F%20CUDA%20GPU-green)](#hardware-profiles)

Leocherd Iris Energy is a serious, extensible, Python-first computer vision and 3D rigging platform designed to convert real-world monocular video footage into fully articulated 3D skeletons, continuous bone transformations, and exportable animation assets.

---

## Key Features

- **Robust Video Processing:** Streams and validates MP4, MOV, WebM, and AVI videos without exhausting memory.
- **Genuine 3D Pose Estimation:** Integrates real neural pose estimation (BlazePose 3D ONNX / MoveNet) with confidence scoring and occlusion tracking.
- **Physically Grounded Kinematics:** Solves monocular depth ambiguity through kinematic bone-length invariance regularizers.
- **Temporal Motion Reconstruction:** Smooths jitter while preserving rapid transients via adaptive One Euro and Savitzky-Golay filtering.
- **Modular Morphology Plugins:** Supports Human, Biped, Quadruped, Serpentine, Tentacular, and custom graph skeletons.
- **Export Formats:**
  - Standard **glTF 2.0 (.gltf / .glb)** with bone hierarchy, skinning data, and animation tracks.
  - Industry-standard **Biovision Hierarchy (.bvh)** motion capture files.
  - Headless **Blender Python scripts** for automated scene and armature generation.
- **Production API & CLI:** FastAPI service with async job orchestration and a direct command-line interface.

---

## Quickstart

### 1. Diagnostics & Hardware Profile Check
```bash
python scripts/system_diagnostics.py
```

### 2. Inspect a Video File
```bash
python main.py inspect-video path/to/input.mp4
```

### 3. Reconstruct a 3D Skeleton and Animation
```bash
python main.py reconstruct path/to/input.mp4 --morphology human --output-dir outputs/
```

### 4. Run the REST API
```bash
python main.py run-server --host 127.0.0.1 --port 8000
```
Open your browser to `http://127.0.0.1:8000/docs` for interactive OpenAPI documentation.

### 5. Run the Test Suite
```bash
pytest tests/
```

---

## Documentation
- [System Architecture](docs/ARCHITECTURE.md)
- [Pose Estimation & Research Benchmark](docs/POSE_ESTIMATION.md)
- [Video Pipeline](docs/VIDEO_PIPELINE.md)
- [Skeleton & Morphology Engine](docs/SKELETON_ENGINE.md)
- [Animation & Retargeting](docs/ANIMATION.md)
- [Blender Automation](docs/BLENDER.md)
- [REST API Specification](docs/API.md)
- [Installation Guide](docs/INSTALLATION.md)
- [Licensing & Compliance](docs/LICENSING.md)
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [Implementation Progress](docs/PROGRESS.md)

---

## License
Apache License 2.0. See [LICENSE](LICENSE) and [docs/LICENSING.md](docs/LICENSING.md).
