# Changelog

All notable changes to **Leocherd Iris Energy** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-06
### Added
- Core pipeline architecture and pipeline stage abstraction (`iris.core.pipeline`).
- System diagnostics suite (`scripts/system_diagnostics.py`) with hardware profiling.
- CPU-first dependency installation system configured for Drive D: isolation (`scripts/install.py`).
- Video ingestion, validation, metadata inspection, and streaming frame extraction (`iris.video`).
- Monocular 3D pose estimation engine with ONNX BlazePose adapter (`iris.pose`).
- Temporal motion engine: velocity, acceleration, One Euro filtering, spline interpolation (`iris.motion`).
- Morphology system with Human, Biped, Quadruped, Serpentine, and Custom plugins (`iris.morphology`).
- Graph-based 3D skeleton builder, kinematic length regularization, and bone quaternion calculator (`iris.rigging`).
- Multi-format asset exporters: standalone glTF 2.0 with skin and animation, BVH, and Blender Python generator (`iris.export`).
- FastAPI REST service (`iris.api`) and CLI interface (`main.py`).
- Comprehensive unit and integration test suite (`tests/`).
