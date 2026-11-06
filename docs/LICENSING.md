# Leocherd Iris Energy - Licensing Policy & Compliance

## 1. Project License
Leocherd Iris Energy is released under the **Apache License 2.0**.
This allows open development, modular extension, and commercial embedding while guaranteeing trademark protection and clear patent grants.

## 2. Third-Party Dependencies and Models

| Component | Upstream Project | License | Compliance Rules |
| :--- | :--- | :--- | :--- |
| **ONNX Runtime** | Microsoft | MIT | Permissive, compatible with commercial and closed distribution. |
| **OpenCV (headless)** | OpenCV Foundation | Apache 2.0 | Permissive, video decoding and image processing. |
| **NumPy & SciPy** | SciPy Community | BSD-3-Clause | Standard scientific Python foundation. |
| **BlazePose Weights** | Google MediaPipe | Apache 2.0 | Permissive, model checkpoints packaged under Apache 2.0. |
| **FastAPI / Pydantic** | Tiangolo / Colvin | MIT | Modern asynchronous REST API framework. |
| **Trimesh** | Dawson-Haggerty | MIT | 3D mesh loading, glTF serialization, math utilities. |
| **Ultralytics (Optional)** | Ultralytics Inc. | AGPL-3.0 | **Isolated Adapter Only:** Kept strictly optional to avoid viral copyleft licensing obligations unless explicitly authorized. |

## 3. Strict Model Governance Policy
- No proprietary or non-commercial weights (such as original OpenPose Academic) are packaged by default.
- Checksums (SHA-256) are calculated and validated for every downloaded model weight.
- Model adapters declare their license in their metadata registration schema.
