# Computer Vision & 3D Pose Estimation Model Evaluation

## 1. Candidate Model Benchmark

| Model / Architecture | Creature Support | Latency (CPU) | Latency (GPU) | Model Size | License | Assessment for Iris Energy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MediaPipe BlazePose 3D (ONNX)** | Humanoid (33 landmarks) | **~35 ms** (Real-time on 2-core CPU) | < 5 ms | 5.3 MB | Apache 2.0 | **Primary Human Estimator.** Lightweight, outputs true 3D metric coordinates in meters. |
| **MoveNet Thunder / Lightning** | Humanoid (17 keypoints) | ~25 ms | < 4 ms | 4–12 MB | Apache 2.0 | **Secondary 2D Estimator.** Robust 2D keypoints, requires kinematic geometric depth lifting. |
| **DeepLabCut / SuperAnimal** | Quadruped, Rodents, Primates | > 400 ms (Too slow on CPU) | ~15 ms | ~150 MB | LGPL-3.0 / Apache | **Cloud/Kaggle Quadruped Target.** High precision for animals, but requires GPU compute. |
| **AP-10K Animal Pose (MMPose)** | 54 Animal Species (17 keypoints) | ~250 ms | ~10 ms | ~85 MB | Apache 2.0 | **Recommended Animal Model.** Ideal for Kaggle GPU benchmark and offline animal rigging. |
| **YOLOv8-Pose / YOLO11-Pose** | Humanoid (17 keypoints + multi-person bbox) | ~80 ms | < 8 ms | ~6–40 MB | AGPL-3.0 | **Evaluation Only.** Licensing restrictions (AGPL) require user opt-in before distribution. |
| **Optical Flow (Farneback / RAFT)** | Motion dense field | ~120 ms (Farneback) | ~10 ms (RAFT) | ~20 MB (RAFT) | BSD / Apache | Farneback used locally for temporal velocity estimation; RAFT suited for Kaggle. |

---

## 2. Creature Morphology Feasibility Analysis

A human pose estimator **cannot** rig a quadruped, snake, or tentacle. Iris Energy enforces strict morphology isolation:
- **Humanoid:** Ingested via BlazePose / MoveNet. Mapped to 19-joint humanoid rig via `HumanPoseSynthesizer`.
- **Quadruped:** Evaluated via AP-10K / DeepLabCut animal keypoints (17 body landmarks including withers, croup, paws, hocks, tail base). Mapped to `QuadrupedMorphology`.
- **Serpentine & Tentacular:** Inferred through foreground skeletonization, medial axis transforms, and dynamic spline contour tracking.
