# Pose Estimation & 3D Reconstruction Solutions: Research & Benchmark

## 1. Comparative Evaluation of Open-Source Solutions

| Model / Framework | License | Keypoints / Output | Runtime / Weights Size | Hardware Requirement | Depth Capability | Fit for Iris Energy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MediaPipe BlazePose (ONNX)** | Apache 2.0 | 33 Landmarks (Full body + face/hands) | ONNX Runtime / ~9 MB | CPU (x86/ARM), minimal RAM (<100MB) | Direct 3D metric landmarks relative to hip center | **PRIMARY ADAPTER** (Ultra-light, true 3D output, permissive) |
| **Google MoveNet (Thunder/Lightning)** | Apache 2.0 | 17 Keypoints (COCO standard) | TFLite / ONNX / ~4-12 MB | CPU / GPU, fast inference | 2D only (requires kinematic geometric 3D lifting) | **SECONDARY ADAPTER** (Fast 2D verification) |
| **Ultralytics YOLOv8/11-Pose** | AGPL-3.0 | 17 Keypoints + Multi-person bbox | PyTorch / ONNX / ~6-40 MB | GPU preferred, CPU acceptable | 2D only | Multi-subject tracking adapter (licensing constraints note) |
| **OpenMMLab RTMPose** | Apache 2.0 | 17/133 Keypoints (Whole-body) | ONNX / TensorRT / ~25 MB | GPU preferred, CPU slower | 2D + Lifted 3D | Complex dependency tree |
| **CMU OpenPose** | Non-Commercial | 25 Body Keypoints (BODY_25) | Caffe / CUDA / >100 MB | High-end NVIDIA GPU mandatory | 2D only | Rejected (license + heavy GPU requirement) |

---

## 2. Selected Architecture: BlazePose 3D ONNX + Kinematic Bone Regularization

For the first production-grade vertical slice, Leocherd Iris Energy implements a **dual-model ONNX pipeline**:
1. **Subject Detector & ROI Aligner:**
   - Detects the primary human or quadruped torso bounding box across frames.
   - Computes square region of interest (ROI) with 20% margin to prevent limb clipping during dynamic movement.
2. **Pose Landmark Model (BlazePose Full 3D):**
   - Inputs: $256 \times 256 \times 3$ normalized RGB tensor.
   - Outputs:
     - 33 2D screen coordinates $(x, y) \in [0, 1]$ with visibility and confidence.
     - 33 3D world landmark coordinates $(X, Y, Z)$ measured in virtual meters with the root origin $(0, 0, 0)$ calibrated at the midpoint between the left and right hip joints.
3. **Biological Kinematic Chain Regularizer:**
   - Monocular neural depth $Z$ can exhibit scaling noise when limbs move directly towards or away from the camera.
   - Iris Energy applies a dynamic bone length stabilization filter:
     $$\vec{P}_{child}^{stabilized} = \vec{P}_{parent} + L_{rest} \cdot \frac{\vec{P}_{child} - \vec{P}_{parent}}{\|\vec{P}_{child} - \vec{P}_{parent}\|}$$
   - This ensures continuous anatomical integrity without bone stretching or dismemberment.

---

## 3. Data Representation Standard

All pose estimators in Iris Energy implement the unified `iris.pose.base.BasePoseEstimator` interface and produce strongly-typed `PoseFrame` objects:

```python
class Joint3D(BaseModel):
    name: str
    x: float          # Meters or normalized coordinate (right +X)
    y: float          # Meters or normalized coordinate (up +Y)
    z: float          # Meters depth (forward +Z)
    confidence: float # Range [0.0, 1.0]
    visibility: float # Range [0.0, 1.0]

class PoseFrame(BaseModel):
    frame_index: int
    timestamp_sec: float
    subject_id: int
    morphology: str
    joints: Dict[str, Joint3D]
    bounding_box: Tuple[float, float, float, float] # [x_min, y_min, x_max, y_max]
```
