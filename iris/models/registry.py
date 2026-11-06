"""
Model registry declaring supported neural checkpoints and their hardware criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelMetadata:
    name: str
    version: str
    source_url: str
    license: str
    size_mb: float
    sha256: Optional[str]
    input_shape: List[int]  # e.g. [1, 3, 256, 256]
    output_type: str        # e.g. "33_3d_landmarks"
    supported_morphologies: List[str]
    hardware_recommendation: str
    description: str


MODEL_REGISTRY: Dict[str, ModelMetadata] = {
    "blazepose-mediapipe-onnx": ModelMetadata(
        name="blazepose-mediapipe-onnx",
        version="2023mar",
        source_url="https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx",
        license="Apache-2.0",
        size_mb=5.3,
        sha256=None,
        input_shape=[1, 3, 256, 256],
        output_type="33_landmarks_3d",
        supported_morphologies=["human", "biped"],
        hardware_recommendation="CPU_LIGHTWEIGHT or CUDA_GPU",
        description="Official OpenCV Zoo MediaPipe BlazePose 3D full-body pose landmark detector.",
    ),
    "geometric-kinematic-tracker": ModelMetadata(
        name="geometric-kinematic-tracker",
        version="1.0",
        source_url="builtin",
        license="Apache-2.0",
        size_mb=0.0,
        sha256=None,
        input_shape=[1, 3, 256, 256],
        output_type="kinematic_3d_landmarks",
        supported_morphologies=["human", "biped", "quadruped", "serpentine", "tentacular", "custom"],
        hardware_recommendation="CPU_LIGHTWEIGHT (offline fallback, zero downloads)",
        description="Offline biological kinematic model enforcing physiological bone length and joint angle constraints.",
    ),
}
