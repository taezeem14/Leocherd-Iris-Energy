"""
Base schemas and abstract interfaces for pose estimation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class Joint3D(BaseModel):
    """Represents a single anatomical joint with 3D spatial coordinates and confidence."""

    name: str
    x: float = Field(description="Coordinate along horizontal X-axis (right +X)")
    y: float = Field(description="Coordinate along vertical Y-axis (up +Y or screen down)")
    z: float = Field(default=0.0, description="Inferred depth along Z-axis (camera forward +Z)")
    confidence: float = Field(default=1.0, description="Detection confidence")
    visibility: float = Field(default=1.0, description="Presence/visibility score")

    def model_post_init(self, __context) -> None:
        object.__setattr__(self, "confidence", min(1.0, max(0.0, float(self.confidence))))
        object.__setattr__(self, "visibility", min(1.0, max(0.0, float(self.visibility))))

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def is_visible(self, threshold: float = 0.35) -> bool:
        return self.confidence >= threshold and self.visibility >= threshold


class PoseFrame(BaseModel):
    """Complete skeletal pose observed at a specific video frame timestamp."""

    frame_index: int
    timestamp_sec: float
    subject_id: int = 1
    morphology: str = "human"
    joints: Dict[str, Joint3D]
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # [xmin, ymin, xmax, ymax]
    is_interpolated: bool = False

    @property
    def joint_names(self) -> List[str]:
        return list(self.joints.keys())

    def get_joint(self, name: str) -> Optional[Joint3D]:
        return self.joints.get(name)

    def get_coordinates_matrix(self) -> np.ndarray:
        """Returns Nx3 matrix of (x, y, z) coordinates for all joints in fixed order."""
        return np.array([j.to_numpy() for j in self.joints.values()], dtype=np.float32)

    def mean_confidence(self) -> float:
        if not self.joints:
            return 0.0
        return sum(j.confidence for j in self.joints.values()) / len(self.joints)


class BasePoseEstimator(ABC):
    """Abstract interface for all pose estimation models and adapters."""

    def __init__(self, name: str, min_confidence: float = 0.35) -> None:
        self.name = name
        self.min_confidence = min_confidence

    @abstractmethod
    def estimate(
        self,
        rgb_frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        subject_id: int = 1,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> PoseFrame:
        """Estimates pose from an RGB image array."""
        pass

    def estimate_batch(
        self,
        frames: List[Tuple[np.ndarray, int, float]],
        subject_id: int = 1,
    ) -> List[PoseFrame]:
        """Default batch estimation sequentially applying estimate()."""
        return [self.estimate(f[0], f[1], f[2], subject_id=subject_id) for f in frames]
