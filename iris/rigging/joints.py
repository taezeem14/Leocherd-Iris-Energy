"""
Joint node and bone edge representations in the skeletal hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from iris.rigging.orientation import QuaternionMath


@dataclass
class JointNode:
    """Represents a node in the articulated skeleton tree."""

    name: str
    parent_name: Optional[str] = None
    children_names: List[str] = field(default_factory=list)

    # Rest pose (bind pose) in world space
    rest_world_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # Current evaluated transforms
    world_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    local_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # Rotations as unit quaternions [w, x, y, z]
    world_rotation: np.ndarray = field(default_factory=QuaternionMath.identity)
    local_rotation: np.ndarray = field(default_factory=QuaternionMath.identity)

    # Affine matrices (4x4)
    local_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    world_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))


@dataclass
class BoneEdge:
    """Represents an articulated bone linking parent to child joint."""

    name: str
    parent_joint: str
    child_joint: str
    rest_length: float
    current_length: float = 0.0

    def compute_rest_direction(self, rest_parent: np.ndarray, rest_child: np.ndarray) -> np.ndarray:
        return QuaternionMath.normalize_vector(rest_child - rest_parent)

    def compute_current_direction(self, cur_parent: np.ndarray, cur_child: np.ndarray) -> np.ndarray:
        diff = cur_child - cur_parent
        self.current_length = float(np.linalg.norm(diff))
        return QuaternionMath.normalize_vector(diff)
