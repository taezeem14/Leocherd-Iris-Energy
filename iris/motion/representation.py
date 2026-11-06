"""
Temporal trajectory representation, velocity, and acceleration mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class JointTrajectory:
    """Continuous temporal trajectory of a single anatomical joint across time."""

    name: str
    timestamps: np.ndarray    # Shape: (T,)
    positions: np.ndarray     # Shape: (T, 3)
    confidences: np.ndarray   # Shape: (T,)
    velocities: Optional[np.ndarray] = None      # Shape: (T, 3)
    accelerations: Optional[np.ndarray] = None   # Shape: (T, 3)

    def compute_derivatives(self) -> None:
        """Computes velocity and acceleration using second-order central difference."""
        T = len(self.timestamps)
        if T < 2:
            self.velocities = np.zeros_like(self.positions)
            self.accelerations = np.zeros_like(self.positions)
            return

        dt = np.gradient(self.timestamps)
        # Avoid division by zero
        dt = np.where(dt <= 1e-6, 1.0 / 30.0, dt)

        # Velocity: dP/dt
        vel = np.zeros_like(self.positions)
        for dim in range(3):
            vel[:, dim] = np.gradient(self.positions[:, dim], self.timestamps)
        self.velocities = vel

        # Acceleration: dV/dt
        acc = np.zeros_like(self.positions)
        for dim in range(3):
            acc[:, dim] = np.gradient(self.velocities[:, dim], self.timestamps)
        self.accelerations = acc

    def speeds(self) -> np.ndarray:
        if self.velocities is None:
            self.compute_derivatives()
        return np.linalg.norm(self.velocities, axis=1)


@dataclass
class MotionSequence:
    """Collection of joint trajectories representing articulated motion over time."""

    timestamps: np.ndarray
    trajectories: Dict[str, JointTrajectory]
    fps: float

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)

    @property
    def duration_sec(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def get_joint_positions_at_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        return {name: traj.positions[frame_idx] for name, traj in self.trajectories.items()}
