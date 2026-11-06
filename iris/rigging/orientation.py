"""
Quaternion and 3D spatial transformation mathematics for rigging and skeletal articulation.
"""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np


class QuaternionMath:
    """Rigorous quaternion operations conforming to standard (w, x, y, z) representation."""

    @staticmethod
    def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < eps:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return (v / norm).astype(np.float32)

    @staticmethod
    def normalize_quat(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm < eps:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return (q / norm).astype(np.float32)

    @staticmethod
    def identity() -> np.ndarray:
        """Returns identity quaternion [w=1, x=0, y=0, z=0]."""
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
        """
        Computes the minimal-arc unit quaternion rotating v_from into v_to.
        Both vectors are normalized internally.
        Returns: [w, x, y, z]
        """
        u1 = QuaternionMath.normalize_vector(v_from)
        u2 = QuaternionMath.normalize_vector(v_to)

        dot = float(np.dot(u1, u2))

        # Parallel identical vectors
        if dot > 0.999999:
            return QuaternionMath.identity()

        # Parallel opposite vectors: 180 degree rotation around arbitrary orthogonal axis
        if dot < -0.999999:
            ortho = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(u1[0]) > 0.9:
                ortho = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis = QuaternionMath.normalize_vector(np.cross(u1, ortho))
            return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)

        axis = np.cross(u1, u2)
        w = math.sqrt(max(0.0, (1.0 + dot) * 0.5))
        inv_denom = 1.0 / (2.0 * w) if w > 1e-6 else 1.0
        q = np.array([w, axis[0] * inv_denom, axis[1] * inv_denom, axis[2] * inv_denom], dtype=np.float32)
        return QuaternionMath.normalize_quat(q)

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Hamilton product q = q1 * q2 (applies q2 first, then q1).
        Format: [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return QuaternionMath.normalize_quat(np.array([w, x, y, z], dtype=np.float32))

    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Inverse of a unit quaternion: [w, -x, -y, -z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotates 3D vector v by unit quaternion q: v' = q * (0, v) * q^-1."""
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
        res = QuaternionMath.multiply(QuaternionMath.multiply(q, qv), QuaternionMath.conjugate(q))
        return res[1:4]

    @staticmethod
    def quat_to_matrix(q: np.ndarray) -> np.ndarray:
        """Converts unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = QuaternionMath.normalize_quat(q)
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
