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
        """
        Rotates 3D vector v by unit quaternion q = [w, x, y, z] using Rodrigues vector formula:
        v' = v + 2 * q_v x (q_v x v + w * v).
        Guarantees exact length preservation of v without quaternion normalization distortion.
        """
        w = float(q[0])
        qv = q[1:4].astype(np.float32)
        v_f = v.astype(np.float32)
        t = 2.0 * np.cross(qv, v_f)
        v_rot = v_f + w * t + np.cross(qv, t)
        return v_rot.astype(np.float32)

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

    @staticmethod
    def matrix_to_quat(R: np.ndarray) -> np.ndarray:
        """
        Converts an orthonormal 3x3 rotation matrix to unit quaternion [w, x, y, z]
        using robust Shepperd's branch-selection algorithm.
        """
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = math.sqrt(max(1e-8, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(max(1e-8, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1e-8, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return QuaternionMath.normalize_quat(np.array([w, x, y, z], dtype=np.float32))

    @staticmethod
    def build_orthonormal_basis(primary_vec: np.ndarray, reference_sec_vec: np.ndarray) -> np.ndarray:
        """
        Builds an orthonormal 3x3 rotation basis [X, Y, Z] where Y is aligned with primary_vec,
        and Z is aligned as closely as possible with reference_sec_vec.
        """
        y = QuaternionMath.normalize_vector(primary_vec)
        # Project secondary vector perpendicular to y
        ref = reference_sec_vec.astype(np.float32)
        dot_ref_y = float(np.dot(ref, y))
        z_proj = ref - dot_ref_y * y
        z = QuaternionMath.normalize_vector(z_proj)
        if np.linalg.norm(z) < 1e-4:
            # Fallback if primary and reference are collinear
            alt_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(y[2]) < 0.85 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            z_proj = alt_ref - float(np.dot(alt_ref, y)) * y
            z = QuaternionMath.normalize_vector(z_proj)

        x = QuaternionMath.normalize_vector(np.cross(y, z))
        z = QuaternionMath.normalize_vector(np.cross(x, y))  # Re-orthogonalize
        return np.column_stack([x, y, z]).astype(np.float32)

    @staticmethod
    def rotation_between_bases(rest_basis: np.ndarray, cur_basis: np.ndarray) -> np.ndarray:
        """
        Calculates the rotation quaternion rotating rest_basis to cur_basis: R = cur_basis @ rest_basis.T.
        Completely eliminates twist/roll ambiguity.
        """
        R = cur_basis @ rest_basis.T
        return QuaternionMath.matrix_to_quat(R)
