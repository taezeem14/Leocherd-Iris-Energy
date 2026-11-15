"""
Quaternion and 3D spatial transformation mathematics for rigging and skeletal articulation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
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
    def clamp_rotation_angle(q: np.ndarray, max_angle_rad: float) -> np.ndarray:
        """
        Clamps a unit quaternion q = [w, x, y, z] so its angular displacement from
        identity does not exceed max_angle_rad.
        Preserves the rotation axis perfectly while limiting extreme deformation.
        """
        q_norm = QuaternionMath.normalize_quat(q)
        w = float(q_norm[0])
        sign = 1.0 if w >= 0.0 else -1.0
        w *= sign
        v = q_norm[1:4] * sign

        angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))
        if angle <= max_angle_rad or angle < 1e-5:
            return q_norm

        half_clamped = max_angle_rad * 0.5
        w_clamped = math.cos(half_clamped)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-6:
            return QuaternionMath.identity()
        v_clamped = (v / v_norm) * math.sin(half_clamped)
        return np.array([w_clamped, v_clamped[0], v_clamped[1], v_clamped[2]], dtype=np.float32)

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


def _unnormalized_quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product without normalization."""
    w1, x1, y1, z1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    w2, x2, y2, z2 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=np.float32)


class DualQuaternion:
    """
    Unit dual quaternion representation (q0, q_eps) for SE(3) rigid transformations.
    q0: real quaternion representing rotation [w, x, y, z]
    q_eps: dual quaternion representing translation [w, x, y, z]
    Conforms to standard Dual Quaternion Skinning (Kavan et al. 2007, 2008).
    """

    def __init__(self, q0: np.ndarray, q_eps: Optional[np.ndarray] = None):
        self.q0 = np.asarray(q0, dtype=np.float32)
        if q_eps is None:
            self.q_eps = np.zeros(4, dtype=np.float32)
        else:
            self.q_eps = np.asarray(q_eps, dtype=np.float32)
        if self.q0.shape != (4,):
            raise ValueError(f"q0 must have shape (4,), got {self.q0.shape}")
        if self.q_eps.shape != (4,):
            raise ValueError(f"q_eps must have shape (4,), got {self.q_eps.shape}")

    def __repr__(self) -> str:
        return f"DualQuaternion(q0={self.q0.tolist()}, q_eps={self.q_eps.tolist()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DualQuaternion):
            return False
        return np.allclose(self.q0, other.q0, atol=1e-6) and np.allclose(self.q_eps, other.q_eps, atol=1e-6)

    @classmethod
    def identity(cls) -> DualQuaternion:
        """Returns identity unit dual quaternion (zero rotation, zero translation)."""
        return cls(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    @classmethod
    def from_rotation_translation(cls, q: np.ndarray, t: np.ndarray) -> DualQuaternion:
        """
        Constructs a unit dual quaternion from unit rotation quaternion q = [w, x, y, z]
        and translation vector t = [tx, ty, tz].
        q0 = normalize(q)
        q_eps = 0.5 * (t_quat * q0)
        """
        q0 = QuaternionMath.normalize_quat(np.asarray(q, dtype=np.float32))
        t_vec = np.asarray(t, dtype=np.float32)
        if t_vec.shape != (3,):
            raise ValueError(f"Translation vector must have shape (3,), got {t_vec.shape}")

        tx, ty, tz = float(t_vec[0]), float(t_vec[1]), float(t_vec[2])
        w0, x0, y0, z0 = float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])

        w_eps = -0.5 * (tx * x0 + ty * y0 + tz * z0)
        x_eps = 0.5 * (tx * w0 + ty * z0 - tz * y0)
        y_eps = 0.5 * (-tx * z0 + ty * w0 + tz * x0)
        z_eps = 0.5 * (tx * y0 - ty * x0 + tz * w0)

        q_eps = np.array([w_eps, x_eps, y_eps, z_eps], dtype=np.float32)
        return cls(q0, q_eps)

    def to_rotation_translation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts unit rotation quaternion q and translation vector t.
        Returns:
            q: [w, x, y, z] normalized rotation quaternion
            t: [tx, ty, tz] 3D translation vector
        """
        q0_norm = QuaternionMath.normalize_quat(self.q0)
        w0, x0, y0, z0 = float(q0_norm[0]), float(q0_norm[1]), float(q0_norm[2]), float(q0_norm[3])
        we, xe, ye, ze = float(self.q_eps[0]), float(self.q_eps[1]), float(self.q_eps[2]), float(self.q_eps[3])

        # t = 2 * (q_eps * conjugate(q0_norm))_vec
        tx = 2.0 * (-we * x0 + xe * w0 - ye * z0 + ze * y0)
        ty = 2.0 * (-we * y0 + xe * z0 + ye * w0 - ze * x0)
        tz = 2.0 * (-we * z0 - xe * y0 + ye * x0 + ze * w0)

        return q0_norm, np.array([tx, ty, tz], dtype=np.float32)

    @classmethod
    def from_matrix_affine(cls, mat: np.ndarray) -> DualQuaternion:
        """Constructs dual quaternion from 4x4 or 3x4 affine transformation matrix."""
        mat_arr = np.asarray(mat, dtype=np.float32)
        R = mat_arr[:3, :3]
        t = mat_arr[:3, 3] if mat_arr.shape[1] >= 4 else np.zeros(3, dtype=np.float32)
        q = QuaternionMath.matrix_to_quat(R)
        return cls.from_rotation_translation(q, t)

    def to_matrix(self) -> np.ndarray:
        """Converts unit dual quaternion to 4x4 affine transformation matrix."""
        q, t = self.to_rotation_translation()
        R = QuaternionMath.quat_to_matrix(q)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R
        mat[:3, 3] = t
        return mat

    def __add__(self, other: DualQuaternion) -> DualQuaternion:
        if not isinstance(other, DualQuaternion):
            return NotImplemented
        return DualQuaternion(self.q0 + other.q0, self.q_eps + other.q_eps)

    def __sub__(self, other: DualQuaternion) -> DualQuaternion:
        if not isinstance(other, DualQuaternion):
            return NotImplemented
        return DualQuaternion(self.q0 - other.q0, self.q_eps - other.q_eps)

    def __mul__(self, other: Union[DualQuaternion, float, int, np.floating, np.integer]) -> DualQuaternion:
        if isinstance(other, (int, float, np.floating, np.integer)):
            s = float(other)
            return DualQuaternion(self.q0 * s, self.q_eps * s)
        if isinstance(other, DualQuaternion):
            # p * q = (p0 * q0) + eps * (p0 * q_eps + p_eps * q0)
            p0 = self.q0
            pe = self.q_eps
            q0 = other.q0
            qe = other.q_eps

            r0 = _unnormalized_quat_mult(p0, q0)
            re = _unnormalized_quat_mult(p0, qe) + _unnormalized_quat_mult(pe, q0)
            return DualQuaternion(r0, re)
        return NotImplemented

    def __rmul__(self, other: Union[float, int, np.floating, np.integer]) -> DualQuaternion:
        return self.__mul__(other)

    def __neg__(self) -> DualQuaternion:
        return DualQuaternion(-self.q0, -self.q_eps)

    def norm(self) -> float:
        """Returns magnitude norm of real quaternion part ||q0||."""
        return float(np.linalg.norm(self.q0))

    def normalize(self) -> DualQuaternion:
        """Returns normalized unit dual quaternion where ||q0|| == 1."""
        n = self.norm()
        if n < 1e-8:
            return DualQuaternion.identity()
        inv_n = 1.0 / n
        return DualQuaternion(self.q0 * inv_n, self.q_eps * inv_n)

    def conjugate(self) -> DualQuaternion:
        """
        Quaternion conjugate of real and dual parts: q0* + eps * q_eps*.
        Satisfies (p * q)* = q* * p*.
        """
        return DualQuaternion(
            QuaternionMath.conjugate(self.q0),
            QuaternionMath.conjugate(self.q_eps),
        )

    def inverse(self) -> DualQuaternion:
        """
        Computes the multiplicative inverse of the dual quaternion.
        For unit dual quaternions, q^{-1} == conjugate(q).
        """
        norm_sq = float(np.dot(self.q0, self.q0))
        if norm_sq < 1e-12:
            return DualQuaternion.identity()
        inv_q0 = QuaternionMath.conjugate(self.q0) / norm_sq
        inv_q_eps = QuaternionMath.conjugate(self.q_eps) / norm_sq - (
            (2.0 * float(np.dot(self.q0, self.q_eps)) / (norm_sq * norm_sq))
            * QuaternionMath.conjugate(self.q0)
        )
        return DualQuaternion(inv_q0, inv_q_eps)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transforms 3D point or (N, 3) points by this dual quaternion."""
        return DualQuaternionMath.transform_point(self, point)

    def transform_normal(self, normal: np.ndarray) -> np.ndarray:
        """Transforms 3D normal or (N, 3) normals by this dual quaternion."""
        return DualQuaternionMath.transform_normal(self, normal)


class DualQuaternionMath:
    """Operations on Dual Quaternions for skinning, transformation, and blending."""

    @staticmethod
    def from_rotation_translation(q: np.ndarray, t: np.ndarray) -> DualQuaternion:
        return DualQuaternion.from_rotation_translation(q, t)

    @staticmethod
    def to_rotation_translation(dq: DualQuaternion) -> Tuple[np.ndarray, np.ndarray]:
        return dq.to_rotation_translation()

    @staticmethod
    def from_matrix_affine(mat: np.ndarray) -> DualQuaternion:
        return DualQuaternion.from_matrix_affine(mat)

    @staticmethod
    def to_matrix_affine(dq: DualQuaternion) -> np.ndarray:
        return dq.to_matrix()

    @staticmethod
    def dlb(
        dual_quaternions: Union[List[DualQuaternion], np.ndarray],
        weights: Union[List[float], np.ndarray],
    ) -> DualQuaternion:
        """
        Dual Quaternion Linear Blending (DLB) with anti-flip dot product pivot check.
        Ensures shortest arc interpolation: if dot(q0_pivot, q0_k) < 0, flips -dq_k.
        Returns normalized blended unit dual quaternion.
        """
        if len(dual_quaternions) == 0:
            return DualQuaternion.identity()

        w_arr = np.asarray(weights, dtype=np.float32)
        if len(dual_quaternions) != len(w_arr):
            raise ValueError(f"Mismatch between DQs ({len(dual_quaternions)}) and weights ({len(w_arr)})")

        # Find pivot: influence with highest weight
        pivot_idx = int(np.argmax(w_arr))
        pivot_dq = dual_quaternions[pivot_idx]
        q0_piv = pivot_dq.q0

        b0 = np.zeros(4, dtype=np.float32)
        b_eps = np.zeros(4, dtype=np.float32)

        for k, (dq, w) in enumerate(zip(dual_quaternions, w_arr)):
            if w <= 0.0:
                continue
            # Anti-flip dot product pivot check
            dot = float(np.dot(q0_piv, dq.q0))
            sign = -1.0 if dot < 0.0 else 1.0
            eff_w = float(w * sign)
            b0 += eff_w * dq.q0
            b_eps += eff_w * dq.q_eps

        norm = float(np.linalg.norm(b0))
        if norm < 1e-8:
            return DualQuaternion.identity()

        inv_norm = 1.0 / norm
        return DualQuaternion(b0 * inv_norm, b_eps * inv_norm)

    @staticmethod
    def transform_point(dq: DualQuaternion, point: np.ndarray) -> np.ndarray:
        """
        Transforms 3D point (or array of points of shape (N, 3)) by dual quaternion.
        v' = q0 * v * q0* + 2 * (q_eps * q0*)_vec
        """
        p = np.asarray(point, dtype=np.float32)
        is_single = (p.ndim == 1)
        if is_single:
            p = p.reshape(1, 3)

        q_rot, t = dq.to_rotation_translation()
        w = float(q_rot[0])
        qv = q_rot[1:4]

        # Rodrigues rotation vectorized
        cross1 = np.cross(qv, p)
        v_rot = p + 2.0 * (w * cross1 + np.cross(qv, cross1))
        v_trans = v_rot + t

        return v_trans[0] if is_single else v_trans

    @staticmethod
    def transform_normal(dq: DualQuaternion, normal: np.ndarray) -> np.ndarray:
        """
        Transforms 3D normal vector (or array of normals of shape (N, 3)) by dual quaternion.
        Applies only the rotational component and normalizes.
        """
        n = np.asarray(normal, dtype=np.float32)
        is_single = (n.ndim == 1)
        if is_single:
            n = n.reshape(1, 3)

        q_rot, _ = dq.to_rotation_translation()
        w = float(q_rot[0])
        qv = q_rot[1:4]

        cross1 = np.cross(qv, n)
        n_rot = n + 2.0 * (w * cross1 + np.cross(qv, cross1))
        norms = np.linalg.norm(n_rot, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        n_norm = (n_rot / norms).astype(np.float32)

        return n_norm[0] if is_single else n_norm
