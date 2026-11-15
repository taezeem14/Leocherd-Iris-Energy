"""
Adaptive filtering and trajectory smoothing for human and quadruped motion.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter


class LowPassFilter:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.hat_x_prev: Optional[np.ndarray] = None

    def filter(self, x: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.hat_x_prev is None:
            self.hat_x_prev = x.copy()
            return x
        hat_x = self.alpha * x + (1.0 - self.alpha) * self.hat_x_prev
        self.hat_x_prev = hat_x
        return hat_x

    def reset(self) -> None:
        self.hat_x_prev = None


class OneEuroFilter:
    """
    One Euro adaptive filter for continuous 3D keypoint tracking.
    Eliminates jitter when stationary, minimizes phase lag during rapid motion.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, timestamp: float) -> np.ndarray:
        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter(x)

        dt = max(1e-4, timestamp - self.last_time)
        self.last_time = timestamp

        # Estimate derivative dx
        prev_x = self.x_filter.hat_x_prev if self.x_filter.hat_x_prev is not None else x
        dx = (x - prev_x) / dt
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, dt))

        # Dynamic cutoff frequency based on speed
        speed = float(np.linalg.norm(edx))
        cutoff = self.min_cutoff + self.beta * speed
        return self.x_filter.filter(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class TemporalSmoother:
    """High-level smoother applying adaptive One Euro or Savitzky-Golay filters across timeseries."""

    @staticmethod
    def smooth_trajectory(
        positions: np.ndarray,
        timestamps: np.ndarray,
        method: str = "one_euro",
    ) -> np.ndarray:
        T, D = positions.shape
        if T < 4:
            return positions.copy()

        if method == "one_euro":
            dt_avg = (timestamps[-1] - timestamps[0]) / max(1, T - 1)
            fps = 1.0 / max(1e-4, dt_avg)
            flt = OneEuroFilter(freq=fps, min_cutoff=1.2, beta=0.005)
            smoothed = np.zeros_like(positions)
            for t in range(T):
                smoothed[t] = flt.filter(positions[t], timestamps[t])
            return smoothed

        elif method == "savgol":
            window = min(7, T if T % 2 == 1 else T - 1)
            if window >= 3:
                smoothed = np.zeros_like(positions)
                for d in range(D):
                    smoothed[:, d] = savgol_filter(positions[:, d], window_length=window, polyorder=2)
                return smoothed

        return positions.copy()


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical Linear Interpolation (Slerp) between two unit quaternions q1 and q2.
    Conforms to standard [w, x, y, z] representation.

    Features:
    - Antipodal sign continuity pivot: if dot(q1, q2) < 0, pivots q2 -> -q2 to take shortest arc.
    - Small-angle Taylor expansion fallback when angle is near zero (dot > 0.9995) to prevent
      division by zero in sin(theta).
    - Exact unit norm preservation.
    """
    t_f = float(np.clip(t, 0.0, 1.0))
    n1 = float(np.linalg.norm(q1))
    n2 = float(np.linalg.norm(q2))
    u1 = (q1 / n1).astype(np.float32) if n1 > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    u2 = (q2 / n2).astype(np.float32) if n2 > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    dot = float(np.dot(u1, u2))
    if dot < 0.0:
        u2 = -u2
        dot = -dot

    dot = min(1.0, max(-1.0, dot))

    # Small-angle linear approximation
    if dot > 0.9995:
        result = (1.0 - t_f) * u1 + t_f * u2
        norm_res = float(np.linalg.norm(result))
        return (result / norm_res).astype(np.float32) if norm_res > 1e-8 else u1

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t_f

    s1 = math.sin(theta_0 - theta) / sin_theta_0
    s2 = math.sin(theta) / sin_theta_0

    result = s1 * u1 + s2 * u2
    norm_res = float(np.linalg.norm(result))
    return (result / norm_res).astype(np.float32) if norm_res > 1e-8 else u1


class QuaternionSmoother:
    """
    Spherical manifold S^3 quaternion temporal smoother.
    Eliminates high-frequency angular jitter on rotational animation tracks using
    bidirectional zero-phase recursive Slerp filtering.
    """

    def __init__(self, cutoff_hz: float = 2.0) -> None:
        self.cutoff_hz = float(cutoff_hz)

    def smooth(
        self,
        rotations: np.ndarray,
        timestamps: np.ndarray,
        cutoff_hz: Optional[float] = None,
    ) -> np.ndarray:
        """
        Applies zero-phase forward-backward recursive Slerp filtering to rotational track.

        Args:
            rotations: Array of shape (T, 4) with unit quaternions [w, x, y, z].
            timestamps: Array of shape (T,) monotonically increasing in seconds.
            cutoff_hz: Optional cutoff frequency in Hz. Defaults to self.cutoff_hz.

        Returns:
            Smoothed rotations array of shape (T, 4), strictly unit normalized.
        """
        T = rotations.shape[0]
        if T < 3:
            out = np.zeros_like(rotations, dtype=np.float32)
            for t in range(T):
                n = float(np.linalg.norm(rotations[t]))
                out[t] = rotations[t] / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return out

        fc = float(cutoff_hz if cutoff_hz is not None else self.cutoff_hz)

        # 1. Enforce antipodal sign continuity and unit normalization
        rots = np.zeros_like(rotations, dtype=np.float32)
        for i in range(T):
            n = float(np.linalg.norm(rotations[i]))
            rots[i] = rotations[i] / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if i > 0 and float(np.dot(rots[i], rots[i - 1])) < 0.0:
                rots[i] = -rots[i]

        # 2. Forward recursive Slerp pass
        q_f = np.zeros_like(rots)
        q_f[0] = rots[0].copy()
        for t in range(1, T):
            dt = max(1e-4, float(timestamps[t] - timestamps[t - 1]))
            tau = 1.0 / (2.0 * math.pi * fc)
            alpha = dt / (dt + tau)
            q_f[t] = quaternion_slerp(q_f[t - 1], rots[t], alpha)

        # 3. Backward recursive Slerp pass
        q_b = np.zeros_like(rots)
        q_b[-1] = rots[-1].copy()
        for t in range(T - 2, -1, -1):
            dt = max(1e-4, float(timestamps[t + 1] - timestamps[t]))
            tau = 1.0 / (2.0 * math.pi * fc)
            alpha = dt / (dt + tau)
            q_b[t] = quaternion_slerp(q_b[t + 1], rots[t], alpha)

        # 4. Zero-phase bilateral merge
        q_smooth = np.zeros_like(rots)
        for t in range(T):
            q_smooth[t] = quaternion_slerp(q_f[t], q_b[t], 0.5)

        return q_smooth


class FootLockingStabilizer:
    """
    Stabilizes ground contact phases to eliminate foot-skating and contact jitter.

    Uses height and velocity thresholding to identify stance phases, clamps end-effector
    to a stationary ground anchor during stance, and applies cubic smoothstep transitions.
    """

    def __init__(
        self,
        height_thresh: float = 0.08,
        vel_thresh: float = 0.20,
        ground_height: float = 0.0,
        blend_frames: int = 3,
    ) -> None:
        self.height_thresh = height_thresh
        self.vel_thresh = vel_thresh
        self.ground_height = ground_height
        self.blend_frames = blend_frames

    def detect_contacts(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray,
        height_thresh: Optional[float] = None,
        vel_thresh: Optional[float] = None,
        ground_height: Optional[float] = None,
    ) -> np.ndarray:
        """
        Detects stance contact frames where foot is near ground and horizontal speed is low.

        Args:
            positions: End-effector positions of shape (T, 3).
            timestamps: Array of shape (T,) in seconds.
            height_thresh: Maximum height above ground for contact.
            vel_thresh: Maximum horizontal velocity (m/s) for contact.
            ground_height: Elevation of ground plane (Y-coordinate).

        Returns:
            Boolean array of shape (T,) where True indicates stance contact.
        """
        T = len(positions)
        if T < 2:
            return np.ones(T, dtype=bool)

        h_thresh = height_thresh if height_thresh is not None else self.height_thresh
        v_thresh = vel_thresh if vel_thresh is not None else self.vel_thresh
        g_height = ground_height if ground_height is not None else self.ground_height

        # Calculate horizontal velocities (in XZ plane) using minimum of forward/backward rates
        # to ensure clean detection at touchdown and liftoff boundaries
        vel_xz = np.zeros(T, dtype=np.float32)
        for t in range(T):
            if t == 0:
                dt = max(1e-4, float(timestamps[1] - timestamps[0]))
                disp = positions[1, [0, 2]] - positions[0, [0, 2]]
                vel_xz[t] = float(np.linalg.norm(disp)) / dt
            elif t == T - 1:
                dt = max(1e-4, float(timestamps[-1] - timestamps[-2]))
                disp = positions[-1, [0, 2]] - positions[-2, [0, 2]]
                vel_xz[t] = float(np.linalg.norm(disp)) / dt
            else:
                dt_fwd = max(1e-4, float(timestamps[t + 1] - timestamps[t]))
                disp_fwd = positions[t + 1, [0, 2]] - positions[t, [0, 2]]
                vel_fwd = float(np.linalg.norm(disp_fwd)) / dt_fwd

                dt_bwd = max(1e-4, float(timestamps[t] - timestamps[t - 1]))
                disp_bwd = positions[t, [0, 2]] - positions[t - 1, [0, 2]]
                vel_bwd = float(np.linalg.norm(disp_bwd)) / dt_bwd

                vel_xz[t] = min(vel_fwd, vel_bwd)

        height_mask = np.abs(positions[:, 1] - g_height) <= h_thresh
        vel_mask = vel_xz <= v_thresh
        return height_mask & vel_mask

    def stabilize_trajectory(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray,
        contacts: Optional[np.ndarray] = None,
        height_thresh: Optional[float] = None,
        vel_thresh: Optional[float] = None,
        ground_height: Optional[float] = None,
        blend_frames: Optional[int] = None,
    ) -> np.ndarray:
        """
        Stabilizes trajectory by clamping stance frames to ground anchor positions
        and smoothly interpolating at phase boundaries.

        Args:
            positions: End-effector positions of shape (T, 3).
            timestamps: Monotonic timestamps of shape (T,).
            contacts: Optional precomputed boolean contact array.
            height_thresh, vel_thresh, ground_height, blend_frames: Override parameters.

        Returns:
            Stabilized trajectory of shape (T, 3).
        """
        T = len(positions)
        if T < 2:
            return positions.copy()

        g_height = ground_height if ground_height is not None else self.ground_height
        b_frames = blend_frames if blend_frames is not None else self.blend_frames

        if contacts is None:
            contacts = self.detect_contacts(
                positions,
                timestamps,
                height_thresh=height_thresh,
                vel_thresh=vel_thresh,
                ground_height=g_height,
            )

        stabilized = positions.copy().astype(np.float32)

        # Identify contiguous stance intervals
        intervals: List[Tuple[int, int]] = []
        in_stance = False
        start_idx = 0
        for t in range(T):
            if contacts[t] and not in_stance:
                in_stance = True
                start_idx = t
            elif not contacts[t] and in_stance:
                in_stance = False
                intervals.append((start_idx, t - 1))
        if in_stance:
            intervals.append((start_idx, T - 1))

        for start, end in intervals:
            # Anchor at the first contact frame horizontal coordinate and ground elevation
            anchor = np.array(
                [positions[start, 0], g_height, positions[start, 2]],
                dtype=np.float32,
            )

            # Clamp core stance frames
            stabilized[start : end + 1] = anchor

            # Cubic smoothstep blending on entrance
            if start > 0 and b_frames > 0:
                actual_blend_in = min(b_frames, start)
                for step in range(actual_blend_in):
                    idx = start - actual_blend_in + step
                    tau = float(step + 1) / float(actual_blend_in + 1)
                    w = 3.0 * tau * tau - 2.0 * tau * tau * tau
                    stabilized[idx] = (1.0 - w) * positions[idx] + w * anchor

            # Cubic smoothstep blending on exit
            if end < T - 1 and b_frames > 0:
                actual_blend_out = min(b_frames, T - 1 - end)
                for step in range(actual_blend_out):
                    idx = end + 1 + step
                    tau = 1.0 - (float(step + 1) / float(actual_blend_out + 1))
                    w = 3.0 * tau * tau - 2.0 * tau * tau * tau
                    stabilized[idx] = w * anchor + (1.0 - w) * positions[idx]

        return stabilized
