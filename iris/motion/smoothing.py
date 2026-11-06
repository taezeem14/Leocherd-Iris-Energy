"""
Adaptive filtering and trajectory smoothing for human and quadruped motion.
"""

from __future__ import annotations

import math
from typing import Optional
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
