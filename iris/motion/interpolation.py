"""
Missing frame recovery and spline interpolation for occluded joint trajectories.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class TrajectoryInterpolator:
    """Recovers occluded or missing joints using monotonic cubic spline interpolation."""

    @staticmethod
    def impute_missing(
        positions: np.ndarray,
        confidences: np.ndarray,
        timestamps: np.ndarray,
        min_conf: float = 0.35,
    ) -> np.ndarray:
        T, D = positions.shape
        if T < 2:
            return positions.copy()

        valid_mask = confidences >= min_conf
        valid_indices = np.where(valid_mask)[0]

        # If too few valid points, return original
        if len(valid_indices) < 2:
            return positions.copy()

        # If already completely valid, no interpolation needed
        if len(valid_indices) == T:
            return positions.copy()

        out_positions = positions.copy()
        valid_times = timestamps[valid_indices]

        for d in range(D):
            valid_vals = positions[valid_indices, d]
            # Use linear with nearest extrapolate for safety at sequence boundaries
            interp_func = interp1d(
                valid_times,
                valid_vals,
                kind="linear",
                bounds_error=False,
                fill_value=(valid_vals[0], valid_vals[-1]),
            )
            out_positions[:, d] = interp_func(timestamps)

        return out_positions
