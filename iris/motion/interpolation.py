"""
Missing frame recovery and spline interpolation for occluded joint trajectories.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.interpolate import interp1d


class TrajectoryInterpolator:
    """Recovers occluded or missing joints using monotonic spline interpolation and rest-pose blending."""

    @staticmethod
    def impute_missing(
        positions: np.ndarray,
        confidences: np.ndarray,
        timestamps: np.ndarray,
        min_conf: float = 0.35,
        default_rest_pos: Optional[np.ndarray] = None,
        blend_frames: int = 8,
    ) -> np.ndarray:
        T, D = positions.shape
        if T < 2:
            return positions.copy()

        valid_mask = confidences >= min_conf
        valid_indices = np.where(valid_mask)[0]

        # If no valid points, fallback to default_rest_pos or original
        if len(valid_indices) == 0:
            if default_rest_pos is not None:
                return np.tile(default_rest_pos, (T, 1)).astype(np.float32)
            return positions.copy()

        # If already completely valid, no interpolation needed
        if len(valid_indices) == T:
            return positions.copy()

        times_list = []
        vals_list = []

        # Handle leading unobserved segment
        first_idx = valid_indices[0]
        if first_idx > 0:
            if default_rest_pos is not None:
                times_list.append(timestamps[0])
                vals_list.append(default_rest_pos)
                blend_start_idx = max(0, first_idx - blend_frames)
                if blend_start_idx > 0:
                    times_list.append(timestamps[blend_start_idx])
                    vals_list.append(default_rest_pos)
            else:
                times_list.append(timestamps[0])
                vals_list.append(positions[first_idx])

        # Include all observed valid points
        for idx in valid_indices:
            times_list.append(timestamps[idx])
            vals_list.append(positions[idx])

        # Handle trailing unobserved segment
        last_idx = valid_indices[-1]
        if last_idx < T - 1:
            if default_rest_pos is not None:
                blend_end_idx = min(T - 1, last_idx + blend_frames)
                if blend_end_idx < T - 1:
                    times_list.append(timestamps[blend_end_idx])
                    vals_list.append(default_rest_pos)
                times_list.append(timestamps[-1])
                vals_list.append(default_rest_pos)
            else:
                times_list.append(timestamps[-1])
                vals_list.append(positions[last_idx])

        t_arr = np.array(times_list, dtype=np.float64)
        v_arr = np.array(vals_list, dtype=np.float32)

        unique_t, unique_indices = np.unique(t_arr, return_index=True)
        unique_v = v_arr[unique_indices]

        if len(unique_t) < 2:
            return positions.copy()

        out_positions = np.zeros_like(positions)
        for d in range(D):
            interp_func = interp1d(
                unique_t,
                unique_v[:, d],
                kind="linear",
                bounds_error=False,
                fill_value=(unique_v[0, d], unique_v[-1, d]),
            )
            out_positions[:, d] = interp_func(timestamps)

        return out_positions
