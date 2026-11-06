"""
Temporal frame rate downsampling and timestamp calculation.
"""

from __future__ import annotations

from typing import List, Tuple


class FrameSampler:
    """Calculates frame sampling schedules to normalize arbitrary video framerates."""

    @staticmethod
    def calculate_sampling_indices(
        total_frames: int, source_fps: float, target_fps: float
    ) -> List[Tuple[int, float]]:
        """
        Returns a list of tuples: (frame_index, timestamp_seconds) for the sampled sequence.
        """
        if total_frames <= 0 or source_fps <= 0 or target_fps <= 0:
            return []

        # If source FPS is lower than or equal to target, keep every frame
        if source_fps <= target_fps:
            return [(i, i / source_fps) for i in range(total_frames)]

        step = source_fps / target_fps
        indices = []
        current = 0.0
        while int(round(current)) < total_frames:
            idx = int(round(current))
            timestamp = idx / source_fps
            indices.append((idx, timestamp))
            current += step

        return indices
