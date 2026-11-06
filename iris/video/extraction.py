"""
Memory-bounded streaming frame extraction using OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from iris.core.exceptions import VideoProcessingError
from iris.core.logging import get_logger

logger = get_logger("iris.video.extraction")


@dataclass
class ExtractedFrame:
    """A single processed video frame with metadata and RGB pixels."""

    index: int
    timestamp_sec: float
    rgb_image: np.ndarray  # Shape: (H, W, 3), dtype: uint8, RGB order
    orig_width: int
    orig_height: int
    scaled_width: int
    scaled_height: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.rgb_image.shape


class FrameExtractor:
    """Extracts, resizes, and color-converts video frames in a memory-bounded stream."""

    def __init__(self, max_dimension: int = 1280) -> None:
        self.max_dimension = max_dimension

    def _calculate_scaled_dims(self, orig_w: int, orig_h: int) -> Tuple[int, int]:
        if orig_w <= 0 or orig_h <= 0:
            return 640, 480
        max_dim = max(orig_w, orig_h)
        if max_dim <= self.max_dimension:
            return orig_w, orig_h
        scale = self.max_dimension / max_dim
        scaled_w = int(round(orig_w * scale))
        scaled_h = int(round(orig_h * scale))
        return scaled_w, scaled_h

    def stream_frames(
        self,
        video_path: str,
        sample_schedule: Optional[List[Tuple[int, float]]] = None,
        max_frames: Optional[int] = None,
    ) -> Generator[ExtractedFrame, None, None]:
        """
        Yields frames sequentially. If sample_schedule is provided, seeks/steps to designated indices.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file for decoding: {video_path}")

        try:
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scaled_w, scaled_h = self._calculate_scaled_dims(orig_w, orig_h)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

            schedule_set = set(idx for idx, _ in sample_schedule) if sample_schedule else None
            schedule_map = dict(sample_schedule) if sample_schedule else {}

            current_idx = 0
            yielded_count = 0

            while True:
                if max_frames is not None and yielded_count >= max_frames:
                    break

                ret, bgr_frame = cap.read()
                if not ret or bgr_frame is None:
                    break

                should_process = True
                if schedule_set is not None:
                    should_process = current_idx in schedule_set

                if should_process:
                    # Resize if needed
                    if (orig_w, orig_h) != (scaled_w, scaled_h):
                        resized = cv2.resize(bgr_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        resized = bgr_frame

                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    timestamp = schedule_map.get(current_idx, current_idx / fps)

                    yield ExtractedFrame(
                        index=current_idx,
                        timestamp_sec=round(timestamp, 4),
                        rgb_image=rgb_frame,
                        orig_width=orig_w,
                        orig_height=orig_h,
                        scaled_width=scaled_w,
                        scaled_height=scaled_h,
                    )
                    yielded_count += 1

                current_idx += 1

        finally:
            cap.release()

    def extract_all(
        self,
        video_path: str,
        sample_schedule: Optional[List[Tuple[int, float]]] = None,
        max_frames: Optional[int] = None,
    ) -> List[ExtractedFrame]:
        """Convenience method to collect extracted frames in memory."""
        return list(self.stream_frames(video_path, sample_schedule, max_frames))
