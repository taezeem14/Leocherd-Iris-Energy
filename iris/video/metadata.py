"""
Video metadata schema and probe definitions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Strongly typed representation of video container and stream attributes."""

    path: str
    filename: str
    file_size_bytes: int
    duration_sec: float
    frame_count: int
    fps: float
    width: int
    height: int
    codec: str = "unknown"
    aspect_ratio: float = Field(default=1.0)
    warnings: List[str] = Field(default_factory=list)

    @classmethod
    def from_opencv(cls, video_path: str, cap: Any) -> "VideoMetadata":
        import cv2

        path_obj = Path(video_path)
        file_size = path_obj.stat().st_size if path_obj.exists() else 0

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Avoid zero or invalid FPS
        if fps <= 0.001 or fps > 240.0:
            fps = 30.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        # Decode fourcc
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        aspect = float(width / height) if height > 0 else 1.0

        warnings = []
        if frame_count <= 0:
            warnings.append("Could not determine total frame count reliably from container header.")
        if duration > 300.0:
            warnings.append(f"Video duration ({duration:.1f}s) is exceptionally long for real-time 3D rigging.")

        return cls(
            path=str(path_obj.resolve()),
            filename=path_obj.name,
            file_size_bytes=file_size,
            duration_sec=round(duration, 3),
            frame_count=frame_count,
            fps=round(fps, 3),
            width=width,
            height=height,
            codec=codec if codec else "unknown",
            aspect_ratio=round(aspect, 4),
            warnings=warnings,
        )

    def summary(self) -> str:
        size_mb = self.file_size_bytes / (1024 * 1024)
        return (
            f"Video: {self.filename} | {self.width}x{self.height} @ {self.fps:.2f}fps | "
            f"{self.frame_count} frames | {self.duration_sec:.2f}s | {size_mb:.2f} MB | Codec: {self.codec}"
        )
