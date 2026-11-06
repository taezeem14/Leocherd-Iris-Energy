"""
Subject detection and bounding box representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = "subject"

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0

    def iou(self, other: "BoundingBox") -> float:
        ixmin = max(self.xmin, other.xmin)
        iymin = max(self.ymin, other.ymin)
        ixmax = min(self.xmax, other.xmax)
        iymax = min(self.ymax, other.ymax)

        iw = max(0.0, ixmax - ixmin)
        ih = max(0.0, iymax - iymin)
        intersection = iw * ih

        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def to_square(self, margin_ratio: float = 0.2, img_w: int = 1, img_h: int = 1) -> "BoundingBox":
        """Expands box to square with margin for optimal neural pose estimation."""
        cx, cy = self.center
        side = max(self.width, self.height) * (1.0 + margin_ratio)
        half = side / 2.0
        return BoundingBox(
            xmin=max(0.0, cx - half),
            ymin=max(0.0, cy - half),
            xmax=min(float(img_w), cx + half),
            ymax=min(float(img_h), cy + half),
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )


class SubjectDetector:
    """Robust subject detector using motion saliency and contour analysis as CPU baseline."""

    def __init__(self, min_area_ratio: float = 0.02) -> None:
        self.min_area_ratio = min_area_ratio
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

    def detect(self, rgb_frame: np.ndarray) -> List[BoundingBox]:
        h, w = rgb_frame.shape[:2]
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        fg_mask = self.bg_subtractor.apply(gray)

        # Morphological filtering to close holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (w * h) * self.min_area_ratio

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                conf = min(1.0, float(area / (w * h * 0.3)))
                boxes.append(
                    BoundingBox(
                        xmin=float(x),
                        ymin=float(y),
                        xmax=float(x + bw),
                        ymax=float(y + bh),
                        confidence=conf,
                    )
                )

        # Fallback to full frame centered crop if no significant motion detected (e.g. static start)
        if not boxes:
            pad_w = w * 0.1
            pad_h = h * 0.05
            boxes.append(
                BoundingBox(
                    xmin=pad_w,
                    ymin=pad_h,
                    xmax=w - pad_w,
                    ymax=h - pad_h,
                    confidence=0.5,
                )
            )

        # Sort by area descending (primary subject first)
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes
