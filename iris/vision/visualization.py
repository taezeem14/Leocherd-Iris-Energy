"""
Computer vision visualization tools for skeletons, bounding boxes, and overlays.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class Visualizer:
    """Overlays bounding boxes, joint keypoints, and bone linkages on video frames."""

    DEFAULT_SKELETON_CONNECTIONS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
    ]

    @staticmethod
    def draw_bbox(
        img: np.ndarray,
        bbox: Any,
        label: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        out = img.copy()
        x1, y1 = int(bbox.xmin), int(bbox.ymin)
        x2, y2 = int(bbox.xmax), int(bbox.ymax)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(out, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out

    @staticmethod
    def draw_pose_2d(
        img: np.ndarray,
        joints: Dict[str, Any],
        connections: Optional[List[Tuple[str, str]]] = None,
        joint_color: Tuple[int, int, int] = (0, 255, 255),
        bone_color: Tuple[int, int, int] = (255, 0, 0),
        min_conf: float = 0.3,
    ) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        conns = connections or Visualizer.DEFAULT_SKELETON_CONNECTIONS

        # Draw bones
        for u, v in conns:
            if u in joints and v in joints:
                ju = joints[u]
                jv = joints[v]
                if ju.confidence >= min_conf and jv.confidence >= min_conf:
                    # Check if coordinates are normalized [0,1] or pixel coordinates
                    ux = int(ju.x * w) if ju.x <= 1.05 and ju.y <= 1.05 else int(ju.x)
                    uy = int(ju.y * h) if ju.x <= 1.05 and ju.y <= 1.05 else int(ju.y)
                    vx = int(jv.x * w) if jv.x <= 1.05 and jv.y <= 1.05 else int(jv.x)
                    vy = int(jv.y * h) if jv.x <= 1.05 and jv.y <= 1.05 else int(jv.y)
                    cv2.line(out, (ux, uy), (vx, vy), bone_color, 2)

        # Draw joints
        for name, j in joints.items():
            if j.confidence >= min_conf:
                jx = int(j.x * w) if j.x <= 1.05 and j.y <= 1.05 else int(j.x)
                jy = int(j.y * h) if j.x <= 1.05 and j.y <= 1.05 else int(j.y)
                cv2.circle(out, (jx, jy), 4, joint_color, -1)

        return out
