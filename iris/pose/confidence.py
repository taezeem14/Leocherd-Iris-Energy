"""
Confidence assessment, occlusion detection, and quality validation.
"""

from __future__ import annotations

from typing import Dict, List, Set
from iris.pose.base import PoseFrame


class ConfidenceFilter:
    """Evaluates skeletal confidence and identifies occluded or missing anatomical nodes."""

    def __init__(self, min_confidence: float = 0.35) -> None:
        self.min_confidence = min_confidence

    def evaluate_frame(self, frame: PoseFrame) -> Dict[str, any]:
        missing: List[str] = []
        low_confidence: List[str] = []
        high_confidence: List[str] = []

        # Biomechanical sanity filters for humanoid poses
        if frame.morphology in ("human", "biped"):
            j = frame.joints
            # Feet must never be higher than knees
            for side in ["left", "right"]:
                foot_k = f"{side}_ankle"
                knee_k = f"{side}_knee"
                if foot_k in j and knee_k in j:
                    if j[foot_k].y >= j[knee_k].y - 0.05:
                        j[foot_k].confidence = 0.0
            # Wrists cannot be high above head in low-confidence distant detections
            for side in ["left", "right"]:
                wrist_k = f"{side}_wrist"
                head_k = "nose"
                if wrist_k in j and head_k in j:
                    if j[wrist_k].y > j[head_k].y + 0.15 and j[wrist_k].confidence < 0.65:
                        j[wrist_k].confidence = 0.0

        for name, joint in frame.joints.items():
            if joint.confidence < self.min_confidence or joint.visibility < self.min_confidence:
                low_confidence.append(name)
            else:
                high_confidence.append(name)

        mean_conf = frame.mean_confidence()
        is_usable = len(high_confidence) >= 6 and mean_conf >= self.min_confidence

        return {
            "mean_confidence": round(mean_conf, 3),
            "high_confidence_count": len(high_confidence),
            "low_confidence_joints": low_confidence,
            "is_usable": is_usable,
        }

    def filter_sequence(self, sequence: List[PoseFrame]) -> List[PoseFrame]:
        """Flags frames with unusable tracking for subsequent temporal imputation."""
        out = []
        for frame in sequence:
            eval_res = self.evaluate_frame(frame)
            if not eval_res["is_usable"]:
                frame.is_interpolated = True
            out.append(frame)
        return out
