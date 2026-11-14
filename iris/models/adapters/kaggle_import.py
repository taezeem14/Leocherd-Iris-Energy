"""
Adapter for importing high-fidelity GPU pose estimation results exported from Kaggle notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from iris.core.exceptions import PoseEstimationError
from iris.core.logging import get_logger
from iris.pose.base import BasePoseEstimator, Joint3D, PoseFrame

logger = get_logger("iris.models.adapters.kaggle")


class KagglePoseImportAdapter(BasePoseEstimator):
    """
    Imports pre-computed, high-fidelity GPU 3D pose landmarks and trajectories
    generated in Kaggle GPU notebooks.
    """

    def __init__(self, json_path: str | Path, name: str = "KaggleGPUImportAdapter") -> None:
        super().__init__(name=name)
        self.json_path = Path(json_path)
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            raise PoseEstimationError(f"Kaggle pose export file not found: {self.json_path}")
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info(
                f"Successfully loaded Kaggle GPU pose data from {self.json_path} "
                f"({len(self.data.get('frames', []))} frames, morphology: {self.data.get('morphology', 'human')})"
            )
        except Exception as e:
            raise PoseEstimationError(f"Failed to parse Kaggle pose export: {e}") from e

    def estimate_sequence(self, frames: Optional[List[Any]] = None) -> List[PoseFrame]:
        kaggle_frames = self.data.get("frames", [])
        if not kaggle_frames:
            raise PoseEstimationError(f"No pose frames found in Kaggle export {self.json_path}")

        poses: List[PoseFrame] = []
        for i, k_frame in enumerate(kaggle_frames):
            idx = k_frame.get("frame_index", i)
            ts = k_frame.get("timestamp_sec", float(i) / float(self.data.get("fps", 15.0)))
            joints_dict: Dict[str, Joint3D] = {}

            for j_name, j_data in k_frame.get("joints", {}).items():
                if isinstance(j_data, dict):
                    joints_dict[j_name] = Joint3D(
                        name=j_name,
                        x=float(j_data.get("x", 0.0)),
                        y=float(j_data.get("y", 0.0)),
                        z=float(j_data.get("z", 0.0)),
                        confidence=float(j_data.get("confidence", 1.0)),
                        visibility=float(j_data.get("visibility", 1.0)),
                    )
                elif isinstance(j_data, (list, tuple)) and len(j_data) >= 3:
                    conf = float(j_data[3]) if len(j_data) > 3 else 1.0
                    joints_dict[j_name] = Joint3D(
                        name=j_name,
                        x=float(j_data[0]),
                        y=float(j_data[1]),
                        z=float(j_data[2]),
                        confidence=conf,
                        visibility=conf,
                    )

            poses.append(
                PoseFrame(
                    frame_index=idx,
                    timestamp_sec=ts,
                    morphology=self.data.get("morphology", "human"),
                    joints=joints_dict,
                )
            )

        return poses

    def estimate(
        self,
        rgb_frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        subject_id: int = 1,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> PoseFrame:
        seq = self.estimate_sequence()
        if frame_index < len(seq):
            return seq[frame_index]
        return seq[-1]
