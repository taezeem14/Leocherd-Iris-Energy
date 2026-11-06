"""
ONNX Runtime adapter for MediaPipe BlazePose 3D Full-Body Landmark Estimator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from iris.core.config import get_settings
from iris.core.exceptions import ModelLoadError
from iris.core.logging import get_logger
from iris.models.downloads import ModelDownloader
from iris.pose.base import BasePoseEstimator, Joint3D, PoseFrame
from iris.pose.human import BLAZEPOSE_KEYPOINTS

logger = get_logger("iris.models.adapters.blazepose")


class ONNXBlazePoseEstimator(BasePoseEstimator):
    """Executes BlazePose 3D ONNX inference on CPU or CUDA."""

    INPUT_SIZE = 256

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        min_confidence: float = 0.35,
    ) -> None:
        super().__init__(name="ONNXBlazePoseEstimator", min_confidence=min_confidence)
        settings = get_settings()

        if model_path is None:
            downloader = ModelDownloader(settings.models_dir)
            resolved_path = downloader.ensure_model("blazepose-mediapipe-onnx")
            if resolved_path == Path("builtin") or not resolved_path.exists():
                raise ModelLoadError("Could not resolve BlazePose ONNX model file.")
            self.model_path = str(resolved_path)
        else:
            self.model_path = str(model_path)

        providers = ["CPUExecutionProvider"]
        if settings.active_profile == "CUDA_GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        logger.info(f"Loading BlazePose ONNX session from {self.model_path} (providers: {providers})")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess_crop(
        self,
        rgb_img: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Crops and pads the region of interest into a square 256x256 tensor."""
        h, w = rgb_img.shape[:2]
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            bw = bx2 - bx1
            bh = by2 - by1
            side = max(bw, bh) * 1.25  # 25% margin
            half = side / 2.0
            x1 = max(0, int(cx - half))
            y1 = max(0, int(cy - half))
            x2 = min(w, int(cx + half))
            y2 = min(h, int(cy + half))
            crop = rgb_img[y1:y2, x1:x2]
            crop_box = (float(x1), float(y1), float(x2), float(y2))
        else:
            crop = rgb_img
            crop_box = (0.0, 0.0, float(w), float(h))

        if crop.size == 0:
            crop = rgb_img
            crop_box = (0.0, 0.0, float(w), float(h))

        # Resize to 256x256
        resized = cv2.resize(crop, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        # Normalize to [0.0, 1.0]
        norm = resized.astype(np.float32) / 255.0
        # Expand batch dim -> shape: (1, 256, 256, 3)
        tensor = np.expand_dims(norm, axis=0)
        return tensor, crop_box

    def estimate(
        self,
        rgb_frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        subject_id: int = 1,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> PoseFrame:
        input_tensor, crop_box = self._preprocess_crop(rgb_frame, bbox)

        # Run forward pass
        # outputs: [0: Identity (1, 195), 1: Identity_1 (1, 1), 2: Identity_2 (seg), 3: heatmap, 4: Identity_4 (1, 117)]
        results = self.session.run(None, {self.input_name: input_tensor})

        landmarks_raw = results[0][0]  # 195 entries = 39 * 5
        pose_score = float(results[1][0, 0])
        world_landmarks = results[4][0]  # 117 entries = 39 * 3

        orig_h, orig_w = rgb_frame.shape[:2]
        cx1, cy1, cx2, cy2 = crop_box
        cw = max(1.0, cx2 - cx1)
        ch = max(1.0, cy2 - cy1)

        joints: Dict[str, Joint3D] = {}

        # 33 BlazePose landmark points
        for i in range(33):
            name = BLAZEPOSE_KEYPOINTS.get(i, f"joint_{i}")

            # 2D screen coordinates from crop
            norm_x = landmarks_raw[i * 5 + 0] / float(self.INPUT_SIZE)
            norm_y = landmarks_raw[i * 5 + 1] / float(self.INPUT_SIZE)
            # Uncrop back to original frame pixel coords
            px = cx1 + norm_x * cw
            py = cy1 + norm_y * ch

            # Normalize to frame dimensions [0.0, 1.0]
            screen_x = px / orig_w
            screen_y = py / orig_h

            # Metric 3D world coordinates (meters)
            world_x = float(world_landmarks[i * 3 + 0])
            world_y = float(-world_landmarks[i * 3 + 1])  # Invert Y to conform to 3D standard (+Y up)
            world_z = float(-world_landmarks[i * 3 + 2])  # Invert Z for right-handed coordinate convention

            # Confidence & visibility (apply sigmoid to raw logits)
            raw_vis = float(landmarks_raw[i * 5 + 3])
            raw_pres = float(landmarks_raw[i * 5 + 4])
            vis_prob = float(1.0 / (1.0 + np.exp(-np.clip(raw_vis, -15.0, 15.0))))
            pres_prob = float(1.0 / (1.0 + np.exp(-np.clip(raw_pres, -15.0, 15.0))))
            conf = min(1.0, max(0.0, pose_score * vis_prob))

            joints[name] = Joint3D(
                name=name,
                x=round(world_x, 4),
                y=round(world_y, 4),
                z=round(world_z, 4),
                confidence=round(conf, 3),
                visibility=round(pres_prob, 3),
            )

        return PoseFrame(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            subject_id=subject_id,
            morphology="human",
            joints=joints,
            bounding_box=crop_box,
        )
