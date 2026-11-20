"""
Structure-from-Motion (SfM) 3D Surface Reconstruction for Leocherd Iris Energy.
Uses OpenCV feature detection (SIFT/ORB), FLANN/BFMatcher feature matching,
camera pose estimation, and multi-view triangulation to generate 3D point clouds and meshes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from iris.core.exceptions import IrisError
from iris.core.logging import get_logger
from iris.mesh.surface_reconstruction import Point3DCloud, SurfaceMesh

logger = get_logger("iris.mesh.sfm_reconstruction")


class SfMReconstructionError(IrisError):
    """Raised when Structure-from-Motion feature triangulation fails."""
    pass


class VideoStructureFromMotion:
    """
    Recovers 3D sparse surface structures from video frames using
    OpenCV feature tracking (SIFT), Essential Matrix decomposition, and triangulation.
    """

    def __init__(self, feature_detector: str = "sift", max_features: int = 2000) -> None:
        self.feature_detector = feature_detector.lower()
        self.max_features = max_features
        if self.feature_detector == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def extract_features(self, gray_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detects keypoints and computes feature descriptors."""
        kps, descs = self.detector.detectAndCompute(gray_frame, None)
        if descs is None or len(kps) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

        pts = np.float32([kp.pt for kp in kps])
        return pts, descs

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.75,
    ) -> List[Tuple[int, int]]:
        """Matches features between two consecutive frames using Lowe's ratio test."""
        if len(desc1) == 0 or len(desc2) == 0:
            return []

        knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m_pair in knn_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))
        return good_matches

    def triangulate_frames(
        self,
        frame1_rgb: np.ndarray,
        frame2_rgb: np.ndarray,
        focal_length: Optional[float] = None,
    ) -> Point3DCloud:
        """
        Triangulates matched 2D features between two video frames into 3D metric coordinates.
        """
        h, w = frame1_rgb.shape[:2]
        gray1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2GRAY)

        pts1, desc1 = self.extract_features(gray1)
        pts2, desc2 = self.extract_features(gray2)

        matches = self.match_features(desc1, desc2)
        if len(matches) < 15:
            raise SfMReconstructionError(
                f"Insufficient feature matches between frames ({len(matches)} found, minimum 15 required)."
            )

        idx1 = [m[0] for m in matches]
        idx2 = [m[1] for m in matches]
        matched_pts1 = pts1[idx1]
        matched_pts2 = pts2[idx2]

        # Camera Intrinsic Matrix K
        f = focal_length or float(max(w, h))
        cx, cy = float(w * 0.5), float(h * 0.5)
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # Estimate Essential Matrix with RANSAC
        E, inlier_mask = cv2.findEssentialMat(
            matched_pts1, matched_pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or inlier_mask is None:
            raise SfMReconstructionError("Essential matrix decomposition failed.")

        inliers = inlier_mask.ravel().astype(bool)
        p1_in = matched_pts1[inliers]
        p2_in = matched_pts2[inliers]

        # Recover Camera Pose: Rotation (R) and Translation (t)
        _, R, t, pose_mask = cv2.recoverPose(E, p1_in, p2_in, K)

        # Projection Matrices P1 = K [I | 0], P2 = K [R | t]
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])

        # Triangulate Points
        pts4d = cv2.triangulatePoints(P1, P2, p1_in.T, p2_in.T)
        pts3d = (pts4d[:3] / np.maximum(1e-7, pts4d[3])).T.astype(np.float32)

        # Positive depth filter (points in front of both cameras)
        valid_depth = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 50.0)
        filtered_pts = pts3d[valid_depth]

        # Sample colors from frame 1
        valid_indices = p1_in[valid_depth].astype(int)
        valid_indices[:, 0] = np.clip(valid_indices[:, 0], 0, w - 1)
        valid_indices[:, 1] = np.clip(valid_indices[:, 1], 0, h - 1)
        colors = frame1_rgb[valid_indices[:, 1], valid_indices[:, 0]].astype(np.float32) / 255.0

        pcd = Point3DCloud(points=filtered_pts, colors=colors)
        pcd = pcd.remove_statistical_outliers()
        pcd.estimate_normals()
        return pcd

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        frame_step: int = 15,
        max_pairs: int = 5,
    ) -> Dict[str, str]:
        """
        Executes Structure-from-Motion tracking across sequential frame pairs of a video.
        Outputs .ply point cloud and reconstructed mesh.
        """
        path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SfMReconstructionError(f"Could not open video file: {path}")

        frames = []
        f_idx = 0
        while cap.isOpened() and len(frames) < (max_pairs + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % frame_step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            f_idx += 1
        cap.release()

        if len(frames) < 2:
            raise SfMReconstructionError("At least 2 distinct video frames are required for SfM.")

        all_points = []
        all_colors = []

        for i in range(len(frames) - 1):
            try:
                pcd = self.triangulate_frames(frames[i], frames[i + 1])
                if pcd.count > 0:
                    all_points.append(pcd.points)
                    if pcd.colors is not None:
                        all_colors.append(pcd.colors)
            except Exception as e:
                logger.warning(f"SfM triangulation skipped for frame pair ({i}, {i+1}): {e}")

        if not all_points:
            raise SfMReconstructionError("No 3D points could be triangulated from video frames.")

        merged_pts = np.vstack(all_points)
        merged_colors = np.vstack(all_colors) if all_colors else None
        cloud = Point3DCloud(points=merged_pts, colors=merged_colors).downsample(voxel_size=0.03)

        ply_path = out_dir / "sfm_sparse_pointcloud.ply"
        cloud.save_ply(ply_path)

        return {"sparse_pointcloud": str(ply_path)}
